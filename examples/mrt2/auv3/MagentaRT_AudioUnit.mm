// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Magenta RT AUv3 Audio Unit — decoder transformer generator (no audio input).
// Uses sequence-layers exported .mlxfn for autoregressive generation.

#import "MagentaRT_AudioUnit.h"
#import <AVFoundation/AVFoundation.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import "MagentaModelManager.h"
#import "MagentaModelDownloader.h"
#include "magenta_paths.h"
#include "audio_level_processor.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using magentart::core::EngineMetrics;

// ─── Dev server probe ────────────────────────────────────────────────────────

static const int kDevServerPort = 62420;

static BOOL isDevServerRunning(void) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return NO;
    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 }; // 100ms
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(kDevServerPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    BOOL up = (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(sock);
    return up;
}

@interface MagentaRTAudioUnit ()
#if MAGENTART_DEBUG_LOG
@property (nonatomic, copy) void (^debugLogHandler)(NSString *);
#endif
@property (nonatomic, strong) NSMutableArray* logHistory;
@end

@implementation MagentaRTAudioUnit {
    RealtimeRunner _engine;
    AUParameterTree* _parameterTree;
    AUAudioUnitBus* _outputBus;
    AUAudioUnitBusArray* _outputBusArray;
    BOOL _modelLoaded;
    AudioConverterRef _resampler;
    float* _resampleBufferL;
    float* _resampleBufferR;
    float* _resampleBufferInterleaved;
    BOOL _isOffline;                 // current state, read by render block
    // Transport and musical context block caching. The host may set these after
    // internalRenderBlock is called, so we can't capture them there. The
    // metrics timer (main thread) caches them once they appear. We keep a
    // strong reference so the blocks stay alive even if the host later
    // sets the property to nil (e.g. during/after offline export).
    AUHostTransportStateBlock _retainedTransportBlock;
    void* _transportBlockPtr;        // raw pointer for render thread (no ARC)
    AUHostMusicalContextBlock _retainedMusicalContextBlock;
    void* _musicalContextBlockPtr;   // raw pointer for render thread (no ARC)
    NSMutableArray* _pendingLogs;
    std::atomic<bool> _midiNotes[128];
    magentart::common::AudioLevelProcessor _levelProcessor;
}

// Fallback init — the extension system may call plain init before the factory method.
// Redirect to the designated initializer with our registered component description.
- (instancetype)init {
    AudioComponentDescription desc = {
        .componentType = kAudioUnitType_MusicDevice,
        .componentSubType = 'MGRT',
        .componentManufacturer = 'Goog',
        .componentFlags = 0,
        .componentFlagsMask = 0
    };
    return [self initWithComponentDescription:desc options:0 error:nil];
}

- (instancetype)initWithComponentDescription:(AudioComponentDescription)componentDescription
                                     options:(AudioComponentInstantiationOptions)options
                                       error:(NSError**)outError {
    self = [super initWithComponentDescription:componentDescription
                                       options:options
                                         error:outError];
    if (!self) return nil;

    _modelLoaded = NO;

    for (int i = 0; i < 128; i++) {
        _midiNotes[i].store(false, std::memory_order_relaxed);
    }

    auto makeParam = ^(NSString* ident, NSString* name, AUParameterAddress addr, float min, float max, float def) {
        AUParameter* p = [AUParameterTree
            createParameterWithIdentifier:ident name:name address:addr min:min max:max
            unit:kAudioUnitParameterUnit_Generic unitName:nil
            flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
            valueStrings:nil dependentParameters:nil];
        p.value = def;
        return p;
    };

    AUParameter* tempParam = makeParam(@"temperature", @"Temperature", 0, 0.0, 3.0, 1.3);
    AUParameter* topkParam = makeParam(@"topk", @"Top-K", 1, 1, 1024, 40);
    AUParameter* cfgMusicCoCaParam = makeParam(@"cfgmusiccoca", @"Prompt Adherence", 3, -1.0, 7.0, 3.0);
    AUParameter* cfgNotesParam = makeParam(@"cfgnotes", @"Note Adherence", 4, -1.0, 7.0, 1.0);
    AUParameter* volParam = makeParam(@"volume", @"Volume", 5, -60.0, 12.0, 0.0);

    AUParameter* muteParam = [AUParameterTree
        createParameterWithIdentifier:@"mute" name:@"Mute" address:6 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    muteParam.value = 0.0;

    AUParameter* unmaskWidthParam = makeParam(@"unmaskwidth", @"Unmask width", 7, 0, 127, 0);

    NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
    float savedBufSize = [defaults objectForKey:@"MagentaRT_AU_BufferSize"] ? [defaults floatForKey:@"MagentaRT_AU_BufferSize"] : 0.0f;

    AUParameter* bufSizeParam = [AUParameterTree
        createParameterWithIdentifier:@"buffersize" name:@"Buffer Size" address:8 min:0.0 max:2.0
        unit:kAudioUnitParameterUnit_Indexed unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:@[@"2048", @"4096", @"8192"] dependentParameters:nil];
    bufSizeParam.value = savedBufSize;

    // Initialize C++ engine's buffer size to match
    size_t initialCap = 8192;
    if (savedBufSize < 0.5f) initialCap = 2048;
    else if (savedBufSize < 1.5f) initialCap = 4096;
    _engine.set_buffer_size(initialCap);

    _engine.set_latency_comp(true);

    AUParameter* latencyCompParam = [AUParameterTree
        createParameterWithIdentifier:@"latencycomp" name:@"Latency Comp" address:9 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    latencyCompParam.value = 1.0; // Default on

    // Blend weight parameters (addresses 10-15)
    NSMutableArray* weightParams = [NSMutableArray array];
    for (int i = 0; i < 6; i++) {
        NSString* ident = [NSString stringWithFormat:@"weight_%d", i];
        NSString* name = [NSString stringWithFormat:@"Weight %d", i];
        [weightParams addObject:makeParam(ident, name, 10 + i, 0.0, 1.0, 0.0)];
    }

    // Reset state (edge-detected boolean, address 31)
    AUParameter* resetParam = [AUParameterTree
        createParameterWithIdentifier:@"resetstate" name:@"Reset State" address:31 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    resetParam.value = 0.0;

    // Bypass (address 32)
    AUParameter* bypassParam = [AUParameterTree
        createParameterWithIdentifier:@"bypass" name:@"Bypass" address:32 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    bypassParam.value = 0.0;

    // Drumless (address 39)
    AUParameter* drumlessParam = [AUParameterTree
        createParameterWithIdentifier:@"drumless" name:@"Filter Drums" address:39 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    drumlessParam.value = 0.0;

    AUParameter* midiGateParam = [AUParameterTree
        createParameterWithIdentifier:@"midigate" name:@"MIDI Gate" address:45 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Boolean unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    midiGateParam.value = 0.0;

    AUParameter* onsetModeParam = [AUParameterTree
        createParameterWithIdentifier:@"onsetmode" name:@"Onset Mode" address:46 min:0.0 max:1.0
        unit:kAudioUnitParameterUnit_Indexed unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:@[@"Mask", @"Unmask"] dependentParameters:nil];
    onsetModeParam.value = 0.0;

    // Seed Rotation (address 47)
    AUParameter* seedRotationParam = [AUParameterTree
        createParameterWithIdentifier:@"seedrotation" name:@"Seed Rotation" address:47 min:0.0 max:1000.0
        unit:kAudioUnitParameterUnit_Indexed unitName:nil
        flags:kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable
        valueStrings:nil dependentParameters:nil];
    seedRotationParam.value = 0.0;

    AUParameter* cfgDrumsParam = makeParam(@"cfgdrums", @"Drums Adherence", 48, -1.0, 7.0, 1.0);

    NSMutableArray* allParams = [NSMutableArray arrayWithArray:@[
        tempParam, topkParam, cfgMusicCoCaParam, cfgNotesParam, volParam, muteParam, unmaskWidthParam, bufSizeParam, latencyCompParam,
        cfgDrumsParam
    ]];
    [allParams addObjectsFromArray:weightParams];
    [allParams addObjectsFromArray:@[resetParam, bypassParam, seedRotationParam]];
    [allParams addObject:drumlessParam];
    [allParams addObject:midiGateParam];
    [allParams addObject:onsetModeParam];

    _parameterTree = [AUParameterTree createTreeWithChildren:allParams];

    __unsafe_unretained MagentaRTAudioUnit* weakSelf = self;
    _parameterTree.implementorValueObserver = ^(AUParameter* param, AUValue value) {
        if (param.address == 0) weakSelf->_engine.set_temperature(value);
        else if (param.address == 1) weakSelf->_engine.set_top_k((int)value);
        else if (param.address == 3) weakSelf->_engine.set_cfg_musiccoca(value);
        else if (param.address == 4) weakSelf->_engine.set_cfg_notes(value);
        else if (param.address == 5) weakSelf->_engine.set_volume_db(value);
        else if (param.address == 6) weakSelf->_engine.set_mute(value > 0.5f);
        else if (param.address == 7) weakSelf->_engine.set_unmask_width((int)value);
        else if (param.address == 8) {
            size_t cap = 8192;
            if (value < 0.5f) cap = 2048;
            else if (value < 1.5f) cap = 4096;
            weakSelf->_engine.set_buffer_size(cap);
            [[NSUserDefaults standardUserDefaults] setFloat:value forKey:@"MagentaRT_AU_BufferSize"];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf willChangeValueForKey:@"latency"];
                [weakSelf didChangeValueForKey:@"latency"];
            });
        }
        else if (param.address == 9) {
            weakSelf->_engine.set_latency_comp(value > 0.5f);
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf willChangeValueForKey:@"latency"];
                [weakSelf didChangeValueForKey:@"latency"];
            });
        }
        else if (param.address >= 10 && param.address <= 15) weakSelf->_engine.set_blend_weight((int)param.address - 10, value);
        else if (param.address == 31) {
            if (value > 0.5f) weakSelf->_engine.trigger_reset();
        }
        else if (param.address == 32) weakSelf->_engine.set_bypass(value > 0.5f);
        else if (param.address == 39) weakSelf->_engine.set_drumless(value > 0.5f);
        else if (param.address == 45) weakSelf->_engine.set_midi_gate_enabled(value > 0.5f);
        else if (param.address == 46) weakSelf->_engine.set_onset_mode(value > 0.5f);
        else if (param.address == 48) weakSelf->_engine.set_cfg_drums(value);
        else if (param.address == 47) weakSelf->_engine.set_seed_rotation((int)value);
    };
    _parameterTree.implementorValueProvider = ^AUValue(AUParameter* param) {
        if (param.address == 0) return weakSelf->_engine.get_temperature();
        else if (param.address == 1) return (AUValue)weakSelf->_engine.get_top_k();
        else if (param.address == 3) return weakSelf->_engine.get_cfg_musiccoca();
        else if (param.address == 4) return weakSelf->_engine.get_cfg_notes();
        else if (param.address == 5) return weakSelf->_engine.get_volume_db();
        else if (param.address == 6) return weakSelf->_engine.get_mute() ? 1.0f : 0.0f;
        else if (param.address == 7) return (AUValue)weakSelf->_engine.get_unmask_width();
        else if (param.address == 8) {
            size_t cap = weakSelf->_engine.get_buffer_size();
            if (cap <= 2048) return 0.0f;
            if (cap <= 4096) return 1.0f;
            return 2.0f;
        }
        else if (param.address == 9) return weakSelf->_engine.get_latency_comp() ? 1.0f : 0.0f;
        else if (param.address >= 10 && param.address <= 15) return weakSelf->_engine.get_blend_weight((int)param.address - 10);
        else if (param.address == 31) return 0.0f; // reset is momentary
        else if (param.address == 32) return weakSelf->_engine.get_bypass() ? 1.0f : 0.0f;
        else if (param.address == 39) return weakSelf->_engine.get_drumless() ? 1.0f : 0.0f;
        else if (param.address == 45) return weakSelf->_engine.get_midi_gate_enabled() ? 1.0f : 0.0f;
        else if (param.address == 46) return weakSelf->_engine.get_onset_mode() ? 1.0f : 0.0f;
        else if (param.address == 48) return weakSelf->_engine.get_cfg_drums();
        else if (param.address == 47) return (AUValue)weakSelf->_engine.get_seed_rotation();
        return 0.0;
    };

    // Output bus: stereo 48000 Hz float
    AVAudioFormat* format = [[AVAudioFormat alloc]
        initStandardFormatWithSampleRate:48000.0 channels:2];
    NSError* busError = nil;
    _outputBus = [[AUAudioUnitBus alloc] initWithFormat:format error:&busError];
    if (busError) {
        if (outError) *outError = busError;
        return nil;
    }
    _outputBusArray = [[AUAudioUnitBusArray alloc] initWithAudioUnit:self
                                                              busType:AUAudioUnitBusTypeOutput
                                                               busses:@[_outputBus]];

    // Load tokenizer and models externally from custom path or ~/Documents/Magenta/resources/ to keep bundle size tiny
    NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
    std::string resourcesPath = customResources ? std::string(customResources.UTF8String) : magentart::paths::get_resources_dir();
    _modelLoaded = _engine.init_assets(resourcesPath.c_str());
    if (_modelLoaded) {
      self.musicCocaModelName = @"musiccoca";
      _engine.load_musiccoca_model(resourcesPath.c_str(), "musiccoca");
    } else {
        NSLog(@"MagentaRT_AU: Failed to load static assets externally from: %s", resourcesPath.c_str());
    }

    self.maximumFramesToRender = 4096;

    return self;
}

- (RealtimeRunner*)engine {
    return &_engine;
}

- (void)pollOfflineState {
    _isOffline = self.isRenderingOffline;

    // Cache the transport block once it becomes available.  The host may
    // set it after internalRenderBlock is called, and may remove it during
    // offline export.  We retain it permanently so the render block can
    // always query transport state.
    if (!_transportBlockPtr) {
        AUHostTransportStateBlock tb = self.transportStateBlock;
        if (tb) {
            _retainedTransportBlock = tb;              // keeps it alive
            _transportBlockPtr = (__bridge void*)tb;   // for render thread
        }
    }

    // Cache the musical context block for the same reasons.
    if (!_musicalContextBlockPtr) {
        AUHostMusicalContextBlock mcb = self.musicalContextBlock;
        if (mcb) {
            _retainedMusicalContextBlock = mcb;
            _musicalContextBlockPtr = (__bridge void*)mcb;
        }
    }
}

- (void)setNoteOn:(uint8_t)note on:(BOOL)on {
    if (note < 128) {
        _midiNotes[note].store(on, std::memory_order_relaxed);
    }
}

- (NSArray<NSNumber*>*)activeNotes {
    NSMutableArray* notes = [NSMutableArray array];
    for (int i = 0; i < 128; i++) {
        if (_midiNotes[i].load(std::memory_order_relaxed)) {
            [notes addObject:@(i)];
        }
    }
    return notes;
}

- (void)readAudioLevels:(float*)outLeft right:(float*)outRight {
    _levelProcessor.read_and_reset_peaks(*outLeft, *outRight);
}


- (NSTimeInterval)latency {
    return _engine.get_latency_samples() / 48000.0;
}

// Called by the host when switching between online/offline rendering.
// Updates _isOffline immediately so the render block uses blocking reads
// from the very first offline render call.
- (void)setRenderingOffline:(BOOL)renderingOffline {
    [super setRenderingOffline:renderingOffline];
    _isOffline = renderingOffline;
}

- (BOOL)shouldBypassEffect {
    return _engine.get_host_bypass();
}

- (void)setShouldBypassEffect:(BOOL)shouldBypassEffect {
    [super setShouldBypassEffect:shouldBypassEffect];
    _engine.set_host_bypass(shouldBypassEffect);
}

- (void)dealloc {
    _engine.stop();
    _engine.unload();
}

// --- Bus Arrays ---------------------------------------------------------------

- (AUAudioUnitBusArray*)outputBusses {
    return _outputBusArray;
}

// --- Parameter Tree -----------------------------------------------------------

- (AUParameterTree*)parameterTree {
    return _parameterTree;
}

// --- State Serialization ------------------------------------------------------

- (void)applyCustomState:(NSDictionary<NSString *, id> *)state {
    if (state[@"MGRT_Prompts"]) self.prompts = state[@"MGRT_Prompts"];
    if (state[@"MGRT_ModelName"]) self.modelName = state[@"MGRT_ModelName"];
    self.musicCocaModelName = @"musiccoca";
    NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
    std::string loadPathStr = customResources ? std::string(customResources.UTF8String) : magentart::paths::get_resources_dir();
    _engine.load_musiccoca_model(loadPathStr.c_str(), "musiccoca");
    if (state[@"MGRT_PromptSurface"]) self.promptSurfaceState = state[@"MGRT_PromptSurface"];
    if (state[@"MGRT_StatePrefix"]) self.statePrefix = state[@"MGRT_StatePrefix"];

    if (state[@"MGRT_AudioEmbeddings"]) {
        NSDictionary* audioEmbeddings = state[@"MGRT_AudioEmbeddings"];
        for (NSString* key in audioEmbeddings) {
            int index = key.intValue;
            NSData* data = audioEmbeddings[key];
            if (data.length == 768 * sizeof(float)) {
                self->_engine.set_audio_embedding(index, (const float*)data.bytes);
            }
        }
    }

    if (state[@"MGRT_ModelBookmark"]) {
        self.modelBookmark = state[@"MGRT_ModelBookmark"];

        // Resolve security scoped bookmark
        BOOL isStale = NO;
        NSError* error = nil;
        NSURL* url = [NSURL URLByResolvingBookmarkData:self.modelBookmark
                                               options:NSURLBookmarkResolutionWithoutUI | NSURLBookmarkResolutionWithSecurityScope
                                         relativeToURL:nil
                                   bookmarkDataIsStale:&isStale
                                                 error:&error];
        if (url && [url startAccessingSecurityScopedResource]) {
            NSString* path = url.path;
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir];

            NSString* mlxfnPath = nil;
            if ([path hasSuffix:@".mlxfn"]) {
                mlxfnPath = path;
            } else if (isDir) {
                NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:path error:nil];
                for (NSString *file in contents) {
                    if ([file hasSuffix:@".mlxfn"]) {
                        mlxfnPath = [path stringByAppendingPathComponent:file];
                        break;
                    }
                }
            }

            if (mlxfnPath) {
                // Skip reload if the engine already has a model loaded —
                // setFullState: can be called during parameter automation
                // (undo management) and must not tear down a running model.
                if (self->_engine.is_loaded()) {
                    [url stopAccessingSecurityScopedResource];
                } else {
                // Perform async load so we don't block AU initialization
                dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                    BOOL success = self->_engine.load_model(mlxfnPath.UTF8String);
                    if (success) {
                        if (self.prompts) {
                            std::vector<std::string> std_texts;
                            std::vector<float> std_weights;
                            for (NSDictionary* p in self.prompts) {
                                NSString* text = p[@"text"];
                                NSNumber* weight = p[@"weight"];
                                BOOL isValid = [text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]];
                                std_texts.push_back(isValid ? text.UTF8String : "");
                                std_weights.push_back(isValid ? weight.floatValue : 0.0f);
                            }
                            self->_engine.set_text_prompts(std_texts, std_weights);
                            self->_engine.set_blend_weights(std_weights.data(), (int)std_weights.size());
                            // Sync to AU parameter tree
                            dispatch_async(dispatch_get_main_queue(), ^{
                                for (int i = 0; i < (int)std_weights.size() && i < 6; i++) {
                                    AUParameter* wp = [self->_parameterTree parameterWithAddress:10 + i];
                                    if (wp) [wp setValue:std_weights[i] originator:nil];
                                }
                            });
                        }
                        NSLog(@"MagentaRT_AU: Successfully auto-loaded model from bookmark.");

                        // Load SpectroStream encoder: model dir → external spectrostream → bundle
                        NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
                        NSString* spectrostreamPath = [parentDir stringByAppendingPathComponent:@"spectrostream_encoder.mlxfn"];
                        if ([[NSFileManager defaultManager] fileExistsAtPath:spectrostreamPath]) {
                            NSLog(@"MagentaRT_AU: Auto-loading spectrostream encoder from model dir: %@", spectrostreamPath.lastPathComponent);
                            self->_engine.load_prefill_model(spectrostreamPath.UTF8String, nullptr);
                        } else {
                            std::string extPath = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";
                            NSString* extNSPath = [NSString stringWithUTF8String:extPath.c_str()];
                            if ([[NSFileManager defaultManager] fileExistsAtPath:extNSPath]) {
                                NSLog(@"MagentaRT_AU: Auto-loading spectrostream encoder from external path: %@", extNSPath);
                                self->_engine.load_prefill_model(extNSPath.UTF8String, nullptr);
                            } else {
                                NSString* fallbackPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"spectrostream_encoder" ofType:@"mlxfn"];
                                if (fallbackPath) {
                                    NSLog(@"MagentaRT_AU: Auto-loading spectrostream encoder from bundle: %@", fallbackPath.lastPathComponent);
                                    self->_engine.load_prefill_model(fallbackPath.UTF8String, nullptr);
                                }
                            }
                        }
                    } else {
                        NSLog(@"MagentaRT_AU: Failed to auto-load model from bookmark.");
                    }
                    [url stopAccessingSecurityScopedResource];
                });
                } // else (not already loaded)
            } else {
                [url stopAccessingSecurityScopedResource];
            }
        } else {
            NSLog(@"MagentaRT_AU: Failed to resolve bookmark: %@", error);
        }
    }
}

- (NSDictionary<NSString *,id> *)fullState {
    NSMutableDictionary *state = [[super fullState] mutableCopy];
    if (!state) state = [NSMutableDictionary dictionary];

    if (self.prompts) state[@"MGRT_Prompts"] = self.prompts;
    if (self.modelName) state[@"MGRT_ModelName"] = self.modelName;
    if (self.musicCocaModelName) state[@"MGRT_MusicCoCaModelName"] = self.musicCocaModelName;
    if (self.modelBookmark) state[@"MGRT_ModelBookmark"] = self.modelBookmark;
    if (self.promptSurfaceState) state[@"MGRT_PromptSurface"] = self.promptSurfaceState;
    if (self.statePrefix) state[@"MGRT_StatePrefix"] = self.statePrefix;

    NSMutableDictionary* audioEmbeddings = [NSMutableDictionary dictionary];
    for (int i = 0; i < 6; ++i) {
        float buffer[768];
        if (self->_engine.get_audio_embedding(i, buffer)) {
            NSData* data = [NSData dataWithBytes:buffer length:768 * sizeof(float)];
            audioEmbeddings[[NSString stringWithFormat:@"%d", i]] = data;
        }
    }
    if (audioEmbeddings.count > 0) state[@"MGRT_AudioEmbeddings"] = audioEmbeddings;

    return state;
}

- (void)setFullState:(NSDictionary<NSString *,id> *)state {
    [super setFullState:state];
    [self applyCustomState:state];
}

- (NSDictionary<NSString *,id> *)fullStateForDocument {
    NSMutableDictionary *state = [[super fullStateForDocument] mutableCopy];
    if (!state) state = [NSMutableDictionary dictionary];

    if (self.prompts) state[@"MGRT_Prompts"] = self.prompts;
    if (self.modelName) state[@"MGRT_ModelName"] = self.modelName;
    if (self.modelBookmark) state[@"MGRT_ModelBookmark"] = self.modelBookmark;
    if (self.promptSurfaceState) state[@"MGRT_PromptSurface"] = self.promptSurfaceState;

    NSMutableDictionary* audioEmbeddings = [NSMutableDictionary dictionary];
    for (int i = 0; i < 6; ++i) {
        float buffer[768];
        if (self->_engine.get_audio_embedding(i, buffer)) {
            NSData* data = [NSData dataWithBytes:buffer length:768 * sizeof(float)];
            audioEmbeddings[[NSString stringWithFormat:@"%d", i]] = data;
        }
    }
    if (audioEmbeddings.count > 0) state[@"MGRT_AudioEmbeddings"] = audioEmbeddings;

    return state;
}

- (void)setFullStateForDocument:(NSDictionary<NSString *,id> *)state {
    [super setFullStateForDocument:state];
    [self applyCustomState:state];
}


// --- Lifecycle ----------------------------------------------------------------

- (BOOL)allocateRenderResourcesAndReturnError:(NSError**)outError {
    if (![super allocateRenderResourcesAndReturnError:outError]) return NO;

    [self pollOfflineState]; // Cache transport block early!
    NSString* logMsg = @"allocateRenderResourcesAndReturnError called";
    if (!self.logHistory) {
        self.logHistory = [NSMutableArray array];
    }
    [self.logHistory addObject:logMsg];
    if (self.logHistory.count > 1000) {
        [self.logHistory removeObjectAtIndex:0];
    }

#if MAGENTART_DEBUG_LOG
    if (self.debugLogHandler) {
        self.debugLogHandler(logMsg);
    }
#endif

    if (!_modelLoaded) {
        NSLog(@"MagentaRT_AU: Assets not fully loaded, but continuing allocation to allow UI interaction.");
    }


    double outSampleRate = self.outputBusses[0].format.sampleRate;
    if (std::abs(outSampleRate - 48000.0) > 1.0) {
        AudioStreamBasicDescription outDesc = *self.outputBusses[0].format.streamDescription;
        AudioStreamBasicDescription inDesc;
        inDesc.mSampleRate = 48000.0;
        inDesc.mFormatID = kAudioFormatLinearPCM;
        inDesc.mFormatFlags = static_cast<UInt32>(kAudioFormatFlagIsFloat) | static_cast<UInt32>(kAudioFormatFlagsNativeEndian);
        inDesc.mBytesPerPacket = 8;
        inDesc.mFramesPerPacket = 1;
        inDesc.mBytesPerFrame = 8;
        inDesc.mChannelsPerFrame = 2;
        inDesc.mBitsPerChannel = 32;

        OSStatus err = AudioConverterNew(&inDesc, &outDesc, &_resampler);
        if (err != noErr) {
            NSLog(@"MagentaRT_AU: AudioConverterNew failed with error %d", (int)err);
            if (outError) *outError = [NSError errorWithDomain:NSOSStatusErrorDomain code:err userInfo:nil];
            return NO;
        }

        // maximumFramesToRender could be max from host like 4096.
        // 8192 is safe up to ~96kHz -> 44.1kHz downsampling blocks
        _resampleBufferL = (float*)calloc(8192, sizeof(float));
        _resampleBufferR = (float*)calloc(8192, sizeof(float));
        _resampleBufferInterleaved = (float*)calloc(16384, sizeof(float));
    }

    _engine.start();
    return YES;
}

- (void)deallocateRenderResources {
    _engine.stop();
    if (_resampler) {
        AudioConverterDispose(_resampler);
        _resampler = NULL;
    }
    if (_resampleBufferL) {
        free(_resampleBufferL);
        _resampleBufferL = NULL;
    }
    if (_resampleBufferR) {
        free(_resampleBufferR);
        _resampleBufferR = NULL;
    }
    [super deallocateRenderResources];
}

// --- Render -------------------------------------------------------------------

struct ResamplerContext {
    RealtimeRunner* engine;
    float* tempL;
    float* tempR;
    float* tempInterleaved;
    UInt32 maxFrames;
    bool blocking;
};

static OSStatus ConverterDataProc(AudioConverterRef inAudioConverter,
                                  UInt32 *ioNumberDataPackets,
                                  AudioBufferList *ioData,
                                  AudioStreamPacketDescription **outDataPacketDescription,
                                  void *inUserData) {
    ResamplerContext* ctx = (ResamplerContext*)inUserData;

    UInt32 framesToRead = *ioNumberDataPackets;
    if (framesToRead > ctx->maxFrames) {
        framesToRead = ctx->maxFrames;
    }

    if (framesToRead == 0) {
        return noErr;
    }

    ctx->engine->read_audio_stereo(ctx->tempL, ctx->tempR, framesToRead,
                                    ctx->blocking);

    // Interleave data
    float* interleaved = ctx->tempInterleaved;
    for (UInt32 i = 0; i < framesToRead; ++i) {
        interleaved[i * 2] = ctx->tempL[i];
        interleaved[i * 2 + 1] = ctx->tempR[i];
    }

    ioData->mBuffers[0].mData = interleaved;
    ioData->mBuffers[0].mDataByteSize = framesToRead * 2 * sizeof(float);
    ioData->mBuffers[0].mNumberChannels = 2;

    *ioNumberDataPackets = framesToRead;

    if (outDataPacketDescription) {
        *outDataPacketDescription = NULL;
    }

    return noErr;
}

- (AUInternalRenderBlock)internalRenderBlock {
    // Capture raw pointers — safe: their lifetimes span allocate → deallocate.
    __unsafe_unretained MagentaRTAudioUnit* unsafeSelf = self;
    RealtimeRunner* engine = &_engine;

    // Snapshot offline state for blocking reads.
    _isOffline = self.isRenderingOffline;

    __block BOOL wasPlaying = YES;
    __block BOOL wasResetHigh = NO;
    __block BOOL wasBeatZero = NO;
    __block BOOL wasDawPlaying = NO;

    return ^AUAudioUnitStatus(AudioUnitRenderActionFlags* actionFlags,
                               const AudioTimeStamp* timestamp,
                               AUAudioFrameCount frameCount,
                               NSInteger outputBusNumber,
                               AudioBufferList* outputData,
                               const AURenderEvent* realtimeEventListHead,
                               AURenderPullInputBlock __unsafe_unretained pullInputBlock) {

        // Read pointers dynamically to avoid capturing NULL if called before allocate.
        AudioConverterRef resampler = unsafeSelf->_resampler;
        float* tempL = unsafeSelf->_resampleBufferL;
        float* tempR = unsafeSelf->_resampleBufferR;
        float* tempInterleaved = unsafeSelf->_resampleBufferInterleaved;
        // Process parameter and MIDI events
        for (const AURenderEvent* event = realtimeEventListHead;
             event != nullptr; event = event->head.next) {
            if (event->head.eventType == AURenderEventParameter) {
                const AUParameterEvent& paramEvent = event->parameter;
                if (paramEvent.parameterAddress == 0) engine->set_temperature(paramEvent.value);
                else if (paramEvent.parameterAddress == 1) engine->set_top_k((int)paramEvent.value);
                else if (paramEvent.parameterAddress == 3) engine->set_cfg_musiccoca(paramEvent.value);
                else if (paramEvent.parameterAddress == 4) engine->set_cfg_notes(paramEvent.value);
                else if (paramEvent.parameterAddress == 5) engine->set_volume_db(paramEvent.value);
                else if (paramEvent.parameterAddress == 6) engine->set_mute(paramEvent.value > 0.5f);
                else if (paramEvent.parameterAddress == 7) engine->set_unmask_width((int)paramEvent.value);
                else if (paramEvent.parameterAddress == 8) {
                    size_t cap = 8192;
                    if (paramEvent.value < 0.5f) cap = 2048;
                    else if (paramEvent.value < 1.5f) cap = 4096;
                    engine->set_buffer_size(cap);
                }
                else if (paramEvent.parameterAddress == 9) {
                    engine->set_latency_comp(paramEvent.value > 0.5f);
                }
                else if (paramEvent.parameterAddress >= 10 && paramEvent.parameterAddress <= 15) engine->set_blend_weight((int)paramEvent.parameterAddress - 10, paramEvent.value);
                else if (paramEvent.parameterAddress == 31) {
                    bool isHigh = paramEvent.value > 0.5f;
                    if (isHigh && !wasResetHigh) engine->trigger_reset();
                    wasResetHigh = isHigh;
                }
                else if (paramEvent.parameterAddress == 32) {
                    engine->set_bypass(paramEvent.value > 0.5f);
                }
                else if (paramEvent.parameterAddress == 39) {
                    engine->set_drumless(paramEvent.value > 0.5f);
                }
                else if (paramEvent.parameterAddress == 45) engine->set_midi_gate_enabled(paramEvent.value > 0.5f);
                else if (paramEvent.parameterAddress == 46) engine->set_onset_mode(paramEvent.value > 0.5f);
                else if (paramEvent.parameterAddress == 48) engine->set_cfg_drums(paramEvent.value);
            } else if (event->head.eventType == AURenderEventMIDI) {
                const AUMIDIEvent& midiEvent = event->MIDI;
                uint8_t status = midiEvent.data[0] & 0xF0;
                uint8_t note = midiEvent.data[1];
                uint8_t velocity = midiEvent.data[2];
                if (status == 0x90 && velocity > 0) { // Note On
                    engine->set_note_on(note);
                    if (note < 128) unsafeSelf->_midiNotes[note].store(true, std::memory_order_relaxed);
                } else if (status == 0x80 || (status == 0x90 && velocity == 0)) { // Note Off
                    engine->set_note_off(note);
                    if (note < 128) unsafeSelf->_midiNotes[note].store(false, std::memory_order_relaxed);
                }

            }
        }

        // Read offline flag from the ivar (plain C memory, no ObjC messaging).
        // Updated every ~200ms by pollOfflineState on the main thread.
        bool isOffline = unsafeSelf->_isOffline;
        engine->set_offline(isOffline);

        // Read the transport block from the cached raw pointer (set once
        // by pollOfflineState on the main thread, never freed).
        __unsafe_unretained AUHostTransportStateBlock transportBlock =
            (__bridge AUHostTransportStateBlock)(unsafeSelf->_transportBlockPtr);

        BOOL isDawPlaying = NO;
        BOOL isPlaying = unsafeSelf->_uiPlaying; // Start with UI play state
        if (transportBlock) {
            AUHostTransportStateFlags flags = 0;
            double currentSamplePosition = 0;
            double cycleStartBeatPosition = 0;
            double cycleEndBeatPosition = 0;
            if (transportBlock(&flags, &currentSamplePosition, &cycleStartBeatPosition, &cycleEndBeatPosition)) {
                isDawPlaying = (flags & AUHostTransportStateMoving) != 0;
                isPlaying = isPlaying || isDawPlaying;
                engine->set_transport_flags((int)flags);
            } else {
                engine->set_transport_flags(-3);
            }
        } else {
            engine->set_transport_flags(-2);
        }

        // If DAW was playing and has now stopped, stop the Audio Unit's playback as well.
        if (wasDawPlaying && !isDawPlaying) {
            unsafeSelf->_uiPlaying = NO;
            isPlaying = NO;
        }
        wasDawPlaying = isDawPlaying;

        // Edge detection from stopped to playing
        if (isPlaying && !wasPlaying) {
            engine->reset_for_playback();
            if (resampler) {
                AudioConverterReset(resampler);
            }
        }
        wasPlaying = isPlaying;

        // Auto-reset when currentBeatPosition is exactly 0. This edge-detects the
        // transition to beat 0.0 so that resets are only triggered once (e.g., when the
        // timeline loops back to start, but not repeatedly if user pauses at the start).
        __unsafe_unretained AUHostMusicalContextBlock musicalContextBlock =
            (__bridge AUHostMusicalContextBlock)(unsafeSelf->_musicalContextBlockPtr);

        if (musicalContextBlock) {
            double currentBeatPosition = 0;
            if (musicalContextBlock(NULL, NULL, NULL, &currentBeatPosition, NULL, NULL)) {
                if (currentBeatPosition == 0.0 && !wasBeatZero) {
                    // Transport-rewind reset (DAW timeline jumped back to
                    // beat 0). Suppressible: a freshly-prefilled context
                    // arms a one-shot skip so re-cuing the DAW after
                    // clicking Audio Prefill / Silent Prefill doesn't wipe
                    // the prefill. User-initiated resets (resetModel,
                    // param 31) take a different path and are never
                    // suppressed.
                    engine->trigger_transport_reset();
                }
                wasBeatZero = (currentBeatPosition == 0.0);
            }
        }

        float* outL = (float*)outputData->mBuffers[0].mData;
        float* outR = outputData->mNumberBuffers > 1 ? (float*)outputData->mBuffers[1].mData : outL;
        // When bypass is active, the engine writes zeros.  Signal
        // OutputIsSilence so the host can stop calling us — this may
        // restore Ableton's pre-export behavior of not rendering while
        // the transport is stopped.
        bool isBypassed = engine->get_bypass();

        if (isPlaying && !isBypassed) {
            if (resampler) {
                ResamplerContext ctx;
                ctx.engine = engine;
                ctx.tempL = tempL;
                ctx.tempR = tempR;
                ctx.tempInterleaved = tempInterleaved;
                ctx.maxFrames = 8192;
                ctx.blocking = isOffline;

                UInt32 outFrames = frameCount;
                OSStatus err = AudioConverterFillComplexBuffer(resampler, ConverterDataProc, &ctx, &outFrames, outputData, NULL);
                if (err != noErr) {
                    NSLog(@"MagentaRT_AU: AudioConverterFillComplexBuffer failed with error %d", (int)err);
                }
            } else {
                engine->read_audio_stereo(outL, outR, frameCount, isOffline);
            }
            unsafeSelf->_levelProcessor.process_block(outL, outR, frameCount);
        } else {
            std::memset(outL, 0, frameCount * sizeof(float));
            if (outputData->mNumberBuffers > 1) {
                std::memset(outR, 0, frameCount * sizeof(float));
            }
            *actionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        }

        return noErr;
    };
}

@end

// --- View Controller (extension principal class) ------------------------------
#import <WebKit/WebKit.h>

@interface MagentaRTWebView : WKWebView
@end

@implementation MagentaRTWebView

// We implement this to get the keyboard shortcuts for copy-paste to work in the text prompts.
- (BOOL)performKeyEquivalent:(NSEvent *)event {
    if ([event modifierFlags] & NSEventModifierFlagCommand) {
        NSString *chars = [event charactersIgnoringModifiers];
        if ([chars isEqualToString:@"c"]) {
            [NSApp sendAction:@selector(copy:) to:nil from:self];
            return YES;
        } else if ([chars isEqualToString:@"v"]) {
            [NSApp sendAction:@selector(paste:) to:nil from:self];
            return YES;
        } else if ([chars isEqualToString:@"a"]) {
            [NSApp sendAction:@selector(selectAll:) to:nil from:self];
            return YES;
        } else if ([chars isEqualToString:@"x"]) {
            [NSApp sendAction:@selector(cut:) to:nil from:self];
            return YES;
        }
    }
    return [super performKeyEquivalent:event];
}

@end

@interface MagentaRTViewController () <WKScriptMessageHandler, WKNavigationDelegate, NSDraggingSource>
- (void)handleSelectModel:(NSString*)modelName;
@end

@implementation MagentaRTViewController {
    AUAudioUnit* _audioUnit;
    WKWebView* _webView;
#if MAGENTART_DEBUG_LOG
    NSTextField* _debugLabel;
#endif
    NSTimer* _metricsTimer;
    NSURL* _activeModelURL;
    NSMutableDictionary* _lastParams;
    int _metricsTicks;
    BOOL _weightChangeFromUI;  // set by textPrompts handler, cleared by polling loop

    NSURL* _modelDirectoryURL;
    NSURL* _pendingDragURL;
}

// ── Bank file paths (emulator-style save states) ─────────────────────────────

static NSString* bankFilePathAU(int index) {
    std::string banksDir = magentart::paths::get_banks_dir();
    NSString* dir = [NSString stringWithUTF8String:banksDir.c_str()];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES attributes:nil error:nil];
    return [dir stringByAppendingPathComponent:
            [NSString stringWithFormat:@"bank_%d.safetensors", index + 1]];
}

- (AUAudioUnit*)createAudioUnitWithComponentDescription:(AudioComponentDescription)desc
                                                  error:(NSError**)error {
    _audioUnit = [[MagentaRTAudioUnit alloc] initWithComponentDescription:desc
                                                                    options:0
                                                                      error:error];
    return _audioUnit;
}

- (void)addDebugLog:(NSString*)msg {
    dispatch_async(dispatch_get_main_queue(), ^{
#if MAGENTART_DEBUG_LOG
        if (self->_debugLabel) {
            self->_debugLabel.stringValue = [self->_debugLabel.stringValue stringByAppendingFormat:@"%@\n", msg];
        }
#endif
        [self sendStateUpdate:@{@"debugLog": msg}];
    });
}

- (void)writeDiskLog:(NSString*)msg {
#if MAGENTART_DEBUG_LOG
    NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"DownloadFolderBookmark"];
    if (!bookmark) {
        bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
    }
    NSURL* modelsDir = nil;
    BOOL accessGranted = NO;
    if (bookmark) {
        BOOL stale = NO;
        modelsDir = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (modelsDir) {
            accessGranted = [modelsDir startAccessingSecurityScopedResource];
        }
    }
    if (!modelsDir) {
        NSArray* paths = [[NSFileManager defaultManager] URLsForDirectory:NSApplicationSupportDirectory inDomains:NSUserDomainMask];
        modelsDir = [[paths firstObject] URLByAppendingPathComponent:@"MagentaRT/models"];
    }
    NSURL* logURL = [modelsDir URLByAppendingPathComponent:@"mrt_debug.log"];
    NSString* line = [NSString stringWithFormat:@"%@: %@\n", [NSDate date], msg];
    NSFileHandle* fileHandle = [NSFileHandle fileHandleForWritingToURL:logURL error:nil];
    if (fileHandle) {
        [fileHandle seekToEndOfFile];
        [fileHandle writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
        [fileHandle closeFile];
    } else {
        [line writeToURL:logURL atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    if (accessGranted) {
        [modelsDir stopAccessingSecurityScopedResource];
    }
#endif

    [self sendStateUpdate:@{@"debugLog": msg}];
}

- (void)loadView {
    NSView* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1075, 470)];
    view.wantsLayer = YES;
    view.layer.backgroundColor = [NSColor colorWithRed:0.15 green:0.15 blue:0.2 alpha:1.0].CGColor;

    self.view = view;
    self.preferredContentSize = NSMakeSize(1075, 470);
}

- (void)viewWillAppear {
    [super viewWillAppear];

    if (!_webView) {
        WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
        [config.preferences setValue:@YES forKey:@"developerExtrasEnabled"];
        [config.preferences setValue:@YES forKey:@"allowFileAccessFromFileURLs"];
        @try {
            [config setValue:@YES forKey:@"allowUniversalAccessFromFileURLs"];
        } @catch (NSException *e) { }

        NSString *js = @"window.__HOST_MODE__ = 'auv3';"
                       @"window.onerror = function(msg, url, line, col, error) { window.webkit.messageHandlers.auHost.postMessage({type:'log', value:'JS Error: '+msg+ ' @ line '+line}); };"
                       @"var origLog = console.log; console.log = function(msg) { window.webkit.messageHandlers.auHost.postMessage({type:'log', value:'JS Log: '+msg}); origLog(msg); };"
                       @"var origErr = console.error; console.error = function(msg) { window.webkit.messageHandlers.auHost.postMessage({type:'log', value:'JS Console.Error: '+msg}); origErr(msg); };";
        WKUserScript *script = [[WKUserScript alloc] initWithSource:js injectionTime:WKUserScriptInjectionTimeAtDocumentStart forMainFrameOnly:YES];
        [config.userContentController addUserScript:script];
        [config.userContentController addScriptMessageHandler:self name:@"auHost"];

        _webView = [[MagentaRTWebView alloc] initWithFrame:self.view.bounds configuration:config];
        _webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        _webView.navigationDelegate = self;
        [_webView setValue:@(NO) forKey:@"drawsBackground"];
        [self.view addSubview:_webView];

        if (isDevServerRunning()) {
            NSLog(@"MagentaRT_AU2: Vite dev server detected on port %d — loading with HMR", kDevServerPort);
            [_webView loadRequest:[NSURLRequest requestWithURL:
                [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%d", kDevServerPort]]]];
        } else {
            NSBundle* bundle = [NSBundle bundleForClass:[self class]];
            NSString* uiPath = [bundle pathForResource:@"index" ofType:@"html" inDirectory:@"ui"];
            if (uiPath) {
                NSURL* url = [NSURL fileURLWithPath:uiPath];
                NSURL* folderUrl = [url URLByDeletingLastPathComponent];
                [_webView loadFileURL:url allowingReadAccessToURL:folderUrl];
            }
        }
    }
}

- (void)viewDidAppear {
    [super viewDidAppear];

    if (self.view.window) {
        self.view.window.minSize = NSMakeSize(1075, 470);
        self.view.window.maxSize = NSMakeSize(1075, 470);
    }

    if (_metricsTimer) {
        [_metricsTimer invalidate];
    }
    _metricsTicks = 0;
    _lastParams = [NSMutableDictionary dictionary];

    _metricsTimer = [NSTimer scheduledTimerWithTimeInterval:1.0/25.0
                                                    target:self
                                                  selector:@selector(updateMetrics)
                                                  userInfo:nil
                                                    repeats:YES];
}

- (void)viewDidDisappear {
    [super viewDidDisappear];

    if (_metricsTimer) {
        [_metricsTimer invalidate];
        _metricsTimer = nil;
    }

    // No parameter observer to remove — we use polling instead.

    if (_webView) {
        [_webView.configuration.userContentController removeScriptMessageHandlerForName:@"auHost"];
        [_webView removeFromSuperview];
        _webView = nil;
    }
}

- (void)updateMetrics {
    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
    if (!au) return;
    RealtimeRunner* engine = [au engine];
    if (!engine) return;

    [au pollOfflineState];

    _metricsTicks++;
    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];

    // Send audio levels and MIDI note activity every frame (25 Hz)
    float pL = 0.0f;
    float pR = 0.0f;
    [au readAudioLevels:&pL right:&pR];
    stateUpdate[@"audioLevels"] = @{
        @"left": @(pL),
        @"right": @(pR)
    };
    stateUpdate[@"activeNotes"] = [au activeNotes];

    if (_metricsTicks >= 5) {
        _metricsTicks = 0;
        EngineMetrics m = engine->get_metrics();
        int textStatus = engine->get_text_encoder_status();
        int quantStatus = engine->get_quantizer_status();
        double sampleRate = au.outputBusses[0].format.sampleRate;

        NSMutableArray* mutablePrompts = [au.prompts mutableCopy];
        if (!mutablePrompts) {
            mutablePrompts = [NSMutableArray array];
            for (int i = 0; i < 6; ++i) [mutablePrompts addObject:@{@"text": @"", @"weight": @0.0}];
        }
        bool changed = false;
        for (int i = 0; i < 6; ++i) {
            std::string text = engine->get_cached_text(i);
            if (text.substr(0, 4) == "Err:") {
                NSMutableDictionary* p = [mutablePrompts[i] mutableCopy];
                NSString* errStr = [NSString stringWithUTF8String:text.c_str()];
                if (![p[@"text"] isEqualToString:errStr]) {
                    p[@"text"] = errStr;
                    mutablePrompts[i] = p;
                    changed = true;
                }
            }
        }
        if (changed) {
            au.prompts = mutablePrompts;
            stateUpdate[@"textPrompts"] = mutablePrompts;
        }

        std::vector<std::string> logs = engine->get_logs();
        if (!logs.empty()) {
            NSMutableArray* logArray = [NSMutableArray array];
            for (const auto& log : logs) {
                [logArray addObject:[NSString stringWithUTF8String:log.c_str()]];
            }
            stateUpdate[@"logs"] = logArray;
        }

        stateUpdate[@"metrics"] = @{
            @"frameMs": @(m.transformer_ms),
            @"bufferAvail": @(m.buffer_available),
            @"bufferCap": @(m.buffer_capacity),
            @"textEncoderStatusColors": @[
                @(engine->get_prompt_status(0)),
                @(engine->get_prompt_status(1)),
                @(engine->get_prompt_status(2)),
                @(engine->get_prompt_status(3)),
                @(engine->get_prompt_status(4)),
                @(engine->get_prompt_status(5))
            ],
            @"quantizerStatusColor": @(quantStatus),
            @"transportFlags": @(m.transport_flags),
            @"sampleRateIncorrect": @(sampleRate != 48000.0),
            @"droppedFrames": @(m.dropped_frames)
        };
    }

    stateUpdate[@"isPlaying"] = @(au.uiPlaying);
    stateUpdate[@"activeNotes"] = [au activeNotes];

    NSMutableDictionary* params = [NSMutableDictionary dictionary];
    NSMutableDictionary* weightChanges = [NSMutableDictionary dictionary];
    for (int i = 0; i <= 46; i++) {
        NSString* key = paramKeyForAddress(i);
        if (!key) continue;
        AUParameter* param = [au.parameterTree parameterWithAddress:i];
        if (param) {
            NSNumber* val = paramIsBool(i) ? @(param.value > 0.5) : @(param.value);
            NSNumber* lastVal = _lastParams[key];
            if (!lastVal || ![lastVal isEqualToNumber:val]) {
                // Separate weight params (addresses 10-15) from regular params
                if (i >= 10 && i <= 15) {
                    weightChanges[key] = val;
                } else {
                    params[key] = val;
                }
                _lastParams[key] = val;
            }
        }
    }

    if (params.count > 0) {
        stateUpdate[@"params"] = params;
    }

    // Weight changes that did NOT originate from the React UI are DAW
    // automation — send as weightAutomation so the UI switches to list mode.
    if (weightChanges.count > 0 && !_weightChangeFromUI) {
        stateUpdate[@"weightAutomation"] = weightChanges;
    }
    _weightChangeFromUI = NO;

    if (stateUpdate.count > 0) {
        [self sendStateUpdate:stateUpdate];
    }
}

static NSString* paramKeyForAddress(AUParameterAddress address) {
    switch (address) {
        case 0: return @"temperature";
        case 1: return @"topk";
        case 3: return @"cfgmusiccoca";
        case 4: return @"cfgnotes";
        case 5: return @"volume";
        case 6: return @"mute";
        case 7: return @"unmaskwidth";
        case 8: return @"buffersize";
        case 9: return @"latencycomp";
        case 10: return @"weight_0";
        case 11: return @"weight_1";
        case 12: return @"weight_2";
        case 13: return @"weight_3";
        case 14: return @"weight_4";
        case 15: return @"weight_5";
        case 31: return @"resetstate";
        case 32: return @"bypass";
        case 39: return @"drumless";
        case 40: return @"drums_mute_all";
        case 41: return @"drums_mute_kick";
        case 42: return @"drums_mute_snare";
        case 43: return @"drums_mute_hihat";
        case 44: return @"drums_mute_other";
        case 45: return @"midigate";
        case 46: return @"onsetmode";
        default:
            return nil;
    }
}

static BOOL paramIsBool(AUParameterAddress address) {
    if (address == 6 || address == 9 || address == 31 || address == 32 || address == 39 || (address >= 40 && address <= 46)) return YES;
    return NO;
}

- (void)connectToAU {
    AUAudioUnit* au = _audioUnit;
    if (!au) return;

    NSMutableDictionary* initialParams = [NSMutableDictionary dictionary];
    for (int i = 0; i <= 46; i++) {
        // Skip weight params — prompts carry their own weights via textPrompts.
        if (i >= 10 && i <= 15) continue;
        AUParameter* param = [au.parameterTree parameterWithAddress:i];
        if (param) {
            NSString* key = paramKeyForAddress(i);
            if (key) {
                NSNumber* val = paramIsBool(i) ? @(param.value > 0.5) : @(param.value);
                initialParams[key] = val;
                _lastParams[key] = val; // Sync with _lastParams
            }
        }
    }

    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];
    stateUpdate[@"params"] = initialParams;

    if ([au isKindOfClass:[MagentaRTAudioUnit class]]) {
        MagentaRTAudioUnit* m_au = (MagentaRTAudioUnit*)au;

        __weak MagentaRTViewController* weakVC = self;
#if MAGENTART_DEBUG_LOG
        m_au.debugLogHandler = ^(NSString* msg) {
            [weakVC sendStateUpdate:@{@"debugLog": msg}];
        };
#endif

        // Replay log history
        if (m_au.logHistory.count > 0) {
            for (NSString* msg in m_au.logHistory) {
                [weakVC sendStateUpdate:@{@"debugLog": msg}];
            }
        }

        if (m_au.prompts) stateUpdate[@"textPrompts"] = m_au.prompts;
        if (m_au.modelName) stateUpdate[@"modelName"] = m_au.modelName;
        if (m_au.promptSurfaceState) stateUpdate[@"prompt_surface"] = m_au.promptSurfaceState;

        // Push bank existence status
        NSFileManager* fm = [NSFileManager defaultManager];
        stateUpdate[@"bankStatus"] = @[
            @([fm fileExistsAtPath:bankFilePathAU(0)]),
            @([fm fileExistsAtPath:bankFilePathAU(1)]),
            @([fm fileExistsAtPath:bankFilePathAU(2)]),
        ];

        RealtimeRunner* engine = [m_au engine];
        if (engine) {
            if (engine->get_recorded_sample_count() > 0) {
                std::vector<float> peaks = engine->get_waveform_peaks(200);
                NSMutableArray* peaksArray = [NSMutableArray arrayWithCapacity:peaks.size()];
                for (float p : peaks) [peaksArray addObject:@(p)];
                stateUpdate[@"waveformPeaks"] = peaksArray;
                stateUpdate[@"recordedSampleCount"] = @(engine->get_recorded_sample_count());
            }
        }
    }

    // Parameter changes from the host (LFO, automation) are picked up by
    // the metrics timer via polling — no observer callback needed.  This
    // avoids dispatch_async / __weak-reference overhead on the audio thread.

    NSString* savedPath = [[NSUserDefaults standardUserDefaults] objectForKey:@"DownloadFolderPath"];
    if (savedPath) {
        stateUpdate[@"downloadPath"] = savedPath;
    } else {
        NSArray* dirs = [[NSFileManager defaultManager] URLsForDirectory:NSApplicationSupportDirectory inDomains:NSUserDomainMask];
        NSURL* appSupport = [dirs firstObject];
        NSURL* defDir = [appSupport URLByAppendingPathComponent:@"MagentaRT/models"];
        stateUpdate[@"downloadPath"] = defDir.path;
    }

    stateUpdate[@"resourcesMissing"] = @(![MagentaModelDownloader areSharedResourcesValid]);

    [self sendStateUpdate:stateUpdate];
    [self handleListLocalModels];

    // Auto-load last model from preferences if not already loaded by DAW/Host state restoration
    if ([au isKindOfClass:[MagentaRTAudioUnit class]]) {
        MagentaRTAudioUnit* m_au = (MagentaRTAudioUnit*)au;
        RealtimeRunner* engine = [m_au engine];
        if (engine && !engine->is_loaded() && !m_au.modelBookmark && !m_au.modelName) {
            NSData* savedBookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"LoadedModelBookmark"];
            if (savedBookmark) {
                [self writeDiskLog:@"connectToAU: Auto-loading default model from bookmark asynchronously..."];
                dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                    BOOL stale = NO;
                    NSError* error = nil;
                    NSURL* url = [NSURL URLByResolvingBookmarkData:savedBookmark
                                                           options:NSURLBookmarkResolutionWithoutUI | NSURLBookmarkResolutionWithSecurityScope
                                                     relativeToURL:nil
                                               bookmarkDataIsStale:&stale
                                                             error:&error];
                    if (url && [url startAccessingSecurityScopedResource]) {
                        NSString* path = url.path;
                        BOOL isDir = NO;
                        [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir];

                        NSString* mlxfnPath = nil;
                        if ([path hasSuffix:@".mlxfn"]) {
                            mlxfnPath = path;
                        } else if (isDir) {
                            NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:path error:nil];
                            for (NSString *file in contents) {
                                if ([file hasSuffix:@".mlxfn"]) {
                                    mlxfnPath = [path stringByAppendingPathComponent:file];
                                    break;
                                }
                            }
                        }

                        if (mlxfnPath) {
                            [self writeDiskLog:[NSString stringWithFormat:@"connectToAU: Found mlxfn path: %@, loading...", mlxfnPath]];
                            BOOL success = [self loadModelAtPath:mlxfnPath];
                            if (success) {
                                m_au.modelBookmark = savedBookmark;
                                NSString* savedModelName = [[NSUserDefaults standardUserDefaults] objectForKey:@"LoadedModelName"];
                                if (savedModelName) {
                                    m_au.modelName = savedModelName;
                                } else {
                                    m_au.modelName = mlxfnPath.lastPathComponent;
                                }
                                [self writeDiskLog:@"connectToAU: Default model auto-loaded successfully."];
                            } else {
                                [self writeDiskLog:@"connectToAU: Default model auto-load failed."];
                            }
                        }
                        [url stopAccessingSecurityScopedResource];
                    } else {
                        [self writeDiskLog:[NSString stringWithFormat:@"connectToAU: Failed to resolve saved model bookmark: %@", error]];
                    }
                });
            }
        }
    }
}

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation {
    NSLog(@"MagentaRT_AU: WKWebView didFinishNavigation");
    // We rely on the 'uiReady' message from React to push state instead
}

- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"MagentaRT_AU: WKWebView didFailProvisionalNavigation: %@", error);
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"MagentaRT_AU: WKWebView didFailNavigation: %@", error);
}

- (void)sendStateUpdate:(NSDictionary*)state {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!self->_webView) return;

        NSError* error = nil;
        NSData* jsonData = [NSJSONSerialization dataWithJSONObject:state options:0 error:&error];
        if (error) {
            NSLog(@"MagentaRT_AU JSON Error: %@", error);
            return;
        }

        NSString* jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        NSString* script = [NSString stringWithFormat:@"if (window.updateState) { window.updateState(%@); }", jsonString];

        [self->_webView evaluateJavaScript:script completionHandler:^(id result, NSError *error) {
            if (error) {
                NSLog(@"MagentaRT JS Eval Error: %@ for script: %@", error.localizedDescription, script);
            }
        }];
    });
}

- (void)userContentController:(WKUserContentController *)userContentController didReceiveScriptMessage:(WKScriptMessage *)message {
    if ([message.name isEqualToString:@"auHost"] && [message.body isKindOfClass:[NSDictionary class]]) {
        NSDictionary* body = message.body;
        NSString* type = body[@"type"];

        if ([type isEqualToString:@"param"]) {
            NSNumber* indexValue = body[@"index"];
            NSNumber* paramValue = body[@"value"];
            if (indexValue && paramValue && _audioUnit) {
                AUParameter* param = [_audioUnit.parameterTree parameterWithAddress:indexValue.unsignedIntValue];
                if (param) {
                    [param setValue:paramValue.floatValue originator:nil];
                }
            }
        }
        else if ([type isEqualToString:@"textPrompts"]) {
            NSArray* promptsArray = body[@"value"];
            if ([promptsArray isKindOfClass:[NSArray class]] && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                au.prompts = promptsArray;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    std::vector<std::string> std_texts;
                    std::vector<float> std_weights;
                    for (NSDictionary* p in promptsArray) {
                        NSString* text = p[@"text"];
                        NSNumber* weight = p[@"weight"];
                        BOOL isValid = [text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]];
                        std_texts.push_back(isValid ? text.UTF8String : "");
                        std_weights.push_back(isValid ? weight.floatValue : 0.0f);
                    }
                    engine->set_text_prompts(std_texts, std_weights);
                    // Push explicit blend weights to the engine
                    engine->set_blend_weights(std_weights.data(), (int)std_weights.size());
                    // Flag so the polling loop knows this weight change came from the UI
                    // (not DAW automation) and should not trigger a mode switch.
                    _weightChangeFromUI = YES;
                    // Sync weights to AU parameter tree so DAW knobs track UI changes
                    for (int i = 0; i < (int)std_weights.size() && i < 6; i++) {
                        AUParameter* wp = [au.parameterTree parameterWithAddress:10 + i];
                        if (wp) [wp setValue:std_weights[i] originator:nil];
                    }
                }
            }
        }
        else if ([type isEqualToString:@"promptSurfaceState"]) {
            NSDictionary* promptSurfaceDict = body[@"value"];
            if ([promptSurfaceDict isKindOfClass:[NSDictionary class]] && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                au.promptSurfaceState = promptSurfaceDict;
            }
        }
        else if ([type isEqualToString:@"setMusicCoCaModel"]) {
            NSString* subfolder = body[@"value"];
            if ([subfolder isKindOfClass:[NSString class]] && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                au.musicCocaModelName = subfolder;
                NSBundle* bundle = [NSBundle bundleForClass:[au class]];
                NSString* resourcePath = bundle.resourcePath;
                if (resourcePath) {
                    NSString* loadPath = resourcePath;
                    if ([subfolder hasPrefix:@"musiccoca"]) {
                        NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
                        loadPath = customResources ?: [NSString stringWithUTF8String:magentart::paths::get_resources_dir().c_str()];
                    }
                    RealtimeRunner* engine = [au engine];
                    if (engine) {
                        engine->load_musiccoca_model(loadPath.UTF8String, subfolder.UTF8String);
                    }
                }
                [self sendStateUpdate:@{ @"musicCocaModelName": subfolder }];
            }
        }
        else if ([type isEqualToString:@"listRemoteModels"]) {
            [MagentaModelDownloader listRemoteModelsWithCompletion:^(NSArray<NSString *> *models, NSError *error) {
                if (error) {
                    [self sendStateUpdate:@{@"remoteModelsError": error.localizedDescription}];
                } else {
                    [self sendStateUpdate:@{@"remoteModels": models}];
                }
            }];
        }
        else if ([type isEqualToString:@"downloadModel"]) {
            NSString* name = body[@"name"];
            if (name) {
                [MagentaModelDownloader downloadModel:name progress:^(double progress, NSString *status) {
                    [self sendStateUpdate:@{
                        @"downloadProgress": @{
                            @"status": @"downloading",
                            @"percent": @(progress),
                            @"text": status,
                            @"modelName": name
                        }
                    }];
                } completion:^(BOOL success, NSError *error) {
                    if (success) {
                        [self sendStateUpdate:@{
                            @"downloadProgress": @{
                                @"status": @"success",
                                @"percent": @(1.0),
                                @"text": @"Download Complete!",
                                @"modelName": name
                            }
                        }];
                        [self handleListLocalModels];
                    } else {
                        [self sendStateUpdate:@{
                            @"downloadProgress": @{
                                @"status": @"error",
                                @"percent": @(0.0),
                                @"text": error.localizedDescription ?: @"Download Failed",
                                @"modelName": name
                            }
                        }];
                    }
                }];
            }
        }
        else if ([type isEqualToString:@"deleteModel"]) {
            NSString* name = body[@"name"];
            if (name) {
                [self handleDeleteModel:name];
            }
        }
        else if ([type isEqualToString:@"listLocalModels"]) {
            [self handleListLocalModels];
        }
        else if ([type isEqualToString:@"selectModel"]) {
            NSString* name = body[@"name"];
            if (name) {
                [self handleSelectModel:name];
            }
        }
        else if ([type isEqualToString:@"initResources"]) {
            NSString* modelName = body[@"modelName"];
            [self handleInitResources:modelName];
        }
        else if ([type isEqualToString:@"audioPrefill"]) {
            [self handleAudioPrefill];
        }
        else if ([type isEqualToString:@"silentPrefill"]) {
            [self handleSilentPrefill];
        }
        else if ([type isEqualToString:@"selectDownloadFolder"]) {
            [self handleSelectDownloadFolder];
        }
        else if ([type isEqualToString:@"loadModel"]) {
            [self handleLoadModel];
        }
        else if ([type isEqualToString:@"saveBank"]) {
            NSNumber* indexVal = body[@"index"];
            if (indexVal && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    int idx = indexVal.intValue;
                    NSString* path = bankFilePathAU(idx);
                    BOOL success = engine->save_state(path.UTF8String);
                    NSLog(@"MagentaRT_AU2: %@ bank %d to %@",
                          success ? @"Saved" : @"Failed to save", idx + 1, path);
                    [self addDebugLog:[NSString stringWithFormat:@"%@ bank %d",
                                      success ? @"Saved" : @"Failed to save", idx + 1]];
                    // Push updated bank status back to UI
                    NSFileManager* fm = [NSFileManager defaultManager];
                    [self sendStateUpdate:@{@"bankStatus": @[
                        @([fm fileExistsAtPath:bankFilePathAU(0)]),
                        @([fm fileExistsAtPath:bankFilePathAU(1)]),
                        @([fm fileExistsAtPath:bankFilePathAU(2)]),
                    ]}];
                }
            }
        }
        else if ([type isEqualToString:@"loadBank"]) {
            NSNumber* indexVal = body[@"index"];
            if (indexVal && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    int idx = indexVal.intValue;
                    NSString* path = bankFilePathAU(idx);
                    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                        BOOL success = engine->load_state(path.UTF8String);
                        NSLog(@"MagentaRT_AU2: %@ bank %d from %@",
                              success ? @"Loaded" : @"Failed to load", idx + 1, path);
                        [self addDebugLog:[NSString stringWithFormat:@"%@ bank %d",
                                          success ? @"Loaded" : @"Failed to load", idx + 1]];
                    } else {
                        NSLog(@"MagentaRT_AU2: Bank %d file does not exist", idx + 1);
                    }
                }
            }
        }
        else if ([type isEqualToString:@"checkBanks"]) {
            NSFileManager* fm = [NSFileManager defaultManager];
            [self sendStateUpdate:@{@"bankStatus": @[
                @([fm fileExistsAtPath:bankFilePathAU(0)]),
                @([fm fileExistsAtPath:bankFilePathAU(1)]),
                @([fm fileExistsAtPath:bankFilePathAU(2)]),
            ]}];
        }
        else if ([type isEqualToString:@"loadAudioPrompt"]) {
            NSNumber* indexValue = body[@"index"];
            if (indexValue && _audioUnit) {
                [self handleLoadAudioPrompt:indexValue.intValue];
            }
        }
        else if ([type isEqualToString:@"clearAudioPrompt"]) {
            NSNumber* indexValue = body[@"index"];
            if (indexValue && _audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    int index = indexValue.intValue;
                    engine->set_audio_prompt(index, "");
                    // Update prompts array to reflect cleared audio state
                    NSMutableArray* mutablePrompts = [au.prompts mutableCopy];
                    if (mutablePrompts && index < (int)mutablePrompts.count) {
                        NSMutableDictionary* p = [mutablePrompts[index] mutableCopy];
                        p[@"text"] = @"";
                        p[@"isAudio"] = @NO;
                        mutablePrompts[index] = p;
                        au.prompts = mutablePrompts;
                        [self sendStateUpdate:@{@"textPrompts": mutablePrompts}];
                    }
                }
            }
        }
        else if ([type isEqualToString:@"resetModel"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    engine->reset();
                }
            }
        }
        else if ([type isEqualToString:@"resetToFactory"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    engine->reset_to_factory();
                    [self addDebugLog:@"Reset to factory state"];
                    // Clear any "loaded reset state file" indicator the host UI
                    // may be showing — that file is no longer the reset target.
                    [self sendStateUpdate:@{@"resetStateFileName": @""}];
                }
            }
        }
        else if ([type isEqualToString:@"log"]) {
            NSString* val = body[@"value"];
            if (val) {
                [self addDebugLog:val];
            }
        }
        else if ([type isEqualToString:@"kbdNote"]) {
            NSNumber* noteVal = body[@"note"];
            NSNumber* onVal = body[@"on"];
            if (!noteVal || !onVal || !_audioUnit) return;
            MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
            RealtimeRunner* engine = [au engine];
            if (!engine) return;
            uint8_t note = (uint8_t)MIN(127, MAX(0, noteVal.intValue));
            BOOL on = onVal.boolValue;
            if (on) {
                engine->set_note_on(note);
                [au setNoteOn:note on:YES];
            } else {
                engine->set_note_off(note);
                [au setNoteOn:note on:NO];
            }
        }
        else if ([type isEqualToString:@"togglePlay"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                au.uiPlaying = !au.uiPlaying;
                [self sendStateUpdate:@{@"isPlaying": @(au.uiPlaying)}];
            }
        }
        else if ([type isEqualToString:@"uiReady"]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self connectToAU];
            });
        }
        else if ([type isEqualToString:@"startRecording"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) engine->start_recording();
            }
        }
        else if ([type isEqualToString:@"stopRecording"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) engine->stop_recording();
            }
        }
        else if ([type isEqualToString:@"clearRecording"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) engine->clear_recording();
            }
        }
        else if ([type isEqualToString:@"getWaveformPeaks"]) {
            if (_audioUnit) {
                MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                RealtimeRunner* engine = [au engine];
                if (engine) {
                    std::vector<float> peaks = engine->get_waveform_peaks(200);
                    NSMutableArray* peaksArray = [NSMutableArray arrayWithCapacity:peaks.size()];
                    for (float p : peaks) [peaksArray addObject:@(p)];
                    [self sendStateUpdate:@{@"waveformPeaks": peaksArray, @"recordedSampleCount": @(engine->get_recorded_sample_count())}];
                }
            }
        }
        else if ([type isEqualToString:@"prepareDragFile"]) {
            NSNumber* startIdx = body[@"startIdx"];
            NSNumber* endIdx = body[@"endIdx"];
            if (startIdx && endIdx && _audioUnit) {
                [self handlePrepareDragFile:startIdx.unsignedIntegerValue endIdx:endIdx.unsignedIntegerValue];
            }
        }
        else if ([type isEqualToString:@"dragStart"]) {
            if (_audioUnit) {
                [self handleDragStart];
            }
        }
    }
}

// ─── Model loading (shared core) ─────────────────────────────────────────────

// Loads a model from the resolved mlxfnPath.
// Loads MusicCoCa and SpectroStream encoder, updates AU properties
// and pushes state to the React UI. Returns YES on success.
- (BOOL)loadModelAtPath:(NSString*)mlxfnPath {
    if (!self->_audioUnit) return NO;
    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)self->_audioUnit;
    RealtimeRunner* engine = [au engine];
    if (!engine) return NO;

    engine->set_drumless(false);

    NSLog(@"MagentaRT_AU: Attempting to load model from path: %@", mlxfnPath);
    BOOL success = engine->load_model(mlxfnPath.UTF8String);

    if (success) {
        NSLog(@"MagentaRT_AU: Successfully loaded model.");
        self->_modelDirectoryURL = [NSURL fileURLWithPath:[mlxfnPath stringByDeletingLastPathComponent]];

        NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionaryWithDictionary:@{
            @"modelName": mlxfnPath.lastPathComponent
        }];

        // Load MusicCoCa model if not already loaded
        if (![au.musicCocaModelName isEqualToString:@"musiccoca"]) {
            NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
            std::string loadPathStr = customResources ? std::string(customResources.UTF8String) : magentart::paths::get_resources_dir();
            BOOL ok = engine->load_musiccoca_model(loadPathStr.c_str(), "musiccoca");
            if (!ok) {
                NSLog(@"MagentaRT_AU: Failed to load MusicCoCa model for subfolder: musiccoca");
            } else {
                au.musicCocaModelName = @"musiccoca";
                stateUpdate[@"musicCocaModelName"] = @"musiccoca";
            }
        } else {
            // Already loaded, just populate stateUpdate
            stateUpdate[@"musicCocaModelName"] = @"musiccoca";
        }

        stateUpdate[@"params"] = @{@"drumless": @NO};

        // Load SpectroStream encoder: model dir → external spectrostream → bundle
        NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
        NSString* spectrostreamPath = [parentDir stringByAppendingPathComponent:@"spectrostream_encoder.mlxfn"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:spectrostreamPath]) {
            NSLog(@"MagentaRT_AU: Found spectrostream encoder in model dir: %@", spectrostreamPath.lastPathComponent);
            engine->load_prefill_model(spectrostreamPath.UTF8String, nullptr);
        } else {
            std::string extPath = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";
            NSString* extNSPath = [NSString stringWithUTF8String:extPath.c_str()];
            if ([[NSFileManager defaultManager] fileExistsAtPath:extNSPath]) {
                NSLog(@"MagentaRT_AU: Loading spectrostream encoder from external path: %@", extNSPath);
                engine->load_prefill_model(extNSPath.UTF8String, nullptr);
            } else {
                NSBundle* bundle = [NSBundle bundleForClass:[self class]];
                NSString* fallbackPath = [bundle pathForResource:@"spectrostream_encoder" ofType:@"mlxfn"];
                if (fallbackPath) {
                    NSLog(@"MagentaRT_AU: Loading spectrostream encoder from bundle resources: %@", fallbackPath.lastPathComponent);
                    engine->load_prefill_model(fallbackPath.UTF8String, nullptr);
                }
            }
        }

        au.modelName = mlxfnPath.lastPathComponent;

        dispatch_async(dispatch_get_main_queue(), ^{
            [self sendStateUpdate:stateUpdate];
        });
    } else {
        NSLog(@"MagentaRT_AU: engine->load_model returned false.");
        dispatch_async(dispatch_get_main_queue(), ^{
            [self sendStateUpdate:@{ @"modelName": [NSString stringWithFormat:@"Failed: %@", mlxfnPath.lastPathComponent] }];
        });
    }

    return success;
}

- (void)handleLoadModel {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:YES];
    [panel setMessage:@"Select the directory containing your model, or the .mlxfn file."];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSURL* url = [panel URL];
            if (url) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    if (self->_activeModelURL) {
                        [self->_activeModelURL stopAccessingSecurityScopedResource];
                        self->_activeModelURL = nil;
                    }

                    BOOL access = [url startAccessingSecurityScopedResource];
                    if (access) {
                        self->_activeModelURL = url;
                    }

                    NSString* path = url.path;
                    BOOL isDir = NO;
                    [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir];

                    NSString* mlxfnPath = nil;
                    if ([path hasSuffix:@".mlxfn"]) {
                        mlxfnPath = path;
                    } else if (isDir) {
                        NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:path error:nil];
                        for (NSString *file in contents) {
                            if ([file hasSuffix:@".mlxfn"]) {
                                mlxfnPath = [path stringByAppendingPathComponent:file];
                                break;
                            }
                        }
                    }

                    if (!mlxfnPath) {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [self sendStateUpdate:@{@"modelName": @"No .mlxfn found"}];
                        });
                        return;
                    }

                    BOOL success = [self loadModelAtPath:mlxfnPath];

                    if (success && self->_audioUnit) {
                        MagentaRTAudioUnit* m_au = (MagentaRTAudioUnit*)self->_audioUnit;
                        NSError* bmErr = nil;
                        NSData* bookmark = [url bookmarkDataWithOptions:NSURLBookmarkCreationWithSecurityScope
                                         includingResourceValuesForKeys:nil
                                                          relativeToURL:nil
                                                                  error:&bmErr];
                        if (bookmark) {
                            m_au.modelBookmark = bookmark;
                            [[NSUserDefaults standardUserDefaults] setObject:bookmark forKey:@"LoadedModelBookmark"];
                        } else {
                            NSLog(@"MagentaRT_AU: Failed to create bookmark: %@", bmErr);
                        }
                    }
                });
            }
        }
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}


- (void)handleLoadAudioPrompt:(int)index {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedContentTypes:@[[UTType typeWithIdentifier:@"public.audio"]]];
    [panel setMessage:@"Select an audio file for the prompt"];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSURL* url = [panel URL];
            if (!url) return;

            dispatch_async(dispatch_get_main_queue(), ^{
                if (self->_audioUnit) {
                    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)self->_audioUnit;
                    RealtimeRunner* engine = [au engine];
                    if (engine) {
                        NSString* filename = url.lastPathComponent;
                        BOOL readSuccess = NO;

                        BOOL accessed = [url startAccessingSecurityScopedResource];
                        ExtAudioFileRef extFile = nullptr;
                        OSStatus status = ExtAudioFileOpenURL((__bridge CFURLRef)url, &extFile);
                        if (status == noErr && extFile) {
                            AudioStreamBasicDescription clientFormat;
                            clientFormat.mSampleRate = 16000.0;
                            clientFormat.mFormatID = kAudioFormatLinearPCM;
                            clientFormat.mFormatFlags = kAudioFormatFlagIsFloat;
                            clientFormat.mBitsPerChannel = 32;
                            clientFormat.mChannelsPerFrame = 1;
                            clientFormat.mBytesPerFrame = 4;
                            clientFormat.mFramesPerPacket = 1;
                            clientFormat.mBytesPerPacket = 4;

                            status = ExtAudioFileSetProperty(extFile, kExtAudioFileProperty_ClientDataFormat, sizeof(clientFormat), &clientFormat);
                            if (status == noErr) {
                                int maxFrames = 160000; // 10s at 16kHz
                                std::vector<float> samples(maxFrames, 0.0f);

                                AudioBufferList bufferList;
                                bufferList.mNumberBuffers = 1;
                                bufferList.mBuffers[0].mNumberChannels = 1;
                                bufferList.mBuffers[0].mDataByteSize = maxFrames * sizeof(float);
                                bufferList.mBuffers[0].mData = samples.data();

                                UInt32 framesToRead = maxFrames;
                                status = ExtAudioFileRead(extFile, &framesToRead, &bufferList);
                                if (status == noErr && framesToRead > 0) {
                                    if (framesToRead < maxFrames) {
                                        for (UInt32 i = framesToRead; i < maxFrames; ++i) {
                                            samples[i] = samples[i % framesToRead];
                                        }
                                    }
                                    engine->set_audio_prompt_samples(index, filename.UTF8String, samples.data(), maxFrames);
                                    readSuccess = YES;
                                }
                            }
                            ExtAudioFileDispose(extFile);
                        }
                        if (accessed) {
                            [url stopAccessingSecurityScopedResource];
                        }

                        NSMutableArray* mutablePrompts = [au.prompts mutableCopy];
                        if (!mutablePrompts) {
                            mutablePrompts = [NSMutableArray array];
                            for (int i = 0; i < 6; ++i) [mutablePrompts addObject:@{@"text": @"", @"weight": @0.0}];
                        }
                        if (index < mutablePrompts.count) {
                            NSMutableDictionary* p = [mutablePrompts[index] mutableCopy];
                            if (readSuccess) {
                                p[@"text"] = filename;
                                p[@"isAudio"] = @YES;
                            } else {
                                p[@"text"] = @"Error: Load failed";
                                p[@"isAudio"] = @NO;
                            }
                            mutablePrompts[index] = p;
                        }
                        au.prompts = mutablePrompts;
                        [self sendStateUpdate:@{@"textPrompts": mutablePrompts}];
                    }
                }
            });
        }
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

- (void)handleAudioPrefill {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedContentTypes:@[[UTType typeWithIdentifier:@"public.audio"]]];
    [panel setMessage:@"Select an audio file for prefill (will be truncated to 28 s; the SpectroStream encoder is traced at that fixed length)"];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSURL* url = [panel URL];
            if (!url) return;

            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                if (self->_audioUnit) {
                    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)self->_audioUnit;
                    RealtimeRunner* engine = [au engine];
                    if (engine) {
                        NSString* filename = url.lastPathComponent;

                        BOOL accessed = [url startAccessingSecurityScopedResource];
                        ExtAudioFileRef extFile = nullptr;
                        OSStatus status = ExtAudioFileOpenURL((__bridge CFURLRef)url, &extFile);
                        if (status == noErr && extFile) {
                            AudioStreamBasicDescription clientFormat;
                            clientFormat.mSampleRate = 48000.0;
                            clientFormat.mFormatID = kAudioFormatLinearPCM;
                            clientFormat.mFormatFlags = kAudioFormatFlagIsFloat;
                            clientFormat.mBitsPerChannel = 32;
                            clientFormat.mChannelsPerFrame = 2;
                            clientFormat.mBytesPerFrame = 8;
                            clientFormat.mFramesPerPacket = 1;
                            clientFormat.mBytesPerPacket = 8;

                            status = ExtAudioFileSetProperty(extFile, kExtAudioFileProperty_ClientDataFormat, sizeof(clientFormat), &clientFormat);
                            if (status == noErr) {
                                // The SpectroStream encoder's exported .mlxfn
                                // is traced with a fixed input shape of
                                // (1, 1344000, 2) = 28 s @ 48 kHz. We read
                                // up to that and let the engine handle the
                                // exact length internally.
                                int maxFrames = 1344000; // 28s at 48kHz.
                                std::vector<float> samples(maxFrames * 2, 0.0f);

                                AudioBufferList bufferList;
                                bufferList.mNumberBuffers = 1;
                                bufferList.mBuffers[0].mNumberChannels = 2;
                                bufferList.mBuffers[0].mDataByteSize = maxFrames * 2 * sizeof(float);
                                bufferList.mBuffers[0].mData = samples.data();

                                UInt32 framesToRead = maxFrames;
                                status = ExtAudioFileRead(extFile, &framesToRead, &bufferList);
                                if (status == noErr && framesToRead > 0) {
                                    NSLog(@"MagentaRT_AU: Read %u frames for prefill", (unsigned int)framesToRead);
                                    bool success = engine->prefill_state(samples.data(), framesToRead, [self](const std::string& msg) {
                                        [self addDebugLog:[NSString stringWithUTF8String:msg.c_str()]];
                                    });
                                    if (success) {
                                        [self addDebugLog:@"Audio prefill successful"];
                                        [self sendStateUpdate:@{@"audioPrefillStatus": @"Success"}];
                                    } else {
                                        [self addDebugLog:@"Audio prefill failed in engine"];
                                        [self sendStateUpdate:@{@"audioPrefillStatus": @"Failed"}];
                                    }
                                } else {
                                    NSLog(@"MagentaRT_AU: Failed to read audio file or empty");
                                    [self addDebugLog:@"Failed to read audio file"];
                                }
                            } else {
                                NSLog(@"MagentaRT_AU: Failed to set client format");
                                [self addDebugLog:@"Failed to set client format"];
                            }
                            ExtAudioFileDispose(extFile);
                        } else {
                            NSLog(@"MagentaRT_AU: Failed to open audio file at %@", url.path);
                            [self addDebugLog:[NSString stringWithFormat:@"Failed to open audio file at %@", url.lastPathComponent]];
                        }
                        if (accessed) {
                            [url stopAccessingSecurityScopedResource];
                        }
                    }
                }
            });
        } else {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self addDebugLog:@"File dialog cancelled or failed"];
            });
        }
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

- (void)handleSilentPrefill {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        if (self->_audioUnit) {
            MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)self->_audioUnit;
            RealtimeRunner* engine = [au engine];
            if (engine) {
                // 550 frames @ 25 Hz = 22 s, comfortably above the model's
                // ~19.7 s effective receptive field (12 layers × 41-frame
                // local-attention window). This guarantees every layer's
                // KV cache is saturated with silence, so any prior
                // generation no longer influences output. The engine
                // resets state, masks MusicCoCa, and broadcasts a cached
                // silent token through `prefill_state_from_tokens` —
                // no SpectroStream encoder pass at every click.
                [self addDebugLog:@"Starting silent prefill (22s)..."];
                bool success = engine->prefill_silence(/*duration_frames=*/550,
                    [self](const std::string& msg) {
                        [self addDebugLog:[NSString stringWithUTF8String:msg.c_str()]];
                    });
                if (success) {
                    [self addDebugLog:@"Silent prefill successful"];
                } else {
                    [self addDebugLog:@"Silent prefill failed in engine"];
                }
            }
        }
    });
}



- (void)handleListLocalModels {
    NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"DownloadFolderBookmark"];
    if (!bookmark) {
        bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
    }
    NSURL* modelsDir = nil;
    BOOL accessGranted = NO;

    if (bookmark) {
        BOOL stale = NO;
        modelsDir = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (modelsDir) {
            accessGranted = [modelsDir startAccessingSecurityScopedResource];
        }
    }

    if (!modelsDir) {
        std::string defaultPath = magentart::paths::get_models_dir();
        modelsDir = [NSURL fileURLWithPath:[NSString stringWithUTF8String:defaultPath.c_str()]];
    }

    // Create directory if it doesn't exist, just in case
    [[NSFileManager defaultManager] createDirectoryAtURL:modelsDir withIntermediateDirectories:YES attributes:nil error:nil];

    NSArray<NSString *> *modelFiles = [MagentaModelManager listLocalModelsInDirectory:modelsDir];

    if (accessGranted) {
        [modelsDir stopAccessingSecurityScopedResource];
    }

    [self sendStateUpdate:@{@"localModels": modelFiles}];
}

- (void)handleSelectModel:(NSString*)modelName {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!self->_audioUnit) return;
        if (![(MagentaRTAudioUnit*)self->_audioUnit engine]) return;

        NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"DownloadFolderBookmark"];
        if (!bookmark) {
            bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
        }
        NSURL* modelsDir = nil;
        BOOL accessGranted = NO;

        if (bookmark) {
            BOOL stale = NO;
            modelsDir = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
            if (modelsDir) {
                accessGranted = [modelsDir startAccessingSecurityScopedResource];
            }
        }

        if (!modelsDir) {
            std::string defaultPath = magentart::paths::get_models_dir();
            modelsDir = [NSURL fileURLWithPath:[NSString stringWithUTF8String:defaultPath.c_str()]];
        }

        NSURL* modelURL = [modelsDir URLByAppendingPathComponent:modelName];
        NSString* path = modelURL.path;
        BOOL isDir = NO;
        [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir];

        NSString* mlxfnPath = nil;
        if ([path hasSuffix:@".mlxfn"]) {
            mlxfnPath = path;
        } else if (isDir) {
            std::string dirPathStr = path.UTF8String;
            std::string foundMlxfn = magentart::paths::find_mlxfn_in_dir(dirPathStr);
            if (!foundMlxfn.empty()) {
                mlxfnPath = [NSString stringWithUTF8String:foundMlxfn.c_str()];
            }
        }

        if (!mlxfnPath) {
            [self sendStateUpdate:@{@"modelName": @"No .mlxfn found"}];
            if (accessGranted) [modelsDir stopAccessingSecurityScopedResource];
            return;
        }

        BOOL success = [self loadModelAtPath:mlxfnPath];

        if (success && self->_audioUnit) {
            MagentaRTAudioUnit* m_au = (MagentaRTAudioUnit*)self->_audioUnit;
            NSError* bmErr = nil;
            NSData* modelBookmark = [modelURL bookmarkDataWithOptions:NSURLBookmarkCreationWithSecurityScope
                                       includingResourceValuesForKeys:nil
                                                        relativeToURL:nil
                                                                error:&bmErr];
            if (modelBookmark) {
                m_au.modelBookmark = modelBookmark;
                [[NSUserDefaults standardUserDefaults] setObject:modelBookmark forKey:@"LoadedModelBookmark"];
                [[NSUserDefaults standardUserDefaults] setObject:modelName forKey:@"LoadedModelName"];
            }
        }

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleDeleteModel:(NSString*)filename {
    [self writeDiskLog:[NSString stringWithFormat:@"handleDeleteModel triggered for %@", filename]];
    NSError* error = nil;
    NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"DownloadFolderBookmark"];
    if (!bookmark) {
        bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
    }
    NSURL* modelsDir = nil;
    BOOL accessGranted = NO;

    if (bookmark) {
        BOOL stale = NO;
        modelsDir = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&stale error:nil];
        if (modelsDir) {
            accessGranted = [modelsDir startAccessingSecurityScopedResource];
            [self writeDiskLog:[NSString stringWithFormat:@"Resolved bookmark to: %@ (stale: %@, accessGranted: %@)", modelsDir.path, stale ? @"YES" : @"NO", accessGranted ? @"YES" : @"NO"]];
        } else {
            [self writeDiskLog:@"Failed to resolve bookmark data"];
        }
    } else {
        [self writeDiskLog:@"No bookmark data found in defaults during delete"];
    }

    if (!modelsDir) {
        NSArray* paths = [[NSFileManager defaultManager] URLsForDirectory:NSApplicationSupportDirectory inDomains:NSUserDomainMask];
        NSURL* appSupportDir = [paths firstObject];
        modelsDir = [appSupportDir URLByAppendingPathComponent:@"MagentaRT/models"];
    }

    NSURL* fileURL = [modelsDir URLByAppendingPathComponent:filename];
    [self writeDiskLog:[NSString stringWithFormat:@"Attempting to delete file: %@", fileURL.path]];

    [[NSFileManager defaultManager] removeItemAtURL:fileURL error:&error];

    if (!error) {
        [self writeDiskLog:[NSString stringWithFormat:@"Successfully deleted %@", filename]];
        [self handleListLocalModels]; // Refresh list
    } else {
        if (error.code == NSFileNoSuchFileError) {
            [self writeDiskLog:[NSString stringWithFormat:@"File not found for deletion: %@", fileURL.path]];
        } else {
            [self writeDiskLog:[NSString stringWithFormat:@"Failed to delete %@ code %ld: %@", filename, (long)error.code, error.localizedDescription]];
        }
    }

    if (accessGranted) {
        [modelsDir stopAccessingSecurityScopedResource];
    }
}

- (void)handleSelectDownloadFolder {
    [MagentaModelManager selectDownloadFolderWithParentWindow:self.view.window
                                                  completion:^(NSString *selectedPath, NSData *bookmarkData, NSError *error) {
        if (selectedPath && bookmarkData) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [[NSUserDefaults standardUserDefaults] setObject:bookmarkData forKey:@"DownloadFolderBookmark"];
                [[NSUserDefaults standardUserDefaults] setObject:selectedPath forKey:@"DownloadFolderPath"];
                [[NSUserDefaults standardUserDefaults] setObject:bookmarkData forKey:@"MagentaRT_ModelFolderBookmark"];
                [[NSUserDefaults standardUserDefaults] setObject:selectedPath forKey:@"MagentaRT_ModelFolderPath"];

                // Check for custom resources folder inside selected path
                NSString *customResourcesPath = [selectedPath stringByAppendingPathComponent:@"resources"];
                BOOL hasCustomResources = [[NSFileManager defaultManager] fileExistsAtPath:customResourcesPath];

                NSString *resourcesPathToLoad = hasCustomResources ? customResourcesPath : nil;
                if (!resourcesPathToLoad) {
                    NSString* home = NSHomeDirectory();
                    NSRange range = [home rangeOfString:@"/Library/Containers/"];
                    NSString* realHome = (range.location != NSNotFound) ? [home substringToIndex:range.location] : home;
                    resourcesPathToLoad = [realHome stringByAppendingPathComponent:@"Documents/Magenta/magenta-rt-v2/resources"];
                }

                if (_audioUnit) {
                    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
                    RealtimeRunner* engine = [au engine];
                    if (engine) {
                        if (!engine->init_assets(resourcesPathToLoad.UTF8String)) {
                            NSLog(@"MagentaRT_AU2: Failed to initialize assets from custom path: %@", resourcesPathToLoad);
                        } else {
                            NSLog(@"MagentaRT_AU2: Successfully initialized assets from path: %@", resourcesPathToLoad);
                            [[NSUserDefaults standardUserDefaults] setObject:resourcesPathToLoad forKey:@"MagentaRT_CustomResourcesPath"];
                        }
                    }
                }

                [self sendStateUpdate:@{
                    @"downloadPath": selectedPath,
                    @"resourcesMissing": @(![MagentaModelDownloader areSharedResourcesValid])
                }];
                [self handleListLocalModels];

                // Auto-load first available model if present
                NSArray<NSString *> *modelFiles = [MagentaModelManager listLocalModelsInDirectory:[NSURL fileURLWithPath:selectedPath]];
                if (modelFiles.count > 0) {
                    [self handleSelectModel:modelFiles[0]];
                }
            });
        } else if (error) {
            [self addDebugLog:[NSString stringWithFormat:@"Failed to create folder bookmark: %@", error.localizedDescription]];
        }
    }];
}

- (void)handleInitResources:(NSString *)modelName {
    BOOL hasModel = modelName && modelName.length > 0;

    [MagentaModelDownloader initializeSharedResourcesWithProgress:^(double progress, NSString *status) {
        double scaledPercent = hasModel ? progress * 0.5 : progress;
        NSString *statusWithProgress = hasModel
            ? [NSString stringWithFormat:@"[1/2] Shared assets: %@", status]
            : status;

        [self sendStateUpdate:@{
            @"resourcesProgress": @{
                @"status": @"downloading",
                @"percent": @(scaledPercent),
                @"text": statusWithProgress
            }
        }];
    } completion:^(BOOL success, NSError *error) {
        if (!success) {
            [self sendStateUpdate:@{
                @"resourcesProgress": @{
                    @"status": @"error",
                    @"percent": @(0.0),
                    @"text": error.localizedDescription ?: @"Initialization Failed"
                }
            }];
            return;
        }

        if (hasModel) {
            [MagentaModelDownloader downloadModel:modelName progress:^(double progress, NSString *status) {
                double scaledPercent = 0.5 + (progress * 0.5);
                [self sendStateUpdate:@{
                    @"resourcesProgress": @{
                        @"status": @"downloading",
                        @"percent": @(scaledPercent),
                        @"text": [NSString stringWithFormat:@"[2/2] Model: %@", status]
                    }
                }];
            } completion:^(BOOL dlSuccess, NSError *dlError) {
                if (dlSuccess) {
                    [self sendStateUpdate:@{
                        @"resourcesProgress": @{
                            @"status": @"success",
                            @"percent": @(1.0),
                            @"text": @"Setup Complete!"
                        },
                        @"resourcesMissing": @NO
                    }];
                    [self handleListLocalModels];
                    [self handleSelectModel:modelName];
                } else {
                    [self sendStateUpdate:@{
                        @"resourcesProgress": @{
                            @"status": @"error",
                            @"percent": @(0.5),
                            @"text": dlError.localizedDescription ?: @"Model Download Failed"
                        }
                    }];
                }
            }];
        } else {
            [self sendStateUpdate:@{
                @"resourcesProgress": @{
                    @"status": @"success",
                    @"percent": @(1.0),
                    @"text": @"Setup Complete!"
                },
                @"resourcesMissing": @NO
            }];
            [self handleListLocalModels];
        }
    }];
}

- (NSURL*)writeAudioRegionToTempFile:(const float*)bufferL right:(const float*)bufferR count:(size_t)count {
    NSString* tempDir = NSTemporaryDirectory();
    NSString* fileName = [NSString stringWithFormat:@"magenta_export_%f.wav", [[NSDate date] timeIntervalSince1970]];
    NSString* filePath = [tempDir stringByAppendingPathComponent:fileName];
    NSURL* url = [NSURL fileURLWithPath:filePath];

    AudioStreamBasicDescription outDesc = {};
    outDesc.mSampleRate = 48000.0;
    outDesc.mFormatID = kAudioFormatLinearPCM;
    outDesc.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    outDesc.mBitsPerChannel = 32;
    outDesc.mChannelsPerFrame = 2;
    outDesc.mFramesPerPacket = 1;
    outDesc.mBytesPerFrame = 8;
    outDesc.mBytesPerPacket = 8;

    ExtAudioFileRef extFile = nullptr;
    OSStatus status = ExtAudioFileCreateWithURL((__bridge CFURLRef)url, kAudioFileWAVEType, &outDesc, nullptr, kAudioFileFlags_EraseFile, &extFile);
    if (status != noErr) return nil;

    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0].mNumberChannels = 2;
    bufferList.mBuffers[0].mDataByteSize = (UInt32)(count * 2 * sizeof(float));
    float* interleaved = new float[count * 2];
    for (size_t i = 0; i < count; ++i) {
        interleaved[i * 2] = bufferL[i];
        interleaved[i * 2 + 1] = bufferR[i];
    }
    bufferList.mBuffers[0].mData = interleaved;

    status = ExtAudioFileWrite(extFile, (UInt32)count, &bufferList);
    ExtAudioFileDispose(extFile);
    delete[] interleaved;

    if (status != noErr) return nil;
    return url;
}

- (void)handlePrepareDragFile:(size_t)startIdx endIdx:(size_t)endIdx {
    MagentaRTAudioUnit* au = (MagentaRTAudioUnit*)_audioUnit;
    RealtimeRunner* engine = [au engine];
    if (!engine) return;

    size_t count = endIdx - startIdx;
    if (count == 0) return;

    float* bufferL = new float[count];
    float* bufferR = new float[count];

    if (engine->get_recorded_audio(bufferL, bufferR, startIdx, count)) {
        _pendingDragURL = [self writeAudioRegionToTempFile:bufferL right:bufferR count:count];
    }

    delete[] bufferL;
    delete[] bufferR;
}

- (void)handleDragStart {
    if (!_pendingDragURL) return;

    dispatch_async(dispatch_get_main_queue(), ^{
        NSDraggingItem* draggingItem = [[NSDraggingItem alloc] initWithPasteboardWriter:self->_pendingDragURL];
        NSRect frame = NSMakeRect(0, 0, 100, 100);
        [draggingItem setDraggingFrame:frame contents:nil];

        NSEvent* currentEvent = [[NSApplication sharedApplication] currentEvent];
        if (currentEvent) {
            [self->_webView beginDraggingSessionWithItems:@[draggingItem] event:currentEvent source:self];
        }
    });
}

- (NSDragOperation)draggingSession:(NSDraggingSession *)session sourceOperationMaskForDraggingContext:(NSDraggingContext)context {
    return NSDragOperationCopy;
}

- (void)dealloc {
    [_metricsTimer invalidate];
    if (_activeModelURL) {
        [_activeModelURL stopAccessingSecurityScopedResource];
    }
}

@end
