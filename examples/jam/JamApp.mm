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

// Jam — standalone app entry point.
// Reuses RealtimeRunner, AVAudioEngine, CoreMIDI from Magenta RT standalone.
// Adds shared state for MIDI note visualization and audio waveform display.

#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMIDI/CoreMIDI.h>
#import <CoreAudio/CoreAudio.h>
#import "JamAppController.h"
#import "../common/objc/MagentaSettings.h"
#include <magentart/realtime_runner.h>
#include "../common/cpp/magenta_paths.h"

using magentart::core::RealtimeRunner;

// ─── Settings Window Controller ─────────────────────────────────────────────
// Full settings panel: Model, Generation params, Audio I/O, MIDI sources.
// Accessible from app menu (Cmd+,) or from the gear icon in the React UI.

@interface JamSettingsController : NSWindowController <NSWindowDelegate>
@property (nonatomic, strong) AVAudioEngine* audioEngine;
@property (nonatomic, weak) JamAppController* appController;
- (void)refreshAll;
@end

@implementation JamSettingsController {
    // Generation
    NSSlider* _temperatureSlider;   NSTextField* _temperatureValue;
    NSSlider* _topkSlider;          NSTextField* _topkValue;

    NSSlider* _cfgMusicCoCaSlider;  NSTextField* _cfgMusicCoCaValue;
    NSSlider* _cfgNotesSlider;      NSTextField* _cfgNotesValue;
    NSSlider* _cfgDrumsSlider;      NSTextField* _cfgDrumsValue;
    NSSlider* _unmaskWidthSlider;   NSTextField* _unmaskWidthValue;
    NSSlider* _volumeSlider;        NSTextField* _volumeValue;
    NSPopUpButton* _bufferSizePopup;
    NSButton* _muteCheckbox;
    NSButton* _drumModeCheckbox;
    // Audio
    NSTextField* _audioDeviceLabel;
    NSTextField* _audioSampleRateLabel;
    NSTextField* _audioBufferSizeLabel;
}

// ── Helpers for building UI ──

static NSTextField* makeLabel(NSString* text, CGFloat x, CGFloat y, CGFloat w) {
    NSTextField* label = [NSTextField labelWithString:text];
    label.frame = NSMakeRect(x, y, w, 16);
    label.font = [NSFont systemFontOfSize:11];
    label.textColor = [NSColor secondaryLabelColor];
    return label;
}

static NSTextField* makeValue(CGFloat x, CGFloat y) {
    NSTextField* label = [NSTextField labelWithString:@"—"];
    label.frame = NSMakeRect(x, y, 50, 16);
    label.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
    label.alignment = NSTextAlignmentRight;
    return label;
}

static NSSlider* makeSlider(CGFloat x, CGFloat y, CGFloat w, double min, double max, double val, id target, SEL action) {
    NSSlider* slider = [[NSSlider alloc] initWithFrame:NSMakeRect(x, y, w, 20)];
    slider.minValue = min;
    slider.maxValue = max;
    slider.doubleValue = val;
    slider.continuous = YES;
    slider.target = target;
    slider.action = action;
    return slider;
}

- (instancetype)init {
    CGFloat W = 480, H = 420;
    NSRect frame = NSMakeRect(0, 0, W, H);
    NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    window.title = @"Settings";
    window.releasedWhenClosed = NO;

    self = [super initWithWindow:window];
    if (!self) return nil;
    window.delegate = self;

    NSView* c = window.contentView;
    CGFloat pad = 20, col2 = 110, sliderW = 280, valX = W - 70;
    CGFloat y = H - 40;

    // ── Generation ──
    NSTextField* genHeader = [NSTextField labelWithString:@"Generation"];
    genHeader.font = [NSFont boldSystemFontOfSize:13];
    genHeader.frame = NSMakeRect(pad, y, 200, 18);
    [c addSubview:genHeader];
    y -= 26;

    // Volume
    [c addSubview:makeLabel(@"Volume (dB)", pad, y, 90)];
    _volumeSlider = makeSlider(col2, y - 2, sliderW, -60, 12, 0, self, @selector(volumeChanged:));
    [c addSubview:_volumeSlider];
    _volumeValue = makeValue(valX, y); [c addSubview:_volumeValue];
    y -= 26;

    // Temperature
    [c addSubview:makeLabel(@"Temperature", pad, y, 90)];
    _temperatureSlider = makeSlider(col2, y - 2, sliderW, 0, 3, kMagentaDefaultTemperature, self, @selector(temperatureChanged:));
    [c addSubview:_temperatureSlider];
    _temperatureValue = makeValue(valX, y); [c addSubview:_temperatureValue];
    y -= 26;

    // Top-K
    [c addSubview:makeLabel(@"Top-K", pad, y, 90)];
    _topkSlider = makeSlider(col2, y - 2, sliderW, 1, 1024, kMagentaDefaultTopK, self, @selector(topkChanged:));
    [c addSubview:_topkSlider];
    _topkValue = makeValue(valX, y); [c addSubview:_topkValue];
    y -= 26;



    // CFG-MusicCoCa
    [c addSubview:makeLabel(@"CFG-MusicCoCa", pad, y, 90)];
    _cfgMusicCoCaSlider = makeSlider(col2, y - 2, sliderW, 0, 5, kMagentaDefaultCfgMusicCoCa, self, @selector(cfgMusicCoCaChanged:));
    [c addSubview:_cfgMusicCoCaSlider];
    _cfgMusicCoCaValue = makeValue(valX, y); [c addSubview:_cfgMusicCoCaValue];
    y -= 26;

    // CFG-Notes
    [c addSubview:makeLabel(@"CFG-Notes", pad, y, 90)];
    _cfgNotesSlider = makeSlider(col2, y - 2, sliderW, 0, 5, kMagentaDefaultCfgNotes, self, @selector(cfgNotesChanged:));
    [c addSubview:_cfgNotesSlider];
    _cfgNotesValue = makeValue(valX, y); [c addSubview:_cfgNotesValue];
    y -= 26;

    // CFG-Drums
    [c addSubview:makeLabel(@"CFG-Drums", pad, y, 90)];
    _cfgDrumsSlider = makeSlider(col2, y - 2, sliderW, 0, 5, 1, self, @selector(cfgDrumsChanged:));
    [c addSubview:_cfgDrumsSlider];
    _cfgDrumsValue = makeValue(valX, y); [c addSubview:_cfgDrumsValue];
    y -= 26;

    // Unmask width
    [c addSubview:makeLabel(@"Unmask width", pad, y, 90)];
    _unmaskWidthSlider = makeSlider(col2, y - 2, sliderW, 0, 127, 0, self, @selector(unmaskWidthChanged:));
    [c addSubview:_unmaskWidthSlider];
    _unmaskWidthValue = makeValue(valX, y); [c addSubview:_unmaskWidthValue];
    y -= 30;

    // Buffer size
    [c addSubview:makeLabel(@"Buffer Size", pad, y + 2, 90)];
    _bufferSizePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(col2, y, 100, 22) pullsDown:NO];
    [_bufferSizePopup addItemsWithTitles:@[@"2048", @"4096", @"8192"]];
    _bufferSizePopup.font = [NSFont systemFontOfSize:11];
    _bufferSizePopup.target = self;
    _bufferSizePopup.action = @selector(bufferSizeChanged:);
    [c addSubview:_bufferSizePopup];

    _muteCheckbox = [NSButton checkboxWithTitle:@"Mute" target:self action:@selector(muteChanged:)];
    _muteCheckbox.frame = NSMakeRect(col2 + 120, y + 1, 60, 18);
    _muteCheckbox.font = [NSFont systemFontOfSize:11];
    [c addSubview:_muteCheckbox];

    _drumModeCheckbox = [NSButton checkboxWithTitle:@"Drum Mode" target:self action:@selector(drumModeChanged:)];
    _drumModeCheckbox.frame = NSMakeRect(col2 + 190, y + 1, 100, 18);
    _drumModeCheckbox.font = [NSFont systemFontOfSize:11];
    [c addSubview:_drumModeCheckbox];
    y -= 20;

    // Reset defaults
    NSButton* resetBtn = [NSButton buttonWithTitle:@"Reset Defaults" target:self action:@selector(resetDefaults:)];
    resetBtn.frame = NSMakeRect(pad, y, 120, 20);
    resetBtn.bezelStyle = NSBezelStyleInline;
    resetBtn.font = [NSFont systemFontOfSize:11];
    [c addSubview:resetBtn];
    y -= 16;

    NSBox* sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(pad, y, W - 2 * pad, 1)];
    sep1.boxType = NSBoxSeparator;
    [c addSubview:sep1];
    y -= 24;

    // ── Audio Output ──
    NSTextField* audioHeader = [NSTextField labelWithString:@"Audio Output"];
    audioHeader.font = [NSFont boldSystemFontOfSize:13];
    audioHeader.frame = NSMakeRect(pad, y, 200, 18);
    [c addSubview:audioHeader];
    y -= 22;

    [c addSubview:makeLabel(@"Device:", pad, y, 55)];
    _audioDeviceLabel = [NSTextField labelWithString:@"—"];
    _audioDeviceLabel.frame = NSMakeRect(pad + 60, y, 350, 16);
    _audioDeviceLabel.font = [NSFont systemFontOfSize:11];
    [c addSubview:_audioDeviceLabel];
    y -= 18;

    [c addSubview:makeLabel(@"Sample Rate:", pad, y, 80)];
    _audioSampleRateLabel = [NSTextField labelWithString:@"—"];
    _audioSampleRateLabel.frame = NSMakeRect(pad + 85, y, 200, 16);
    _audioSampleRateLabel.font = [NSFont systemFontOfSize:11];
    [c addSubview:_audioSampleRateLabel];
    y -= 18;

    [c addSubview:makeLabel(@"Buffer Size:", pad, y, 80)];
    _audioBufferSizeLabel = [NSTextField labelWithString:@"—"];
    _audioBufferSizeLabel.frame = NSMakeRect(pad + 85, y, 200, 16);
    _audioBufferSizeLabel.font = [NSFont systemFontOfSize:11];
    [c addSubview:_audioBufferSizeLabel];

    return self;
}

// ── Show / refresh ──

- (void)showWindow:(id)sender {
    [self refreshAll];
    [super showWindow:sender];
    [self.window center];
}

- (void)refreshAll {
    [self refreshParams];
    [self refreshAudioInfo];
}

- (void)refreshParams {
    JamAppController* ctrl = _appController;
    if (!ctrl) return;

    _temperatureSlider.doubleValue = [ctrl readParamFromEngine:0];
    _temperatureValue.stringValue = [NSString stringWithFormat:@"%.2f", _temperatureSlider.doubleValue];

    _topkSlider.doubleValue = [ctrl readParamFromEngine:1];
    _topkValue.stringValue = [NSString stringWithFormat:@"%d", (int)_topkSlider.doubleValue];



    _cfgMusicCoCaSlider.doubleValue = [ctrl readParamFromEngine:3];
    _cfgMusicCoCaValue.stringValue = [NSString stringWithFormat:@"%.2f", _cfgMusicCoCaSlider.doubleValue];

    _cfgNotesSlider.doubleValue = [ctrl readParamFromEngine:4];
    _cfgNotesValue.stringValue = [NSString stringWithFormat:@"%.2f", _cfgNotesSlider.doubleValue];

    _cfgDrumsSlider.doubleValue = [ctrl readParamFromEngine:48];
    _cfgDrumsValue.stringValue = [NSString stringWithFormat:@"%.2f", _cfgDrumsSlider.doubleValue];

    _unmaskWidthSlider.doubleValue = [ctrl readParamFromEngine:7];
    _unmaskWidthValue.stringValue = [NSString stringWithFormat:@"%d", (int)_unmaskWidthSlider.doubleValue];

    _volumeSlider.doubleValue = [ctrl readParamFromEngine:5];
    _volumeValue.stringValue = [NSString stringWithFormat:@"%.1f", _volumeSlider.doubleValue];

    float bufVal = [ctrl readParamFromEngine:8];
    [_bufferSizePopup selectItemAtIndex:(bufVal < 0.5 ? 0 : (bufVal < 1.5 ? 1 : 2))];

    _muteCheckbox.state = ([ctrl readParamFromEngine:6] > 0.5) ? NSControlStateValueOn : NSControlStateValueOff;
    _drumModeCheckbox.state = ([ctrl readParamFromEngine:39] > 0.5) ? NSControlStateValueOn : NSControlStateValueOff;
}

// ── Slider / control actions ──

- (void)temperatureChanged:(NSSlider*)sender {
    _temperatureValue.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    [_appController applyParamToEngine:0 value:(float)sender.doubleValue];
}
- (void)topkChanged:(NSSlider*)sender {
    int v = (int)sender.doubleValue;
    _topkValue.stringValue = [NSString stringWithFormat:@"%d", v];
    [_appController applyParamToEngine:1 value:(float)v];
}

- (void)cfgMusicCoCaChanged:(NSSlider*)sender {
    _cfgMusicCoCaValue.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    [_appController applyParamToEngine:3 value:(float)sender.doubleValue];
}
- (void)cfgNotesChanged:(NSSlider*)sender {
    _cfgNotesValue.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    [_appController applyParamToEngine:4 value:(float)sender.doubleValue];
}
- (void)cfgDrumsChanged:(NSSlider*)sender {
    _cfgDrumsValue.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
    [_appController applyParamToEngine:48 value:(float)sender.doubleValue];
}
- (void)unmaskWidthChanged:(NSSlider*)sender {
    int v = (int)sender.doubleValue;
    _unmaskWidthValue.stringValue = [NSString stringWithFormat:@"%d", v];
    [_appController applyParamToEngine:7 value:(float)v];
}
- (void)volumeChanged:(NSSlider*)sender {
    _volumeValue.stringValue = [NSString stringWithFormat:@"%.1f", sender.doubleValue];
    [_appController applyParamToEngine:5 value:(float)sender.doubleValue];
}
- (void)bufferSizeChanged:(NSPopUpButton*)sender {
    [_appController applyParamToEngine:8 value:(float)sender.indexOfSelectedItem];
}
- (void)muteChanged:(NSButton*)sender {
    [_appController applyParamToEngine:6 value:(sender.state == NSControlStateValueOn) ? 1.0f : 0.0f];
}
- (void)drumModeChanged:(NSButton*)sender {
    [_appController applyParamToEngine:39 value:(sender.state == NSControlStateValueOn) ? 1.0f : 0.0f];
}

- (void)resetDefaults:(id)sender {
    [MagentaSettings resetDefaultsOnEngine:_appController.engine
                              prefixString:@"Jam"];
    [self refreshParams];
}

// ── Audio info ──

- (void)refreshAudioInfo {
    if (!_audioEngine) return;
    AVAudioFormat* outputFormat = [_audioEngine.outputNode outputFormatForBus:0];
    double sampleRate = outputFormat.sampleRate;

    AudioDeviceID deviceID = 0;
    UInt32 propSize = sizeof(deviceID);
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &propSize, &deviceID);

    NSString* deviceName = @"Unknown";
    if (deviceID != 0) {
        CFStringRef cfName = NULL;
        propSize = sizeof(cfName);
        AudioObjectPropertyAddress nameAddr = {
            kAudioDevicePropertyDeviceNameCFString,
            kAudioObjectPropertyScopeOutput,
            kAudioObjectPropertyElementMain
        };
        if (AudioObjectGetPropertyData(deviceID, &nameAddr, 0, NULL, &propSize, &cfName) == noErr && cfName) {
            deviceName = (__bridge_transfer NSString*)cfName;
        }
    }

    UInt32 bufferFrames = 0;
    propSize = sizeof(bufferFrames);
    AudioObjectPropertyAddress bufAddr = {
        kAudioDevicePropertyBufferFrameSize,
        kAudioObjectPropertyScopeOutput,
        kAudioObjectPropertyElementMain
    };
    if (deviceID != 0) {
        AudioObjectGetPropertyData(deviceID, &bufAddr, 0, NULL, &propSize, &bufferFrames);
    }

    _audioDeviceLabel.stringValue = deviceName;
    _audioSampleRateLabel.stringValue = [NSString stringWithFormat:@"%.0f Hz (engine: 48000 Hz)", sampleRate];
    _audioBufferSizeLabel.stringValue = [NSString stringWithFormat:@"%u frames", (unsigned)bufferFrames];
}
@end

// ─── AppDelegate ─────────────────────────────────────────────────────────────

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
    RealtimeRunner _engine;
    JamSharedState _sharedState;
    AVAudioEngine* _audioEngine;
    AVAudioSourceNode* _sourceNode;
    MIDIClientRef _midiClient;
    MIDIPortRef _midiInputPort;
    MIDIEndpointRef _midiVirtualDest;
    NSWindow* _window;
    JamAppController* _controller;
    JamSettingsController* _settingsController;
    BOOL _isPlaying;
    NSMenuItem* _playStopMenuItem;
    std::atomic<float> _gateLevel;
    std::atomic<float> _gateDecaySeconds;
    std::atomic<bool> _soloMode;
    std::atomic<float> _cfgNotesSliderValue;
    std::atomic<float> _cfgNotesCurrentLevel;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    // Initialize ML assets from ~/Documents/Magenta/magenta-rt-v2/resources (centralized path) or saved custom folder.
    // Model files should be placed in ~/Documents/Magenta/magenta-rt-v2/models/.
    NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
    std::string resources = customResources ? customResources.UTF8String : magentart::paths::get_resources_dir();
    if (!_engine.init_assets(resources.c_str())) {
        NSLog(@"Jam: Failed to load static assets from %s", resources.c_str());
    }

    _gateLevel.store(1.0f);
    _gateDecaySeconds.store(2.0f);

    BOOL savedSoloMode = NO;
    if ([[NSUserDefaults standardUserDefaults] objectForKey:@"Jam_SoloMode"]) {
        savedSoloMode = [[NSUserDefaults standardUserDefaults] boolForKey:@"Jam_SoloMode"];
    }
    _soloMode.store(savedSoloMode);
    _cfgNotesCurrentLevel.store(50.0f);

    float savedCfgNotes = [[NSUserDefaults standardUserDefaults] floatForKey:@"Jam_Param_cfgnotes"];
    _cfgNotesSliderValue.store(savedCfgNotes > 0.0f ? savedCfgNotes : kMagentaDefaultCfgNotes);

    _controller = [[JamAppController alloc] init];
    _controller.engine = &_engine;
    _controller.sharedState = &_sharedState;
    _controller.soloMode = &_soloMode;
    _controller.cfgNotesSliderValue = &_cfgNotesSliderValue;
    _controller.gateDecaySeconds = &_gateDecaySeconds;

    // Restore saved parameters immediately so the engine has them from start
    [_controller restoreSavedParams];

    _isPlaying = NO;
    _engine.set_bypass(true);

    NSRect frame = NSMakeRect(0, 0, 850, 605);
    _window = [[NSWindow alloc] initWithContentRect:frame
                                           styleMask:NSWindowStyleMaskTitled |
                                                     NSWindowStyleMaskClosable |
                                                     NSWindowStyleMaskMiniaturizable
                                             backing:NSBackingStoreBuffered
                                               defer:NO];
    _window.title = @"MRT2 - Jam";
    _window.minSize = NSMakeSize(850, 605);
    _window.maxSize = NSMakeSize(850, 605);
    _window.contentViewController = _controller;
    [_window center];
    [_window makeKeyAndOrderFront:nil];

    [self setupAudioEngine];
    [self setupMIDI];
    [self setupMenuBar];

    _controller.midiInputPort = _midiInputPort;
    _controller.connectedSources = [NSMutableSet set];

    _settingsController = [[JamSettingsController alloc] init];
    _settingsController.audioEngine = _audioEngine;
    _settingsController.appController = _controller;

    [self autoLoadModel];
}

// ─── AVAudioEngine ───────────────────────────────────────────────────────────

- (void)setupAudioEngine {
    _audioEngine = [[AVAudioEngine alloc] init];
    AVAudioFormat* format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:48000.0 channels:2];

    RealtimeRunner* engine = &_engine;
    JamSharedState* shared = &_sharedState;
    auto* gateLevel = &_gateLevel;
    auto* gateDecaySec = &_gateDecaySeconds;
    auto* soloMode = &_soloMode;
    auto* cfgNotesSliderVal = &_cfgNotesSliderValue;
    auto* cfgNotesLevel = &_cfgNotesCurrentLevel;

    // Request 64 frame buffer size on default output device
    AudioDeviceID deviceID = 0;
    UInt32 propSize = sizeof(deviceID);
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &propSize, &deviceID) == noErr && deviceID != 0) {
        UInt32 bufferFrames = 64;
        AudioObjectPropertyAddress bufAddr = {
            kAudioDevicePropertyBufferFrameSize,
            kAudioObjectPropertyScopeOutput,
            kAudioObjectPropertyElementMain
        };
        OSStatus status = AudioObjectSetPropertyData(deviceID, &bufAddr, 0, NULL, sizeof(bufferFrames), &bufferFrames);
        if (status != noErr) {
            NSLog(@"Jam: Failed to set buffer size to 64, error %d", (int)status);
        } else {
            NSLog(@"Jam: Requested buffer size of 64 frames");
        }
    }

    _sourceNode = [[AVAudioSourceNode alloc]
        initWithFormat:format
        renderBlock:^OSStatus(BOOL* isSilence, const AudioTimeStamp* timestamp,
                              AVAudioFrameCount frameCount, AudioBufferList* outputData) {
        float* outL = (float*)outputData->mBuffers[0].mData;
        float* outR = (outputData->mNumberBuffers > 1)
                      ? (float*)outputData->mBuffers[1].mData : outL;

        if (!engine->is_loaded()) {
            memset(outL, 0, frameCount * sizeof(float));
            if (outputData->mNumberBuffers > 1) memset(outR, 0, frameCount * sizeof(float));
            *isSilence = YES;
            return noErr;
        }

        engine->read_audio_stereo(outL, outR, frameCount, false);

        // Check if any MIDI note is currently held
        bool anyNoteHeld = false;
        for (int n = 0; n < 128 && !anyNoteHeld; ++n) {
            anyNoteHeld = shared->midiNotes[n].load(std::memory_order_relaxed);
        }

        bool isSolo = soloMode->load(std::memory_order_relaxed);

        if (isSolo) {
            const float sliderVal = cfgNotesSliderVal->load(std::memory_order_relaxed);
            const float decaySec = gateDecaySec->load(std::memory_order_relaxed);

            // Volume gate: open instantly when any note is held, decay to 0 when released
            float gate = gateLevel->load(std::memory_order_relaxed);
            const float decayPerSample = (decaySec > 0.0f) ? (1.0f / (48000.0f * decaySec)) : 1.0f;

            for (AVAudioFrameCount i = 0; i < frameCount; ++i) {
                if (anyNoteHeld) {
                    gate = 1.0f;
                } else {
                    gate -= decayPerSample;
                    if (gate < 0.0f) gate = 0.0f;
                }
                outL[i] *= gate;
                outR[i] *= gate;
            }
            gateLevel->store(gate, std::memory_order_relaxed);

            // CFG Notes ramp: snap to slider value when note held, ramp to 50.0f over decaySec when released
            float cfgNotes = cfgNotesLevel->load(std::memory_order_relaxed);
            const float targetVal = 50.0f;
            const float rampPerFrame = (decaySec > 0.0f) ? ((targetVal - sliderVal) / (48000.0f * decaySec)) : targetVal;

            if (anyNoteHeld) {
                cfgNotes = sliderVal;
            } else if (cfgNotes < targetVal) {
                cfgNotes += rampPerFrame * (float)frameCount;
                if (cfgNotes > targetVal) cfgNotes = targetVal;
            }
            cfgNotesLevel->store(cfgNotes, std::memory_order_relaxed);
            engine->set_cfg_notes(cfgNotes);

            // DEBUG: throttled log (~1Hz)
            static int _dbgCounter = 0;
            if (++_dbgCounter >= (int)(48000.0f / frameCount)) {
                _dbgCounter = 0;
                NSLog(@"[Solo ramp] noteHeld=%d slider=%.2f cfgNotes=%.2f gate=%.3f",
                      anyNoteHeld, sliderVal, cfgNotes, gate);
            }
        } else {
            // Accompany mode: reset gate/cfg notes level to slider value, bypass gate
            cfgNotesLevel->store(cfgNotesSliderVal->load(std::memory_order_relaxed), std::memory_order_relaxed);
            gateLevel->store(1.0f, std::memory_order_relaxed);
            engine->set_cfg_notes(cfgNotesSliderVal->load(std::memory_order_relaxed));
        }

        shared->pushAudioSamples(outL, outR, frameCount);
        return noErr;
    }];

    [_audioEngine attachNode:_sourceNode];
    [_audioEngine connect:_sourceNode to:_audioEngine.mainMixerNode format:format];

    NSError* error = nil;
    if (![_audioEngine startAndReturnError:&error]) {
        NSLog(@"Jam: AVAudioEngine failed to start: %@", error);
    }
}

// ─── CoreMIDI ────────────────────────────────────────────────────────────────

- (void)setupMIDI {
    RealtimeRunner* engine = &_engine;
    JamSharedState* shared = &_sharedState;

    __weak JamAppController* weakController = _controller;
    OSStatus status = MIDIClientCreateWithBlock(
        CFSTR("MRT2 - Jam"),
        &_midiClient,
        ^(const MIDINotification* notification) {
            if (notification->messageID == kMIDIMsgSetupChanged) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakController handleMIDIStructureChanged];
                });
            }
        }
    );
    if (status != noErr) { NSLog(@"Jam: MIDIClientCreate failed: %d", (int)status); return; }

    status = MIDIInputPortCreateWithProtocol(
        _midiClient, CFSTR("MRT2 - Jam In"), kMIDIProtocol_1_0, &_midiInputPort,
        ^(const MIDIEventList* evtList, void* srcConnRefCon) {
            const MIDIEventPacket* pkt = &evtList->packet[0];
            for (UInt32 i = 0; i < evtList->numPackets; ++i) {
                for (UInt32 w = 0; w < pkt->wordCount; ++w) {
                    uint32_t word = pkt->words[w];
                    uint8_t msgType = (word >> 28) & 0xF;
                    if (msgType == 0x2) {
                        uint8_t statusByte = (word >> 16) & 0xFF;
                        uint8_t statusNibble = statusByte & 0xF0;
                        uint8_t note = (word >> 8) & 0x7F;
                        uint8_t velocity = word & 0x7F;
                        if (statusNibble == 0x90 && velocity > 0) {
                            engine->set_note_on(note);
                            shared->noteOn(note);
                        } else if (statusNibble == 0x80 || (statusNibble == 0x90 && velocity == 0)) {
                            engine->set_note_off(note);
                            shared->noteOff(note);
                        }
                    }
                }
                pkt = MIDIEventPacketNext(pkt);
            }
        }
    );
    if (status != noErr) { NSLog(@"Jam: MIDIInputPortCreate failed: %d", (int)status); return; }

    status = MIDIDestinationCreateWithProtocol(
        _midiClient, CFSTR("MRT2 - Jam Input"), kMIDIProtocol_1_0, &_midiVirtualDest,
        ^(const MIDIEventList* evtList, void* srcConnRefCon) {
            const MIDIEventPacket* pkt = &evtList->packet[0];
            for (UInt32 i = 0; i < evtList->numPackets; ++i) {
                for (UInt32 w = 0; w < pkt->wordCount; ++w) {
                    uint32_t word = pkt->words[w];
                    uint8_t msgType = (word >> 28) & 0xF;
                    if (msgType == 0x2) {
                        uint8_t statusByte = (word >> 16) & 0xFF;
                        uint8_t statusNibble = statusByte & 0xF0;
                        uint8_t note = (word >> 8) & 0x7F;
                        uint8_t velocity = word & 0x7F;
                        if (statusNibble == 0x90 && velocity > 0) {
                            engine->set_note_on(note);
                            shared->noteOn(note);
                        } else if (statusNibble == 0x80 || (statusNibble == 0x90 && velocity == 0)) {
                            engine->set_note_off(note);
                            shared->noteOff(note);
                        }
                    }
                }
                pkt = MIDIEventPacketNext(pkt);
            }
        }
    );
    if (status != noErr) {
        NSLog(@"Jam: MIDIDestinationCreate failed: %d", (int)status);
    }
}

// ─── Menu bar ────────────────────────────────────────────────────────────────

- (void)setupMenuBar {
    NSMenu* menuBar = [[NSMenu alloc] init];

    NSMenuItem* appMenuItem = [[NSMenuItem alloc] init];
    NSMenu* appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"About MRT2 - Jam" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Settings..." action:@selector(menuShowSettings:) keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit MRT2 - Jam" action:@selector(terminate:) keyEquivalent:@"q"];
    appMenuItem.submenu = appMenu;
    [menuBar addItem:appMenuItem];

    NSMenuItem* fileMenuItem = [[NSMenuItem alloc] init];
    NSMenu* fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"Load Model..." action:@selector(menuLoadModel:) keyEquivalent:@"o"];
    fileMenuItem.submenu = fileMenu;
    [menuBar addItem:fileMenuItem];

    NSMenuItem* editMenuItem = [[NSMenuItem alloc] init];
    NSMenu* editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    editMenuItem.submenu = editMenu;
    [menuBar addItem:editMenuItem];

    NSMenuItem* transportMenuItem = [[NSMenuItem alloc] init];
    NSMenu* transportMenu = [[NSMenu alloc] initWithTitle:@"Transport"];
    _playStopMenuItem = [transportMenu addItemWithTitle:@"Play"
                                                  action:@selector(menuTogglePlayStop:)
                                           keyEquivalent:@" "];
    _isPlaying = NO;
    transportMenuItem.submenu = transportMenu;
    [menuBar addItem:transportMenuItem];

    [NSApp setMainMenu:menuBar];
}

- (void)menuTogglePlayStop:(id)sender {
    if (_isPlaying) {
        _engine.set_bypass(true);
        _isPlaying = NO;
        _playStopMenuItem.title = @"Play";
    } else {
        _engine.set_bypass(false);
        _engine.trigger_reset();
        _isPlaying = YES;
        _playStopMenuItem.title = @"Pause";
    }
    [_controller sendPlayState:_isPlaying];
}

- (void)menuShowSettings:(id)sender {
    if (_controller) {
        [_controller showReactSettings];
    }
}

- (void)menuLoadModel:(id)sender {
    [_controller handleLoadModel];
}

// ─── Auto-load model ─────────────────────────────────────────────────────────

- (void)autoLoadModel {
    NSString* modelPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"Jam_ModelPath"];
    if (!modelPath) return;

    if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) return;

    NSLog(@"Jam: Auto-loading model from %@", modelPath);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        BOOL success = self->_engine.load_model(modelPath.UTF8String);
        if (success) {
            NSLog(@"Jam: Model loaded successfully.");

            NSString* parentDir = [modelPath stringByDeletingLastPathComponent];
            NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
                self->_engine.load_pca_file(corpusPath.UTF8String);
            }

            [self->_controller restoreSavedParams];

            dispatch_async(dispatch_get_main_queue(), ^{
                [self->_controller notifyModelLoaded:modelPath.lastPathComponent];
            });
        } else {
            NSLog(@"Jam: Failed to auto-load model from %@", modelPath);
            dispatch_async(dispatch_get_main_queue(), ^{
                [self->_controller sendStateUpdate:@{@"modelName": @"No model loaded"}];
            });
        }
    });
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

- (void)applicationWillTerminate:(NSNotification*)notification {
    _engine.stop();
    _engine.unload();
    [_audioEngine stop];
    if (_midiVirtualDest) MIDIEndpointDispose(_midiVirtualDest);
    if (_midiInputPort) MIDIPortDispose(_midiInputPort);
    if (_midiClient) MIDIClientDispose(_midiClient);
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender { return YES; }
- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app { return YES; }

@end

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate* delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
