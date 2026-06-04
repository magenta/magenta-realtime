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

// Jam view controller — hosts the Jam React UI in a WKWebView.
// Simplified from MagentaRTAppController: single prompt, MIDI/waveform visualization.

#import "JamAppController.h"
#import <WebKit/WebKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <AudioToolbox/AudioToolbox.h>
#import "MagentaModelManager.h"
#import "MagentaModelDownloader.h"
#import "MagentaSettings.h"
#include "magenta_paths.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using magentart::core::RealtimeRunner;
using magentart::core::EngineMetrics;

// ─── Dev server probe ────────────────────────────────────────────────────────

static const int kDevServerPort = 62421;

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

// ─── WKWebView subclass for keyboard shortcuts ──────────────────────────────

@interface JamWebView : WKWebView
@end

@implementation JamWebView
- (BOOL)performKeyEquivalent:(NSEvent *)event {
    if ([event modifierFlags] & NSEventModifierFlagCommand) {
        NSString *chars = [event charactersIgnoringModifiers];
        if ([chars isEqualToString:@"c"]) { [NSApp sendAction:@selector(copy:) to:nil from:self]; return YES; }
        else if ([chars isEqualToString:@"v"]) { [NSApp sendAction:@selector(paste:) to:nil from:self]; return YES; }
        else if ([chars isEqualToString:@"a"]) { [NSApp sendAction:@selector(selectAll:) to:nil from:self]; return YES; }
        else if ([chars isEqualToString:@"x"]) { [NSApp sendAction:@selector(cut:) to:nil from:self]; return YES; }
    }
    return [super performKeyEquivalent:event];
}
@end

// ─── Param helpers ───────────────────────────────────────────────────────────





// Addresses of params to persist across launches


// ─── View Controller ─────────────────────────────────────────────────────────

@interface JamAppController () <WKScriptMessageHandler, WKNavigationDelegate>
- (void)handleSelectDownloadFolder;
- (void)handleListLocalModels;
- (void)handleSelectModel:(NSString*)modelName;
- (void)handleDeleteModel:(NSString*)modelName;
- (void)handleInitResources:(NSString*)modelName;
@end

@implementation JamAppController {
    WKWebView* _webView;
    NSTimer* _metricsTimer;
    NSMutableDictionary* _lastParams;
    int _metricsTicks;

    NSString* _modelName;
    NSString* _currentPromptText;
    BOOL _isPlaying;
}

// ─── Parameter bridging ──────────────────────────────────────────────────────

- (void)applyParamToEngine:(int)address value:(float)value {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    [MagentaSettings applyParamToEngine:engine address:address value:value prefixString:@"Jam"];

    if (address == 4) {
        if (self.cfgNotesSliderValue) {
            self.cfgNotesSliderValue->store(value, std::memory_order_relaxed);
        }
    }
}

- (void)restoreSavedParams {
    [MagentaSettings restoreSavedParams:self.engine prefixString:@"Jam"];
}

- (float)readParamFromEngine:(int)address {
    if (address == 4) {
        return self.cfgNotesSliderValue ? self.cfgNotesSliderValue->load(std::memory_order_relaxed) : kMagentaDefaultCfgNotes;
    }
    return [MagentaSettings readParamFromEngine:self.engine address:address];
}

// ─── View lifecycle ──────────────────────────────────────────────────────────

- (void)loadView {
    NSView* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 850, 605)];
    view.wantsLayer = YES;
    view.layer.backgroundColor = [NSColor colorWithRed:0.96 green:0.94 blue:0.94 alpha:1.0].CGColor;
    self.view = view;
}

- (void)viewDidAppear {
    [super viewDidAppear];
    _isPlaying = NO;

    if (!_webView) {
        WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
        [config.preferences setValue:@YES forKey:@"developerExtrasEnabled"];
        [config.preferences setValue:@YES forKey:@"allowFileAccessFromFileURLs"];
        @try { [config setValue:@YES forKey:@"allowUniversalAccessFromFileURLs"]; } @catch (NSException *e) { }

        NSString *js = @"window.onerror = function(msg, url, line, col, error) { window.webkit.messageHandlers.auHost.postMessage({type:'log', value:'JS Error: '+msg+ ' @ line '+line}); };"
                       @"var origLog = console.log; console.log = function(msg) { window.webkit.messageHandlers.auHost.postMessage({type:'log', value:''+msg}); origLog(msg); };";
        WKUserScript *script = [[WKUserScript alloc] initWithSource:js injectionTime:WKUserScriptInjectionTimeAtDocumentStart forMainFrameOnly:YES];
        [config.userContentController addUserScript:script];
        [config.userContentController addScriptMessageHandler:self name:@"auHost"];

        _webView = [[JamWebView alloc] initWithFrame:self.view.bounds configuration:config];
        _webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        _webView.navigationDelegate = self;
        [_webView setValue:@(NO) forKey:@"drawsBackground"];
        [self.view addSubview:_webView];

        if (isDevServerRunning()) {
            NSLog(@"Jam: Vite dev server detected on port %d — loading with HMR", kDevServerPort);
            [_webView loadRequest:[NSURLRequest requestWithURL:
                [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%d", kDevServerPort]]]];
        } else {
            NSBundle* bundle = [NSBundle mainBundle];
            NSString* uiPath = [bundle pathForResource:@"index" ofType:@"html" inDirectory:@"jam_ui"];
            if (uiPath) {
                NSURL* url = [NSURL fileURLWithPath:uiPath];
                [_webView loadFileURL:url allowingReadAccessToURL:[url URLByDeletingLastPathComponent]];
            } else {
                NSLog(@"Jam: jam_ui/index.html not found in bundle");
            }
        }
    }

    if (_metricsTimer) [_metricsTimer invalidate];
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
    if (_metricsTimer) { [_metricsTimer invalidate]; _metricsTimer = nil; }
    if (_webView) {
        [_webView.configuration.userContentController removeScriptMessageHandlerForName:@"auHost"];
        [_webView removeFromSuperview];
        _webView = nil;
    }
}

// ─── Metrics polling (25 Hz) ─────────────────────────────────────────────────

- (void)updateMetrics {
    RealtimeRunner* engine = self.engine;
    JamSharedState* shared = self.sharedState;
    if (!engine) return;

    _metricsTicks++;
    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];

    // Send MIDI active notes and audio levels every frame
    if (shared) {
        NSMutableArray* notes = [NSMutableArray array];
        for (int i = 0; i < 128; i++) {
            if (shared->midiNotes[i].load(std::memory_order_relaxed)) {
                [notes addObject:@(i)];
            }
        }
        stateUpdate[@"activeNotes"] = notes;

        float pL = 0.0f;
        float pR = 0.0f;
        shared->levelProcessor.read_and_reset_peaks(pL, pR);
        stateUpdate[@"audioLevels"] = @{
            @"left": @(pL),
            @"right": @(pR)
        };
    }

    // Metrics every 5th tick (~5 Hz)
    if (_metricsTicks >= 5) {
        _metricsTicks = 0;
        EngineMetrics m = engine->get_metrics();

        stateUpdate[@"metrics"] = @{
            @"frameMs": @(m.transformer_ms),
            @"bufferAvail": @(m.buffer_available),
            @"bufferCap": @(m.buffer_capacity),
            @"textEncoderStatus": @(engine->get_text_encoder_status()),
            @"droppedFrames": @(m.dropped_frames)
        };
    }

    // Params — send only changed values
    NSMutableDictionary* params = [NSMutableDictionary dictionary];
    int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,46,48};
    for (int addr : addresses) {
        NSString* key = [MagentaSettings paramKeyForAddress:addr];
        if (!key) continue;
        float rawVal = [self readParamFromEngine:addr];
        NSNumber* val = [MagentaSettings paramIsBool:addr] ? @(rawVal > 0.5) : @(rawVal);
        NSNumber* lastVal = _lastParams[key];
        if (!lastVal || ![lastVal isEqualToNumber:val]) {
            params[key] = val;
            _lastParams[key] = val;
        }
    }
    // cfgnotesuser: the user's chosen note-adherence slider value, unaffected
    // by the solo-mode ramp that animates the engine's internal cfg_notes.
    if (self.cfgNotesSliderValue) {
        NSNumber* sliderVal = @(self.cfgNotesSliderValue->load(std::memory_order_relaxed));
        NSNumber* lastSlider = _lastParams[@"cfgnotesuser"];
        if (!lastSlider || ![lastSlider isEqualToNumber:sliderVal]) {
            params[@"cfgnotesuser"] = sliderVal;
            _lastParams[@"cfgnotesuser"] = sliderVal;
        }
    }
    if (params.count > 0) stateUpdate[@"params"] = params;

    if (stateUpdate.count > 0) [self sendStateUpdate:stateUpdate];
}

// ─── State push to React ─────────────────────────────────────────────────────

- (void)sendStateUpdate:(NSDictionary*)state {
    if (!_webView) return;
    NSError* error = nil;
    NSData* jsonData = [NSJSONSerialization dataWithJSONObject:state options:0 error:&error];
    if (error) return;
    NSString* jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    NSString* script = [NSString stringWithFormat:@"if (window.updateState) { window.updateState(%@); }", jsonString];
    [_webView evaluateJavaScript:script completionHandler:nil];
}

- (void)sendPlayState:(BOOL)playing {
    _isPlaying = playing;
    [self sendStateUpdate:@{@"isPlaying": @(playing)}];
}

- (void)showReactSettings {
    [self sendStateUpdate:@{@"openSettings": @YES}];
}

- (void)connectToEngine {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    NSMutableDictionary* initialParams = [NSMutableDictionary dictionary];
    int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,46,48};
    for (int addr : addresses) {
        NSString* key = [MagentaSettings paramKeyForAddress:addr];
        if (!key) continue;
        float rawVal = [self readParamFromEngine:addr];
        NSNumber* val = [MagentaSettings paramIsBool:addr] ? @(rawVal > 0.5) : @(rawVal);
        initialParams[key] = val;
        _lastParams[key] = val;
    }

    // Include stable slider value for note adherence
    if (self.cfgNotesSliderValue) {
        initialParams[@"cfgnotesuser"] = @(self.cfgNotesSliderValue->load(std::memory_order_relaxed));
    }

    NSMutableDictionary* state = [NSMutableDictionary dictionary];
    state[@"params"] = initialParams;
    state[@"isPlaying"] = @(_isPlaying);
    state[@"solomode"] = @(self.soloMode ? self.soloMode->load(std::memory_order_relaxed) : NO);
    if (_modelName) state[@"modelName"] = _modelName;

    // Restore saved prompt (always send, empty string if nothing saved)
    NSString* savedPrompt = [[NSUserDefaults standardUserDefaults] stringForKey:@"Jam_Prompt"];
    state[@"prompt"] = savedPrompt ?: @"";

    // Restore saved rocker index
    NSNumber* savedRockerIndex = [[NSUserDefaults standardUserDefaults] objectForKey:@"Jam_RockerIndex"];
    if (savedRockerIndex) state[@"savedRockerIndex"] = savedRockerIndex;

    // Restore saved prompt history
    NSArray* savedHistory = [[NSUserDefaults standardUserDefaults] arrayForKey:@"Jam_PromptHistory"];
    if (savedHistory) {
        state[@"savedPromptHistory"] = savedHistory;
        state[@"savedHistoryIndex"] = [[NSUserDefaults standardUserDefaults] objectForKey:@"Jam_HistoryIndex"] ?: @0;
    }

    state[@"computerKeyboardMidi"] = @([[NSUserDefaults standardUserDefaults] boolForKey:@"Jam_ComputerKeyboardMidi"]);

    // Restore user preset overrides
    NSDictionary* savedSolo = [[NSUserDefaults standardUserDefaults] dictionaryForKey:@"Jam_UserPresetsSolo"];
    NSDictionary* savedJam = [[NSUserDefaults standardUserDefaults] dictionaryForKey:@"Jam_UserPresetsJam"];
    if (savedSolo || savedJam) {
        NSMutableDictionary* presets = [NSMutableDictionary dictionary];
        if (savedSolo) presets[@"solo"] = savedSolo;
        if (savedJam) presets[@"jam"] = savedJam;
        state[@"savedUserPresets"] = presets;
    }

    NSString* searchPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_ModelFolderPath"];
    if (!searchPath) {
        searchPath = [NSString stringWithUTF8String:magentart::paths::get_models_dir().c_str()];
    }
    state[@"downloadPath"] = searchPath;

    // Connect to the saved MIDI endpoint
    NSInteger savedEndpoint = [[NSUserDefaults standardUserDefaults] integerForKey:@"Jam_SelectedMidiEndpoint"];
    if (![[NSUserDefaults standardUserDefaults] objectForKey:@"Jam_SelectedMidiEndpoint"]) {
        // Default to Computer Keyboard (0) if nothing is saved
        savedEndpoint = 0;
    }
    [self selectMidiInput:(uint32_t)savedEndpoint];

    state[@"resourcesMissing"] = @(![MagentaModelDownloader areSharedResourcesValid]);

    [self sendStateUpdate:state];
    [self handleListLocalModels];
    [self handleMIDIStructureChanged];
}

- (void)setComputerKeyboardMidiEnabled:(BOOL)enabled {
    [[NSUserDefaults standardUserDefaults] setBool:enabled forKey:@"Jam_ComputerKeyboardMidi"];
    [self sendStateUpdate:@{@"computerKeyboardMidi": @(enabled)}];
}

- (void)notifyModelLoaded:(NSString*)modelName {
    _modelName = modelName;

    dispatch_async(dispatch_get_main_queue(), ^{
        NSMutableDictionary* state = [NSMutableDictionary dictionary];
        state[@"modelName"] = modelName;

        NSMutableDictionary* params = [NSMutableDictionary dictionary];
        int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,46,48};
        for (int addr : addresses) {
            NSString* key = [MagentaSettings paramKeyForAddress:addr];
            if (!key) continue;
            float rawVal = [self readParamFromEngine:addr];
            params[key] = [MagentaSettings paramIsBool:addr] ? @(rawVal > 0.5) : @(rawVal);
            self->_lastParams[key] = params[key];
        }
        if (self.cfgNotesSliderValue) {
            params[@"cfgnotesuser"] = @(self.cfgNotesSliderValue->load(std::memory_order_relaxed));
        }
        state[@"params"] = params;

        // Re-apply current prompt to the freshly loaded model.
        // _currentPromptText may have been set by the frontend via textPrompts IPC
        // before the model finished loading, or from a previous saved prompt.
        if (self.engine) {
            NSString* promptToUse = self->_currentPromptText.length > 0
                ? self->_currentPromptText
                : ([[NSUserDefaults standardUserDefaults] stringForKey:@"Jam_Prompt"] ?: @"");
            BOOL isSolo = self.soloMode ? self.soloMode->load(std::memory_order_relaxed) : YES;
            NSString* cleanPrompt = [promptToUse stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            NSString* engineText = @"";
            if (cleanPrompt.length == 0) {
                engineText = @"silence";
            } else {
                engineText = isSolo ? [NSString stringWithFormat:@"SOLO %@", cleanPrompt] : cleanPrompt;
            }
            std::vector<std::string> texts = {engineText.UTF8String, "", "", "", "", ""};
            std::vector<float> weights = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            self.engine->set_text_prompts(texts, weights);
            self.engine->set_blend_weights(weights.data(), (int)weights.size());
            self->_currentPromptText = promptToUse;
            state[@"prompt"] = promptToUse;
        }

        [self sendStateUpdate:state];
    });
}

// ─── Navigation delegate ─────────────────────────────────────────────────────

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation {
    NSLog(@"Jam: WKWebView loaded");
}

// ─── Script message handler ──────────────────────────────────────────────────

- (void)userContentController:(WKUserContentController *)userContentController didReceiveScriptMessage:(WKScriptMessage *)message {
    if (![message.name isEqualToString:@"auHost"] || ![message.body isKindOfClass:[NSDictionary class]]) return;
    NSDictionary* body = message.body;
    NSString* type = body[@"type"];

    if ([type isEqualToString:@"param"]) {
        NSNumber* indexValue = body[@"index"];
        NSNumber* paramValue = body[@"value"];
        if (indexValue && paramValue) {
            [self applyParamToEngine:indexValue.intValue value:paramValue.floatValue];
        }
    }
    else if ([type isEqualToString:@"setSoloMode"]) {
        NSNumber* valueVal = body[@"value"];
        if (valueVal) {
            BOOL solo = valueVal.boolValue;
            if (self.soloMode) {
                self.soloMode->store(solo, std::memory_order_relaxed);
            }
            [[NSUserDefaults standardUserDefaults] setBool:solo forKey:@"Jam_SoloMode"];
        }
    }
    else if ([type isEqualToString:@"textPrompts"]) {
        NSArray* promptsArray = body[@"value"];
        if ([promptsArray isKindOfClass:[NSArray class]] && self.engine) {
            std::vector<std::string> texts;
            std::vector<float> weights;
            for (NSDictionary* p in promptsArray) {
                NSString* text = p[@"text"];
                NSNumber* weight = p[@"weight"];
                if ([text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]]) {
                    NSString* trimmed = [text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
                    if (trimmed.length == 0 || [trimmed isEqualToString:@"SOLO"]) {
                        texts.push_back("silence");
                    } else {
                        texts.push_back(text.UTF8String);
                    }
                    weights.push_back(weight.floatValue);
                }
            }
            self.engine->set_text_prompts(texts, weights);
            self.engine->set_blend_weights(weights.data(), (int)weights.size());

            // Persist current prompt and history
            if (promptsArray.count > 0) {
                NSDictionary* p0 = promptsArray[0];
                NSString* prompt = p0[@"text"];
                if ([prompt isKindOfClass:[NSString class]]) {
                    if ([prompt hasPrefix:@"SOLO "]) {
                        prompt = [prompt substringFromIndex:5];
                    } else if ([prompt isEqualToString:@"SOLO"]) {
                        prompt = @"";
                    }
                    _currentPromptText = prompt;
                    [[NSUserDefaults standardUserDefaults] setObject:prompt forKey:@"Jam_Prompt"];
                }
            }
        }
    }
    else if ([type isEqualToString:@"loadModel"]) {
        [self handleLoadModel];
    }
    else if ([type isEqualToString:@"listLocalModels"]) {
        [self handleListLocalModels];
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
    else if ([type isEqualToString:@"selectDownloadFolder"]) {
        [self handleSelectDownloadFolder];
    }
    else if ([type isEqualToString:@"selectModel"]) {
        NSString* name = body[@"name"];
        if (name) {
            [self handleSelectModel:name];
        }
    }
    else if ([type isEqualToString:@"deleteModel"]) {
        NSString* name = body[@"name"];
        if (name) {
            [self handleDeleteModel:name];
        }
    }
    else if ([type isEqualToString:@"initResources"]) {
        NSString* modelName = body[@"modelName"];
        [self handleInitResources:modelName];
    }
    else if ([type isEqualToString:@"loadAudioPrompt"]) {
        [self handleLoadAudioPrompt:0];
    }
    else if ([type isEqualToString:@"clearAudioPrompt"]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            RealtimeRunner* engine = self.engine;
            if (engine) {
                engine->set_audio_prompt(0, "");
            }
            [self sendStateUpdate:@{
                @"prompt": self->_currentPromptText ?: @"",
                @"isAudioPrompt": @NO,
            }];
        });
    }
    else if ([type isEqualToString:@"kbdNote"]) {
        NSNumber* noteVal = body[@"note"];
        NSNumber* onVal = body[@"on"];
        if (!noteVal || !onVal || !self.engine) return;
        uint8_t note = (uint8_t)MIN(127, MAX(0, noteVal.intValue));
        BOOL on = onVal.boolValue;
        if (on) {
            self.engine->set_note_on(note);
            if (self.sharedState) self.sharedState->noteOn(note);
        } else {
            self.engine->set_note_off(note);
            if (self.sharedState) self.sharedState->noteOff(note);
        }
    }
    else if ([type isEqualToString:@"togglePlay"]) {
        NSNumber* valueVal = body[@"value"];
        if (valueVal != nil) {
            BOOL target = valueVal.boolValue;
            if (target != _isPlaying) {
                [NSApp sendAction:@selector(menuTogglePlayStop:) to:nil from:self];
            }
        } else {
            [NSApp sendAction:@selector(menuTogglePlayStop:) to:nil from:self];
        }
    }
    else if ([type isEqualToString:@"openSettings"]) {
        [NSApp sendAction:@selector(menuShowSettings:) to:nil from:self];
    }
    else if ([type isEqualToString:@"savePromptHistory"]) {
        NSArray* history = body[@"history"];
        NSNumber* index = body[@"index"];
        if (history) [[NSUserDefaults standardUserDefaults] setObject:history forKey:@"Jam_PromptHistory"];
        if (index) [[NSUserDefaults standardUserDefaults] setObject:index forKey:@"Jam_HistoryIndex"];
    }
    else if ([type isEqualToString:@"saveUserPresets"]) {
        NSDictionary* solo = body[@"solo"];
        NSDictionary* jam = body[@"jam"];
        if ([solo isKindOfClass:[NSDictionary class]]) {
            [[NSUserDefaults standardUserDefaults] setObject:solo forKey:@"Jam_UserPresetsSolo"];
        }
        if ([jam isKindOfClass:[NSDictionary class]]) {
            [[NSUserDefaults standardUserDefaults] setObject:jam forKey:@"Jam_UserPresetsJam"];
        }
    }
    else if ([type isEqualToString:@"saveRockerIndex"]) {
        NSNumber* value = body[@"value"];
        if (value) {
            [[NSUserDefaults standardUserDefaults] setObject:value forKey:@"Jam_RockerIndex"];
        }
    }
    else if ([type isEqualToString:@"log"]) {
        NSString* val = body[@"value"];
        if (val) NSLog(@"Jam UI: %@", val);
    }
    else if ([type isEqualToString:@"selectMidiSource"]) {
        NSNumber* endpointVal = body[@"endpoint"];
        if (endpointVal) {
            uint32_t endpoint = endpointVal.unsignedIntValue;
            [self selectMidiInput:endpoint];
        }
    }
    else if ([type isEqualToString:@"uiReady"]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self connectToEngine];
        });
    }
}

// ─── Model loading (shared core) ─────────────────────────────────────────────

- (void)loadModelAtPath:(NSString*)mlxfnPath {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    NSLog(@"Jam: Loading model from %@", mlxfnPath);
    BOOL success = engine->load_model(mlxfnPath.UTF8String);

    if (success) {
        _modelName = mlxfnPath.lastPathComponent;

        // Auto-load corpus
        NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
        NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
            engine->load_pca_file(corpusPath.UTF8String);
        }

        // Re-apply prompt to engine with proper SOLO prefix
        NSString* savedPrompt = [[NSUserDefaults standardUserDefaults] stringForKey:@"Jam_Prompt"];
        NSString* promptToUse = _currentPromptText.length > 0 ? _currentPromptText
                                : (savedPrompt.length > 0 ? savedPrompt : @"");
        _currentPromptText = promptToUse;
        BOOL isSolo = self.soloMode ? self.soloMode->load(std::memory_order_relaxed) : YES;
        NSString* cleanPrompt = [promptToUse stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        NSString* engineText = @"";
        if (cleanPrompt.length == 0) {
            engineText = @"silence";
        } else {
            engineText = isSolo ? [NSString stringWithFormat:@"SOLO %@", cleanPrompt] : cleanPrompt;
        }
        std::vector<std::string> texts = {engineText.UTF8String, "", "", "", "", ""};
        std::vector<float> weights = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        engine->set_text_prompts(texts, weights);
        engine->set_blend_weights(weights.data(), (int)weights.size());

        [self sendStateUpdate:@{
            @"modelName": mlxfnPath.lastPathComponent,
            @"prompt": promptToUse
        }];

        [[NSUserDefaults standardUserDefaults] setObject:mlxfnPath forKey:@"Jam_ModelPath"];
    } else {
        [self sendStateUpdate:@{@"modelName": [NSString stringWithFormat:@"Failed: %@", mlxfnPath.lastPathComponent]}];
    }
}

- (void)handleLoadModel {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:YES];
    [panel setMessage:@"Select the directory containing your model, or the .mlxfn file."];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result != NSModalResponseOK) return;
        NSURL* url = [panel URL];
        if (!url) return;

        dispatch_async(dispatch_get_main_queue(), ^{
            NSString* path = url.path;
            BOOL isDir = NO;
            [[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDir];

            NSString* mlxfnPath = nil;
            if (isDir) {
                NSArray *contents = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:path error:nil];
                for (NSString *file in contents) {
                    if ([file hasSuffix:@".mlxfn"]) {
                        mlxfnPath = [path stringByAppendingPathComponent:file];
                        break;
                    }
                }
            } else if ([path hasSuffix:@".mlxfn"]) {
                mlxfnPath = path;
            }

            if (!mlxfnPath) {
                [self sendStateUpdate:@{@"modelName": @"No .mlxfn found"}];
                return;
            }

            [self loadModelAtPath:mlxfnPath];
        });
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [panel beginWithCompletionHandler:completionBlock];
    }
}

// ─── Audio prompt loading ────────────────────────────────────────────────────

- (void)loadAudioPromptFileAtPath:(NSString*)path index:(int)index {
    dispatch_async(dispatch_get_main_queue(), ^{
        RealtimeRunner* engine = self.engine;
        if (!engine) return;

        NSString* filename = path.lastPathComponent;
        BOOL readSuccess = NO;
        NSURL* url = [NSURL fileURLWithPath:path];

        ExtAudioFileRef extFile = nullptr;
        OSStatus status = ExtAudioFileOpenURL((__bridge CFURLRef)url, &extFile);
        if (status == noErr && extFile) {
            AudioStreamBasicDescription clientFormat = {};
            clientFormat.mSampleRate = 16000.0;
            clientFormat.mFormatID = kAudioFormatLinearPCM;
            clientFormat.mFormatFlags = kAudioFormatFlagIsFloat;
            clientFormat.mBitsPerChannel = 32;
            clientFormat.mChannelsPerFrame = 1;
            clientFormat.mBytesPerFrame = 4;
            clientFormat.mFramesPerPacket = 1;
            clientFormat.mBytesPerPacket = 4;

            status = ExtAudioFileSetProperty(extFile, kExtAudioFileProperty_ClientDataFormat,
                                              sizeof(clientFormat), &clientFormat);
            if (status == noErr) {
                int maxFrames = 160000;
                std::vector<float> samples(maxFrames, 0.0f);
                AudioBufferList bufferList;
                bufferList.mNumberBuffers = 1;
                bufferList.mBuffers[0].mNumberChannels = 1;
                bufferList.mBuffers[0].mDataByteSize = maxFrames * sizeof(float);
                bufferList.mBuffers[0].mData = samples.data();

                UInt32 framesToRead = maxFrames;
                status = ExtAudioFileRead(extFile, &framesToRead, &bufferList);
                if (status == noErr && framesToRead > 0) {
                    if (framesToRead < (UInt32)maxFrames) {
                        for (UInt32 i = framesToRead; i < (UInt32)maxFrames; ++i)
                            samples[i] = samples[i % framesToRead];
                    }
                    engine->set_audio_prompt_samples(index, filename.UTF8String, samples.data(), maxFrames);
                    readSuccess = YES;
                }
            }
            ExtAudioFileDispose(extFile);
        }

        [self sendStateUpdate:@{
            @"prompt": readSuccess ? filename : @"Error: Load failed",
            @"isAudioPrompt": @(readSuccess),
        }];
    });
}

- (void)handleLoadAudioPrompt:(int)index {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedContentTypes:@[[UTType typeWithIdentifier:@"public.audio"]]];
    [panel setMessage:@"Select an audio file for the prompt"];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result != NSModalResponseOK) return;
        NSURL* url = [panel URL];
        if (!url) return;

        [self loadAudioPromptFileAtPath:url.path index:index];
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [panel beginWithCompletionHandler:completionBlock];
    }
}

- (void)handleSelectDownloadFolder {
    [MagentaModelManager selectDownloadFolderWithParentWindow:self.view.window
                                                  completion:^(NSString *selectedPath, NSData *bookmarkData, NSError *error) {
        if (selectedPath) {
            dispatch_async(dispatch_get_main_queue(), ^{
                // Save custom path bookmarks
                [[NSUserDefaults standardUserDefaults] setObject:bookmarkData forKey:@"MagentaRT_ModelFolderBookmark"];
                [[NSUserDefaults standardUserDefaults] setObject:selectedPath forKey:@"MagentaRT_ModelFolderPath"];

                // Determine if custom resources folder exists inside the selected path
                NSString *customResourcesPath = [selectedPath stringByAppendingPathComponent:@"resources"];
                BOOL hasCustomResources = [[NSFileManager defaultManager] fileExistsAtPath:customResourcesPath];

                NSString *resourcesPathToLoad = hasCustomResources ? customResourcesPath : [NSString stringWithUTF8String:magentart::paths::get_resources_dir().c_str()];

                // Re-initialize the C++ engine with this selected resources folder!
                if (!self.engine->init_assets(resourcesPathToLoad.UTF8String)) {
                    NSLog(@"Jam: Failed to initialize C++ assets from custom path: %@", resourcesPathToLoad);
                } else {
                    NSLog(@"Jam: Successfully initialized C++ assets from path: %@", resourcesPathToLoad);
                    // Save custom resources path for subsequent launches!
                    [[NSUserDefaults standardUserDefaults] setObject:resourcesPathToLoad forKey:@"MagentaRT_CustomResourcesPath"];
                }
                // Force close the onboarding modal!
                [self sendStateUpdate:@{
                    @"downloadPath": selectedPath,
                    @"resourcesMissing": @NO // Close onboarding modal instantly!
                }];

                [self handleListLocalModels];

                // Programmatically auto-load the first available model in the newly selected folder if present!
                NSArray<NSString *> *modelFiles = [MagentaModelManager listLocalModelsInDirectory:[NSURL fileURLWithPath:selectedPath]];
                if (modelFiles.count > 0) {
                    [self handleSelectModel:modelFiles[0]];
                }
            });
        } else if (error) {
            NSLog(@"Jam: Failed to create folder bookmark: %@", error.localizedDescription);
        }
    }];
}

- (void)handleListLocalModels {
    NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
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

    [[NSFileManager defaultManager] createDirectoryAtURL:modelsDir withIntermediateDirectories:YES attributes:nil error:nil];

    NSArray<NSString *> *modelFiles = [MagentaModelManager listLocalModelsInDirectory:modelsDir];

    if (accessGranted) {
        [modelsDir stopAccessingSecurityScopedResource];
    }

    [self sendStateUpdate:@{@"localModels": modelFiles}];
}

- (void)handleSelectModel:(NSString*)modelName {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!self.engine) return;

        NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_ModelFolderBookmark"];
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

        [self loadModelAtPath:mlxfnPath];
        [[NSUserDefaults standardUserDefaults] setObject:modelName forKey:@"Jam_LoadedModelName"];

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleDeleteModel:(NSString *)modelName {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"Jam_ModelSearchBookmark"];
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

        NSError *error = nil;
        [[NSFileManager defaultManager] removeItemAtPath:path error:&error];
        if (error) {
            NSLog(@"Jam: Failed to delete model %@: %@", modelName, error.localizedDescription);
        } else {
            NSLog(@"Jam: Successfully deleted model %@", modelName);
            [self handleListLocalModels];
        }

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleInitResources:(NSString *)modelName {
    BOOL hasModel = modelName && modelName.length > 0;

    [MagentaModelDownloader initializeSharedResourcesWithProgress:^(double progress, NSString *status) {
        double scaledPercent = hasModel ? progress * 0.5 : progress;
        NSString *statusWithProgress = [NSString stringWithFormat:@"[1/2] Shared assets: %@", status];
        if (!hasModel) statusWithProgress = status;

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
            // Start downloading the selected model
            [MagentaModelDownloader downloadModel:modelName progress:^(double progress, NSString *status) {
                double scaledPercent = 0.5 + (progress * 0.5);
                [self sendStateUpdate:@{
                    @"resourcesProgress": @{
                        @"status": @"downloading",
                        @"percent": @(scaledPercent),
                        @"text": [NSString stringWithFormat:@"[2/2] Model: %@", status]
                    }
                }];
            } completion:^(BOOL success, NSError *error) {
                if (success) {
                    // Re-initialize the C++ engine assets with the newly downloaded resources!
                    std::string resources = magentart::paths::get_resources_dir();
                    if (!self.engine->init_assets(resources.c_str())) {
                        NSLog(@"Jam: Failed to re-initialize C++ assets after onboarding download");
                    } else {
                        NSLog(@"Jam: Successfully initialized C++ assets after onboarding download");
                    }

                    [self sendStateUpdate:@{
                        @"resourcesProgress": @{
                            @"status": @"success",
                            @"percent": @(1.0),
                            @"text": @"Onboarding Completed!"
                        },
                        @"resourcesMissing": @NO
                    }];
                    // Re-list local models so it immediately appears in local list
                    [self handleListLocalModels];

                    // Programmatically select and load the newly downloaded model into the C++ engine
                    [self handleSelectModel:modelName];
                } else {
                    [self sendStateUpdate:@{
                        @"resourcesProgress": @{
                            @"status": @"error",
                            @"percent": @(0.5),
                            @"text": error.localizedDescription ?: @"Model download failed"
                        }
                    }];
                }
            }];
        } else {
            // Finished resources download only
            // Re-initialize the C++ engine assets with the newly downloaded resources!
            std::string resources = magentart::paths::get_resources_dir();
            if (!self.engine->init_assets(resources.c_str())) {
                NSLog(@"Jam: Failed to re-initialize C++ assets after onboarding download");
            } else {
                NSLog(@"Jam: Successfully initialized C++ assets after onboarding download");
            }

            [self sendStateUpdate:@{
                @"resourcesProgress": @{
                    @"status": @"success",
                    @"percent": @(1.0),
                    @"text": @"Initialization Completed!"
                },
                @"resourcesMissing": @NO
            }];
        }
    }];
}

- (void)dealloc {
    [_metricsTimer invalidate];
}

// ─── MIDI management ──────────────────────────────────────────────────────────

- (NSArray<NSDictionary*>*)getMIDISourcesList {
    NSMutableArray* sources = [NSMutableArray array];
    ItemCount sourceCount = MIDIGetNumberOfSources();
    for (ItemCount i = 0; i < sourceCount; ++i) {
        MIDIEndpointRef src = MIDIGetSource(i);
        CFStringRef cfName = NULL;
        MIDIObjectGetStringProperty(src, kMIDIPropertyDisplayName, &cfName);
        NSString* name = cfName ? (__bridge_transfer NSString*)cfName : @"Unknown MIDI Source";
        BOOL connected = [self.connectedSources containsObject:@((uint32_t)src)];
        [sources addObject:@{
            @"name": name,
            @"endpoint": @((uint32_t)src),
            @"connected": @(connected)
        }];
    }
    return sources;
}

- (void)handleMIDIStructureChanged {
    NSArray* sources = [self getMIDISourcesList];

    BOOL connectedSourceStillExists = NO;
    for (NSNumber* srcNum in [self.connectedSources allObjects]) {
        uint32_t endpoint = [srcNum unsignedIntValue];
        for (NSDictionary* srcInfo in sources) {
            if ([srcInfo[@"endpoint"] unsignedIntValue] == endpoint) {
                connectedSourceStillExists = YES;
                break;
            }
        }
    }

    if (self.connectedSources.count > 0 && !connectedSourceStillExists) {
        [self selectMidiInput:0];
        return;
    }

    [self sendStateUpdate:@{@"midiSources": sources}];
}

- (void)selectMidiInput:(uint32_t)selectedEndpoint {
    if (!self.midiInputPort || !self.connectedSources) return;

    // 1. Disconnect all currently connected physical MIDI sources
    for (NSNumber* srcNum in [self.connectedSources allObjects]) {
        MIDIEndpointRef endpoint = (MIDIEndpointRef)[srcNum unsignedIntValue];
        MIDIPortDisconnectSource(self.midiInputPort, endpoint);
    }
    [self.connectedSources removeAllObjects];

    if (selectedEndpoint == 0) {
        // "Computer Keyboard" selected
        [self setComputerKeyboardMidiEnabled:YES];
    } else {
        // Physical MIDI input selected
        [self setComputerKeyboardMidiEnabled:NO];
        BOOL success = NO;
        if (selectedEndpoint != 0xFFFFFFFF) { // 0xFFFFFFFF can mean "None"
            MIDIEndpointRef endpoint = (MIDIEndpointRef)selectedEndpoint;
            if (MIDIPortConnectSource(self.midiInputPort, endpoint, NULL) == noErr) {
                [self.connectedSources addObject:@(selectedEndpoint)];
                success = YES;
            }
        }
        if (!success) {
            [self setComputerKeyboardMidiEnabled:YES];
            selectedEndpoint = 0;
        }
    }

    [[NSUserDefaults standardUserDefaults] setInteger:selectedEndpoint forKey:@"Jam_SelectedMidiEndpoint"];
    [self handleMIDIStructureChanged];
}

@end
