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

// Collider view controller — hosts the Collider React UI in a WKWebView.
// Simplified from MagentaRTAppController: single prompt, MIDI/waveform visualization.

#import "ColliderAppController.h"
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

// ─── TEMP Dev server probe ────────────────────────────────────────────────────────

static const int kDevServerPort = 62419;

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

@interface ColliderWebView : WKWebView
@end

@implementation ColliderWebView
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

@interface ColliderAppController () <WKScriptMessageHandler, WKNavigationDelegate>
- (void)handleSelectDownloadFolder;
- (void)handleListLocalModels;
- (void)handleSelectModel:(NSString*)modelName;
- (void)handleDeleteModel:(NSString*)modelName;
- (void)handleInitResources:(NSString*)modelName;
@end

@implementation ColliderAppController {
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
    [MagentaSettings applyParamToEngine:self.engine address:address value:value prefixString:@"Collider"];
}

- (void)restoreSavedParams {
    [MagentaSettings restoreSavedParams:self.engine prefixString:@"Collider"];
}

- (float)readParamFromEngine:(int)address {
    return [MagentaSettings readParamFromEngine:self.engine address:address];
}

// ─── View lifecycle ──────────────────────────────────────────────────────────

- (void)loadView {
    NSView* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 700, 505)];
    view.wantsLayer = YES;
    view.layer.backgroundColor = [NSColor colorWithRed:0.125 green:0.129 blue:0.141 alpha:1.0].CGColor;
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

        _webView = [[ColliderWebView alloc] initWithFrame:self.view.bounds configuration:config];
        _webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        _webView.navigationDelegate = self;
        [_webView setValue:@(NO) forKey:@"drawsBackground"];
        [self.view addSubview:_webView];

        if (isDevServerRunning()) {
            NSLog(@"Collider: Vite dev server detected on port %d — loading with HMR", kDevServerPort);
            [_webView loadRequest:[NSURLRequest requestWithURL:
                [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%d", kDevServerPort]]]];
        } else {
            NSBundle* bundle = [NSBundle mainBundle];
            NSString* uiPath = [bundle pathForResource:@"index" ofType:@"html" inDirectory:@"collider_ui"];
            if (uiPath) {
                NSURL* url = [NSURL fileURLWithPath:uiPath];
                [_webView loadFileURL:url allowingReadAccessToURL:[url URLByDeletingLastPathComponent]];
            } else {
                NSLog(@"Collider: collider_ui/index.html not found in bundle");
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
    ColliderSharedState* shared = self.sharedState;
    if (!engine) return;

    _metricsTicks++;
    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];

    // Send MIDI active notes every frame
    if (shared) {
        NSMutableArray* notes = [NSMutableArray array];
        for (int i = 0; i < 128; i++) {
            if (shared->midiNotes[i].load(std::memory_order_relaxed)) {
                [notes addObject:@(i)];
            }
        }
        stateUpdate[@"activeNotes"] = notes;
    }

    // Send audio level every frame (single scalar — negligible bridge cost)
    if (shared) {
        int head = shared->vizHead.load(std::memory_order_acquire);
        static constexpr int WINDOW = 2048; // ~42ms at 48kHz
        float peak = 0;
        for (int i = 0; i < WINDOW; i++) {
            int idx = (head - WINDOW + i + ColliderSharedState::VIZ_BUF_SIZE) % ColliderSharedState::VIZ_BUF_SIZE;
            float v = fabsf(shared->vizRing[idx]);
            if (v > peak) peak = v;
        }
        stateUpdate[@"audioLevel"] = @(peak);
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
    int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,48};
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
    int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,48};
    for (int addr : addresses) {
        NSString* key = [MagentaSettings paramKeyForAddress:addr];
        if (!key) continue;
        float rawVal = [self readParamFromEngine:addr];
        NSNumber* val = [MagentaSettings paramIsBool:addr] ? @(rawVal > 0.5) : @(rawVal);
        initialParams[key] = val;
        _lastParams[key] = val;
    }

    NSMutableDictionary* state = [NSMutableDictionary dictionary];
    state[@"params"] = initialParams;
    state[@"isPlaying"] = @(_isPlaying);
    if (_modelName) state[@"modelName"] = _modelName;

    // Restore saved prompt
    NSString* savedPrompt = [[NSUserDefaults standardUserDefaults] stringForKey:@"Collider_Prompt"];
    if (savedPrompt) state[@"prompt"] = savedPrompt;

    // Restore saved prompt history
    NSArray* savedHistory = [[NSUserDefaults standardUserDefaults] arrayForKey:@"Collider_PromptHistory"];
    if (savedHistory) {
        state[@"savedPromptHistory"] = savedHistory;
        state[@"savedHistoryIndex"] = [[NSUserDefaults standardUserDefaults] objectForKey:@"Collider_HistoryIndex"] ?: @0;
    }

    NSNumber* savedPalette = [[NSUserDefaults standardUserDefaults] objectForKey:@"Collider_PaletteIndex"];
    if (savedPalette) state[@"savedPaletteIndex"] = savedPalette;

    state[@"computerKeyboardMidi"] = @([[NSUserDefaults standardUserDefaults] boolForKey:@"Collider_ComputerKeyboardMidi"]);

    NSString* searchPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_ModelFolderPath"];
    if (!searchPath) {
        searchPath = [NSString stringWithUTF8String:magentart::paths::get_models_dir().c_str()];
    }
    state[@"resourcesMissing"] = @(![MagentaModelDownloader areSharedResourcesValid]);

    [self sendStateUpdate:state];
    [self handleListLocalModels];
}

- (void)setComputerKeyboardMidiEnabled:(BOOL)enabled {
    [[NSUserDefaults standardUserDefaults] setBool:enabled forKey:@"Collider_ComputerKeyboardMidi"];
    [self sendStateUpdate:@{@"computerKeyboardMidi": @(enabled)}];
}

- (void)notifyModelLoaded:(NSString*)modelName {
    _modelName = modelName;

    dispatch_async(dispatch_get_main_queue(), ^{
        NSMutableDictionary* state = [NSMutableDictionary dictionary];
        state[@"modelName"] = modelName;

        NSMutableDictionary* params = [NSMutableDictionary dictionary];
        int addresses[] = {0,1,3,4,5,6,7,8,9,32,39,48};
        for (int addr : addresses) {
            NSString* key = [MagentaSettings paramKeyForAddress:addr];
            if (!key) continue;
            float rawVal = [self readParamFromEngine:addr];
            params[key] = [MagentaSettings paramIsBool:addr] ? @(rawVal > 0.5) : @(rawVal);
            self->_lastParams[key] = params[key];
        }
        state[@"params"] = params;

        // Always push a prompt to the engine after model load so embeddings are
        // computed from the current text (UI default, saved value, or current
        // in-memory prompt). Without this, the engine runs on hardcoded fallback
        // musiccoca tokens until the user edits the prompt field.
        if (self.engine) {
            NSString* savedPrompt = [[NSUserDefaults standardUserDefaults] stringForKey:@"Collider_Prompt"];
            NSString* promptToUse = self->_currentPromptText.length > 0 ? self->_currentPromptText
                                    : (savedPrompt.length > 0 ? savedPrompt : @"funky bass guitar");
            self->_currentPromptText = promptToUse;
            state[@"prompt"] = promptToUse;
            std::vector<std::string> texts = {promptToUse.UTF8String, "", "", "", "", ""};
            std::vector<float> weights = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            self.engine->set_text_prompts(texts, weights);
        }

        [self sendStateUpdate:state];
    });
}

// ─── Navigation delegate ─────────────────────────────────────────────────────

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation {
    NSLog(@"Collider: WKWebView loaded");
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
    else if ([type isEqualToString:@"textPrompts"]) {
        NSArray* promptsArray = body[@"value"];
        if ([promptsArray isKindOfClass:[NSArray class]] && self.engine) {
            std::vector<std::string> texts;
            std::vector<float> weights;
            for (NSDictionary* p in promptsArray) {
                NSString* text = p[@"text"];
                NSNumber* weight = p[@"weight"];
                if ([text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]]) {
                    texts.push_back(text.UTF8String);
                    weights.push_back(weight.floatValue);
                }
            }
            self.engine->set_text_prompts(texts, weights);
            self.engine->set_blend_weights(weights.data(), (int)weights.size());

            // Persist current prompt and history
            if (texts.size() > 0) {
                NSString* prompt = [NSString stringWithUTF8String:texts[0].c_str()];
                _currentPromptText = prompt;
                [[NSUserDefaults standardUserDefaults] setObject:prompt forKey:@"Collider_Prompt"];
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
    else if ([type isEqualToString:@"deleteModel"]) {
        NSString* name = body[@"name"];
        if (name) {
            [self handleDeleteModel:name];
        }
    }
    else if ([type isEqualToString:@"initResources"]) {
        NSString* name = body[@"modelName"];
        [self handleInitResources:name];
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
    else if ([type isEqualToString:@"loadAudioPrompt"]) {
        [self handleLoadAudioPrompt:0];
    }
    else if ([type isEqualToString:@"clearAudioPrompt"]) {
        if (self.engine) {
            self.engine->set_audio_prompt(0, "");
        }
        [self sendStateUpdate:@{
            @"prompt": _currentPromptText ?: @"funky bass guitar",
            @"isAudioPrompt": @NO,
        }];
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
        [NSApp sendAction:@selector(menuTogglePlayStop:) to:nil from:self];
    }
    else if ([type isEqualToString:@"openSettings"]) {
        [NSApp sendAction:@selector(menuShowSettings:) to:nil from:self];
    }
    else if ([type isEqualToString:@"savePromptHistory"]) {
        NSArray* history = body[@"history"];
        NSNumber* index = body[@"index"];
        NSNumber* palette = body[@"paletteIndex"];
        if (history) [[NSUserDefaults standardUserDefaults] setObject:history forKey:@"Collider_PromptHistory"];
        if (index) [[NSUserDefaults standardUserDefaults] setObject:index forKey:@"Collider_HistoryIndex"];
        if (palette) [[NSUserDefaults standardUserDefaults] setObject:palette forKey:@"Collider_PaletteIndex"];
    }
    else if ([type isEqualToString:@"log"]) {
        NSString* val = body[@"value"];
        if (val) NSLog(@"Collider UI: %@", val);
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

    NSLog(@"Collider: Loading model from %@", mlxfnPath);
    BOOL success = engine->load_model(mlxfnPath.UTF8String);

    if (success) {
        self->_modelName = mlxfnPath.lastPathComponent;

        // Auto-load corpus
        NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
        NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
            engine->load_pca_file(corpusPath.UTF8String);
        }

        // Always push a prompt to the engine, falling back through
        // current → saved → bundled default.
        NSString* savedPrompt = [[NSUserDefaults standardUserDefaults] stringForKey:@"Collider_Prompt"];
        NSString* promptToUse = self->_currentPromptText.length > 0 ? self->_currentPromptText
                                : (savedPrompt.length > 0 ? savedPrompt : @"funky bass guitar");
        self->_currentPromptText = promptToUse;
        {
            std::vector<std::string> texts = {promptToUse.UTF8String, "", "", "", "", ""};
            std::vector<float> weights = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            engine->set_text_prompts(texts, weights);
        }

        [self sendStateUpdate:@{
            @"modelName": mlxfnPath.lastPathComponent,
            @"prompt": promptToUse
        }];

        [[NSUserDefaults standardUserDefaults] setObject:mlxfnPath forKey:@"Collider_ModelPath"];
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
            RealtimeRunner* engine = self.engine;
            if (!engine) return;

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
                    NSLog(@"Collider: Failed to initialize C++ assets from custom path: %@", resourcesPathToLoad);
                } else {
                    NSLog(@"Collider: Successfully initialized C++ assets from path: %@", resourcesPathToLoad);
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
            NSLog(@"Collider: Failed to create folder bookmark: %@", error.localizedDescription);
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
        [[NSUserDefaults standardUserDefaults] setObject:modelName forKey:@"Collider_LoadedModelName"];

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

// ─── Audio prompt loading ────────────────────────────────────────────────────

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

        dispatch_async(dispatch_get_main_queue(), ^{
            RealtimeRunner* engine = self.engine;
            if (!engine) return;

            NSString* filename = url.lastPathComponent;
            BOOL readSuccess = NO;

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
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [panel beginWithCompletionHandler:completionBlock];
    }
}

- (void)handleDeleteModel:(NSString *)modelName {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:@"Collider_ModelSearchBookmark"];
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
            NSLog(@"Collider: Failed to delete model %@: %@", modelName, error.localizedDescription);
        } else {
            NSLog(@"Collider: Successfully deleted model %@", modelName);
            [self handleListLocalModels];
        }

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleInitResources:(NSString *)modelName {
    BOOL hasModel = modelName && modelName.length > 0;

    NSArray *resourceFiles = @[
        @"resources/musiccoca/text_encoder.tflite",
        @"resources/musiccoca/pretrained_vector_quantizer.tflite",
        @"resources/musiccoca/audio_preprocessor.tflite",
        @"resources/musiccoca/music_encoder.tflite",
        @"resources/musiccoca/spm.model",
        @"resources/spectrostream/spectrostream_encoder.mlxfn",
        @"resources/spectrostream/decoder.safetensors",
        @"resources/spectrostream/encoder.safetensors",
        @"resources/spectrostream/quantizer.safetensors"
    ];

    NSMutableArray *allFiles = [NSMutableArray arrayWithArray:resourceFiles];
    if (hasModel) {
        NSString *prefix = [NSString stringWithFormat:@"models/%@", modelName];
        [allFiles addObject:[NSString stringWithFormat:@"%@/%@.mlxfn", prefix, modelName]];
        [allFiles addObject:[NSString stringWithFormat:@"%@/%@_state.safetensors", prefix, modelName]];
    }

    NSMutableArray *basenames = [NSMutableArray array];
    for (NSString *path in allFiles) {
        [basenames addObject:[path lastPathComponent]];
    }

    [self sendStateUpdate:@{
        @"onboardingFiles": basenames,
        @"resourcesProgress": @{
            @"status": @"downloading",
            @"percent": @(0.0),
            @"currentFile": basenames[0],
            @"currentIndex": @(0)
        }
    }];

    [MagentaModelDownloader initializeSharedResourcesWithProgress:^(double progress, NSString *status) {
        NSInteger resourceIndex = 0;
        NSString *currentBasename = [status lastPathComponent];
        for (NSInteger i = 0; i < resourceFiles.count; ++i) {
            if ([[resourceFiles[i] lastPathComponent] isEqualToString:currentBasename]) {
                resourceIndex = i;
                break;
            }
        }

        double scaledPercent = hasModel ? progress * 0.5 : progress;
        NSString *statusWithProgress = hasModel
            ? [NSString stringWithFormat:@"[1/2] Shared assets: %@", status]
            : status;

        [self sendStateUpdate:@{
            @"resourcesProgress": @{
                @"status": @"downloading",
                @"percent": @(scaledPercent),
                @"currentFile": currentBasename,
                @"currentIndex": @(resourceIndex),
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
                NSString *currentBasename = [status lastPathComponent];

                NSInteger modelIndex = 0;
                if ([currentBasename containsString:@"_state.safetensors"]) {
                    modelIndex = 1;
                }
                NSInteger overallIndex = 9 + modelIndex;

                [self sendStateUpdate:@{
                    @"resourcesProgress": @{
                        @"status": @"downloading",
                        @"percent": @(scaledPercent),
                        @"currentFile": currentBasename,
                        @"currentIndex": @(overallIndex),
                        @"text": [NSString stringWithFormat:@"[2/2] Model: %@", status]
                    }
                }];
            } completion:^(BOOL success, NSError *error) {
                if (success) {
                    // Re-initialize the C++ engine assets with the newly downloaded resources!
                    std::string resources = magentart::paths::get_resources_dir();
                    if (!self.engine->init_assets(resources.c_str())) {
                        NSLog(@"Collider: Failed to re-initialize C++ assets after onboarding download");
                    } else {
                        NSLog(@"Collider: Successfully initialized C++ assets after onboarding download");
                    }

                    [self sendStateUpdate:@{
                        @"resourcesProgress": @{
                            @"status": @"success",
                            @"percent": @(1.0),
                            @"text": @"Onboarding Completed!"
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
                            @"text": error.localizedDescription ?: @"Model download failed"
                        }
                    }];
                }
            }];
        } else {
            std::string resources = magentart::paths::get_resources_dir();
            self.engine->init_assets(resources.c_str());

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

@end
