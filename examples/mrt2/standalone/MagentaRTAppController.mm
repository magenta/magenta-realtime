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

// Standalone app view controller — hosts the React UI in a WKWebView
// and bridges parameters directly to RealtimeRunner (no AUParameterTree).
//
// ─── INTENTIONAL DUPLICATION WITH examples/auv3/MagentaRT_AudioUnit.mm ─────
//
// The WebView IPC glue (WKScriptMessageHandler dispatch, React state push via
// `window.updateState(...)`, parameter mirroring, load-model file dialogs)
// exists in two places on purpose. Consolidating into a shared Objective-C++
// helper is possible but deliberately skipped: each host stays compilable on
// its own, the AU sandbox constraints don't leak into the standalone, and the
// message shapes happen to differ in small ways (AU parameters route through
// AUParameterTree observers; standalone writes atomically to RealtimeRunner).
//
// If the two ever drift, treat examples/auv3/MagentaRT_AudioUnit.mm as the
// canonical reference — it's ~3× larger and receives the most development.
// ────────────────────────────────────────────────────────────────────────────

#import "MagentaRTAppController.h"
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

// ─── WKWebView subclass for keyboard shortcuts ──────────────────────────────

@interface MagentaRTWebView : WKWebView
@end

@implementation MagentaRTWebView

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

// ─── Helper functions ────────────────────────────────────────────────────────

static NSString* paramKeyForAddress(int address) {
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
        case 45: return @"midigate";
        case 46: return @"onsetmode";
        case 48: return @"cfgdrums";
        default:
            if (address >= 33 && address <= 38) {
                return [NSString stringWithFormat:@"pca_coeff_%d", address - 33];
            }
            return nil;
    }
}

static BOOL paramIsBool(int address) {
    if (address == 6 || address == 9 || address == 31 || address == 32 || address == 39 || address == 45 || address == 46) return YES;
    return NO;
}

// ─── View Controller ─────────────────────────────────────────────────────────

@interface MagentaRTAppController () <WKScriptMessageHandler, WKNavigationDelegate>
@end

@implementation MagentaRTAppController {
    WKWebView* _webView;
    NSTimer* _metricsTimer;
    NSMutableDictionary* _lastParams;
    int _metricsTicks;
    NSString* _modelName;
    NSString* _pcaFileName;
    NSArray<NSDictionary*>* _prompts;
    NSURL* _modelDirectoryURL;
    BOOL _isPlaying;
}

// ── Bank file paths (emulator-style save states) ─────────────────────────────

static NSString* bankFilePath(int index) {
    std::string banksDir = magentart::paths::get_banks_dir();
    NSString* dir = [NSString stringWithUTF8String:banksDir.c_str()];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES attributes:nil error:nil];
    return [dir stringByAppendingPathComponent:
            [NSString stringWithFormat:@"bank_%d.safetensors", index + 1]];
}

// ─── Parameter bridging ──────────────────────────────────────────────────────

// Addresses of "advanced controls" that should persist across app launches.
// Excludes spatial/cursor params (10-30), one-shot reset (31), bypass (32), and PCA coeffs (33-38).
static BOOL shouldPersistParam(int address) {
    return (address >= 0 && address <= 9 && address != 2) || address == 39 || address == 48;
}

- (void)applyParamToEngine:(int)address value:(float)value {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    if (address == 0) engine->set_temperature(value);
    else if (address == 1) engine->set_top_k((int)value);
    else if (address == 3) engine->set_cfg_musiccoca(value);
    else if (address == 4) engine->set_cfg_notes(value);
    else if (address == 5) engine->set_volume_db(value);
    else if (address == 6) engine->set_mute(value > 0.5f);
    else if (address == 7) engine->set_unmask_width((int)value);
    else if (address == 8) {
        size_t cap = 8192;
        if (value < 0.5f) cap = 2048;
        else if (value < 1.5f) cap = 4096;
        engine->set_buffer_size(cap);
    }
    else if (address == 9) engine->set_latency_comp(value > 0.5f);
    else if (address >= 10 && address <= 15) engine->set_blend_weight(address - 10, value);
    else if (address == 31) {
        if (value > 0.5f) engine->trigger_reset();
    }
    else if (address == 32) engine->set_bypass(value > 0.5f);
    else if (address >= 33 && address <= 38) {
        engine->set_pca_coeff(address - 33, value);
    }
    else if (address == 39) engine->set_drumless(value > 0.5f);
    else if (address == 45) engine->set_midi_gate_enabled(value > 0.5f);
    else if (address == 46) engine->set_onset_mode(value > 0.5f);
    else if (address == 48) engine->set_cfg_drums(value);

    // Persist advanced control values to NSUserDefaults
    if (shouldPersistParam(address)) {
        NSString* key = paramKeyForAddress(address);
        if (key) {
            NSString* defaultsKey = [NSString stringWithFormat:@"MagentaRT_Param_%@", key];
            [[NSUserDefaults standardUserDefaults] setFloat:value forKey:defaultsKey];
        }
    }
}

- (void)restoreSavedParams {
    NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
    for (int i = 0; i <= 48; i++) {
        if (!shouldPersistParam(i)) continue;
        NSString* key = paramKeyForAddress(i);
        if (!key) continue;
        NSString* defaultsKey = [NSString stringWithFormat:@"MagentaRT_Param_%@", key];
        if ([defaults objectForKey:defaultsKey]) {
            float value = [defaults floatForKey:defaultsKey];
            [self applyParamToEngine:i value:value];
        } else {
            float value = 0.0f;
            if (i == 0) value = kMagentaDefaultTemperature;
            else if (i == 1) value = kMagentaDefaultTopK;
            else if (i == 3) value = kMagentaDefaultCfgMusicCoCa;
            else if (i == 4) value = kMagentaDefaultCfgNotes;
            else if (i == 5) value = kMagentaDefaultVolume;
            else if (i == 7) value = kMagentaDefaultUnmaskWidth;
            else if (i == 8) value = kMagentaDefaultBufferSize;
            else if (i == 48) value = kMagentaDefaultCfgDrums;

            [self applyParamToEngine:i value:value];
        }
    }
}

- (void)restorePrompts:(NSArray*)prompts {
    if (!prompts || ![prompts isKindOfClass:[NSArray class]]) return;
    _prompts = prompts;

    // Re-apply to engine so embeddings are freshly computed
    RealtimeRunner* engine = self.engine;
    if (engine) {
        std::vector<std::string> std_texts;
        std::vector<float> std_weights;
        int idx = 0;
        for (NSDictionary* p in prompts) {
            NSString* text = p[@"text"];
            NSNumber* weight = p[@"weight"];
            BOOL isValid = [text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]];

            std_texts.push_back(isValid ? text.UTF8String : "");
            std_weights.push_back(isValid ? weight.floatValue : 0.0f);
            idx++;
        }
        engine->set_text_prompts(std_texts, std_weights);
        engine->set_blend_weights(std_weights.data(), (int)std_weights.size());
    }
}

- (float)readParamFromEngine:(int)address {
    RealtimeRunner* engine = self.engine;
    if (!engine) return 0;

    if (address == 0) return engine->get_temperature();
    else if (address == 1) return (float)engine->get_top_k();
    else if (address == 3) return engine->get_cfg_musiccoca();
    else if (address == 4) return engine->get_cfg_notes();
    else if (address == 5) return engine->get_volume_db();
    else if (address == 6) return engine->get_mute() ? 1.0f : 0.0f;
    else if (address == 7) return (float)engine->get_unmask_width();
    else if (address == 8) {
        size_t cap = engine->get_buffer_size();
        if (cap <= 2048) return 0.0f;
        if (cap <= 4096) return 1.0f;
        return 2.0f;
    }
    else if (address == 9) return engine->get_latency_comp() ? 1.0f : 0.0f;
    else if (address >= 10 && address <= 15) return engine->get_blend_weight(address - 10);
    else if (address == 31) return 0.0f;
    else if (address == 32) return engine->get_bypass() ? 1.0f : 0.0f;
    else if (address >= 33 && address <= 38) {
        return engine->get_pca_coeff(address - 33);
    }
    else if (address == 39) return engine->get_drumless() ? 1.0f : 0.0f;
    else if (address == 45) return engine->get_midi_gate_enabled() ? 1.0f : 0.0f;
    else if (address == 46) return engine->get_onset_mode() ? 1.0f : 0.0f;
    else if (address == 48) return engine->get_cfg_drums();
    return 0.0f;
}

// ─── View lifecycle ──────────────────────────────────────────────────────────

- (void)loadView {
    NSView* view = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 1075, 470)];
    view.wantsLayer = YES;
    view.layer.backgroundColor = [NSColor colorWithRed:0.15 green:0.15 blue:0.2 alpha:1.0].CGColor;
    self.view = view;
}

- (void)viewDidAppear {
    [super viewDidAppear];

    if (!_webView) {
        WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
        [config.preferences setValue:@YES forKey:@"developerExtrasEnabled"];
        [config.preferences setValue:@YES forKey:@"allowFileAccessFromFileURLs"];
        @try {
            [config setValue:@YES forKey:@"allowUniversalAccessFromFileURLs"];
        } @catch (NSException *e) { }

        NSString *js = @"window.__HOST_MODE__ = 'standalone';"
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
            NSLog(@"MagentaRT Standalone 2: Vite dev server detected on port %d — loading with HMR", kDevServerPort);
            [_webView loadRequest:[NSURLRequest requestWithURL:
                [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%d", kDevServerPort]]]];
        } else {
            NSBundle* bundle = [NSBundle mainBundle];
            NSString* uiPath = [bundle pathForResource:@"index" ofType:@"html" inDirectory:@"ui"];
            if (uiPath) {
                NSURL* url = [NSURL fileURLWithPath:uiPath];
                NSURL* folderUrl = [url URLByDeletingLastPathComponent];
                [_webView loadFileURL:url allowingReadAccessToURL:folderUrl];
            } else {
                NSLog(@"MagentaRT Standalone 2: ui/index.html not found in bundle");
            }
        }
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

    if (_webView) {
        [_webView.configuration.userContentController removeScriptMessageHandlerForName:@"auHost"];
        [_webView removeFromSuperview];
        _webView = nil;
    }
}

// ─── Metrics polling (25 Hz) ────────────────────────────────────────────────

- (void)updateMetrics {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    _metricsTicks++;
    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];

    // Send audio levels and MIDI note activity every frame (25 Hz)
    StandaloneSharedState* shared = self.sharedState;
    if (shared) {
        float pL = 0.0f;
        float pR = 0.0f;
        shared->levelProcessor.read_and_reset_peaks(pL, pR);
        stateUpdate[@"audioLevels"] = @{
            @"left": @(pL),
            @"right": @(pR)
        };

        // MIDI note activity for the green LED indicator
        NSMutableArray* notes = [NSMutableArray array];
        for (int i = 0; i < 128; i++) {
            if (shared->midiNotes[i].load(std::memory_order_relaxed)) {
                [notes addObject:@(i)];
            }
        }
        stateUpdate[@"activeNotes"] = notes;
    }

    if (_metricsTicks >= 5) {
        _metricsTicks = 0;
        EngineMetrics m = engine->get_metrics();
        int textStatus = engine->get_text_encoder_status();
        int quantStatus = engine->get_quantizer_status();

        // Check for error text from the engine (e.g. text encoder failures)
        NSMutableArray* mutablePrompts = [_prompts mutableCopy];
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
            _prompts = mutablePrompts;
            stateUpdate[@"textPrompts"] = mutablePrompts;
        }

        // Forward engine logs to the UI
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
            @"transportFlags": @(0),  // standalone: no transport
            @"droppedFrames": @(m.dropped_frames)
        };
    }

    NSMutableDictionary* params = [NSMutableDictionary dictionary];
    for (int i = 0; i <= 48; i++) {
        // Skip weight params (10-15): the UI is the source of truth for
        // slider positions.  The engine normalises raw weights internally.
        if (i >= 10 && i <= 15) continue;
        NSString* key = paramKeyForAddress(i);
        if (!key) continue;
        float rawVal = [self readParamFromEngine:i];
        NSNumber* val = paramIsBool(i) ? @(rawVal > 0.5) : @(rawVal);
        NSNumber* lastVal = _lastParams[key];
        if (!lastVal || ![lastVal isEqualToNumber:val]) {
            params[key] = val;
            _lastParams[key] = val;
        }
    }

    if (params.count > 0) {
        stateUpdate[@"params"] = params;
    }

    if (stateUpdate.count > 0) {
        [self sendStateUpdate:stateUpdate];
    }
}

// ─── State push to React ────────────────────────────────────────────────────

- (void)sendStateUpdate:(NSDictionary*)state {
    if (!_webView) return;

    NSError* error = nil;
    NSData* jsonData = [NSJSONSerialization dataWithJSONObject:state options:0 error:&error];
    if (error) {
        NSLog(@"MagentaRT Standalone JSON Error: %@", error);
        return;
    }

    NSString* jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
    NSString* script = [NSString stringWithFormat:@"if (window.updateState) { window.updateState(%@); }", jsonString];

    [_webView evaluateJavaScript:script completionHandler:nil];
}

- (void)sendPlayState:(BOOL)playing {
    _isPlaying = playing;
    [self sendStateUpdate:@{@"isPlaying": @(playing)}];
}

- (void)connectToEngine {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    NSMutableDictionary* initialParams = [NSMutableDictionary dictionary];
    for (int i = 0; i <= 48; i++) {
        // Skip weight params — prompts carry their own weights via textPrompts.
        if (i >= 10 && i <= 15) continue;
        NSString* key = paramKeyForAddress(i);
        if (!key) continue;
        float rawVal = [self readParamFromEngine:i];
        NSNumber* val = paramIsBool(i) ? @(rawVal > 0.5) : @(rawVal);
        initialParams[key] = val;
        _lastParams[key] = val;
    }

    NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];
    stateUpdate[@"params"] = initialParams;
    stateUpdate[@"isPlaying"] = @(_isPlaying);

    if (_modelName) stateUpdate[@"modelName"] = _modelName;
    if (_prompts) stateUpdate[@"textPrompts"] = _prompts;

    // Push bank existence status
    NSFileManager* fm = [NSFileManager defaultManager];
    stateUpdate[@"bankStatus"] = @[
        @([fm fileExistsAtPath:bankFilePath(0)]),
        @([fm fileExistsAtPath:bankFilePath(1)]),
        @([fm fileExistsAtPath:bankFilePath(2)]),
    ];

    // Restore persisted prompt surface layout
    NSDictionary* savedPromptSurface = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_PromptSurface"];
    if (savedPromptSurface) stateUpdate[@"prompt_surface"] = savedPromptSurface;

    if (engine->is_pca_loaded()) {
        stateUpdate[@"pcaLoaded"] = @YES;
        stateUpdate[@"pcaComponentCount"] = @(engine->pca_component_count());
        stateUpdate[@"pcaCentroidCount"] = @(engine->pca_centroid_count());
        stateUpdate[@"pcaFileName"] = _pcaFileName ?: @"corpus.safetensors";
    }

    // Model management state
    NSString* searchPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_ModelFolderPath"];
    if (!searchPath) {
        searchPath = [NSString stringWithUTF8String:magentart::paths::get_models_dir().c_str()];
    }
    stateUpdate[@"downloadPath"] = searchPath;
    stateUpdate[@"resourcesMissing"] = @(![MagentaModelDownloader areSharedResourcesValid]);

    [self sendStateUpdate:stateUpdate];
    [self handleListLocalModels];
    [self handleMIDIStructureChanged];
}

// ─── Notification helpers (called by AppDelegate after auto-load) ───────────

- (void)notifyModelLoaded:(NSString*)modelName {
    _modelName = modelName;
    dispatch_async(dispatch_get_main_queue(), ^{
        // Full state push — the webview may have already sent uiReady before model load finished,
        // so connectToEngine would have sent stale/empty state. Re-send everything now.
        NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionary];
        stateUpdate[@"modelName"] = modelName;

        if (self->_prompts) stateUpdate[@"textPrompts"] = self->_prompts;

        // Send current param values
        NSMutableDictionary* params = [NSMutableDictionary dictionary];
        for (int i = 0; i <= 48; i++) {
            // Skip weight params — prompts carry their own weights via textPrompts.
            if (i >= 10 && i <= 15) continue;
            NSString* key = paramKeyForAddress(i);
            if (!key) continue;
            float rawVal = [self readParamFromEngine:i];
            NSNumber* val = paramIsBool(i) ? @(rawVal > 0.5) : @(rawVal);
            params[key] = val;
            self->_lastParams[key] = val;
        }
        stateUpdate[@"params"] = params;

        [self sendStateUpdate:stateUpdate];
    });
}

- (void)notifyPCALoaded:(int)componentCount centroidCount:(int)centroidCount fileName:(NSString*)fileName {
    _pcaFileName = fileName;
    dispatch_async(dispatch_get_main_queue(), ^{
        [self sendStateUpdate:@{
            @"pcaLoaded": @YES,
            @"pcaComponentCount": @(componentCount),
            @"pcaCentroidCount": @(centroidCount),
            @"pcaFileName": fileName ?: @""
        }];
    });
}

// ─── WKWebView navigation delegate ──────────────────────────────────────────

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation {
    NSLog(@"MagentaRT Standalone: WKWebView didFinishNavigation");
}

- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"MagentaRT Standalone: WKWebView didFailProvisionalNavigation: %@", error);
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error {
    NSLog(@"MagentaRT Standalone: WKWebView didFailNavigation: %@", error);
}

// ─── WKScriptMessageHandler — messages from React UI ────────────────────────

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
            _prompts = promptsArray;
            std::vector<std::string> std_texts;
            std::vector<float> std_weights;
            for (NSDictionary* p in promptsArray) {
                NSString* text = p[@"text"];
                NSNumber* weight = p[@"weight"];
                BOOL isValid = [text isKindOfClass:[NSString class]] && [weight isKindOfClass:[NSNumber class]];
                std_texts.push_back(isValid ? text.UTF8String : "");
                std_weights.push_back(isValid ? weight.floatValue : 0.0f);
            }
            self.engine->set_text_prompts(std_texts, std_weights);
            // Push explicit blend weights to the engine
            self.engine->set_blend_weights(std_weights.data(), (int)std_weights.size());

            // Persist prompts
            [[NSUserDefaults standardUserDefaults] setObject:promptsArray forKey:@"MagentaRT_Prompts"];
        }
    }
    else if ([type isEqualToString:@"promptSurfaceState"]) {
        // Persist prompt surface layout
        NSDictionary* promptSurfaceDict = body[@"value"];
        if ([promptSurfaceDict isKindOfClass:[NSDictionary class]]) {
            [[NSUserDefaults standardUserDefaults] setObject:promptSurfaceDict forKey:@"MagentaRT_PromptSurface"];
        }
    }
    else if ([type isEqualToString:@"loadModel"]) {
        [self handleLoadModel];
    }
    else if ([type isEqualToString:@"loadPCAFile"]) {
        [self handleLoadPCAFile];
    }
    else if ([type isEqualToString:@"setMusicCoCaModel"]) {
        NSString* subfolder = body[@"value"];
        if ([subfolder isKindOfClass:[NSString class]] && self.engine) {
            NSString* loadPath = [NSString stringWithUTF8String:magentart::paths::get_resources_dir().c_str()];
            if (![[NSFileManager defaultManager] fileExistsAtPath:[loadPath stringByAppendingPathComponent:subfolder]]) {
                loadPath = [NSBundle mainBundle].resourcePath;
            }
            self.engine->load_musiccoca_model(loadPath.UTF8String, subfolder.UTF8String);
            [self sendStateUpdate:@{@"musicCocaModelName": subfolder}];
        }
    }
    else if ([type isEqualToString:@"saveBank"]) {
        NSNumber* indexVal = body[@"index"];
        if (indexVal && self.engine) {
            int idx = indexVal.intValue;
            NSString* path = bankFilePath(idx);
            BOOL success = self.engine->save_state(path.UTF8String);
            NSLog(@"MagentaRT Standalone 2: %@ bank %d to %@",
                  success ? @"Saved" : @"Failed to save", idx + 1, path);
            // Push updated bank status back to UI
            NSFileManager* fm = [NSFileManager defaultManager];
            [self sendStateUpdate:@{@"bankStatus": @[
                @([fm fileExistsAtPath:bankFilePath(0)]),
                @([fm fileExistsAtPath:bankFilePath(1)]),
                @([fm fileExistsAtPath:bankFilePath(2)]),
            ]}];
        }
    }
    else if ([type isEqualToString:@"loadBank"]) {
        NSNumber* indexVal = body[@"index"];
        if (indexVal && self.engine) {
            int idx = indexVal.intValue;
            NSString* path = bankFilePath(idx);
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                BOOL success = self.engine->load_state(path.UTF8String);
                NSLog(@"MagentaRT Standalone 2: %@ bank %d from %@",
                      success ? @"Loaded" : @"Failed to load", idx + 1, path);
            } else {
                NSLog(@"MagentaRT Standalone 2: Bank %d file does not exist", idx + 1);
            }
        }
    }
    else if ([type isEqualToString:@"checkBanks"]) {
        NSFileManager* fm = [NSFileManager defaultManager];
        [self sendStateUpdate:@{@"bankStatus": @[
            @([fm fileExistsAtPath:bankFilePath(0)]),
            @([fm fileExistsAtPath:bankFilePath(1)]),
            @([fm fileExistsAtPath:bankFilePath(2)]),
        ]}];
    }
    else if ([type isEqualToString:@"loadAudioPrompt"]) {
        NSNumber* indexValue = body[@"index"];
        if (indexValue && self.engine) {
            [self handleLoadAudioPrompt:indexValue.intValue];
        }
    }
    else if ([type isEqualToString:@"clearAudioPrompt"]) {
        NSNumber* indexValue = body[@"index"];
        if (indexValue && self.engine) {
            int index = indexValue.intValue;
            self.engine->set_audio_prompt(index, "");
            // Update prompts array to reflect cleared audio state
            NSMutableArray* mutablePrompts = [_prompts mutableCopy];
            if (mutablePrompts && index < (int)mutablePrompts.count) {
                NSMutableDictionary* p = [mutablePrompts[index] mutableCopy];
                p[@"text"] = @"";
                p[@"isAudio"] = @NO;
                mutablePrompts[index] = p;
                _prompts = mutablePrompts;
                [self sendStateUpdate:@{@"textPrompts": mutablePrompts}];
            }
        }
    }
    else if ([type isEqualToString:@"resetModel"]) {
        if (self.engine) {
            self.engine->reset();
        }
    }
    else if ([type isEqualToString:@"resetToFactory"]) {
        if (self.engine) {
            self.engine->reset_to_factory();
            NSLog(@"MagentaRT Standalone 2: Reset to factory state");
        }
    }
    else if ([type isEqualToString:@"silentPrefill"]) {
        [self handleSilentPrefill];
    }
    else if ([type isEqualToString:@"audioPrefill"]) {
        [self handleAudioPrefill];
    }
    else if ([type isEqualToString:@"log"]) {
        NSString* val = body[@"value"];
        if (val) NSLog(@"MagentaRT UI: %@", val);
    }
    else if ([type isEqualToString:@"togglePlay"]) {
        NSLog(@"MagentaRT Standalone 2: togglePlay received from JS");
        [NSApp sendAction:@selector(menuTogglePlayStop:) to:nil from:self];
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
    else if ([type isEqualToString:@"selectDownloadFolder"]) {
        [self handleSelectDownloadFolder];
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
    else if ([type isEqualToString:@"selectMidiSource"]) {
        NSNumber* endpointVal = body[@"endpoint"];
        if (endpointVal) {
            uint32_t endpoint = endpointVal.unsignedIntValue;
            [self selectMidiInput:endpoint];
        }
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
    else if ([type isEqualToString:@"uiReady"]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self connectToEngine];
        });
    }
}

// ─── Model loading ──────────────────────────────────────────────────────────

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

            NSLog(@"MagentaRT Standalone: Loading model from %@", mlxfnPath);
            BOOL success = engine->load_model(mlxfnPath.UTF8String);

            if (success) {
                NSLog(@"MagentaRT Standalone: Model loaded successfully.");
                self->_modelName = mlxfnPath.lastPathComponent;
                self->_modelDirectoryURL = [NSURL fileURLWithPath:[mlxfnPath stringByDeletingLastPathComponent]];

                NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionaryWithDictionary:@{
                    @"modelName": mlxfnPath.lastPathComponent
                }];

                // Auto-load corpus.safetensors if present
                NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
                NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
                if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
                    if (engine->load_pca_file(corpusPath.UTF8String)) {
                        NSLog(@"MagentaRT Standalone: Auto-loaded corpus from %@", corpusPath.lastPathComponent);
                        self->_pcaFileName = corpusPath.lastPathComponent;
                        [stateUpdate addEntriesFromDictionary:@{
                            @"pcaLoaded": @YES,
                            @"pcaComponentCount": @(engine->pca_component_count()),
                            @"pcaCentroidCount": @(engine->pca_centroid_count()),
                            @"pcaFileName": corpusPath.lastPathComponent ?: @""
                        }];
                    }
                }

                // Re-apply saved prompts
                if (self->_prompts) {
                    [self restorePrompts:self->_prompts];
                }

                [self sendStateUpdate:stateUpdate];

                // Persist model path for auto-load on next launch
                [[NSUserDefaults standardUserDefaults] setObject:mlxfnPath forKey:@"MagentaRT_ModelPath"];
            } else {
                NSLog(@"MagentaRT Standalone: Model load failed.");
                [self sendStateUpdate:@{@"modelName": [NSString stringWithFormat:@"Failed: %@", mlxfnPath.lastPathComponent]}];
            }
        });
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

// ─── PCA / corpus loading ───────────────────────────────────────────────────

- (void)handleLoadPCAFile {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedContentTypes:@[[UTType typeWithFilenameExtension:@"safetensors"]]];
    [panel setMessage:@"Select a .safetensors corpus file (PCA + centroids)"];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result != NSModalResponseOK) return;
        NSURL* url = [panel URL];
        if (!url) return;

        dispatch_async(dispatch_get_main_queue(), ^{
            RealtimeRunner* engine = self.engine;
            if (!engine) return;

            NSString* path = url.path;
            BOOL success = engine->load_pca_file(path.UTF8String);

            if (success) {
                NSLog(@"MagentaRT Standalone: Loaded PCA corpus from %@", path.lastPathComponent);
                self->_pcaFileName = path.lastPathComponent;
                [self sendStateUpdate:@{
                    @"pcaLoaded": @YES,
                    @"pcaComponentCount": @(engine->pca_component_count()),
                    @"pcaCentroidCount": @(engine->pca_centroid_count()),
                    @"pcaFileName": path.lastPathComponent ?: @""
                }];
            } else {
                NSLog(@"MagentaRT Standalone: Failed to load PCA corpus");
                [self sendStateUpdate:@{@"pcaLoaded": @NO}];
            }
        });
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}



// ─── Audio prompt loading ───────────────────────────────────────────────────

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
                        if (framesToRead < (UInt32)maxFrames) {
                            for (UInt32 i = framesToRead; i < (UInt32)maxFrames; ++i) {
                                samples[i] = samples[i % framesToRead];
                            }
                        }
                        engine->set_audio_prompt_samples(index, filename.UTF8String, samples.data(), maxFrames);
                        readSuccess = YES;
                    }
                }
            ExtAudioFileDispose(extFile);
            }

            // Update prompts array
            NSMutableArray* mutablePrompts = [self->_prompts mutableCopy];
            if (!mutablePrompts) {
                mutablePrompts = [NSMutableArray array];
                for (int i = 0; i < 6; ++i) [mutablePrompts addObject:@{@"text": @"", @"weight": @0.0}];
            }
            if (index < (int)mutablePrompts.count) {
                NSMutableDictionary* p = [mutablePrompts[index] mutableCopy];
                p[@"text"] = readSuccess ? filename : @"Error: Load failed";
                p[@"isAudio"] = @(readSuccess);
                mutablePrompts[index] = p;
            }
            self->_prompts = mutablePrompts;
            [self sendStateUpdate:@{@"textPrompts": mutablePrompts}];
        });
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

// ─── Silent / Audio prefill ─────────────────────────────────────────────────

- (void)handleSilentPrefill {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        RealtimeRunner* engine = self.engine;
        if (engine) {
            // 550 frames @ 25 Hz = 22 s, comfortably above the model's
            // ~19.7 s effective receptive field (12 layers × 41-frame
            // local-attention window).
            NSLog(@"MagentaRT Standalone 2: Starting silent prefill (22s)...");
            bool success = engine->prefill_silence(/*duration_frames=*/550,
                [](const std::string& msg) {
                    NSLog(@"MagentaRT Standalone 2: %s", msg.c_str());
                });
            NSLog(@"MagentaRT Standalone 2: Silent prefill %@",
                  success ? @"successful" : @"failed");
        }
    });
}

- (void)handleAudioPrefill {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowedContentTypes:@[[UTType typeWithIdentifier:@"public.audio"]]];
    [panel setMessage:@"Select an audio file for prefill (will be truncated to 28 s)"];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (result != NSModalResponseOK) return;
        NSURL* url = [panel URL];
        if (!url) return;

        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            RealtimeRunner* engine = self.engine;
            if (!engine) return;

            ExtAudioFileRef extFile = nullptr;
            OSStatus status = ExtAudioFileOpenURL((__bridge CFURLRef)url, &extFile);
            if (status == noErr && extFile) {
                AudioStreamBasicDescription clientFormat = {};
                clientFormat.mSampleRate = 48000.0;
                clientFormat.mFormatID = kAudioFormatLinearPCM;
                clientFormat.mFormatFlags = kAudioFormatFlagIsFloat;
                clientFormat.mBitsPerChannel = 32;
                clientFormat.mChannelsPerFrame = 2;
                clientFormat.mBytesPerFrame = 8;
                clientFormat.mFramesPerPacket = 1;
                clientFormat.mBytesPerPacket = 8;

                status = ExtAudioFileSetProperty(extFile, kExtAudioFileProperty_ClientDataFormat,
                                                  sizeof(clientFormat), &clientFormat);
                if (status == noErr) {
                    // 28 s @ 48 kHz — matches the SpectroStream encoder's
                    // fixed input shape (1, 1344000, 2).
                    int maxFrames = 1344000;
                    std::vector<float> samples(maxFrames * 2, 0.0f);

                    AudioBufferList bufferList;
                    bufferList.mNumberBuffers = 1;
                    bufferList.mBuffers[0].mNumberChannels = 2;
                    bufferList.mBuffers[0].mDataByteSize = maxFrames * 2 * sizeof(float);
                    bufferList.mBuffers[0].mData = samples.data();

                    UInt32 framesToRead = maxFrames;
                    status = ExtAudioFileRead(extFile, &framesToRead, &bufferList);
                    if (status == noErr && framesToRead > 0) {
                        NSLog(@"MagentaRT Standalone 2: Read %u frames for prefill",
                              (unsigned int)framesToRead);
                        bool success = engine->prefill_state(samples.data(), framesToRead,
                            [](const std::string& msg) {
                                NSLog(@"MagentaRT Standalone 2: %s", msg.c_str());
                            });
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [self sendStateUpdate:@{@"audioPrefillStatus":
                                success ? @"Success" : @"Failed"}];
                        });
                    } else {
                        NSLog(@"MagentaRT Standalone 2: Failed to read audio file or empty");
                    }
                } else {
                    NSLog(@"MagentaRT Standalone 2: Failed to set client format");
                }
                ExtAudioFileDispose(extFile);
            } else {
                NSLog(@"MagentaRT Standalone 2: Failed to open audio file at %@", url.path);
            }
        });
    };

    if (self.view.window) {
        [panel beginSheetModalForWindow:self.view.window completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

// ─── Model management (shared pattern from Jam) ─────────────────────────────

- (void)loadModelAtPath:(NSString*)mlxfnPath {
    RealtimeRunner* engine = self.engine;
    if (!engine) return;

    NSLog(@"MagentaRT Standalone 2: Loading model from %@", mlxfnPath);
    BOOL success = engine->load_model(mlxfnPath.UTF8String);

    if (success) {
        _modelName = mlxfnPath.lastPathComponent;
        _modelDirectoryURL = [NSURL fileURLWithPath:[mlxfnPath stringByDeletingLastPathComponent]];

        NSMutableDictionary* stateUpdate = [NSMutableDictionary dictionaryWithDictionary:@{
            @"modelName": mlxfnPath.lastPathComponent
        }];

        // Load MusicCoCa model
        std::string resourcesPath = magentart::paths::get_resources_dir();
        BOOL ok =
            engine->load_musiccoca_model(resourcesPath.c_str(), "musiccoca");
        if (!ok) {
          NSLog(@"MagentaRT Standalone 2: Failed to load MusicCoCa model for "
                @"subfolder: musiccoca");
        } else {
          stateUpdate[@"musicCocaModelName"] = @"musiccoca";
        }

        // Auto-load corpus.safetensors if present
        NSString* parentDir = [mlxfnPath stringByDeletingLastPathComponent];
        NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
            if (engine->load_pca_file(corpusPath.UTF8String)) {
                NSLog(@"MagentaRT Standalone 2: Auto-loaded corpus from %@", corpusPath.lastPathComponent);
                _pcaFileName = corpusPath.lastPathComponent;
                [stateUpdate addEntriesFromDictionary:@{
                    @"pcaLoaded": @YES,
                    @"pcaComponentCount": @(engine->pca_component_count()),
                    @"pcaCentroidCount": @(engine->pca_centroid_count()),
                    @"pcaFileName": corpusPath.lastPathComponent ?: @""
                }];
            }
        }

        // Load SpectroStream encoder: model dir → external spectrostream → bundle
        NSString* spectrostreamPath = [parentDir stringByAppendingPathComponent:@"spectrostream_encoder.mlxfn"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:spectrostreamPath]) {
            NSLog(@"MagentaRT Standalone 2: Found spectrostream encoder in model dir: %@", spectrostreamPath.lastPathComponent);
            engine->load_prefill_model(spectrostreamPath.UTF8String, nullptr);
        } else {
            std::string extPath = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";
            NSString* extNSPath = [NSString stringWithUTF8String:extPath.c_str()];
            if ([[NSFileManager defaultManager] fileExistsAtPath:extNSPath]) {
                NSLog(@"MagentaRT Standalone 2: Loading spectrostream encoder from external path: %@", extNSPath);
                engine->load_prefill_model(extNSPath.UTF8String, nullptr);
            } else {
                NSString* fallbackPath = [[NSBundle mainBundle] pathForResource:@"spectrostream_encoder" ofType:@"mlxfn"];
                if (fallbackPath) {
                    NSLog(@"MagentaRT Standalone 2: Loading spectrostream encoder from bundle resources: %@", fallbackPath.lastPathComponent);
                    engine->load_prefill_model(fallbackPath.UTF8String, nullptr);
                }
            }
        }

        // Re-apply saved prompts
        if (_prompts) {
            [self restorePrompts:_prompts];
        }

        [self sendStateUpdate:stateUpdate];

        // Persist model path for auto-load on next launch
        [[NSUserDefaults standardUserDefaults] setObject:mlxfnPath forKey:@"MagentaRT_ModelPath"];
    } else {
        [self sendStateUpdate:@{@"modelName": [NSString stringWithFormat:@"Failed: %@", mlxfnPath.lastPathComponent]}];
    }
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

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleDeleteModel:(NSString *)modelName {
    dispatch_async(dispatch_get_main_queue(), ^{
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

        NSError *error = nil;
        [[NSFileManager defaultManager] removeItemAtPath:path error:&error];
        if (error) {
            NSLog(@"MagentaRT Standalone 2: Failed to delete model %@: %@", modelName, error.localizedDescription);
        } else {
            NSLog(@"MagentaRT Standalone 2: Successfully deleted model %@", modelName);
            [self handleListLocalModels];
        }

        if (accessGranted) {
            [modelsDir stopAccessingSecurityScopedResource];
        }
    });
}

- (void)handleSelectDownloadFolder {
    [MagentaModelManager selectDownloadFolderWithParentWindow:self.view.window
                                                  completion:^(NSString *selectedPath, NSData *bookmarkData, NSError *error) {
        if (selectedPath) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [[NSUserDefaults standardUserDefaults] setObject:bookmarkData forKey:@"MagentaRT_ModelFolderBookmark"];
                [[NSUserDefaults standardUserDefaults] setObject:selectedPath forKey:@"MagentaRT_ModelFolderPath"];

                // Check for custom resources folder inside selected path
                NSString *customResourcesPath = [selectedPath stringByAppendingPathComponent:@"resources"];
                BOOL hasCustomResources = [[NSFileManager defaultManager] fileExistsAtPath:customResourcesPath];

                NSString *resourcesPathToLoad = hasCustomResources ? customResourcesPath : [NSString stringWithUTF8String:magentart::paths::get_resources_dir().c_str()];

                if (!self.engine->init_assets(resourcesPathToLoad.UTF8String)) {
                    NSLog(@"MagentaRT Standalone 2: Failed to initialize assets from custom path: %@", resourcesPathToLoad);
                } else {
                    NSLog(@"MagentaRT Standalone 2: Successfully initialized assets from path: %@", resourcesPathToLoad);
                    [[NSUserDefaults standardUserDefaults] setObject:resourcesPathToLoad forKey:@"MagentaRT_CustomResourcesPath"];
                }

                [self sendStateUpdate:@{
                    @"downloadPath": selectedPath,
                    @"resourcesMissing": @NO
                }];

                [self handleListLocalModels];

                // Auto-load first available model
                NSArray<NSString *> *modelFiles = [MagentaModelManager listLocalModelsInDirectory:[NSURL fileURLWithPath:selectedPath]];
                if (modelFiles.count > 0) {
                    [self handleSelectModel:modelFiles[0]];
                }
            });
        } else if (error) {
            NSLog(@"MagentaRT Standalone 2: Failed to create folder bookmark: %@", error.localizedDescription);
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
    [self sendStateUpdate:@{@"midiSources": sources}];
}

- (void)setComputerKeyboardMidiEnabled:(BOOL)enabled {
    [[NSUserDefaults standardUserDefaults] setBool:enabled forKey:@"MagentaRT_ComputerKeyboardMidi"];
    [self sendStateUpdate:@{@"computerKeyboardMidi": @(enabled)}];
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
        if (selectedEndpoint != 0xFFFFFFFF) { // 0xFFFFFFFF can mean "None"
            MIDIEndpointRef endpoint = (MIDIEndpointRef)selectedEndpoint;
            if (MIDIPortConnectSource(self.midiInputPort, endpoint, NULL) == noErr) {
                [self.connectedSources addObject:@(selectedEndpoint)];
            }
        }
    }

    [[NSUserDefaults standardUserDefaults] setInteger:selectedEndpoint forKey:@"MagentaRT_SelectedMidiEndpoint"];
    [self handleMIDIStructureChanged];
}

- (void)dealloc {
    [_metricsTimer invalidate];
}

@end
