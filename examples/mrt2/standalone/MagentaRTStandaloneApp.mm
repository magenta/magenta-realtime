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

// Magenta RT Standalone App — main entry point, AppDelegate, AVAudioEngine, CoreMIDI.
//
// Owns the RealtimeRunner and outputs audio via AVAudioSourceNode.
// MIDI input via CoreMIDI: virtual destination + connectable physical sources.
// Settings window for Audio I/O and MIDI source selection.
//
// Note: this file and MagentaRTAppController.mm intentionally duplicate some
// WebView IPC and parameter-mirroring glue with examples/auv3/MagentaRT_AudioUnit.mm
// rather than sharing a helper. See the block comment at the top of
// MagentaRTAppController.mm for why, and for which file to treat as the
// canonical reference if the two ever drift.

#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMIDI/CoreMIDI.h>
#import <CoreAudio/CoreAudio.h>
#import "MagentaRTAppController.h"
#include <magentart/realtime_runner.h>
#include "../../common/cpp/magenta_paths.h"

using magentart::core::RealtimeRunner;

// ─── Settings Window Controller ─────────────────────────────────────────────

@interface MagentaRTSettingsController : NSWindowController <NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate>
@property (nonatomic, assign) MIDIClientRef midiClient;
@property (nonatomic, assign) MIDIPortRef midiInputPort;
@property (nonatomic, assign) RealtimeRunner* engine;
@property (nonatomic, strong) AVAudioEngine* audioEngine;
// Currently connected MIDI source endpoints
@property (nonatomic, strong) NSMutableSet<NSNumber*>* connectedSources;
- (void)refreshMIDISources;
- (void)refreshAudioInfo;
@end

@implementation MagentaRTSettingsController {
    NSTextField* _audioDeviceLabel;
    NSTextField* _audioSampleRateLabel;
    NSTextField* _audioBufferSizeLabel;
    NSTextField* _midiVirtualLabel;
    NSTableView* _midiTableView;
    NSMutableArray<NSDictionary*>* _midiSources;
}

- (instancetype)init {
    NSRect frame = NSMakeRect(0, 0, 460, 400);
    NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:NSWindowStyleMaskTitled |
                                                             NSWindowStyleMaskClosable
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    window.title = @"Settings";
    window.releasedWhenClosed = NO;

    self = [super initWithWindow:window];
    if (!self) return nil;

    _connectedSources = [NSMutableSet set];
    _midiSources = [NSMutableArray array];
    window.delegate = self;

    NSView* content = window.contentView;

    // ── Audio section ──
    NSTextField* audioHeader = [NSTextField labelWithString:@"Audio Output"];
    audioHeader.font = [NSFont boldSystemFontOfSize:13];
    audioHeader.frame = NSMakeRect(20, 350, 200, 20);
    [content addSubview:audioHeader];

    NSTextField* deviceTitle = [NSTextField labelWithString:@"Device:"];
    deviceTitle.frame = NSMakeRect(20, 322, 60, 18);
    deviceTitle.font = [NSFont systemFontOfSize:12];
    [content addSubview:deviceTitle];

    _audioDeviceLabel = [NSTextField labelWithString:@"—"];
    _audioDeviceLabel.frame = NSMakeRect(85, 322, 350, 18);
    _audioDeviceLabel.font = [NSFont systemFontOfSize:12];
    [content addSubview:_audioDeviceLabel];

    NSTextField* srTitle = [NSTextField labelWithString:@"Sample Rate:"];
    srTitle.frame = NSMakeRect(20, 300, 80, 18);
    srTitle.font = [NSFont systemFontOfSize:12];
    [content addSubview:srTitle];

    _audioSampleRateLabel = [NSTextField labelWithString:@"—"];
    _audioSampleRateLabel.frame = NSMakeRect(105, 300, 200, 18);
    _audioSampleRateLabel.font = [NSFont systemFontOfSize:12];
    [content addSubview:_audioSampleRateLabel];

    NSTextField* bufTitle = [NSTextField labelWithString:@"Buffer Size:"];
    bufTitle.frame = NSMakeRect(20, 278, 80, 18);
    bufTitle.font = [NSFont systemFontOfSize:12];
    [content addSubview:bufTitle];

    _audioBufferSizeLabel = [NSTextField labelWithString:@"—"];
    _audioBufferSizeLabel.frame = NSMakeRect(105, 278, 200, 18);
    _audioBufferSizeLabel.font = [NSFont systemFontOfSize:12];
    [content addSubview:_audioBufferSizeLabel];

    // Separator
    NSBox* sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(20, 265, 420, 1)];
    sep1.boxType = NSBoxSeparator;
    [content addSubview:sep1];

    // ── MIDI section ──
    NSTextField* midiHeader = [NSTextField labelWithString:@"MIDI Input"];
    midiHeader.font = [NSFont boldSystemFontOfSize:13];
    midiHeader.frame = NSMakeRect(20, 238, 200, 20);
    [content addSubview:midiHeader];

    _midiVirtualLabel = [NSTextField labelWithString:@"Virtual port: MRT2 Input"];
    _midiVirtualLabel.frame = NSMakeRect(20, 216, 400, 18);
    _midiVirtualLabel.font = [NSFont systemFontOfSize:11];
    _midiVirtualLabel.textColor = [NSColor secondaryLabelColor];
    [content addSubview:_midiVirtualLabel];

    NSTextField* sourcesLabel = [NSTextField labelWithString:@"Connect to MIDI sources (click to toggle):"];
    sourcesLabel.frame = NSMakeRect(20, 192, 400, 18);
    sourcesLabel.font = [NSFont systemFontOfSize:12];
    [content addSubview:sourcesLabel];

    // Table view for MIDI sources
    NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:NSMakeRect(20, 20, 420, 168)];
    scrollView.hasVerticalScroller = YES;
    scrollView.autohidesScrollers = YES;
    scrollView.borderType = NSBezelBorder;

    _midiTableView = [[NSTableView alloc] initWithFrame:scrollView.bounds];

    NSTableColumn* checkCol = [[NSTableColumn alloc] initWithIdentifier:@"connected"];
    checkCol.title = @"";
    checkCol.width = 30;
    checkCol.minWidth = 30;
    checkCol.maxWidth = 30;
    [_midiTableView addTableColumn:checkCol];

    NSTableColumn* nameCol = [[NSTableColumn alloc] initWithIdentifier:@"name"];
    nameCol.title = @"Source";
    nameCol.width = 360;
    [_midiTableView addTableColumn:nameCol];

    _midiTableView.dataSource = self;
    _midiTableView.delegate = self;
    _midiTableView.headerView = nil;
    _midiTableView.rowHeight = 22;
    _midiTableView.target = self;
    _midiTableView.action = @selector(midiTableClicked:);

    scrollView.documentView = _midiTableView;
    [content addSubview:scrollView];

    return self;
}

- (void)showWindow:(id)sender {
    [self refreshAudioInfo];
    [self refreshMIDISources];
    [super showWindow:sender];
    [self.window center];
}

- (void)refreshAudioInfo {
    if (!_audioEngine) return;

    // Get the output node's actual format
    AVAudioFormat* outputFormat = [_audioEngine.outputNode outputFormatForBus:0];
    double sampleRate = outputFormat.sampleRate;

    // Get the source node format
    AVAudioFormat* sourceFormat = [_audioEngine.mainMixerNode outputFormatForBus:0];

    // Get output device name via CoreAudio
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

    // Get buffer size
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

- (void)refreshMIDISources {
    [_midiSources removeAllObjects];

    ItemCount sourceCount = MIDIGetNumberOfSources();
    for (ItemCount i = 0; i < sourceCount; ++i) {
        MIDIEndpointRef src = MIDIGetSource(i);
        CFStringRef cfName = NULL;
        MIDIObjectGetStringProperty(src, kMIDIPropertyDisplayName, &cfName);

        NSString* name = cfName ? (__bridge_transfer NSString*)cfName : @"Unknown MIDI Source";
        BOOL connected = [_connectedSources containsObject:@((uint32_t)src)];

        [_midiSources addObject:@{
            @"name": name,
            @"endpoint": @((uint32_t)src),
            @"connected": @(connected)
        }];
    }

    [_midiTableView reloadData];
}

// ── NSTableViewDataSource ──

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
    return (NSInteger)_midiSources.count;
}

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    if (row >= (NSInteger)_midiSources.count) return nil;
    NSDictionary* source = _midiSources[(NSUInteger)row];

    if ([tableColumn.identifier isEqualToString:@"connected"]) {
        NSTextField* cell = [tableView makeViewWithIdentifier:@"checkCell" owner:self];
        if (!cell) {
            cell = [NSTextField labelWithString:@""];
            cell.identifier = @"checkCell";
            cell.alignment = NSTextAlignmentCenter;
        }
        cell.stringValue = [source[@"connected"] boolValue] ? @"\u2713" : @"";
        cell.font = [NSFont systemFontOfSize:14];
        return cell;
    } else {
        NSTextField* cell = [tableView makeViewWithIdentifier:@"nameCell" owner:self];
        if (!cell) {
            cell = [NSTextField labelWithString:@""];
            cell.identifier = @"nameCell";
            cell.bordered = NO;
            cell.editable = NO;
            cell.drawsBackground = NO;
        }
        cell.stringValue = source[@"name"];
        cell.font = [NSFont systemFontOfSize:12];
        return cell;
    }
}

- (void)midiTableClicked:(id)sender {
    NSInteger row = _midiTableView.clickedRow;
    if (row < 0 || row >= (NSInteger)_midiSources.count) return;

    NSDictionary* source = _midiSources[(NSUInteger)row];
    MIDIEndpointRef endpoint = (MIDIEndpointRef)[source[@"endpoint"] unsignedIntValue];
    BOOL wasConnected = [source[@"connected"] boolValue];

    if (wasConnected) {
        // Disconnect
        OSStatus status = MIDIPortDisconnectSource(_midiInputPort, endpoint);
        if (status == noErr) {
            [_connectedSources removeObject:@((uint32_t)endpoint)];
            NSLog(@"MagentaRT: Disconnected MIDI source: %@", source[@"name"]);
        }
    } else {
        // Connect
        OSStatus status = MIDIPortConnectSource(_midiInputPort, endpoint, NULL);
        if (status == noErr) {
            [_connectedSources addObject:@((uint32_t)endpoint)];
            NSLog(@"MagentaRT: Connected MIDI source: %@", source[@"name"]);
        }
    }

    [self refreshMIDISources];
}

@end

// ─── AppDelegate ─────────────────────────────────────────────────────────────

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
    RealtimeRunner _engine;
    StandaloneSharedState _sharedState;
    AVAudioEngine* _audioEngine;
    AVAudioSourceNode* _sourceNode;
    MIDIClientRef _midiClient;
    MIDIPortRef _midiInputPort;
    MIDIEndpointRef _midiVirtualDest;
    NSWindow* _window;
    MagentaRTAppController* _controller;
    MagentaRTSettingsController* _settingsController;
    BOOL _isPlaying;
    NSMenuItem* _playStopMenuItem;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    // Initialize ML assets from ~/Documents/Magenta/resources (centralized path) or saved custom folder.
    // Model files should be placed in ~/Documents/Magenta/models/.
    NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
    std::string resources = customResources ? customResources.UTF8String : magentart::paths::get_resources_dir();
    if (!_engine.init_assets(resources.c_str())) {
        NSLog(@"MagentaRT Standalone: Failed to load static assets from %s", resources.c_str());
    }

    // Create the view controller and give it the engine
    _controller = [[MagentaRTAppController alloc] init];
    _controller.engine = &_engine;
    _controller.sharedState = &_sharedState;

    // Restore saved parameters and prompts immediately so the engine has them from start
    [_controller restoreSavedParams];
    NSArray* savedPrompts = [[NSUserDefaults standardUserDefaults] arrayForKey:@"MagentaRT_Prompts"];
    [_controller restorePrompts:savedPrompts];

    // Create the main window
    NSRect frame = NSMakeRect(0, 0, 1075, 470);
    _window = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:NSWindowStyleMaskTitled |
                                                    NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskMiniaturizable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"MRT2";
    _window.minSize = NSMakeSize(1075, 470);
    _window.maxSize = NSMakeSize(1075, 470);
    _window.contentViewController = _controller;
    [_window center];
    [_window makeKeyAndOrderFront:nil];

    // Set up audio output
    [self setupAudioEngine];

    // Set up MIDI input
    [self setupMIDI];

    // Build the menu bar
    [self setupMenuBar];

    // Create settings controller (shares references)
    _settingsController = [[MagentaRTSettingsController alloc] init];
    _settingsController.midiClient = _midiClient;
    _settingsController.midiInputPort = _midiInputPort;
    _settingsController.engine = &_engine;
    _settingsController.audioEngine = _audioEngine;

    // Give the view controller access to MIDI port for in-app source selection
    _controller.midiInputPort = _midiInputPort;
    _controller.connectedSources = [NSMutableSet set];

    // Restore saved MIDI endpoint
    NSInteger savedEndpoint = [[NSUserDefaults standardUserDefaults] integerForKey:@"MagentaRT_SelectedMidiEndpoint"];
    if ([[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_SelectedMidiEndpoint"] && savedEndpoint != 0) {
        [_controller selectMidiInput:(uint32_t)savedEndpoint];
    }

    // Auto-load last model from preferences
    [self autoLoadModel];
}

// ─── AVAudioEngine + AVAudioSourceNode ──────────────────────────────────────

- (void)setupAudioEngine {
    _audioEngine = [[AVAudioEngine alloc] init];

    AVAudioFormat* format = [[AVAudioFormat alloc]
        initStandardFormatWithSampleRate:48000.0 channels:2];

    RealtimeRunner* engine = &_engine;
    StandaloneSharedState* shared = &_sharedState;

    _sourceNode = [[AVAudioSourceNode alloc]
        initWithFormat:format
        renderBlock:^OSStatus(BOOL* isSilence, const AudioTimeStamp* timestamp,
                              AVAudioFrameCount frameCount, AudioBufferList* outputData) {
        float* outL = (float*)outputData->mBuffers[0].mData;
        float* outR = (outputData->mNumberBuffers > 1)
                      ? (float*)outputData->mBuffers[1].mData : outL;

        if (!engine->is_loaded()) {
            memset(outL, 0, frameCount * sizeof(float));
            if (outputData->mNumberBuffers > 1) {
                memset(outR, 0, frameCount * sizeof(float));
            }
            *isSilence = YES;
        } else {
            engine->read_audio_stereo(outL, outR, frameCount, /*blocking=*/false);
        }

        shared->levelProcessor.process_block(outL, outR, frameCount);
        return noErr;
    }];

    [_audioEngine attachNode:_sourceNode];
    [_audioEngine connect:_sourceNode
                       to:_audioEngine.mainMixerNode
                   format:format];

    NSError* error = nil;
    if (![_audioEngine startAndReturnError:&error]) {
        NSLog(@"MagentaRT Standalone: AVAudioEngine failed to start: %@", error);
    } else {
        NSLog(@"MagentaRT Standalone: Audio engine started (48kHz stereo)");
    }
}

// ─── CoreMIDI ───────────────────────────────────────────────────────────────

- (void)setupMIDI {
    RealtimeRunner* engine = &_engine;

    // Create MIDI client
    OSStatus status = MIDIClientCreateWithBlock(
        CFSTR("MRT2"),
        &_midiClient,
        ^(const MIDINotification* notification) {
            if (notification->messageID == kMIDIMsgSetupChanged) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self->_settingsController refreshMIDISources];
                    [self->_controller handleMIDIStructureChanged];
                });
            }
        }
    );
    if (status != noErr) {
        NSLog(@"MagentaRT Standalone: MIDIClientCreate failed: %d", (int)status);
        return;
    }

    // Create input port for connecting to physical MIDI sources
    StandaloneSharedState* shared = &_sharedState;
    status = MIDIInputPortCreateWithProtocol(
        _midiClient,
        CFSTR("MRT2 In"),
        kMIDIProtocol_1_0,
        &_midiInputPort,
        ^(const MIDIEventList* evtList, void* srcConnRefCon) {
            const MIDIEventPacket* pkt = &evtList->packet[0];
            for (UInt32 i = 0; i < evtList->numPackets; ++i) {
                for (UInt32 w = 0; w < pkt->wordCount; ++w) {
                    uint32_t word = pkt->words[w];
                    // UMP Message Type 0x2 = MIDI 1.0 Channel Voice
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
        NSLog(@"MagentaRT Standalone: MIDIInputPortCreate failed: %d", (int)status);
        return;
    }

    // Create virtual destination so other apps can send MIDI to us
    status = MIDIDestinationCreateWithProtocol(
        _midiClient,
        CFSTR("MRT2 Input"),
        kMIDIProtocol_1_0,
        &_midiVirtualDest,
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
        NSLog(@"MagentaRT Standalone: MIDIDestinationCreate failed: %d", (int)status);
    } else {
        NSLog(@"MRT2 Standalone: MIDI virtual destination created ('MRT2 Input')");
    }

    NSLog(@"MagentaRT Standalone: MIDI input port created, %lu sources available",
          (unsigned long)MIDIGetNumberOfSources());
}

// ─── Menu bar ───────────────────────────────────────────────────────────────

- (void)setupMenuBar {
    NSMenu* menuBar = [[NSMenu alloc] init];

    // App menu
    NSMenuItem* appMenuItem = [[NSMenuItem alloc] init];
    NSMenu* appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"About MRT2"
                       action:@selector(orderFrontStandardAboutPanel:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Settings..."
                       action:@selector(menuShowSettings:)
                keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit MRT2"
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
    appMenuItem.submenu = appMenu;
    [menuBar addItem:appMenuItem];

    // File menu
    NSMenuItem* fileMenuItem = [[NSMenuItem alloc] init];
    NSMenu* fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"Load Model..."
                        action:@selector(menuLoadModel:)
                 keyEquivalent:@"o"];
    [fileMenu addItemWithTitle:@"Load Corpus..."
                        action:@selector(menuLoadCorpus:)
                 keyEquivalent:@""];
    fileMenuItem.submenu = fileMenu;
    [menuBar addItem:fileMenuItem];

    // Edit menu (for copy/paste to work in WebView text fields)
    NSMenuItem* editMenuItem = [[NSMenuItem alloc] init];
    NSMenu* editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    editMenuItem.submenu = editMenu;
    [menuBar addItem:editMenuItem];

    // Transport menu
    NSMenuItem* transportMenuItem = [[NSMenuItem alloc] init];
    NSMenu* transportMenu = [[NSMenu alloc] initWithTitle:@"Transport"];
    _playStopMenuItem = [transportMenu addItemWithTitle:@"Start"
                                                 action:@selector(menuTogglePlayStop:)
                                          keyEquivalent:@"z"];
    _isPlaying = NO;
    _engine.set_bypass(true);
    transportMenuItem.submenu = transportMenu;
    [menuBar addItem:transportMenuItem];

    [NSApp setMainMenu:menuBar];
}

- (void)menuTogglePlayStop:(id)sender {
    if (_isPlaying) {
        // Stop: bypass the engine (silence + pause inference GPU usage)
        _engine.set_bypass(true);
        _isPlaying = NO;
        _playStopMenuItem.title = @"Start";
        NSLog(@"MagentaRT Standalone: Stopped");
    } else {
        // Start: un-bypass and reset to get fresh audio
        _engine.set_bypass(false);
        _engine.trigger_reset();
        _isPlaying = YES;
        _playStopMenuItem.title = @"Stop";
        NSLog(@"MagentaRT Standalone: Started");
    }
    [_controller sendPlayState:_isPlaying];
}

- (void)menuShowSettings:(id)sender {
    [_settingsController showWindow:nil];
    [_settingsController.window makeKeyAndOrderFront:nil];
}

- (void)menuLoadModel:(id)sender {
    [_controller handleLoadModel];
}

- (void)menuLoadCorpus:(id)sender {
    [_controller handleLoadPCAFile];
}

// ─── Auto-load model from preferences ───────────────────────────────────────

- (void)autoLoadModel {
    NSString* modelPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_ModelPath"];
    if (!modelPath) return;

    if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
        NSLog(@"MagentaRT Standalone: Saved model path no longer exists: %@", modelPath);
        return;
    }

    NSLog(@"MagentaRT Standalone: Auto-loading model from %@", modelPath);

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        BOOL success = self->_engine.load_model(modelPath.UTF8String);
        if (success) {
            NSLog(@"MagentaRT Standalone: Auto-loaded model successfully.");

            NSString* parentDir = [modelPath stringByDeletingLastPathComponent];
            NSString* corpusPath = [parentDir stringByAppendingPathComponent:@"corpus.safetensors"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:corpusPath]) {
                if (self->_engine.load_pca_file(corpusPath.UTF8String)) {
                    NSLog(@"MagentaRT Standalone: Auto-loaded corpus.");
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self->_controller notifyPCALoaded:self->_engine.pca_component_count()
                                            centroidCount:self->_engine.pca_centroid_count()
                                                 fileName:corpusPath.lastPathComponent];
                    });
                }
            }

            NSArray* savedPrompts = [[NSUserDefaults standardUserDefaults] arrayForKey:@"MagentaRT_Prompts"];

            // Restore persisted advanced controls (buffer size, temperature, etc.)
            [self->_controller restoreSavedParams];

            dispatch_async(dispatch_get_main_queue(), ^{
                // Set saved prompts on the controller, then notify model loaded.
                // notifyModelLoaded triggers connectToEngine which reads _prompts and
                // sends them to the UI + re-applies to the engine for fresh embedding computation.
                [self->_controller restorePrompts:savedPrompts];
                [self->_controller notifyModelLoaded:modelPath.lastPathComponent];
            });
        } else {
            NSLog(@"MagentaRT Standalone: Auto-load failed for %@", modelPath);
        }
    });
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

- (void)applicationWillTerminate:(NSNotification*)notification {
    _engine.stop();
    _engine.unload();
    [_audioEngine stop];

    if (_midiVirtualDest) {
        MIDIEndpointDispose(_midiVirtualDest);
    }
    if (_midiInputPort) {
        MIDIPortDispose(_midiInputPort);
    }
    if (_midiClient) {
        MIDIClientDispose(_midiClient);
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
    return YES;
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
    return YES;
}

@end

// ─── main ───────────────────────────────────────────────────────────────────

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
