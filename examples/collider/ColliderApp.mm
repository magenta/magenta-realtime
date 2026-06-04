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

// Collider — standalone app entry point.
// Reuses RealtimeRunner, AVAudioEngine, CoreMIDI from Magenta RT standalone.
// Adds shared state for MIDI note visualization and audio waveform display.

#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMIDI/CoreMIDI.h>
#import <CoreAudio/CoreAudio.h>
#import "ColliderAppController.h"
#import "../common/objc/MagentaSettings.h"
#include <magentart/realtime_runner.h>
#include "../common/cpp/magenta_paths.h"

using magentart::core::RealtimeRunner;

// ─── Settings Window Controller ─────────────────────────────────────────────
// Full settings panel: Model, Generation params, Audio I/O, MIDI sources.
// Accessible from app menu (Cmd+,) or from the gear icon in the React UI.

@interface ColliderSettingsController : NSWindowController <NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate>
@property (nonatomic, assign) MIDIClientRef midiClient;
@property (nonatomic, assign) MIDIPortRef midiInputPort;
@property (nonatomic, strong) AVAudioEngine* audioEngine;
@property (nonatomic, strong) NSMutableSet<NSNumber*>* connectedSources;
@property (nonatomic, weak) ColliderAppController* appController;
- (void)refreshMIDISources;
- (void)refreshAll;
@end

@implementation ColliderSettingsController {
    // Model
    NSTextField* _modelNameLabel;
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
    // MIDI
    NSTextField* _midiVirtualLabel;
    NSTableView* _midiTableView;
    NSMutableArray<NSDictionary*>* _midiSources;
    NSButton* _computerKeyboardMidiCheckbox;
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
    CGFloat W = 480, H = 740;
    NSRect frame = NSMakeRect(0, 0, W, H);
    NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    window.title = @"Settings";
    window.releasedWhenClosed = NO;

    self = [super initWithWindow:window];
    if (!self) return nil;
    _connectedSources = [NSMutableSet set];
    _midiSources = [NSMutableArray array];
    window.delegate = self;

    NSView* c = window.contentView;
    CGFloat pad = 20, col2 = 110, sliderW = 280, valX = W - 70;
    CGFloat y = H - 40;

    // ── Model ──
    NSTextField* modelHeader = [NSTextField labelWithString:@"Model"];
    modelHeader.font = [NSFont boldSystemFontOfSize:13];
    modelHeader.frame = NSMakeRect(pad, y, 200, 18);
    [c addSubview:modelHeader];
    y -= 28;

    NSButton* loadBtn = [NSButton buttonWithTitle:@"Load Model..." target:self action:@selector(loadModelClicked:)];
    loadBtn.frame = NSMakeRect(pad, y, 120, 24);
    loadBtn.bezelStyle = NSBezelStyleRounded;
    loadBtn.font = [NSFont systemFontOfSize:12];
    [c addSubview:loadBtn];

    _modelNameLabel = [NSTextField labelWithString:@"No model loaded"];
    _modelNameLabel.frame = NSMakeRect(pad + 128, y + 3, W - pad - 148, 16);
    _modelNameLabel.font = [NSFont systemFontOfSize:11];
    _modelNameLabel.textColor = [NSColor secondaryLabelColor];
    _modelNameLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
    [c addSubview:_modelNameLabel];
    y -= 24;



    NSBox* sep0 = [[NSBox alloc] initWithFrame:NSMakeRect(pad, y, W - 2 * pad, 1)];
    sep0.boxType = NSBoxSeparator;
    [c addSubview:sep0];
    y -= 24;

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
    _cfgMusicCoCaSlider = makeSlider(col2, y - 2, sliderW, 0, 5, kColliderDefaultCfgMusicCoCa, self, @selector(cfgMusicCoCaChanged:));
    [c addSubview:_cfgMusicCoCaSlider];
    _cfgMusicCoCaValue = makeValue(valX, y); [c addSubview:_cfgMusicCoCaValue];
    y -= 26;

    // CFG-Notes
    [c addSubview:makeLabel(@"CFG-Notes", pad, y, 90)];
    _cfgNotesSlider = makeSlider(col2, y - 2, sliderW, 0, 5, kColliderDefaultCfgNotes, self, @selector(cfgNotesChanged:));
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
    y -= 16;

    NSBox* sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(pad, y, W - 2 * pad, 1)];
    sep2.boxType = NSBoxSeparator;
    [c addSubview:sep2];
    y -= 24;

    // ── MIDI Input ──
    NSTextField* midiHeader = [NSTextField labelWithString:@"MIDI Input"];
    midiHeader.font = [NSFont boldSystemFontOfSize:13];
    midiHeader.frame = NSMakeRect(pad, y, 200, 18);
    [c addSubview:midiHeader];
    y -= 20;

    _midiVirtualLabel = [NSTextField labelWithString:@"Virtual port: MRT2 - Collider Input"];
    _midiVirtualLabel.frame = NSMakeRect(pad, y, 400, 16);
    _midiVirtualLabel.font = [NSFont systemFontOfSize:10];
    _midiVirtualLabel.textColor = [NSColor tertiaryLabelColor];
    [c addSubview:_midiVirtualLabel];
    y -= 20;

    _computerKeyboardMidiCheckbox = [NSButton checkboxWithTitle:@"Use computer keyboard as MIDI input (Ableton layout)"
                                                         target:self
                                                         action:@selector(computerKeyboardMidiChanged:)];
    _computerKeyboardMidiCheckbox.frame = NSMakeRect(pad, y, 400, 18);
    _computerKeyboardMidiCheckbox.font = [NSFont systemFontOfSize:11];
    [c addSubview:_computerKeyboardMidiCheckbox];
    y -= 20;

    [c addSubview:makeLabel(@"Connect to MIDI sources (click to toggle):", pad, y, 400)];
    y -= 6;

    NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:NSMakeRect(pad, 12, W - 2 * pad, y - 12)];
    scrollView.hasVerticalScroller = YES;
    scrollView.autohidesScrollers = YES;
    scrollView.borderType = NSBezelBorder;

    _midiTableView = [[NSTableView alloc] initWithFrame:scrollView.bounds];
    NSTableColumn* checkCol = [[NSTableColumn alloc] initWithIdentifier:@"connected"];
    checkCol.title = @""; checkCol.width = 30; checkCol.minWidth = 30; checkCol.maxWidth = 30;
    [_midiTableView addTableColumn:checkCol];
    NSTableColumn* nameCol = [[NSTableColumn alloc] initWithIdentifier:@"name"];
    nameCol.title = @"Source"; nameCol.width = W - 2 * pad - 50;
    [_midiTableView addTableColumn:nameCol];

    _midiTableView.dataSource = self;
    _midiTableView.delegate = self;
    _midiTableView.headerView = nil;
    _midiTableView.rowHeight = 22;
    _midiTableView.target = self;
    _midiTableView.action = @selector(midiTableClicked:);
    scrollView.documentView = _midiTableView;
    [c addSubview:scrollView];

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
    [self refreshMIDISources];
    [self refreshModelName];
    BOOL kbdMidi = [[NSUserDefaults standardUserDefaults] boolForKey:@"Collider_ComputerKeyboardMidi"];
    _computerKeyboardMidiCheckbox.state = kbdMidi ? NSControlStateValueOn : NSControlStateValueOff;
}

- (void)computerKeyboardMidiChanged:(NSButton*)sender {
    BOOL enabled = (sender.state == NSControlStateValueOn);
    [_appController setComputerKeyboardMidiEnabled:enabled];
}

- (void)refreshModelName {
    NSString* modelPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"Collider_ModelPath"];
    _modelNameLabel.stringValue = modelPath ? modelPath.lastPathComponent : @"No model loaded";
}

- (void)refreshParams {
    ColliderAppController* ctrl = _appController;
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


- (void)loadModelClicked:(id)sender {
    [_appController handleLoadModel];
    // Refresh model name after a short delay (loading is async)
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [self refreshModelName];
    });
}

- (void)resetDefaults:(id)sender {
    [MagentaSettings resetDefaultsOnEngine:_appController.engine
                              prefixString:@"Collider"
                                  cfgNotes:kColliderDefaultCfgNotes
                              cfgMusicCoCa:kColliderDefaultCfgMusicCoCa];
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

// ── MIDI sources ──

- (void)refreshMIDISources {
    [_midiSources removeAllObjects];
    ItemCount sourceCount = MIDIGetNumberOfSources();
    for (ItemCount i = 0; i < sourceCount; ++i) {
        MIDIEndpointRef src = MIDIGetSource(i);
        CFStringRef cfName = NULL;
        MIDIObjectGetStringProperty(src, kMIDIPropertyDisplayName, &cfName);
        NSString* name = cfName ? (__bridge_transfer NSString*)cfName : @"Unknown MIDI Source";
        BOOL connected = [_connectedSources containsObject:@((uint32_t)src)];
        [_midiSources addObject:@{ @"name": name, @"endpoint": @((uint32_t)src), @"connected": @(connected) }];
    }
    [_midiTableView reloadData];
}

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView { return (NSInteger)_midiSources.count; }

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    if (row >= (NSInteger)_midiSources.count) return nil;
    NSDictionary* source = _midiSources[(NSUInteger)row];
    if ([tableColumn.identifier isEqualToString:@"connected"]) {
        NSTextField* cell = [tableView makeViewWithIdentifier:@"checkCell" owner:self];
        if (!cell) { cell = [NSTextField labelWithString:@""]; cell.identifier = @"checkCell"; cell.alignment = NSTextAlignmentCenter; }
        cell.stringValue = [source[@"connected"] boolValue] ? @"\u2713" : @"";
        cell.font = [NSFont systemFontOfSize:14];
        return cell;
    } else {
        NSTextField* cell = [tableView makeViewWithIdentifier:@"nameCell" owner:self];
        if (!cell) { cell = [NSTextField labelWithString:@""]; cell.identifier = @"nameCell"; cell.bordered = NO; cell.editable = NO; cell.drawsBackground = NO; }
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
        if (MIDIPortDisconnectSource(_midiInputPort, endpoint) == noErr)
            [_connectedSources removeObject:@((uint32_t)endpoint)];
    } else {
        if (MIDIPortConnectSource(_midiInputPort, endpoint, NULL) == noErr)
            [_connectedSources addObject:@((uint32_t)endpoint)];
    }
    [self refreshMIDISources];
}
@end

// ─── AppDelegate ─────────────────────────────────────────────────────────────

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
    RealtimeRunner _engine;
    ColliderSharedState _sharedState;
    AVAudioEngine* _audioEngine;
    AVAudioSourceNode* _sourceNode;
    MIDIClientRef _midiClient;
    MIDIPortRef _midiInputPort;
    MIDIEndpointRef _midiVirtualDest;
    NSWindow* _window;
    ColliderAppController* _controller;
    ColliderSettingsController* _settingsController;
    BOOL _isPlaying;
    NSMenuItem* _playStopMenuItem;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    // Initialize ML assets from ~/Documents/Magenta/magenta-rt-v2/resources (centralized path) or saved custom folder.
    // Model files should be placed in ~/Documents/Magenta/magenta-rt-v2/models/.
    NSString *customResources = [[NSUserDefaults standardUserDefaults] stringForKey:@"MagentaRT_CustomResourcesPath"];
    std::string resources = customResources ? customResources.UTF8String : magentart::paths::get_resources_dir();
    if (!_engine.init_assets(resources.c_str())) {
        NSLog(@"Collider: Failed to load static assets from %s", resources.c_str());
    }

    _controller = [[ColliderAppController alloc] init];
    _controller.engine = &_engine;
    _controller.sharedState = &_sharedState;

    // Restore saved parameters immediately so the engine has them from start
    [_controller restoreSavedParams];

    // Start bypassed — user must press Play
    _engine.set_bypass(true);
    _engine.set_cfg_musiccoca(kColliderDefaultCfgMusicCoCa);
    _engine.set_cfg_notes(kColliderDefaultCfgNotes);

    // 500×500 window, resizable for testing
    NSRect frame = NSMakeRect(0, 0, 700, 505);
    _window = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:NSWindowStyleMaskTitled |
                                                    NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskMiniaturizable |
                                                    NSWindowStyleMaskResizable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"MRT2 - Collider";
    _window.restorable = NO;
    _window.contentMinSize = NSMakeSize(310, 310);
    _window.contentViewController = _controller;
    [_window center];
    [_window makeKeyAndOrderFront:nil];

    [self setupAudioEngine];
    [self setupMIDI];
    [self setupMenuBar];

    _settingsController = [[ColliderSettingsController alloc] init];
    _settingsController.midiClient = _midiClient;
    _settingsController.midiInputPort = _midiInputPort;
    _settingsController.audioEngine = _audioEngine;
    _settingsController.appController = _controller;

    [self autoLoadModel];
}

// ─── AVAudioEngine ───────────────────────────────────────────────────────────

- (void)setupAudioEngine {
    _audioEngine = [[AVAudioEngine alloc] init];
    AVAudioFormat* format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:48000.0 channels:2];

    RealtimeRunner* engine = &_engine;
    ColliderSharedState* shared = &_sharedState;

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
        shared->pushAudioSamples(outL, outR, frameCount);
        return noErr;
    }];

    [_audioEngine attachNode:_sourceNode];
    [_audioEngine connect:_sourceNode to:_audioEngine.mainMixerNode format:format];

    NSError* error = nil;
    if (![_audioEngine startAndReturnError:&error]) {
        NSLog(@"Collider: AVAudioEngine failed to start: %@", error);
    }
}

// ─── CoreMIDI ────────────────────────────────────────────────────────────────

- (void)setupMIDI {
    RealtimeRunner* engine = &_engine;
    ColliderSharedState* shared = &_sharedState;

    OSStatus status = MIDIClientCreateWithBlock(
        CFSTR("MRT2 - Collider"),
        &_midiClient,
        ^(const MIDINotification* notification) {
            if (notification->messageID == kMIDIMsgSetupChanged) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self->_settingsController refreshMIDISources];
                });
            }
        }
    );
    if (status != noErr) { NSLog(@"Collider: MIDIClientCreate failed: %d", (int)status); return; }

    status = MIDIInputPortCreateWithProtocol(
        _midiClient, CFSTR("MRT2 - Collider In"), kMIDIProtocol_1_0, &_midiInputPort,
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
    if (status != noErr) { NSLog(@"Collider: MIDIInputPortCreate failed: %d", (int)status); return; }

    status = MIDIDestinationCreateWithProtocol(
        _midiClient, CFSTR("MRT2 - Collider Input"), kMIDIProtocol_1_0, &_midiVirtualDest,
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
        NSLog(@"Collider: MIDIDestinationCreate failed: %d", (int)status);
    }
}

// ─── Menu bar ────────────────────────────────────────────────────────────────

- (void)setupMenuBar {
    NSMenu* menuBar = [[NSMenu alloc] init];

    NSMenuItem* appMenuItem = [[NSMenuItem alloc] init];
    NSMenu* appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"About MRT2 - Collider" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Settings..." action:@selector(menuShowSettings:) keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit MRT2 - Collider" action:@selector(terminate:) keyEquivalent:@"q"];
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
    NSString* modelPath = [[NSUserDefaults standardUserDefaults] stringForKey:@"Collider_ModelPath"];
    if (!modelPath) return;

    if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) return;

    NSLog(@"Collider: Auto-loading model from %@", modelPath);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        BOOL success = self->_engine.load_model(modelPath.UTF8String);
        if (success) {
            NSLog(@"Collider: Model loaded successfully.");

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
            NSLog(@"Collider: Failed to auto-load model from %@", modelPath);
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
