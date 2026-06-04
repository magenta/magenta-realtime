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

#pragma once
#import <Cocoa/Cocoa.h>
#import <CoreMIDI/CoreMIDI.h>
#include <magentart/realtime_runner.h>
#include <atomic>
#include "audio_level_processor.h"

using magentart::core::RealtimeRunner;

struct StandaloneSharedState {
    magentart::common::AudioLevelProcessor levelProcessor;
    std::atomic<bool> midiNotes[128] = {};

    void noteOn(uint8_t note) { if (note < 128) midiNotes[note].store(true, std::memory_order_relaxed); }
    void noteOff(uint8_t note) { if (note < 128) midiNotes[note].store(false, std::memory_order_relaxed); }
};

@interface MagentaRTAppController : NSViewController
@property (nonatomic, assign) RealtimeRunner* engine;
@property (nonatomic, assign) StandaloneSharedState* sharedState;
// MIDI source management (set by AppDelegate after setupMIDI)
@property (nonatomic, assign) MIDIPortRef midiInputPort;
@property (nonatomic, strong) NSMutableSet<NSNumber*>* connectedSources;
// Persistence helpers — called by AppDelegate after model auto-load
- (void)notifyModelLoaded:(NSString*)modelName;
- (void)notifyPCALoaded:(int)componentCount centroidCount:(int)centroidCount fileName:(NSString*)fileName;
- (void)restoreSavedParams;
- (void)restorePrompts:(NSArray*)prompts;
// File dialog handlers — also called from menu bar actions
- (void)handleLoadModel;
- (void)handleLoadPCAFile;
// State push helpers — called by AppDelegate for play/stop feedback
- (void)sendPlayState:(BOOL)playing;
- (void)sendStateUpdate:(NSDictionary*)state;
// MIDI source management bridging to UI
- (void)handleMIDIStructureChanged;
- (void)selectMidiInput:(uint32_t)endpoint;
@end
