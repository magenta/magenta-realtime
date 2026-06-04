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

// Shared state between audio/MIDI threads and the UI controller
struct JamSharedState {
    std::atomic<bool> midiNotes[128] = {};

    static constexpr int VIZ_BUF_SIZE = 8192;
    float vizRing[VIZ_BUF_SIZE] = {};
    std::atomic<int> vizHead{0};

    magentart::common::AudioLevelProcessor levelProcessor;

    void pushAudioSamples(const float* left, const float* right, int count) {
        int h = vizHead.load(std::memory_order_relaxed);
        for (int i = 0; i < count; i++) {
            vizRing[h] = (left[i] + right[i]) * 0.5f;
            h = (h + 1) % VIZ_BUF_SIZE;
        }
        vizHead.store(h, std::memory_order_release);

        levelProcessor.process_block(left, right, count);
    }

    void noteOn(uint8_t note) { if (note < 128) midiNotes[note].store(true, std::memory_order_relaxed); }
    void noteOff(uint8_t note) { if (note < 128) midiNotes[note].store(false, std::memory_order_relaxed); }
};

@interface JamAppController : NSViewController
@property (nonatomic, assign) RealtimeRunner* engine;
@property (nonatomic, assign) JamSharedState* sharedState;
@property (nonatomic, assign) MIDIPortRef midiInputPort;
@property (nonatomic, strong) NSMutableSet<NSNumber*>* connectedSources;
@property (nonatomic, assign) std::atomic<bool>* soloMode;
@property (nonatomic, assign) std::atomic<float>* cfgNotesSliderValue;
@property (nonatomic, assign) std::atomic<float>* gateDecaySeconds;

- (void)notifyModelLoaded:(NSString*)modelName;
- (void)sendStateUpdate:(NSDictionary*)state;
- (void)restoreSavedParams;
- (void)handleLoadModel;
- (void)showReactSettings;
- (void)sendPlayState:(BOOL)playing;
// Param bridging — also used by settings window
- (void)applyParamToEngine:(int)address value:(float)value;
- (float)readParamFromEngine:(int)address;
// Computer-keyboard-as-MIDI (toggled from settings window)
- (void)setComputerKeyboardMidiEnabled:(BOOL)enabled;
// MIDI source management bridging to UI
- (void)handleMIDIStructureChanged;
- (void)selectMidiInput:(uint32_t)endpoint;
@end
