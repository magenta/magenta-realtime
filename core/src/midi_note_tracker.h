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

#include <atomic>
#include <cstdint>

namespace magentart {
namespace core {

enum NoteState : uint8_t {
    NOTE_IDLE = 0,
    NOTE_ONSET = 1,    // Note turned on, but not yet processed by inference.
    NOTE_SUSTAIN = 2,  // Onset processed, note is still held down.

    // A note-off occurred before the inference thread could process the NOTE_ONSET.
    // We keep it in this state so the inference thread sees the onset for one frame,
    // then it resets to NOTE_IDLE.
    NOTE_ONSET_RELEASED = 3
};

constexpr int kNumStandardMidiNotes = 128;
constexpr int kNumDrumTriggers = 4;
constexpr int kTotalPitches = kNumStandardMidiNotes + kNumDrumTriggers; // 132

class MidiNoteTracker {
public:
    MidiNoteTracker();
    ~MidiNoteTracker() = default;

    void noteOn(int pitch);
    void noteOff(int pitch);

    // Advances the state machine for a pitch and returns the state observed.
    // CRITICAL: This must be called EXACTLY ONCE per pitch per inference frame
    // to ensure the "latch" behavior works correctly.
    NoteState evaluateAndUpdate(int pitch);

private:
    std::atomic<NoteState> note_states_[kTotalPitches]{};
};

} // namespace core
} // namespace magentart
