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

#include "midi_note_tracker.h"

namespace magentart {
namespace core {

MidiNoteTracker::MidiNoteTracker() {
    for (int i = 0; i < kTotalPitches; ++i) {
        note_states_[i].store(NOTE_IDLE, std::memory_order_relaxed);
    }
}

void MidiNoteTracker::noteOn(int pitch) {
    if (pitch >= 0 && pitch < kTotalPitches) {
        note_states_[pitch].store(NOTE_ONSET, std::memory_order_relaxed);
    }
}

void MidiNoteTracker::noteOff(int pitch) {
    if (pitch >= 0 && pitch < kTotalPitches) {
        NoteState expected = note_states_[pitch].load(std::memory_order_relaxed);
        while (true) {
            NoteState desired = expected;

            // State transitions on Note Off:
            // - If it was ONSET (not yet processed by inference), it becomes ONSET_RELEASED.
            //   This "latches" the onset so the inference thread doesn't miss it.
            // - If it was SUSTAIN (already processed by inference), it becomes IDLE.
            if (expected == NOTE_ONSET) desired = NOTE_ONSET_RELEASED;
            else if (expected == NOTE_SUSTAIN) desired = NOTE_IDLE;
            else break; // Already IDLE or ONSET_RELEASED, no update needed.

            // Attempt to atomically update the state.
            // If it fails (spurious failure or concurrent update), 'expected' is
            // updated with the new current value, and we retry.
            if (note_states_[pitch].compare_exchange_weak(expected, desired, std::memory_order_relaxed)) {
                break;
            }
        }
    }
}

NoteState MidiNoteTracker::evaluateAndUpdate(int pitch) {
    // Read the current state atomically.
    NoteState expected = note_states_[pitch].load(std::memory_order_relaxed);

    // Lock-free loop to update state. Retries if compare_exchange fails.
    while (true) {
        NoteState desired = expected;

        // Define state transitions:
        // - ONSET progresses to SUSTAIN after being observed once.
        // - ONSET_RELEASED progresses to IDLE after being observed once.
        if (expected == NOTE_ONSET) desired = NOTE_SUSTAIN;
        else if (expected == NOTE_ONSET_RELEASED) desired = NOTE_IDLE;

        // If the state is already IDLE or SUSTAIN, no transition is needed.
        if (desired == expected) break;

        // Attempt to atomically update the state.
        // If it fails (because another thread modified it or spuriously),
        // 'expected' is updated with the new current value, and we retry.
        if (note_states_[pitch].compare_exchange_weak(expected, desired, std::memory_order_relaxed)) {
            break;
        }
    }

    // Return the state that was read before the successful update (or no update).
    // This ensures the inference thread sees the "latched" onset state.
    return expected;
}

} // namespace core
} // namespace magentart
