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

/// @file envelope.h
/// @brief One-pole attack/release envelope for use on the audio thread only.
///
/// Extracted from realtime_runner.h so it can be included and tested
/// independently without pulling in the full MLX/TFLite machinery.

#include <cmath>

namespace magentart {
namespace core {

/// One-pole attack/release envelope for use on the audio thread only.
/// `value` is a plain float -- not atomic. All reads and writes happen
/// exclusively on the audio thread inside `read_audio_stereo`. The UI thread
/// communicates resets via `RealtimeRunner::reset_env_trigger_`, an atomic
/// bool that the audio thread exchanges-to-consume.
struct ExponentialEnvelope {
    float value{0.0f};
    float alpha_attack  = 0.0f;
    float alpha_release = 0.0f;

    void set_attack_samples(float samples) {
        alpha_attack = 1.0f - std::exp(-4.60517f / samples);
    }
    void set_release_samples(float samples) {
        alpha_release = 1.0f - std::exp(-4.60517f / samples);
    }
    float tick(float target) {
        float alpha = (target > value) ? alpha_attack : alpha_release;
        value = value + (target - value) * alpha;
        return value;
    }
};

}  // namespace core
}  // namespace magentart
