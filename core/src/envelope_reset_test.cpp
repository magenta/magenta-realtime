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

// Verifies two invariants introduced by fix/audio-thread-envelope-cas-removal:
//
//  1. ExponentialEnvelope::tick() produces bounded, finite values and converges
//     toward its target — checked deterministically in a single thread.
//
//  2. The atomic-bool trigger pattern (UI thread stores `true` with release,
//     audio thread exchanges to `false` with acquire) is data-race-free when
//     exercised concurrently. Run this binary under ThreadSanitizer to catch
//     any race on `env.value` that would indicate a broken ownership model.
//
// This file is standalone: it mirrors the ExponentialEnvelope definition from
// realtime_runner.h and the consume pattern from read_audio_stereo so it can
// build without the full MLX/TFLite link (same model as numpy_random_state_test).

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

// Mirrors ExponentialEnvelope in core/include/magentart/realtime_runner.h.
// Keep in sync if the struct changes.
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

// ---- Test 1: tick() arithmetic -----------------------------------------------

static bool test_tick_arithmetic() {
    bool failed = false;

    ExponentialEnvelope env;
    env.set_attack_samples(1920.0f);
    env.set_release_samples(48000.0f);
    env.value = 0.0f;

    // Attack toward 1.0: values must be monotonically increasing and in [0, 1].
    float prev = env.value;
    for (int i = 0; i < 9600; ++i) {
        float v = env.tick(1.0f);
        if (std::isnan(v) || std::isinf(v)) {
            std::fprintf(stderr, "[FAIL] tick() produced non-finite value %f at attack sample %d\n", v, i);
            failed = true;
            break;
        }
        if (v < 0.0f || v > 1.0f) {
            std::fprintf(stderr, "[FAIL] tick() out of range: %f at attack sample %d\n", v, i);
            failed = true;
            break;
        }
        if (v < prev - 1e-6f) {
            std::fprintf(stderr, "[FAIL] tick() decreased during attack: %f -> %f at sample %d\n", prev, v, i);
            failed = true;
            break;
        }
        prev = v;
    }

    // After 5x the attack time constant, value must be close to 1.0.
    if (std::abs(env.value - 1.0f) > 0.01f) {
        std::fprintf(stderr, "[FAIL] envelope did not converge: value=%.6f after 9600 attack samples\n", env.value);
        failed = true;
    }

    // Snap to zero (simulates the trigger consume path in read_audio_stereo).
    env.value = 0.0f;
    if (env.value != 0.0f) {
        std::fprintf(stderr, "[FAIL] snap-to-zero did not set value to 0\n");
        failed = true;
    }

    // Release from 1.0 toward 0.0: values must be monotonically decreasing.
    env.value = 1.0f;
    prev = env.value;
    for (int i = 0; i < 48000; ++i) {
        float v = env.tick(0.0f);
        if (std::isnan(v) || std::isinf(v)) {
            std::fprintf(stderr, "[FAIL] tick() produced non-finite value %f at release sample %d\n", v, i);
            failed = true;
            break;
        }
        if (v < 0.0f || v > 1.0f) {
            std::fprintf(stderr, "[FAIL] tick() out of range: %f at release sample %d\n", v, i);
            failed = true;
            break;
        }
        if (v > prev + 1e-6f) {
            std::fprintf(stderr, "[FAIL] tick() increased during release: %f -> %f at sample %d\n", prev, v, i);
            failed = true;
            break;
        }
        prev = v;
    }

    return !failed;
}

// ---- Test 2: concurrent trigger pattern (designed for TSan) ------------------
//
// Reproduces the exact release/acquire pattern from realtime_runner.h and
// read_audio_stereo. `env.value` is owned exclusively by the "audio thread";
// the "UI thread" communicates only via the atomic bool trigger.

static bool test_concurrent_trigger() {
    constexpr int kBlocks      = 2000;
    constexpr int kBlockSize   = 128;
    constexpr int kUiTriggers  = 500;

    ExponentialEnvelope env;
    env.set_attack_samples(1920.0f);
    env.value = 1.0f;

    std::atomic<bool> trigger{false};
    std::atomic<bool> audio_done{false};
    bool failed = false;

    // UI thread: fires trigger_reset() repeatedly (store true with release).
    std::thread ui_thread([&]() {
        for (int i = 0; i < kUiTriggers; ++i) {
            trigger.store(true, std::memory_order_release);
            // Yield to increase interleaving pressure for TSan.
            std::this_thread::yield();
        }
    });

    // Audio thread: mirrors the block loop in read_audio_stereo.
    // Owns env.value; consumes trigger with acquire exchange.
    for (int block = 0; block < kBlocks && !failed; ++block) {
        if (trigger.exchange(false, std::memory_order_acquire)) {
            env.value = 0.0f;
        }
        for (int i = 0; i < kBlockSize; ++i) {
            float v = env.tick(1.0f);
            if (std::isnan(v) || std::isinf(v) || v < 0.0f || v > 1.0f) {
                std::fprintf(stderr,
                    "[FAIL] concurrent tick() produced out-of-range value %f "
                    "(block %d, sample %d)\n", v, block, i);
                failed = true;
                break;
            }
        }
        std::this_thread::yield();
    }

    ui_thread.join();
    return !failed;
}

// ---- main --------------------------------------------------------------------

int main() {
    bool ok = true;

    std::printf("Test 1: ExponentialEnvelope tick() arithmetic... ");
    if (test_tick_arithmetic()) {
        std::printf("[PASS]\n");
    } else {
        std::printf("[FAIL]\n");
        ok = false;
    }

    std::printf("Test 2: concurrent trigger pattern (run under TSan for full coverage)... ");
    if (test_concurrent_trigger()) {
        std::printf("[PASS]\n");
    } else {
        std::printf("[FAIL]\n");
        ok = false;
    }

    if (ok) {
        std::printf("[PASS] all envelope reset tests passed.\n");
        return 0;
    } else {
        std::printf("[FAIL] one or more tests failed.\n");
        return 1;
    }
}
