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
#include <cmath>

namespace magentart {
namespace common {

class AudioLevelProcessor {
public:
    AudioLevelProcessor() : peakLeft_{0.0f}, peakRight_{0.0f} {}

    // Process a block of left and right samples, computing their peaks atomically
    void process_block(const float* left, const float* right, int count) {
        float localMaxL = 0.0f;
        float localMaxR = 0.0f;
        for (int i = 0; i < count; i++) {
            float absL = std::abs(left[i]);
            float absR = std::abs(right[i]);
            if (absL > localMaxL) localMaxL = absL;
            if (absR > localMaxR) localMaxR = absR;
        }

        // Atomically update peakLeft
        float curL = peakLeft_.load(std::memory_order_relaxed);
        while (localMaxL > curL && !peakLeft_.compare_exchange_weak(curL, localMaxL, std::memory_order_relaxed)) {
            // Loop until successful exchange or curL is already greater or equal
        }

        // Atomically update peakRight
        float curR = peakRight_.load(std::memory_order_relaxed);
        while (localMaxR > curR && !peakRight_.compare_exchange_weak(curR, localMaxR, std::memory_order_relaxed)) {
            // Loop until successful exchange or curR is already greater or equal
        }
    }

    // Retrieve the peak levels since the last read, and atomically reset them to 0.0f
    void read_and_reset_peaks(float& outLeft, float& outRight) {
        outLeft = peakLeft_.exchange(0.0f, std::memory_order_relaxed);
        outRight = peakRight_.exchange(0.0f, std::memory_order_relaxed);
    }

private:
    std::atomic<float> peakLeft_;
    std::atomic<float> peakRight_;
};

} // namespace common
} // namespace magentart
