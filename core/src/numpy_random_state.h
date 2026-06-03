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

#include <cmath>
#include <cstdint>

namespace magentart {
namespace core {
namespace detail {

// Reproduces numpy's legacy `np.random.RandomState(seed).randn(n)` exactly, so
// the C++ MusicCoCa mapper path generates the same Gaussian noise — and hence
// the same refined embedding and RVQ tokens — as the Python reference in
// magenta_rt/musiccoca.py. It implements MT19937 with numpy's scalar integer
// seeding (`mt19937_seed`) plus the polar Box-Muller variant (`legacy_gauss`)
// that RandomState uses for `randn`.
class NumpyRandomState {
public:
  explicit NumpyRandomState(uint32_t seed) {
    key_[0] = seed;
    for (int i = 1; i < kN; ++i) {
      key_[i] = (1812433253UL * (key_[i - 1] ^ (key_[i - 1] >> 30)) +
                 static_cast<uint32_t>(i));
    }
    pos_ = kN;
  }

  // Fills out[0..n) with standard-normal samples in numpy's `randn` order.
  void randn(float *out, int n) {
    for (int i = 0; i < n; ++i)
      out[i] = static_cast<float>(next_gauss());
  }

private:
  static constexpr int kN = 624;
  static constexpr int kM = 397;
  static constexpr uint32_t kMatrixA = 0x9908b0dfUL;
  static constexpr uint32_t kUpperMask = 0x80000000UL;
  static constexpr uint32_t kLowerMask = 0x7fffffffUL;

  void generate() {
    uint32_t y;
    int i;
    for (i = 0; i < kN - kM; ++i) {
      y = (key_[i] & kUpperMask) | (key_[i + 1] & kLowerMask);
      key_[i] = key_[i + kM] ^ (y >> 1) ^ ((y & 1) ? kMatrixA : 0);
    }
    for (; i < kN - 1; ++i) {
      y = (key_[i] & kUpperMask) | (key_[i + 1] & kLowerMask);
      key_[i] = key_[i + (kM - kN)] ^ (y >> 1) ^ ((y & 1) ? kMatrixA : 0);
    }
    y = (key_[kN - 1] & kUpperMask) | (key_[0] & kLowerMask);
    key_[kN - 1] = key_[kM - 1] ^ (y >> 1) ^ ((y & 1) ? kMatrixA : 0);
    pos_ = 0;
  }

  uint32_t next_uint32() {
    if (pos_ == kN)
      generate();
    uint32_t y = key_[pos_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
  }

  double next_double() {
    uint32_t a = next_uint32() >> 5;
    uint32_t b = next_uint32() >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  double next_gauss() {
    if (has_gauss_) {
      has_gauss_ = false;
      return gauss_;
    }
    double f, x1, x2, r2;
    do {
      x1 = 2.0 * next_double() - 1.0;
      x2 = 2.0 * next_double() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    f = std::sqrt(-2.0 * std::log(r2) / r2);
    gauss_ = f * x1;
    has_gauss_ = true;
    return f * x2;
  }

  uint32_t key_[kN];
  int pos_ = kN;
  bool has_gauss_ = false;
  double gauss_ = 0.0;
};

} // namespace detail
} // namespace core
} // namespace magentart
