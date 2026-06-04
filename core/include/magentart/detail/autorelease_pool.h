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

/// @file autorelease_pool.h
/// @brief Portable RAII wrapper around an Objective-C autorelease pool.
///
/// On Apple platforms this pushes/pops an Objective-C autorelease pool around
/// its scope; on other platforms it is a no-op. MLX on Metal produces
/// autoreleased Objective-C objects as a side effect of `mlx::core::eval`, so
/// long-running loops need to drain the pool every iteration to avoid memory
/// growth. We expose this through a plain C++ class rather than the
/// `@autoreleasepool { ... }` syntax so consumers of `magentart::core` can
/// stay in pure C++ (no `.mm` requirement on the caller side).

namespace magentart {
namespace detail {

class AutoreleasePool {
public:
    AutoreleasePool();
    ~AutoreleasePool();
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
    AutoreleasePool(AutoreleasePool&&) = delete;
    AutoreleasePool& operator=(AutoreleasePool&&) = delete;

private:
    void* pool_ = nullptr;
};

}  // namespace detail
}  // namespace magentart
