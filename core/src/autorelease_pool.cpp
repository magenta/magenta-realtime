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

// Implementation of `magentart::detail::AutoreleasePool`.
//
// On Apple platforms we bind directly to the Objective-C runtime's pool
// push/pop C entry points (the same primitives `@autoreleasepool` desugars
// to), which are ARC-safe and callable from plain C++ — so this translation
// unit does not need to be compiled as Objective-C++.
//
// On non-Apple platforms the class is a no-op; the type exists so callers
// can unconditionally construct one on the stack.

#include <magentart/detail/autorelease_pool.h>

#if defined(__APPLE__)
extern "C" {
    // Declared in <objc/objc-internal.h>; redeclared here so we don't have to
    // pull a private header into every TU that uses the pool.
    void* objc_autoreleasePoolPush(void);
    void  objc_autoreleasePoolPop(void*);
}
#endif

namespace magentart {
namespace detail {

AutoreleasePool::AutoreleasePool()
#if defined(__APPLE__)
    : pool_(objc_autoreleasePoolPush())
#endif
{
}

AutoreleasePool::~AutoreleasePool() {
#if defined(__APPLE__)
    objc_autoreleasePoolPop(pool_);
#endif
}

}  // namespace detail
}  // namespace magentart
