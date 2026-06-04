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

#include "numpy_random_state.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using magentart::core::detail::NumpyRandomState;

static const float kExpected[100] = {
    1.7640523460f, 0.4001572084f, 0.9787379841f, 2.2408931992f, 1.8675579901f,
    -0.9772778799f, 0.9500884175f, -0.1513572083f, -0.1032188518f, 0.4105985019f,
    0.1440435712f, 1.4542735070f, 0.7610377251f, 0.1216750165f, 0.4438632327f,
    0.3336743274f, 1.4940790732f, -0.2051582638f, 0.3130677017f, -0.8540957393f,
    -2.5529898158f, 0.6536185954f, 0.8644361989f, -0.7421650204f, 2.2697546240f,
    -1.4543656746f, 0.0457585173f, -0.1871838500f, 1.5327792144f, 1.4693587699f,
    0.1549474257f, 0.3781625196f, -0.8877857476f, -1.9807964682f, -0.3479121493f,
    0.1563489691f, 1.2302906807f, 1.2023798488f, -0.3873268174f, -0.3023027506f,
    -1.0485529651f, -1.4200179372f, -1.7062701906f, 1.9507753952f, -0.5096521818f,
    -0.4380743016f, -1.2527953600f, 0.7774903558f, -1.6138978476f, -0.2127402802f,
    -0.8954665612f, 0.3869024979f, -0.5108051376f, -1.1806321841f, -0.0281822283f,
    0.4283318705f, 0.0665172224f, 0.3024718977f, -0.6343220937f, -0.3627411660f,
    -0.6724604478f, -0.3595531615f, -0.8131462820f, -1.7262826023f, 0.1774261423f,
    -0.4017809362f, -1.6301983470f, 0.4627822555f, -0.9072983644f, 0.0519453958f,
    0.7290905622f, 0.1289829108f, 1.1394006845f, -1.2348258204f, 0.4023416412f,
    -0.6848100909f, -0.8707971492f, -0.5788496648f, -0.3115525321f, 0.0561653422f,
    -1.1651498408f, 0.9008264870f, 0.4656624397f, -1.5362436863f, 1.4882521938f,
    1.8958891760f, 1.1787795712f, -0.1799248358f, -1.0707526215f, 1.0544517269f,
    -0.4031769470f, 1.2224450704f, 0.2082749781f, 0.9766390365f, 0.3563663972f,
    0.7065731682f, 0.0105000207f, 1.7858704939f, 0.1269120927f, 0.4019893634f
};

int main() {
    NumpyRandomState rng(0);
    float actual[100];
    rng.randn(actual, 100);

    bool failed = false;
    printf("Verifying NumpyRandomState bit-exactness...\n");
    for (int i = 0; i < 100; ++i) {
        float diff = std::abs(actual[i] - kExpected[i]);
        if (diff > 1e-7f) {
            std::fprintf(stderr, "Mismatch at index %d: actual=%.10f, expected=%.10f, diff=%.10f\n",
                         i, actual[i], kExpected[i], diff);
            failed = true;
        }
    }

    if (failed) {
        printf("[FAIL] NumpyRandomState is not bit-exact with numpy legacy randn.\n");
        return 1;
    }

    printf("[PASS] NumpyRandomState is perfectly bit-exact with numpy legacy randn!\n");
    return 0;
}
