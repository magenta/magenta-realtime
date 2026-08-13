# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for the SFT POC tests.

The MLX trainer builds models from MLX's global RNG state (``build_model`` takes
no seed and ``mx.nn`` init draws from ``mx.random``). Seed it before every test
so numeric assertions — e.g. ``test_trainstep_decreases_loss`` — are
reproducible run to run rather than dependent on system entropy. The NNX path
is already seeded explicitly (``build_model(spec, seed=0)`` / ``nnx.Rngs``), so
this is a no-op there.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _seed_mlx_global_rng():
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.random.seed(0)
