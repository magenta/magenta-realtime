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

"""Smoke test for the pure-MLX generate.py.

Drives ``magenta_rt.mlx_pure.generate.main`` with the tiny config and
asserts the pipeline runs end-to-end and is bit-deterministic across
runs at the same seed. No ``sequence_layers`` runtime dependency.
"""

from __future__ import annotations

import numpy as np

from magenta_rt.mlx_pure import generate as pure_gen


def test_generate_tiny_runs():
    audio = pure_gen.main(
        model_name="tiny", seed=0, num_steps=3, quiet=True, num_cfgs=0,
    )
    arr = np.array(audio)
    assert arr.ndim in (2, 3)
    assert arr.shape[0] == 1
    assert arr.shape[1] > 0


def test_generate_tiny_deterministic():
    a = pure_gen.main(
        model_name="tiny", seed=42, num_steps=3, quiet=True, num_cfgs=0,
    )
    b = pure_gen.main(
        model_name="tiny", seed=42, num_steps=3, quiet=True, num_cfgs=0,
    )
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_generate_forced_tokens_shortcut():
    """Verifies the forced_tokens path: depthformer.step returns exactly the
    forced codes (bypassing sampling), built through MagentaRT2Sampler.

    Note: this exercises only the depthformer code-forcing path; it does not
    run the spectrostream decoder.
    """
    import mlx.core as mx
    from magenta_rt.mlx_pure.load_weights import init_random_params
    from magenta_rt.mlx_pure.model import MagentaRT2Sampler

    mrt = MagentaRT2Sampler.from_preset("tiny", int16_outputs=True)
    # tiny RVQ overrides codebook_size to 4
    mrt.codebook_size = 4
    num_codebooks = getattr(mrt.depthformer.decoder, "num_codebooks", 3)

    init_random_params(mrt, seed=1)
    state = mrt.make_initial_state(batch_size=1, seed=1)
    src = mx.array([[[1]]], dtype=mx.int32)
    encoded = mrt.depthformer.encode(src)

    forced = mx.zeros((1, 1, num_codebooks), dtype=mx.int32)
    codes, state = mrt.depthformer.step(state, source_frame=encoded, forced_tokens=forced)
    assert codes.tolist() == forced.tolist()
