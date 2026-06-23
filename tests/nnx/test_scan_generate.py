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

"""``MagentaRT2System.generate(scan=True)`` must be bit-identical to the
per-step loop — it is the lossless fast path (one ``nnx.scan`` over all frames,
state threaded through the carry), not an approximation."""

from __future__ import annotations

import jax
import numpy as np

from magenta_rt.nnx.system import MagentaRT2System

jax.config.update("jax_enable_x64", False)


def test_scan_generate_matches_per_step_loop():
    # Tiny untrained preset (random weights) — exercises the full
    # depthformer + codec generate path without a checkpoint.
    mrt = MagentaRT2System(size="tiny", restore=False, seed=0)

    frames = 6
    # Each generate(state=None) resets the stream from the same seed, so the
    # per-step loop and the scan must consume the identical RNG sequence.
    loop_tree, _ = mrt.generate(frames=frames, temperature=1.0, top_k=4, scan=False)
    scan_tree, _ = mrt.generate(frames=frames, temperature=1.0, top_k=4, scan=True)

    loop_codes, scan_codes = np.asarray(loop_tree.codes), np.asarray(scan_tree.codes)
    loop_wav, scan_wav = np.asarray(loop_tree.waveform), np.asarray(scan_tree.waveform)

    assert loop_codes.shape == scan_codes.shape
    assert loop_wav.shape == scan_wav.shape
    # Codes are the model's discrete output: bit-identical (same math, same RNG).
    assert np.array_equal(loop_codes, scan_codes), "scan codes differ from per-step loop"
    # The codec audio matches to fp32 round-off — fusing the per-frame streaming
    # convolutions inside the scan reorders some reductions (bit-identical on GPU,
    # ~1e-4 on CPU); this is float non-associativity, not a different result.
    assert float(np.abs(loop_wav - scan_wav).max()) < 2e-3, "scan audio differs from per-step loop"
