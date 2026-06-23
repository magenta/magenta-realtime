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

"""Tests for ``magenta_rt.nnx.system.MagentaRT2System`` (tiny, random weights).

Exercises the jax/mlx-shaped system API — ``generate -> (AudioTree, state)``,
state continuation, and interleaved streams (each state carries its own stream
pytree) — on the hand-built tiny model from the parity e2e test (full-size
codec construction is too slow for CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from magenta_rt.nnx.system import MagentaRT2System
from tests.nnx.parity.test_system_e2e import _build_tiny_system


def _make_system(jit: bool = False, seed: int = 0) -> MagentaRT2System:
    return MagentaRT2System(
        size="tiny",
        model=_build_tiny_system(seed),
        restore=False,
        temperature=1.0,
        top_k=4,
        seed=seed,
        jit=jit,
    )


@pytest.mark.parametrize("jit", [False, True])
def test_generate_shapes_and_tokens(jit):
    sys = _make_system(jit=jit)
    frames = 3
    wav, state = sys.generate(frames=frames)

    # Channel-major [N, C, T] audio.
    assert wav.waveform.shape[0] == 1
    assert wav.waveform.ndim == 3
    assert wav.waveform.shape[-1] % frames == 0
    assert wav.waveform.dtype == np.float32
    assert np.all(np.isfinite(wav.waveform))
    assert np.abs(wav.waveform).max() <= 1.0

    # Codes are the per-codebook RVQ indices that produced the audio.
    assert wav.codes.shape == (1, frames, 4)
    assert wav.codes.min() >= 0
    assert wav.codes.max() < 8  # tiny codebook_size

    assert state.batch_size == 1


def test_state_continuation_matches_one_shot():
    """generate(2) + generate(2, state) == generate(4) — the threaded state
    (KV caches, codec buffers, sampling rng) continues the stream exactly."""
    sys = _make_system(jit=False)

    wav_full, _ = sys.generate(frames=4)

    wav_a, state = sys.generate(frames=2)  # state=None -> fresh stream
    wav_b, _ = sys.generate(frames=2, state=state)

    np.testing.assert_allclose(
        np.concatenate([wav_a.waveform, wav_b.waveform], axis=-1),
        wav_full.waveform,
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.concatenate([wav_a.codes, wav_b.codes], axis=1),
        wav_full.codes,
    )


@pytest.mark.parametrize("jit", [False, True])
def test_interleaved_streams(jit):
    """State carries the stream pytree, so a handle stays continuable across an
    intervening fresh stream (the old module-held design could not do this)."""
    sys = _make_system(jit=jit)

    # Reference: a single uninterrupted 4-frame stream from the seed.
    ref_full, _ = sys.generate(frames=4)

    # Stream A (same seed -> same first 2 frames as the reference).
    wav_a0, state_a = sys.generate(frames=2)
    # An independent fresh stream B in between must not disturb A.
    sys.generate(frames=2)
    # Continue A past B; it must match the reference's last 2 frames.
    wav_a1, _ = sys.generate(frames=2, state=state_a)

    np.testing.assert_array_equal(
        np.concatenate([wav_a0.codes, wav_a1.codes], axis=1),
        ref_full.codes,
    )


def test_per_element_temperature_rejected():
    sys = _make_system(jit=False)
    with pytest.raises(ValueError, match="per-element"):
        sys.generate(frames=1, temperature=[1.0, 2.0])


class _StubStyleModel:
    """Tokenizes any input into ``n`` fixed RVQ rows (no MusicCoCa load).

    ``generate`` calls ``style_model.tokenize(style)`` and derives the batch
    size from the number of rows, so this drives N>1 generation without the
    TFLite style model.
    """

    def __init__(self, n: int):
        self._n = n

    def tokenize(self, style):
        # [n, k] zeros -> n masked-but-valid musiccoca rows; normalize_style_rows
        # pads/truncates k to the preset's musiccoca length.
        return np.zeros((self._n, 64), dtype=np.int32)


@pytest.mark.slow
def test_batched_functional_generate():
    """N>1 batched generate threads batched stream leaves through the functional
    split/merge/donate path. Needs a multi-channel preset (the tiny preset is
    single-channel -> N=1 only), so use mrt2_small with random weights (no
    checkpoint) and a stub style model that yields two style rows."""
    N = 2
    sys = MagentaRT2System(
        size="mrt2_small", restore=False, style_model=_StubStyleModel(N),
        temperature=1.0, top_k=8, seed=0, jit=True,
    )

    # Shapes + batch size: waveform [N, 2, T], codes [N, frames, Q].
    wav, state = sys.generate(style="x", frames=4)
    assert wav.waveform.shape[0] == N
    assert wav.waveform.ndim == 3
    assert wav.codes.shape[0] == N
    assert wav.codes.shape[1] == 4
    assert state.batch_size == N
    assert np.all(np.isfinite(wav.waveform))
    assert np.abs(wav.waveform).max() <= 1.0

    # Continuation matches one-shot at N>1 (batched donation + threaded stream):
    # both start from state=None (same seed), so the codes are bit-identical.
    full, _ = sys.generate(style="x", frames=4)
    part_a, st = sys.generate(style="x", frames=2)
    part_b, _ = sys.generate(style="x", frames=2, state=st)
    np.testing.assert_array_equal(
        np.concatenate([part_a.codes, part_b.codes], axis=1), full.codes,
    )

    # Functional scan is lossless vs the per-step loop at N>1.
    scan_tree, _ = sys.generate(style="x", frames=4, scan=True)
    np.testing.assert_array_equal(scan_tree.codes, full.codes)
