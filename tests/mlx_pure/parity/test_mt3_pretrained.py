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

"""Parity: pure-MLX MT3 vs the nnx port with pretrained weights.

Both backends load the same converted safetensors (``python -m
magenta_rt.mt3.download``); this asserts the MLX encode / teacher-forced
logits / greedy tokens match the nnx reference (which is itself verified
bit-exact against the original Linen network). Gated like the other
real-weight tests; opt in with ``pytest -m checkpoint``.
"""

from __future__ import annotations

import numpy as np
import pytest

from magenta_rt import paths

pytestmark = pytest.mark.checkpoint

_WEIGHTS = paths.mt3_dir() / "mt3_mt3.safetensors"
if not _WEIGHTS.exists():
    pytest.skip(f"mt3 checkpoint not found at {_WEIGHTS}", allow_module_level=True)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("jax")
    from magenta_rt.mlx_pure import mt3 as mlx_mt3
    from magenta_rt.nnx import mt3 as nnx_mt3

    return nnx_mt3.load_model("mt3"), mlx_mt3.load_model("mt3")


def test_logits_match_nnx(models):
    import jax.numpy as jnp
    import mlx.core as mx

    nnx_model, mlx_model = models
    rng = np.random.RandomState(0)
    spec = rng.randn(2, 256, 512).astype(np.float32) * 2.0
    dec_in = np.concatenate(
        [np.zeros((2, 1), np.int32), rng.randint(3, 1000, (2, 15)).astype(np.int32)],
        axis=1,
    )
    dec_tgt = np.concatenate([dec_in[:, 1:], np.ones((2, 1), np.int32)], axis=1)

    ref_enc = np.asarray(nnx_model.encode(jnp.asarray(spec), deterministic=True))
    got_enc = np.asarray(mlx_model.encode(mx.array(spec)))
    np.testing.assert_allclose(got_enc, ref_enc, atol=5e-4)

    ref = np.asarray(
        nnx_model(jnp.asarray(spec), jnp.asarray(dec_in), jnp.asarray(dec_tgt),
                  deterministic=True)
    )
    got = np.asarray(mlx_model(mx.array(spec), mx.array(dec_in), mx.array(dec_tgt)))
    np.testing.assert_allclose(got, ref, atol=5e-4)


def test_greedy_tokens_match_nnx(models):
    import jax.numpy as jnp
    import mlx.core as mx

    from magenta_rt.mlx_pure.mt3 import greedy_decode as mlx_greedy
    from magenta_rt.nnx.mt3 import greedy_decode as nnx_greedy

    nnx_model, mlx_model = models
    rng = np.random.RandomState(1)
    spec = rng.randn(2, 256, 512).astype(np.float32) * 2.0
    ref_enc = np.asarray(nnx_model.encode(jnp.asarray(spec), deterministic=True))

    ref_tokens = np.asarray(nnx_greedy(nnx_model, jnp.asarray(ref_enc), max_decode_length=64))
    got_tokens = mlx_greedy(mlx_model, mx.array(ref_enc), max_decode_length=64)
    np.testing.assert_array_equal(got_tokens, ref_tokens)


def test_transcribe_produces_notes(models):
    """End-to-end smoke test: transcribe a synthesized arpeggio (MLX only)."""
    from magenta_rt.mlx_pure.mt3 import transcribe

    _, mlx_model = models
    sr = 16000

    def tone(pitch, start, dur, total):
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        t = np.arange(int(dur * sr)) / sr
        x = sum((0.6**k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        x = x * np.exp(-2.5 * t) * 0.3
        out = np.zeros(int(total * sr), np.float32)
        i = int(start * sr)
        out[i : i + len(x)] += x.astype(np.float32)
        return out

    onsets = [(60, 0.25), (64, 1.0), (67, 1.75), (72, 2.5)]
    audio = sum(tone(p, s, 1.0, 4.0) for p, s in onsets)

    ns = transcribe(mlx_model, audio)

    assert len(ns.notes) > 0
    note_onsets = sorted(n.start_time for n in ns.notes)
    for _, expected_onset in onsets:
        assert any(abs(t - expected_onset) < 0.1 for t in note_onsets), (
            f"no note onset near {expected_onset}s in {note_onsets}"
        )
