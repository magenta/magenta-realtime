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

"""Unit tests for the functional jax MusicCoCa (no external resources)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from magenta_rt.jax import musiccoca as mc


def test_init_params_shapes_run_end_to_end():
    params = mc.init_params()
    wav = jnp.zeros((1, mc.CLIP_SAMPLES), jnp.float32)
    # log(0 + 1e-3) is finite; the whole pipeline must run on zeros.
    emb = mc.embed_audio(params, wav)
    assert emb.shape == (1, 768)
    assert np.isfinite(np.asarray(emb)).all()

    ids = jnp.zeros((2, 128), jnp.int32)
    paddings = jnp.concatenate(
        [jnp.zeros((2, 5)), jnp.ones((2, 123))], axis=-1
    ).astype(jnp.float32)
    emb = mc.embed_text(params, ids, paddings)
    assert emb.shape == (2, 768)
    assert np.isfinite(np.asarray(emb)).all()

    tokens = mc.tokenize(params, emb)
    assert tokens.shape == (2, 12)


def test_quantizer_matches_numpy_reference():
    rng = np.random.RandomState(0)
    params = mc.init_params()
    codebooks = rng.randn(12, 1024, 768).astype(np.float32)
    params["quantizer"]["codebooks"] = jnp.asarray(codebooks)
    x = rng.randn(3, 768).astype(np.float32)

    tokens = np.asarray(mc.tokenize(params, jnp.asarray(x)))

    residual = x.astype(np.float64)
    want = []
    for stage in range(12):
        cb = codebooks[stage].astype(np.float64)
        d = ((residual[:, None, :] - cb[None]) ** 2).sum(-1)
        idx = d.argmin(-1)
        want.append(idx)
        residual = residual - cb[idx]
    np.testing.assert_array_equal(tokens, np.stack(want, axis=-1))

    recon = np.asarray(mc.decode_tokens(params, jnp.asarray(tokens)))
    np.testing.assert_allclose(recon, x - residual, rtol=1e-5, atol=1e-3)


class _FakeVocab:
    def EncodeAsIds(self, text):  # noqa: N802 (sentencepiece API)
        assert text == text.lower()
        return list(range(2, 2 + len(text.split())))


def test_encode_text_padding():
    ids, paddings = mc.encode_text(_FakeVocab(), "Three Word Prompt")
    assert ids.shape == (128,) and paddings.shape == (128,)
    assert ids[0] == 1  # SOS
    np.testing.assert_array_equal(ids[1:4], [2, 3, 4])
    assert (ids[4:] == 0).all()
    np.testing.assert_array_equal(paddings[:4], 0.0)
    np.testing.assert_array_equal(paddings[4:], 1.0)


def test_public_entry_points_jit():
    params = mc.init_params()
    emb = jax.jit(mc.embed_text)(
        params, jnp.zeros((1, 128), jnp.int32), jnp.zeros((1, 128), jnp.float32)
    )
    tokens = jax.jit(mc.tokenize)(params, emb)
    assert tokens.shape == (1, 12)
    mapped = jax.jit(mc.map_text_embedding)(
        params, jnp.ones((1, 768)), jnp.ones((1, 768))
    )
    assert mapped.shape == (1, 768)
