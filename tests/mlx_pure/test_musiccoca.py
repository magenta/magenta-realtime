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

"""Unit tests for the mlx_pure MusicCoCa components (no external resources)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from magenta_rt.mlx_pure.musiccoca import (
    AttentionPooler,
    EmbeddingQuantizer,
    LayerNorm,
    LogMelFrontend,
    TransformerLayer,
    encode_text,
)
from magenta_rt.mlx_pure.musiccoca.frontend import NUM_FRAMES, NUM_MEL_BINS


def test_layer_norm_matches_numpy():
    rng = np.random.RandomState(0)
    x = rng.randn(2, 5, 768).astype(np.float32)
    ln = LayerNorm(768)
    ln.scale = mx.array(rng.randn(768).astype(np.float32))
    ln.bias = mx.array(rng.randn(768).astype(np.float32))
    got = np.array(ln(mx.array(x)))
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    want = (x - mean) / np.sqrt(var + 1e-6)
    want = want * np.array(ln.scale) + np.array(ln.bias)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_transformer_layer_shapes_and_mask():
    rng = np.random.RandomState(0)
    layer = TransformerLayer()
    x = mx.array(rng.randn(2, 7, 768).astype(np.float32))
    paddings = np.zeros((2, 7), np.float32)
    paddings[:, 5:] = 1.0
    attn_mask = mx.array(paddings < 0.5)[:, None, None, :]
    ffn_mask = mx.array(1.0 - paddings)[..., None]
    out = layer(x, attn_mask=attn_mask, ffn_mask=ffn_mask)
    assert out.shape == (2, 7, 768)
    assert np.isfinite(np.array(out)).all()


def test_attention_pooler_shape():
    rng = np.random.RandomState(0)
    pooler = AttentionPooler()
    x = mx.array(rng.randn(2, 9, 768).astype(np.float32))
    out = pooler(x)
    assert out.shape == (2, 768)


def test_quantizer_matches_numpy_reference():
    rng = np.random.RandomState(0)
    quantizer = EmbeddingQuantizer()
    codebooks = rng.randn(12, 1024, 768).astype(np.float32)
    quantizer.codebooks = mx.array(codebooks)
    x = rng.randn(3, 768).astype(np.float32)

    tokens = np.array(quantizer.tokenize(mx.array(x)))
    assert tokens.shape == (3, 12)
    assert (tokens >= 0).all() and (tokens < 1024).all()

    # Greedy nearest-neighbor RVQ in numpy.
    residual = x.astype(np.float64)
    want = []
    for stage in range(12):
        cb = codebooks[stage].astype(np.float64)
        d = ((residual[:, None, :] - cb[None]) ** 2).sum(-1)
        idx = d.argmin(-1)
        want.append(idx)
        residual = residual - cb[idx]
    np.testing.assert_array_equal(tokens, np.stack(want, axis=-1))

    # decode() sums one codeword per stage; check against the residual path.
    recon = np.array(quantizer.decode(mx.array(tokens)))
    np.testing.assert_allclose(recon, x - residual, rtol=1e-5, atol=1e-3)

    # Leading-shape preservation.
    tokens2 = np.array(quantizer.tokenize(mx.array(x.reshape(1, 3, 768))))
    np.testing.assert_array_equal(tokens2, tokens[None])


def test_frontend_shape():
    frontend = LogMelFrontend()
    frontend.window = mx.ones((400,), dtype=mx.float32)
    frontend.mel_matrix = mx.full((1025, 128), 1e-3, dtype=mx.float32)
    wav = mx.zeros((2, 160000), dtype=mx.float32)
    out = frontend(wav)
    assert out.shape == (2, NUM_FRAMES, NUM_MEL_BINS)
    assert np.isfinite(np.array(out)).all()


class _FakeVocab:
    def EncodeAsIds(self, text):  # noqa: N802 (sentencepiece API)
        assert text == text.lower()
        return list(range(2, 2 + len(text.split())))


def test_encode_text_padding():
    ids, paddings = encode_text(_FakeVocab(), "Three Word Prompt")
    assert ids.shape == (128,) and paddings.shape == (128,)
    assert ids[0] == 1  # SOS
    np.testing.assert_array_equal(ids[1:4], [2, 3, 4])
    assert (ids[4:] == 0).all()
    np.testing.assert_array_equal(paddings[:4], 0.0)
    np.testing.assert_array_equal(paddings[4:], 1.0)
