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

"""Tests for embedding-space style augmentation (StyleEmbeddingJitter)."""

from __future__ import annotations

import numpy as np
import pytest
from audiotree import AudioTree

from magenta_rt.config import MUSICCOCA as _MUSICCOCA
from magenta_rt.sft.transforms import StyleEmbeddingJitter, rvq_tokenize

DIM, DEPTH, SIZE, FRAMES = 32, 4, 16, 50


@pytest.fixture(scope="module")
def codebooks():
    return np.random.RandomState(0).randn(DEPTH, SIZE, DIM).astype(np.float32)


def _example(codebooks, seed=1):
    rng = np.random.RandomState(seed)
    embedding = rng.randn(DIM).astype(np.float32)
    tokens = rvq_tokenize(embedding, codebooks)
    return AudioTree(
        waveform=None,
        sample_rate=48_000,
        codes=rng.randint(0, 16, (1, FRAMES, 4)).astype(np.int32),
        extras={
            "musiccoca_embedding": embedding[None],
            _MUSICCOCA.key: np.tile(tokens, (1, FRAMES, 1)),
        },
    )


def test_rvq_tokenize_matches_nnx_quantizer(codebooks):
    """The numpy RVQ must agree with the jax EmbeddingQuantizer algorithm."""
    jnp = pytest.importorskip("jax.numpy")
    from magenta_rt.nnx.musiccoca.quantizer import EmbeddingQuantizer

    # The nnx module hardcodes the production shape; emulate its math here
    # via the same greedy loop on a production-shaped random codebook.
    rng = np.random.RandomState(3)
    big = rng.randn(12, 1024, 768).astype(np.float32)
    quantizer = EmbeddingQuantizer()
    quantizer.codebooks.value = jnp.asarray(big)
    emb = rng.randn(768).astype(np.float32)
    np.testing.assert_array_equal(
        rvq_tokenize(emb, big), np.asarray(quantizer.tokenize(jnp.asarray(emb[None])))[0]
    )


def test_zero_std_is_identity(codebooks):
    audio = _example(codebooks)
    out = StyleEmbeddingJitter(0.0, codebooks=codebooks).random_map(
        audio, np.random.default_rng(0)
    )
    assert out is audio


def test_jitter_rewrites_tokens_consistently(codebooks):
    audio = _example(codebooks)
    out = StyleEmbeddingJitter(2.0, codebooks=codebooks).random_map(
        audio, np.random.default_rng(0)
    )
    emb = out.extras["musiccoca_embedding"]
    tokens = out.extras[_MUSICCOCA.key]
    assert emb.shape == (1, DIM)
    assert tokens.shape == (1, FRAMES, DEPTH)
    # The stored embedding and tokens stay mutually consistent.
    np.testing.assert_array_equal(tokens[0, 0], rvq_tokenize(emb[0], codebooks))
    np.testing.assert_array_equal(tokens, np.tile(tokens[:, :1], (1, FRAMES, 1)))
    # Large jitter on a tiny codebook should move at least one level.
    assert not np.array_equal(tokens, audio.extras[_MUSICCOCA.key])


def test_tiny_jitter_keeps_tokens(codebooks):
    """RVQ cells are coarse: negligible noise re-quantizes to the same row."""
    audio = _example(codebooks)
    out = StyleEmbeddingJitter(1e-6, codebooks=codebooks).random_map(
        audio, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(
        out.extras[_MUSICCOCA.key], audio.extras[_MUSICCOCA.key]
    )


def test_noop_without_embedding(codebooks):
    audio = _example(codebooks)
    audio = audio.replace(
        extras={_MUSICCOCA.key: audio.extras[_MUSICCOCA.key]}
    )
    out = StyleEmbeddingJitter(2.0, codebooks=codebooks).random_map(
        audio, np.random.default_rng(0)
    )
    assert out is audio


def test_prob_zero_is_noop(codebooks):
    audio = _example(codebooks)
    out = StyleEmbeddingJitter(2.0, prob=0.0, codebooks=codebooks).random_map(
        audio, np.random.default_rng(0)
    )
    assert out is audio
