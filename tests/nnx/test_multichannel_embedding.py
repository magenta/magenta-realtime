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

"""Tests for ``magenta_rt.nnx.transformer.MultiChannelEmbedding``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.transformer import MultiChannelEmbedding


def test_multichannel_embedding_shape_no_reduction(rng_key):
    embed = MultiChannelEmbedding(
        dimension=8,
        num_embeddings_per_channel=[16, 16, 16],
        num_channels=3,
        num_reserved_embeddings=4,
        round_num_embeddings_to_multiple_of_128=True,
        rngs=nnx.Rngs(0),
    )
    embed.embedding = nnx.Param(
        jax.random.normal(rng_key, embed.embedding[...].shape) * 0.02
    )
    ids = jnp.array([[[0, 1, 2], [3, 5, 7]]], dtype=jnp.int32)
    y = embed(ids)
    # No reduction: [B, T, num_channels, dim]
    assert y.shape == (1, 2, 3, 8)


def test_multichannel_embedding_with_mean_reduction(rng_key):
    embed = MultiChannelEmbedding(
        dimension=8,
        num_embeddings_per_channel=[16, 16],
        num_channels=2,
        num_reserved_embeddings=4,
        reduction_fn=jnp.mean,
        round_num_embeddings_to_multiple_of_128=True,
        rngs=nnx.Rngs(0),
    )
    embed.embedding = nnx.Param(
        jax.random.normal(rng_key, embed.embedding[...].shape) * 0.02
    )
    ids = jnp.array([[[0, 5], [3, 7]]], dtype=jnp.int32)
    y = embed(ids)
    # With reduction: [B, T, dim].
    assert y.shape == (1, 2, 8)


def test_multichannel_embedding_reserved_token_offset_skip(rng_key):
    """Reserved tokens (id < num_reserved_embeddings) bypass the
    per-channel offset (sl convention)."""
    embed = MultiChannelEmbedding(
        dimension=4,
        num_embeddings_per_channel=[8, 8],
        num_channels=2,
        num_reserved_embeddings=4,
        reduction_fn=None,
        round_num_embeddings_to_multiple_of_128=False,
        rngs=nnx.Rngs(0),
    )
    # Embeddings indexed by absolute position in the table.
    table = jnp.arange(20).astype(jnp.float32).reshape(20, 1)
    table = jnp.broadcast_to(table, (20, 4))
    embed.embedding = nnx.Param(table)
    # ids: a reserved token (1) and a real token (5) on each channel.
    ids = jnp.array([[[1, 5]]], dtype=jnp.int32)
    y = embed(ids)  # [1, 1, 2, 4]
    # Channel 0: id=1 < num_reserved(4) → offset 0 → row 1.
    # Channel 1: id=5 ≥ 4 → offset cumsum[1]=8 → row 5+8=13.
    np.testing.assert_array_equal(np.asarray(y[0, 0, 0]), np.full((4,), 1.0))
    np.testing.assert_array_equal(np.asarray(y[0, 0, 1]), np.full((4,), 13.0))


def test_multichannel_embedding_rejects_wrong_channel_count():
    embed = MultiChannelEmbedding(
        dimension=4,
        num_embeddings_per_channel=[8, 8],
        num_channels=2,
        num_reserved_embeddings=0,
        rngs=nnx.Rngs(0),
    )
    with pytest.raises(ValueError):
        embed(jnp.zeros((1, 1, 3), dtype=jnp.int32))
