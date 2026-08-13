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

"""Parity tests for MultiChannelEmbedding."""

from __future__ import annotations

import mlx.core as mx
import sequence_layers.mlx as sl

from magenta_rt.mlx import transformer as mrt_t
from magenta_rt.mlx_pure.transformer import MultiChannelEmbedding
from .conftest import assert_close, tol


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


def test_multichannel_embedding_parity_no_reduction(rng_key):
    dim, num_channels = 16, 4
    num_per = [10, 10, 10, 10]
    sl_layer = mrt_t.MultiChannelEmbedding.Config(
        num_embeddings_per_channel=num_per,
        dimension=dim,
        num_channels=num_channels,
        num_reserved_embeddings=0,
        reduction_fn=None,
        param_dtype=mx.float32,
        compute_dtype=mx.float32,
    ).make()
    ids = mx.random.randint(0, 10, (2, 5, num_channels), key=rng_key)
    sample = sl.Sequence(ids, mx.ones(ids.shape[:2], dtype=mx.bool_))
    _ = sl_layer.layer(sample)

    sub = mx.random.split(rng_key)[0]
    sl_layer.embedding = mx.random.normal(sl_layer.embedding.shape, dtype=mx.float32, key=sub) * 0.05

    pure = MultiChannelEmbedding(
        dimension=dim,
        num_embeddings_per_channel=num_per,
        num_channels=num_channels,
        num_reserved_embeddings=0,
        reduction_fn=None,
        compute_dtype=mx.float32,
        param_dtype=mx.float32,
    )
    pure.embedding = sl_layer.embedding

    sl_y = sl_layer.layer(sample).values
    pure_y = pure(ids)
    a, r = tol(mx.float32, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="multichannel_emb_no_reduction")


def test_multichannel_embedding_parity_mean_reduction(rng_key):
    dim, num_channels = 16, 6
    num_per = [12, 12, 12, 12, 12, 12]

    def reduction_fn(x, axis):
        return mx.mean(x.astype(mx.float32), axis=axis).astype(x.dtype)

    sl_layer = mrt_t.MultiChannelEmbedding.Config(
        num_embeddings_per_channel=num_per,
        dimension=dim,
        num_channels=num_channels,
        num_reserved_embeddings=6,
        reduction_fn=reduction_fn,
        param_dtype=mx.float32,
        compute_dtype=mx.float32,
    ).make()
    ids = mx.random.randint(0, 12, (2, 7, num_channels), key=rng_key)
    sample = sl.Sequence(ids, mx.ones(ids.shape[:2], dtype=mx.bool_))
    _ = sl_layer.layer(sample)
    sl_layer.embedding = mx.random.normal(sl_layer.embedding.shape, dtype=mx.float32, key=mx.random.split(rng_key)[0]) * 0.05

    pure = MultiChannelEmbedding(
        dimension=dim,
        num_embeddings_per_channel=num_per,
        num_channels=num_channels,
        num_reserved_embeddings=6,
        reduction_fn=reduction_fn,
        compute_dtype=mx.float32,
        param_dtype=mx.float32,
    )
    pure.embedding = sl_layer.embedding

    sl_y = sl_layer.layer(sample).values
    pure_y = pure(ids)
    a, r = tol(mx.float32, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="multichannel_emb_mean_reduction")
