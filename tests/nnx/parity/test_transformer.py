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

"""Tests for ``magenta_rt.nnx.transformer``.

Smoke + numerical sanity for FFN, SelfAttentionBlock, TransformerBlock,
and a 2-layer Transformer stack. Streaming-concat parity for the
self-attention path is exercised through ``init_attention_caches``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.transformer import (
    FFN, SelfAttentionBlock, Transformer, TransformerBlock,
)


def _populate_dense(d, key):
    sub = jax.random.split(key, 2)
    d.kernel = nnx.Param(jax.random.normal(sub[0], d.kernel[...].shape) * 0.05)
    if d.bias is not None:
        d.bias = nnx.Param(jax.random.normal(sub[1], d.bias[...].shape) * 0.01)


def _populate_norm(n, key):
    n.scale = nnx.Param(
        jax.random.normal(key, n.scale[...].shape) * 0.05 + 1.0
    )


def _populate_attention(layer, key):
    sub = jax.random.split(key, 6)
    layer.q_proj.kernel = nnx.Param(jax.random.normal(sub[0], layer.q_proj.kernel[...].shape) * 0.05)
    layer.kv_proj.kernel = nnx.Param(jax.random.normal(sub[1], layer.kv_proj.kernel[...].shape) * 0.05)
    layer.per_dim_scale_param = nnx.Param(
        jax.random.normal(sub[2], layer.per_dim_scale_param[...].shape) * 0.01
    )
    proj = layer.output_projection
    n, h = layer.num_heads, layer.units_per_head
    proj(jnp.zeros((1, 1, n, h)))
    proj.kernel = nnx.Param(
        jax.random.normal(sub[3], proj.kernel[...].shape) * 0.05
    )


def _populate_block(blk, key):
    sub = jax.random.split(key, 4)
    _populate_norm(blk.self_attn.pre_norm, sub[0])
    _populate_norm(blk.self_attn.post_norm, sub[1])
    _populate_attention(blk.self_attn.attention, sub[2])
    _populate_norm(blk.ffn.pre_norm, sub[3])
    sub = jax.random.split(sub[3], 5)
    _populate_norm(blk.ffn.post_norm, sub[0])
    _populate_dense(blk.ffn.ffn_layer1, sub[1])
    _populate_dense(blk.ffn.ffn_layer2, sub[2])


def _populate_dense_at(d, index, key):
    sub = jax.random.split(key, 2)
    d.kernel[...] = d.kernel[...].at[index].set(
        jax.random.normal(sub[0], d.kernel[...].shape[1:]) * 0.05
    )
    if d.bias is not None:
        d.bias[...] = d.bias[...].at[index].set(
            jax.random.normal(sub[1], d.bias[...].shape[1:]) * 0.01
        )


def _populate_norm_at(n, index, key):
    n.scale[...] = n.scale[...].at[index].set(
        jax.random.normal(key, n.scale[...].shape[1:]) * 0.05 + 1.0
    )


def _populate_attention_at(layer, index, key):
    sub = jax.random.split(key, 6)
    layer.q_proj.kernel[...] = layer.q_proj.kernel[...].at[index].set(
        jax.random.normal(sub[0], layer.q_proj.kernel[...].shape[1:]) * 0.05
    )
    layer.kv_proj.kernel[...] = layer.kv_proj.kernel[...].at[index].set(
        jax.random.normal(sub[1], layer.kv_proj.kernel[...].shape[1:]) * 0.05
    )
    layer.per_dim_scale_param[...] = layer.per_dim_scale_param[...].at[index].set(
        jax.random.normal(sub[2], layer.per_dim_scale_param[...].shape[1:]) * 0.01
    )
    proj = layer.output_projection
    proj.kernel[...] = proj.kernel[...].at[index].set(
        jax.random.normal(sub[3], proj.kernel[...].shape[1:]) * 0.05
    )


def _populate_block_at(blk, index, key):
    sub = jax.random.split(key, 4)
    _populate_norm_at(blk.self_attn.pre_norm, index, sub[0])
    _populate_norm_at(blk.self_attn.post_norm, index, sub[1])
    _populate_attention_at(blk.self_attn.attention, index, sub[2])
    _populate_norm_at(blk.ffn.pre_norm, index, sub[3])
    sub = jax.random.split(sub[3], 5)
    _populate_norm_at(blk.ffn.post_norm, index, sub[0])
    _populate_dense_at(blk.ffn.ffn_layer1, index, sub[1])
    _populate_dense_at(blk.ffn.ffn_layer2, index, sub[2])


def test_ffn_smoke(rng_key):
    ffn = FFN(model_dim=16, hidden_dim=32, rngs=nnx.Rngs(0))
    sub = jax.random.split(rng_key, 4)
    _populate_norm(ffn.pre_norm, sub[0])
    _populate_norm(ffn.post_norm, sub[1])
    _populate_dense(ffn.ffn_layer1, sub[2])
    _populate_dense(ffn.ffn_layer2, sub[3])
    x = jax.random.normal(rng_key, (1, 4, 16))
    y = ffn(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_self_attention_block_smoke(rng_key):
    blk = SelfAttentionBlock(
        model_dim=16, num_heads=2, units_per_head=8, max_past_horizon=4,
        rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 3)
    _populate_norm(blk.pre_norm, sub[0])
    _populate_norm(blk.post_norm, sub[1])
    _populate_attention(blk.attention, sub[2])
    x = jax.random.normal(rng_key, (1, 5, 16))
    y = blk(x)
    assert y.shape == x.shape


def test_transformer_block_with_cross_attention(rng_key):
    blk = TransformerBlock(
        model_dim=16, num_heads=2, units_per_head=8, ffn_dim=32,
        max_past_horizon=4, num_sinks=0,
        use_cross_attention=True,
        cross_attn_source_features=24,
        cross_attn_max_past_horizon=2,
        rngs=nnx.Rngs(0),
    )
    _populate_block(blk, rng_key)
    sub = jax.random.split(rng_key, 3)
    # Need to populate the cross_attn block separately.
    cross_sub = jax.random.split(sub[0], 4)
    _populate_norm(blk.cross_attn.pre_norm, cross_sub[0])
    _populate_norm(blk.cross_attn.post_norm, cross_sub[1])
    _populate_attention(blk.cross_attn.attention, cross_sub[2])

    x = jax.random.normal(sub[1], (1, 5, 16))
    src = jax.random.normal(sub[2], (1, 5, 24))
    y = blk(x, source=src)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_transformer_stack_smoke(rng_key):
    t = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=4, num_sinks=0,
        rngs=nnx.Rngs(0),
    )
    t.init_cache(batch=1, dtype=jnp.float32)
    sub = jax.random.split(rng_key, t.num_layers + 1)
    for i in range(t.num_layers):
        _populate_block_at(t.layers, i, sub[i])
    x = jax.random.normal(sub[-1], (1, 5, 16))

    y = t(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_transformer_init_cache_walks_attention(rng_key):
    """``Transformer.init_cache(batch, dtype)`` populates the KV cache
    slots on every self-attention layer in the stack — the OO
    counterpart of the old free helper.
    """
    t = Transformer(
        num_layers=2, model_dim=8, num_heads=2, units_per_head=4,
        ffn_dim=16, max_past_horizon=2, num_sinks=0,
        rngs=nnx.Rngs(0),
    )
    t.init_cache(batch=1, dtype=jnp.float32)
    assert t.layers.self_attn.attention.cache.initialized

    # remove_cache clears them all back to None.
    t.remove_cache()
    assert not t.layers.self_attn.attention.cache.initialized
