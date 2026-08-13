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

"""Parity tests for `mlx_pure.attention` vs `sequence_layers.mlx`."""

from __future__ import annotations

import mlx.core as mx
import pytest
import sequence_layers.mlx as sl

from magenta_rt.mlx_pure import attention as pure_attn
from .conftest import assert_close, tol


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


def _make_sl_local_self_attn(
    *,
    in_features: int,
    num_heads: int,
    units_per_head: int,
    max_past_horizon: int,
    num_sinks: int,
    dtype,
):
    block_size = max(1, max_past_horizon)
    cfg = sl.LocalDotProductSelfAttention.Config(
        num_heads=num_heads,
        units_per_head=units_per_head,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=0,
        use_bias=False,
        per_dim_scale=True,
        attention_logits_soft_cap=None,
        compute_dtype=dtype,
        param_dtype=dtype,
        num_sink_embeddings=num_sinks,
        use_sink_scalars=False,
        use_kv_cache_ringbuffer=False,
        input_projection=sl.SeparateQueryKeyValueProjection(),
    )
    return cfg.make(backend="mlx")


def _materialize(sl_layer, x):
    sample = _seq(x)
    _ = sl_layer.layer(sample)


def _bridge_self_attn(sl_layer, pure: pure_attn.LocalSelfAttention) -> None:
    """Copy weights sl → pure for self-attention."""
    inner = sl_layer.inner.inner if hasattr(sl_layer.inner, "inner") else sl_layer.inner
    pure.q_proj = inner.q_proj
    pure.kv_proj = inner.kv_proj
    pure.per_dim_scale = inner._per_dim_scale
    if pure.num_sink_embeddings > 0:
        pure.sink_key_embeddings = inner.sink_key_embeddings
        pure.sink_value_embeddings = inner.sink_value_embeddings


# -----------------------------------------------------------------------------
# Local self-attention — full sequence, no sinks
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_sinks", [0, 1])
def test_local_self_attn_full_seq_parity(num_sinks, rng_key):
    dtype = mx.float32
    B, T, in_features = 2, 9, 32
    num_heads, units_per_head = 4, 8
    max_past_horizon = 5

    sl_layer = _make_sl_local_self_attn(
        in_features=in_features,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        dtype=dtype,
    )
    x = mx.random.normal((B, T, in_features), dtype=dtype, key=rng_key) * 0.1
    _materialize(sl_layer, x)

    # Randomize sl weights to avoid trivial zero-equivalence.
    sub = mx.random.split(rng_key, 4)
    inner = sl_layer.inner.inner if hasattr(sl_layer.inner, "inner") else sl_layer.inner
    inner.q_proj = mx.random.normal(inner.q_proj.shape, dtype=dtype, key=sub[0]) * 0.05
    inner.kv_proj = mx.random.normal(inner.kv_proj.shape, dtype=dtype, key=sub[1]) * 0.05
    if num_sinks > 0:
        inner.sink_key_embeddings = (
            mx.random.normal(inner.sink_key_embeddings.shape, dtype=dtype, key=sub[2]) * 0.05
        )
        inner.sink_value_embeddings = (
            mx.random.normal(inner.sink_value_embeddings.shape, dtype=dtype, key=sub[3]) * 0.05
        )
    inner._per_dim_scale = (
        mx.random.normal(inner._per_dim_scale.shape, dtype=dtype, key=rng_key) * 0.1
    )

    pure = pure_attn.LocalSelfAttention(
        in_features=in_features,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        per_dim_scale=True,
        compute_dtype=dtype,
        param_dtype=dtype,
        num_sink_embeddings=num_sinks,
        model_dimension=in_features,
    )

    # Reference: run sl's inner attention module directly. The full Residual
    # also applies pre/post-norm + output projection, which this test doesn't
    # compare — it checks the raw attention context against pure._attend.
    sample = _seq(x)
    sl_attn_module = sl_layer.inner  # DeferredLocalDotProductSelfAttention
    ref_ctx = sl_attn_module.layer(sample).values  # [B, T, num_heads, units_per_head]

    _bridge_self_attn(sl_layer, pure)
    pure_q, pure_k, pure_v = pure._project_qkv(x)
    # Build mask matching sl's full-sequence path.
    row = mx.arange(T)[:, None]
    col = mx.arange(T)[None, :]
    banded = (col <= row) & (col >= row - max_past_horizon)
    mask = banded.reshape(1, 1, T, T)
    pure_ctx = pure._attend(pure_q, pure_k, pure_v, mask)

    a, r = tol(dtype, "block")
    assert_close(ref_ctx, pure_ctx, atol=a, rtol=r, name=f"local_self_attn_sinks={num_sinks}")


# -----------------------------------------------------------------------------
# Local self-attention — streaming step parity (one-token decode)
# -----------------------------------------------------------------------------


def test_local_self_attn_streaming_step_parity(rng_key):
    dtype = mx.float32
    B, in_features = 2, 32
    num_heads, units_per_head = 4, 8
    max_past_horizon = 5
    num_sinks = 1

    sl_layer = _make_sl_local_self_attn(
        in_features=in_features,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        dtype=dtype,
    )
    # Materialize.
    x_warm = mx.random.normal((B, 1, in_features), dtype=dtype, key=rng_key) * 0.1
    _materialize(sl_layer, x_warm)

    sub = mx.random.split(rng_key, 5)
    inner = sl_layer.inner.inner if hasattr(sl_layer.inner, "inner") else sl_layer.inner
    inner.q_proj = mx.random.normal(inner.q_proj.shape, dtype=dtype, key=sub[0]) * 0.05
    inner.kv_proj = mx.random.normal(inner.kv_proj.shape, dtype=dtype, key=sub[1]) * 0.05
    inner.sink_key_embeddings = (
        mx.random.normal(inner.sink_key_embeddings.shape, dtype=dtype, key=sub[2]) * 0.05
    )
    inner.sink_value_embeddings = (
        mx.random.normal(inner.sink_value_embeddings.shape, dtype=dtype, key=sub[3]) * 0.05
    )
    inner._per_dim_scale = (
        mx.random.normal(inner._per_dim_scale.shape, dtype=dtype, key=sub[4]) * 0.1
    )

    pure = pure_attn.LocalSelfAttention(
        in_features=in_features,
        num_heads=num_heads,
        units_per_head=units_per_head,
        max_past_horizon=max_past_horizon,
        per_dim_scale=True,
        compute_dtype=dtype,
        param_dtype=dtype,
        num_sink_embeddings=num_sinks,
        model_dimension=in_features,
    )
    _bridge_self_attn(sl_layer, pure)

    # Drive both sl and pure step-by-step on a sequence of single-token frames.
    T = 8
    x = mx.random.normal((B, T, in_features), dtype=dtype, key=mx.random.split(sub[0])[0]) * 0.1

    # sl side: get_initial_state then step_with_emits one frame at a time.
    sl_attn_module = sl_layer.inner  # DeferredLocalDotProductSelfAttention
    spec = sl.ChannelSpec(shape=(in_features,), dtype=dtype)
    sl_state = sl_attn_module.get_initial_state(B, spec, constants=None)

    # pure side: LocalKVCache, prime once.
    from magenta_rt.mlx_pure.cache import LocalKVCache

    cache = LocalKVCache(window_size=max_past_horizon + 1, num_sinks=num_sinks)
    cache.prime_sinks(pure.sink_key_embeddings, pure.sink_value_embeddings)

    # Step-by-step.
    for t in range(T):
        xt = x[:, t : t + 1, :]
        # sl
        sl_y_seq, sl_state, _ = sl_attn_module.step_with_emits(_seq(xt), sl_state)
        sl_ctx = sl_y_seq.values  # [B, 1, num_heads, units_per_head]
        # pure
        q, k, v = pure._project_qkv(xt)
        full_k, full_v = cache.update_and_fetch(k, v)
        # Drop sink slots (they get re-prepended inside _attend)
        full_k_no_sink = full_k[:, :, num_sinks:, :]
        full_v_no_sink = full_v[:, :, num_sinks:, :]
        # Cache returns the full fixed-shape [num_sinks + max_past] slice;
        # mask hides unfilled slots. Drop sink columns so _attend's
        # own sink-prepend doesn't double-count them.
        cache_mask = cache.make_mask(1)[:, num_sinks:]  # [1, max_past]
        mask = mx.broadcast_to(cache_mask[None, None, :, :], (B, 1, 1, cache_mask.shape[1]))
        pure_ctx = pure._attend(q, full_k_no_sink, full_v_no_sink, mask)

        a, r = tol(dtype, "block")
        assert_close(sl_ctx, pure_ctx, atol=a, rtol=r, name=f"streaming_step[t={t}]")
