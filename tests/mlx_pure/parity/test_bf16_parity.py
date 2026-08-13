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

"""bf16 parity tests for the core layers.

Production uses ``compute_dtype=bfloat16`` while keeping ``param_dtype=float32``.
This file verifies that ``mlx_pure`` matches ``sequence_layers.mlx``
within a sensible bf16 tolerance for that mixed-precision configuration.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import sequence_layers.mlx as sl

from magenta_rt.mlx_pure import attention as pure_attn
from magenta_rt.mlx_pure import layers
from .conftest import assert_close, tol


pytestmark = pytest.mark.bf16


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


# -----------------------------------------------------------------------------
# Dense
# -----------------------------------------------------------------------------


def test_dense_bf16_compute_parity(rng_key):
    in_features, out_features = 32, 48
    sl_layer = sl.Dense.Config(
        features=out_features, use_bias=True,
        param_dtype=mx.float32, compute_dtype=mx.bfloat16,
    ).make(backend="mlx")
    x = mx.random.normal((2, 5, in_features), dtype=mx.float32, key=rng_key) * 0.1
    sample = _seq(x)
    _ = sl_layer.layer(sample)

    pure = layers.Dense(
        in_features, out_features, bias=True,
        compute_dtype=mx.bfloat16, param_dtype=mx.float32,
    )
    pure.linear.weight = sl_layer.inner._linear.weight
    pure.linear.bias = sl_layer.inner._linear.bias

    sl_y = sl_layer.layer(sample).values  # bf16
    pure_y = pure(x)
    a, r = tol(mx.bfloat16, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="dense_bf16")


# -----------------------------------------------------------------------------
# Norms (param fp32, compute bf16) — the production mixed-precision config.
#
# sl's RMSNormalization builtin path runs nn.RMSNorm directly on the input
# dtype (no fp32 upcast), while LayerNormalization upcasts to fp32 first
# when reductions_in_at_least_fp32=True (the default, used by encoder_ln
# and final_ln). mlx_pure mirrors this: bare nn.RMSNorm, but a custom
# layers.LayerNorm subclass that upcasts. These two tests pin both.
# -----------------------------------------------------------------------------


def test_rmsnorm_bf16_parity(rng_key):
    """mlx_pure ``layers.RMSNorm`` vs sl ``RMSNormalization`` on bf16
    input with fp32 scale. sl's builtin path does no fp32 upcast but
    *does* force the result back to the input dtype
    (``result.astype(x.dtype)``). A bare ``nn.RMSNorm`` with an fp32
    scale would instead promote the output to fp32 — diverging from sl
    and silently turning the residual stream fp32. The mlx_pure subclass
    restores the input dtype; this pins that."""
    from magenta_rt.mlx_pure.layers import RMSNorm

    dim = 48
    sl_norm = sl.RMSNormalization.Config(
        axis=-1, epsilon=1e-6, use_scale=True, param_dtype=mx.float32,
    ).make(backend="mlx")
    x = mx.random.normal((2, 5, dim), dtype=mx.bfloat16, key=rng_key) * 0.5
    _ = sl_norm.layer(_seq(x))  # materialize the lazy nn.RMSNorm
    sl_norm._rms_norm.weight = (
        mx.random.normal((dim,), dtype=mx.float32, key=mx.random.split(rng_key)[0]) * 0.1 + 1
    )

    pure = RMSNorm(dim, eps=1e-6)
    pure.weight = sl_norm._rms_norm.weight

    sl_y = sl_norm.layer(_seq(x)).values
    pure_y = pure(x)
    assert pure_y.dtype == mx.bfloat16, (
        f"RMSNorm must return input dtype (bf16), got {pure_y.dtype}"
    )
    a, r = tol(mx.float32, "leaf")  # ops are identical -> effectively bit-exact
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="rmsnorm_bf16")


def test_layernorm_bf16_parity(rng_key):
    """mlx_pure ``layers.LayerNorm`` vs sl ``LayerNormalization`` on bf16
    input with fp32 scale/bias. sl upcasts to fp32 before ``nn.LayerNorm``
    (``reductions_in_at_least_fp32=True``); the mlx_pure subclass does the
    same. Regression guard for the fp32-upcast fix — a bare ``nn.LayerNorm``
    here would lose bf16 precision and drift from sl."""
    from magenta_rt.mlx_pure.layers import LayerNorm

    dim = 48
    sl_norm = sl.LayerNormalization.Config(
        axis=-1, epsilon=1e-6, use_bias=True, use_scale=True,
        reductions_in_at_least_fp32=True, param_dtype=mx.float32,
    ).make(backend="mlx")
    x = mx.random.normal((2, 5, dim), dtype=mx.bfloat16, key=rng_key) * 0.5
    _ = sl_norm.layer(_seq(x))  # materialize the lazy nn.LayerNorm
    sub = mx.random.split(rng_key, 2)
    sl_norm._layer_norm.weight = mx.random.normal((dim,), dtype=mx.float32, key=sub[0]) * 0.1 + 1
    sl_norm._layer_norm.bias = mx.random.normal((dim,), dtype=mx.float32, key=sub[1]) * 0.05

    pure = LayerNorm(dim, eps=1e-6, affine=True, bias=True)
    pure.weight = sl_norm._layer_norm.weight
    pure.bias = sl_norm._layer_norm.bias

    sl_y = sl_norm.layer(_seq(x)).values
    pure_y = pure(x)
    assert pure_y.dtype == mx.bfloat16, (
        f"LayerNorm must return input dtype (bf16), got {pure_y.dtype}"
    )
    a, r = tol(mx.float32, "leaf")  # ops are identical -> effectively bit-exact
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="layernorm_bf16")


# -----------------------------------------------------------------------------
# EinsumDense mixed-precision (attention output-projection layout)
# -----------------------------------------------------------------------------


def test_einsumdense_mixed_precision_parity(rng_key):
    """Regression test for the EinsumDense mixed-precision bug:
    pure was casting ``kernel`` and ``bias`` to ``compute_dtype``
    (bf16), while sl leaves them in ``param_dtype`` (fp32) and lets
    MLX promote during the einsum reduction. The pure version's
    extra cast lost precision and broke real-checkpoint parity for
    the attention output projection.

    This test replays sl's behaviour: input as bf16, kernel/bias as
    fp32. Pure must match.
    """
    n, h, d = 4, 16, 32
    pure = layers.EinsumDense(
        equation="...nh,dnh->...d", output_shape=(d,), bias_axes="d",
        compute_dtype=mx.bfloat16, param_dtype=mx.float32,
    )
    x_bf16 = mx.random.normal((2, 5, n, h), dtype=mx.bfloat16, key=rng_key) * 0.1
    _ = pure(x_bf16)  # materialize
    sub = mx.random.split(rng_key, 2)
    pure.kernel = mx.random.normal(pure.kernel.shape, dtype=mx.float32, key=sub[0]) * 0.05
    pure.bias = mx.random.normal(pure.bias.shape, dtype=mx.float32, key=sub[1]) * 0.01

    # Reference: replicate sl's exact code path.
    ref = (
        mx.einsum("...nh,dnh->...d", x_bf16.astype(mx.bfloat16), pure.kernel)
        + pure.bias
    )
    out = pure(x_bf16)
    np.testing.assert_array_equal(np.array(out), np.array(ref))


def test_local_self_attn_bf16_parity(rng_key):
    """Mixed-precision (param fp32, compute bf16) self-attention parity."""
    in_features = 32
    num_heads, units_per_head = 4, 8
    max_past_horizon = 5
    num_sinks = 1

    cfg = sl.LocalDotProductSelfAttention.Config(
        num_heads=num_heads, units_per_head=units_per_head,
        block_size=max_past_horizon,
        max_past_horizon=max_past_horizon, max_future_horizon=0,
        use_bias=False, per_dim_scale=True, attention_logits_soft_cap=None,
        param_dtype=mx.float32, compute_dtype=mx.bfloat16,
        num_sink_embeddings=num_sinks, use_sink_scalars=False,
        use_kv_cache_ringbuffer=False,
        input_projection=sl.SeparateQueryKeyValueProjection(),
    )
    sl_layer = cfg.make(backend="mlx")

    B, T = 2, 9
    x = mx.random.normal((B, T, in_features), dtype=mx.bfloat16, key=rng_key) * 0.1
    sample = _seq(x)
    _ = sl_layer.layer(sample)

    sub = mx.random.split(rng_key, 6)
    inner = sl_layer.inner.inner if hasattr(sl_layer.inner, "inner") else sl_layer.inner
    inner.q_proj = mx.random.normal(inner.q_proj.shape, dtype=mx.float32, key=sub[0]) * 0.05
    inner.kv_proj = mx.random.normal(inner.kv_proj.shape, dtype=mx.float32, key=sub[1]) * 0.05
    inner._per_dim_scale = mx.random.normal(inner._per_dim_scale.shape, dtype=mx.float32, key=sub[2]) * 0.1
    inner.sink_key_embeddings = mx.random.normal(inner.sink_key_embeddings.shape, dtype=mx.float32, key=sub[3]) * 0.05
    inner.sink_value_embeddings = mx.random.normal(inner.sink_value_embeddings.shape, dtype=mx.float32, key=sub[4]) * 0.05

    pure = pure_attn.LocalSelfAttention(
        in_features=in_features, num_heads=num_heads, units_per_head=units_per_head,
        max_past_horizon=max_past_horizon, per_dim_scale=True,
        compute_dtype=mx.bfloat16, param_dtype=mx.float32,
        num_sink_embeddings=num_sinks, model_dimension=in_features,
    )
    pure.q_proj = inner.q_proj
    pure.kv_proj = inner.kv_proj
    pure.per_dim_scale = inner._per_dim_scale
    pure.sink_key_embeddings = inner.sink_key_embeddings
    pure.sink_value_embeddings = inner.sink_value_embeddings

    # Reference: just the inner attention, comparing the raw context.
    sl_attn_module = sl_layer.inner
    ref_ctx = sl_attn_module.layer(sample).values

    pure_q, pure_k, pure_v = pure._project_qkv(x)
    row = mx.arange(T)[:, None]
    col = mx.arange(T)[None, :]
    banded = (col <= row) & (col >= row - max_past_horizon)
    mask = banded.reshape(1, 1, T, T)
    pure_ctx = pure._attend(pure_q, pure_k, pure_v, mask)

    a, r = tol(mx.bfloat16, "block")
    assert_close(ref_ctx, pure_ctx, atol=a, rtol=r, name="local_self_attn_bf16")
