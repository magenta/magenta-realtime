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

"""Parity tests for `mlx_pure.transformer` vs the sl-backed
`magenta_rt.mlx.transformer.SLTransformer`."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
import sequence_layers.mlx as sl
from mlx.utils import tree_flatten

from magenta_rt.mlx_pure.transformer import Transformer
from .conftest import assert_close, tol


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


def _mk_sl_transformer(
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    units_per_head: int,
    ffn_dim: int,
    max_past_horizon: int,
    num_sinks: int,
    dtype,
    compute_dtype=None,
    use_cross_attention: bool = False,
    cross_max_past_horizon: int | None = None,
):
    # Build via the magenta_rt SLTransformer.Config used in production.
    # ``dtype`` is the param dtype; ``compute_dtype`` defaults to it but can
    # be set independently (e.g. param fp32 / compute bf16 — the production
    # mixed-precision config).
    from magenta_rt.mlx import transformer as mrt_transformer

    if compute_dtype is None:
        compute_dtype = dtype

    cfg = mrt_transformer.SLTransformer.Config(
        model_dimension=model_dim,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        units_per_head=units_per_head,
        dropout_rate=0.0,
        self_attention_dropout_rate=None,
        max_past_horizon=max_past_horizon,
        max_future_horizon=0,
        use_rope=False,
        rope_only_advance_position_for_valid_timesteps=True,
        rope_positions_in_at_least_fp32=None,
        reductions_in_at_least_fp32=None,
        attention_logits_soft_cap=None,
        ffn_activation=nn.gelu_approx,
        ffn_use_bias=True,
        ffn_gated=False,
        attention_use_bias=False,
        attention_per_dim_scale=True,
        attention_zero_fully_masked=False,
        broadcast_dropout_across_time=False,
        use_cross_attention=use_cross_attention,
        cross_attention_source_name="source" if use_cross_attention else None,
        norm_type="rms_normalization",
        norm_policy="primer_hybrid",
        use_local_attention=True,
        use_streaming_cross_attention=use_cross_attention,
        streaming_cross_attention_max_past_horizon=cross_max_past_horizon,
        streaming_cross_attention_max_future_horizon=0,
        streaming_cross_attention_use_query_delay_buffer=False,
        num_attention_sink_embeddings=num_sinks,
        use_attention_sink_scalars=False,
        self_attention_use_separate_qkv=True,
        cross_attention_use_separate_kv=True,
        self_attention_use_kv_cache_ringbuffer=False,
        streaming_cross_attention_use_kv_cache_ringbuffer=False,
        param_dtype=dtype,
        compute_dtype=compute_dtype,
        name="transformer",
    )
    return cfg.make()


def _mat_sl_transformer(slt, x, *, source=None):
    """Materialize sl Transformer's deferred init by running one full forward."""
    constants = None
    if source is not None:
        constants = {"source": _seq(source)}
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)
    return s


def _residual_body_layers(residual):
    """Return body layers of a sl.Residual."""
    return residual.body.layers


def _bridge_self_attn(sl_self_attn_residual, pure_block):
    body = _residual_body_layers(sl_self_attn_residual)
    pre_norm = body[0]
    attn = body[1]
    output_proj = body[2]
    post_norm = body[4]

    pure_block.pre_norm.weight = pre_norm._rms_norm.weight
    pure_block.post_norm.weight = post_norm._rms_norm.weight

    inner = attn.inner.inner if hasattr(attn.inner, "inner") else attn.inner
    pure_block.attention.q_proj = inner.q_proj
    pure_block.attention.kv_proj = inner.kv_proj
    pure_block.attention.per_dim_scale = inner._per_dim_scale
    if pure_block.attention.num_sink_embeddings > 0:
        pure_block.attention.sink_key_embeddings = inner.sink_key_embeddings
        pure_block.attention.sink_value_embeddings = inner.sink_value_embeddings
    pure_block.attention.output_projection.kernel = output_proj.kernel


def _bridge_ffn(sl_ffn_residual, pure_ffn):
    body = _residual_body_layers(sl_ffn_residual)
    pre_norm = body[0]
    layer1 = body[1]
    layer2 = body[3]
    post_norm = body[5]

    pure_ffn.pre_norm.weight = pre_norm._rms_norm.weight
    pure_ffn.post_norm.weight = post_norm._rms_norm.weight
    pure_ffn.ffn_layer1.linear.weight = layer1.inner._linear.weight
    pure_ffn.ffn_layer1.linear.bias = layer1.inner._linear.bias
    pure_ffn.ffn_layer2.linear.weight = layer2.inner._linear.weight
    pure_ffn.ffn_layer2.linear.bias = layer2.inner._linear.bias


def _walk_block_chain(sl_block_serial, *, has_cross: bool):
    children = sl_block_serial.layers
    self_attn = children[0]
    cross = children[1] if has_cross else None
    ffn = children[2]
    return self_attn, cross, ffn


# -----------------------------------------------------------------------------
# Single-block parity, no cross-attention (depth-decoder style)
# -----------------------------------------------------------------------------


def test_transformer_single_block_no_cross_parity(rng_key):
    dtype = mx.float32
    B, T, model_dim = 1, 6, 32
    num_heads, units_per_head = 4, 8
    ffn_dim = 64
    max_past_horizon = 5
    num_sinks = 0  # depth body uses 0 sinks
    num_layers = 1

    slt = _mk_sl_transformer(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        units_per_head=units_per_head,
        ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        dtype=dtype,
    )
    x = mx.random.normal((B, T, model_dim), dtype=dtype, key=rng_key) * 0.1
    _ = _mat_sl_transformer(slt, x)

    # Randomize sl weights.
    sub = mx.random.split(rng_key, 16)
    sl_block = slt.layers[0]
    self_attn_res, _, ffn_res = _walk_block_chain(sl_block, has_cross=False)

    # Self-attention internals.
    sa_body = _residual_body_layers(self_attn_res)
    pre_n = sa_body[0]
    attn = sa_body[1]
    out_p = sa_body[2]
    post_n = sa_body[4]
    pre_n._rms_norm.weight = mx.random.normal(pre_n._rms_norm.weight.shape, dtype=dtype, key=sub[0]) * 0.1 + 1
    post_n._rms_norm.weight = mx.random.normal(post_n._rms_norm.weight.shape, dtype=dtype, key=sub[1]) * 0.1 + 1
    inner = attn.inner.inner if hasattr(attn.inner, "inner") else attn.inner
    inner.q_proj = mx.random.normal(inner.q_proj.shape, dtype=dtype, key=sub[2]) * 0.05
    inner.kv_proj = mx.random.normal(inner.kv_proj.shape, dtype=dtype, key=sub[3]) * 0.05
    inner._per_dim_scale = mx.random.normal(inner._per_dim_scale.shape, dtype=dtype, key=sub[4]) * 0.1
    out_p.kernel = mx.random.normal(out_p.kernel.shape, dtype=dtype, key=sub[5]) * 0.05

    # FFN internals.
    ffn_body = _residual_body_layers(ffn_res)
    pre_n_ffn = ffn_body[0]
    layer1 = ffn_body[1]
    layer2 = ffn_body[3]
    post_n_ffn = ffn_body[5]
    pre_n_ffn._rms_norm.weight = mx.random.normal(pre_n_ffn._rms_norm.weight.shape, dtype=dtype, key=sub[6]) * 0.1 + 1
    post_n_ffn._rms_norm.weight = mx.random.normal(post_n_ffn._rms_norm.weight.shape, dtype=dtype, key=sub[7]) * 0.1 + 1
    layer1.inner._linear.weight = mx.random.normal(layer1.inner._linear.weight.shape, dtype=dtype, key=sub[8]) * 0.05
    layer1.inner._linear.bias = mx.random.normal(layer1.inner._linear.bias.shape, dtype=dtype, key=sub[9]) * 0.01
    layer2.inner._linear.weight = mx.random.normal(layer2.inner._linear.weight.shape, dtype=dtype, key=sub[10]) * 0.05
    layer2.inner._linear.bias = mx.random.normal(layer2.inner._linear.bias.shape, dtype=dtype, key=sub[11]) * 0.01

    # Build pure.
    pure = Transformer(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        units_per_head=units_per_head,
        ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        use_cross_attention=False,
        compute_dtype=dtype,
        param_dtype=dtype,
    )
    # Trigger lazy init of EinsumDense.
    _ = pure(x)

    pure_block = pure.layers[0]
    _bridge_self_attn(self_attn_res, pure_block.self_attn)
    _bridge_ffn(ffn_res, pure_block.ffn)

    # Forward both.
    sl_y = _mat_sl_transformer(slt, x).values
    pure_y = pure(x)

    a, r = tol(dtype, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="transformer_block_no_cross")


def _bridge_cross_attn(sl_cross_residual, pure_block):
    body = _residual_body_layers(sl_cross_residual)
    pre_norm = body[0]
    attn = body[1]   # DeferredStreamingDotProductAttention
    output_proj = body[2]  # EinsumDense
    post_norm = body[4]

    pure_block.pre_norm.weight = pre_norm._rms_norm.weight
    pure_block.post_norm.weight = post_norm._rms_norm.weight

    inner = attn.inner.inner if hasattr(attn.inner, "inner") else attn.inner
    pure_block.attention.q_proj = inner.q_proj
    pure_block.attention.kv_proj = inner.kv_proj
    pure_block.attention.per_dim_scale = inner._per_dim_scale
    if pure_block.attention.num_sink_embeddings > 0:
        pure_block.attention.sink_key_embeddings = inner.sink_key_embeddings
        pure_block.attention.sink_value_embeddings = inner.sink_value_embeddings
    pure_block.attention.output_projection.kernel = output_proj.kernel


# -----------------------------------------------------------------------------
# Single-block parity, with streaming cross-attention (temporal-decoder style)
# -----------------------------------------------------------------------------


def test_transformer_single_block_with_cross_parity(rng_key):
    dtype = mx.float32
    B, T, model_dim = 1, 6, 32
    num_heads, units_per_head = 4, 8
    ffn_dim = 64
    max_past_horizon = 5
    cross_max_past = 4
    num_sinks = 1
    num_layers = 1
    source_features = model_dim  # source dim matches model dim in our configs.

    slt = _mk_sl_transformer(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        units_per_head=units_per_head,
        ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        dtype=dtype,
        use_cross_attention=True,
        cross_max_past_horizon=cross_max_past,
    )
    x = mx.random.normal((B, T, model_dim), dtype=dtype, key=rng_key) * 0.1
    source = mx.random.normal((B, T, source_features), dtype=dtype, key=mx.random.split(rng_key)[0]) * 0.1
    constants = {"source": _seq(source)}

    # Materialize sl by running once.
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)

    sl_block = slt.layers[0]
    self_attn_res, cross_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)

    # Randomize all weights so non-trivial.
    sub = mx.random.split(rng_key, 32)
    # Self-attn
    sa_body = _residual_body_layers(self_attn_res)
    sa_body[0]._rms_norm.weight = mx.random.normal(sa_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[0]) * 0.1 + 1
    sa_body[4]._rms_norm.weight = mx.random.normal(sa_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[1]) * 0.1 + 1
    sa_inner = sa_body[1].inner.inner if hasattr(sa_body[1].inner, "inner") else sa_body[1].inner
    sa_inner.q_proj = mx.random.normal(sa_inner.q_proj.shape, dtype=dtype, key=sub[2]) * 0.05
    sa_inner.kv_proj = mx.random.normal(sa_inner.kv_proj.shape, dtype=dtype, key=sub[3]) * 0.05
    sa_inner._per_dim_scale = mx.random.normal(sa_inner._per_dim_scale.shape, dtype=dtype, key=sub[4]) * 0.1
    sa_inner.sink_key_embeddings = mx.random.normal(sa_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[5]) * 0.05
    sa_inner.sink_value_embeddings = mx.random.normal(sa_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[6]) * 0.05
    sa_body[2].kernel = mx.random.normal(sa_body[2].kernel.shape, dtype=dtype, key=sub[7]) * 0.05
    # Cross-attn
    ca_body = _residual_body_layers(cross_res)
    ca_body[0]._rms_norm.weight = mx.random.normal(ca_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[8]) * 0.1 + 1
    ca_body[4]._rms_norm.weight = mx.random.normal(ca_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[9]) * 0.1 + 1
    ca_inner = ca_body[1].inner.inner if hasattr(ca_body[1].inner, "inner") else ca_body[1].inner
    ca_inner.q_proj = mx.random.normal(ca_inner.q_proj.shape, dtype=dtype, key=sub[10]) * 0.05
    ca_inner.kv_proj = mx.random.normal(ca_inner.kv_proj.shape, dtype=dtype, key=sub[11]) * 0.05
    ca_inner._per_dim_scale = mx.random.normal(ca_inner._per_dim_scale.shape, dtype=dtype, key=sub[12]) * 0.1
    if num_sinks > 0:
        ca_inner.sink_key_embeddings = mx.random.normal(ca_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[20]) * 0.05
        ca_inner.sink_value_embeddings = mx.random.normal(ca_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[21]) * 0.05
    ca_body[2].kernel = mx.random.normal(ca_body[2].kernel.shape, dtype=dtype, key=sub[13]) * 0.05
    # FFN
    ffn_body = _residual_body_layers(ffn_res)
    ffn_body[0]._rms_norm.weight = mx.random.normal(ffn_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[14]) * 0.1 + 1
    ffn_body[5]._rms_norm.weight = mx.random.normal(ffn_body[5]._rms_norm.weight.shape, dtype=dtype, key=sub[15]) * 0.1 + 1
    ffn_body[1].inner._linear.weight = mx.random.normal(ffn_body[1].inner._linear.weight.shape, dtype=dtype, key=sub[16]) * 0.05
    ffn_body[1].inner._linear.bias = mx.random.normal(ffn_body[1].inner._linear.bias.shape, dtype=dtype, key=sub[17]) * 0.01
    ffn_body[3].inner._linear.weight = mx.random.normal(ffn_body[3].inner._linear.weight.shape, dtype=dtype, key=sub[18]) * 0.05
    ffn_body[3].inner._linear.bias = mx.random.normal(ffn_body[3].inner._linear.bias.shape, dtype=dtype, key=sub[19]) * 0.01

    pure = Transformer(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        units_per_head=units_per_head,
        ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon,
        num_sinks=num_sinks,
        use_cross_attention=True,
        cross_attn_source_features=source_features,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=dtype,
        param_dtype=dtype,
    )
    _ = pure(x, source=source)  # lazy init of EinsumDense

    pure_block = pure.layers[0]
    _bridge_self_attn(self_attn_res, pure_block.self_attn)
    _bridge_cross_attn(cross_res, pure_block.cross_attn)
    _bridge_ffn(ffn_res, pure_block.ffn)

    # Reference forward via sl.
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)
    sl_y = s.values

    pure_y = pure(x, source=source)

    a, r = tol(dtype, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="transformer_block_with_cross")


def test_transformer_production_shape_parity(rng_key):
    """4-layer transformer at production-style dims (model_dim=256,
    num_heads=8, ffn_dim=1024) with cross-attention + 1 sink. Catches
    issues that only manifest at scale."""
    dtype = mx.float32
    B, T = 1, 8
    model_dim = 256
    num_heads, units_per_head = 8, 32
    ffn_dim = 1024
    max_past_horizon = 7
    cross_max_past = 6
    num_sinks = 1
    num_layers = 4

    slt = _mk_sl_transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks, dtype=dtype,
        use_cross_attention=True, cross_max_past_horizon=cross_max_past,
    )
    x = mx.random.normal((B, T, model_dim), dtype=dtype, key=rng_key) * 0.05
    source = mx.random.normal((B, T, model_dim), dtype=dtype, key=mx.random.split(rng_key)[0]) * 0.05
    constants = {"source": _seq(source)}

    # Materialize.
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)

    sub = mx.random.split(rng_key, 200)
    for li, sl_block in enumerate(slt.layers):
        sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)
        offs = li * 32
        sa_body = _residual_body_layers(sa_res)
        sa_inner = sa_body[1].inner.inner if hasattr(sa_body[1].inner, "inner") else sa_body[1].inner
        sa_body[0]._rms_norm.weight = mx.random.normal(sa_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+0]) * 0.05 + 1
        sa_body[4]._rms_norm.weight = mx.random.normal(sa_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+1]) * 0.05 + 1
        sa_inner.q_proj = mx.random.normal(sa_inner.q_proj.shape, dtype=dtype, key=sub[offs+2]) * 0.02
        sa_inner.kv_proj = mx.random.normal(sa_inner.kv_proj.shape, dtype=dtype, key=sub[offs+3]) * 0.02
        sa_inner._per_dim_scale = mx.random.normal(sa_inner._per_dim_scale.shape, dtype=dtype, key=sub[offs+4]) * 0.05
        sa_inner.sink_key_embeddings = mx.random.normal(sa_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[offs+5]) * 0.02
        sa_inner.sink_value_embeddings = mx.random.normal(sa_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[offs+6]) * 0.02
        sa_body[2].kernel = mx.random.normal(sa_body[2].kernel.shape, dtype=dtype, key=sub[offs+7]) * 0.02

        ca_body = _residual_body_layers(ca_res)
        ca_inner = ca_body[1].inner.inner if hasattr(ca_body[1].inner, "inner") else ca_body[1].inner
        ca_body[0]._rms_norm.weight = mx.random.normal(ca_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+8]) * 0.05 + 1
        ca_body[4]._rms_norm.weight = mx.random.normal(ca_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+9]) * 0.05 + 1
        ca_inner.q_proj = mx.random.normal(ca_inner.q_proj.shape, dtype=dtype, key=sub[offs+10]) * 0.02
        ca_inner.kv_proj = mx.random.normal(ca_inner.kv_proj.shape, dtype=dtype, key=sub[offs+11]) * 0.02
        ca_inner._per_dim_scale = mx.random.normal(ca_inner._per_dim_scale.shape, dtype=dtype, key=sub[offs+12]) * 0.05
        ca_inner.sink_key_embeddings = mx.random.normal(ca_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[offs+14]) * 0.02
        ca_inner.sink_value_embeddings = mx.random.normal(ca_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[offs+15]) * 0.02
        ca_body[2].kernel = mx.random.normal(ca_body[2].kernel.shape, dtype=dtype, key=sub[offs+13]) * 0.02

        ffn_body = _residual_body_layers(ffn_res)
        ffn_body[0]._rms_norm.weight = mx.random.normal(ffn_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+16]) * 0.05 + 1
        ffn_body[5]._rms_norm.weight = mx.random.normal(ffn_body[5]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+17]) * 0.05 + 1
        ffn_body[1].inner._linear.weight = mx.random.normal(ffn_body[1].inner._linear.weight.shape, dtype=dtype, key=sub[offs+18]) * 0.02
        ffn_body[1].inner._linear.bias = mx.random.normal(ffn_body[1].inner._linear.bias.shape, dtype=dtype, key=sub[offs+19]) * 0.005
        ffn_body[3].inner._linear.weight = mx.random.normal(ffn_body[3].inner._linear.weight.shape, dtype=dtype, key=sub[offs+20]) * 0.02
        ffn_body[3].inner._linear.bias = mx.random.normal(ffn_body[3].inner._linear.bias.shape, dtype=dtype, key=sub[offs+21]) * 0.005

    pure = Transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        use_cross_attention=True, cross_attn_source_features=model_dim,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=dtype, param_dtype=dtype,
    )
    _ = pure(x, source=source)

    for li, sl_block in enumerate(slt.layers):
        sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)
        _bridge_self_attn(sa_res, pure.layers[li].self_attn)
        _bridge_cross_attn(ca_res, pure.layers[li].cross_attn)
        _bridge_ffn(ffn_res, pure.layers[li].ffn)

    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)
    sl_y = s.values
    pure_y = pure(x, source=source)

    a, r = tol(dtype, "stack")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="transformer_production_shape")


def test_transformer_multi_layer_parity(rng_key):
    """Two layers stacked, with cross-attention. Tests block composition."""
    dtype = mx.float32
    B, T, model_dim = 1, 5, 32
    num_heads, units_per_head = 4, 8
    ffn_dim = 64
    max_past_horizon = 4
    cross_max_past = 3
    num_sinks = 1
    num_layers = 2

    slt = _mk_sl_transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks, dtype=dtype,
        use_cross_attention=True, cross_max_past_horizon=cross_max_past,
    )
    x = mx.random.normal((B, T, model_dim), dtype=dtype, key=rng_key) * 0.1
    source = mx.random.normal((B, T, model_dim), dtype=dtype, key=mx.random.split(rng_key)[0]) * 0.1
    constants = {"source": _seq(source)}

    # Materialize.
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)

    # Randomize all weights for both layers.
    sub = mx.random.split(rng_key, 64)
    for li, sl_block in enumerate(slt.layers):
        sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)
        offs = li * 32
        sa_body = _residual_body_layers(sa_res)
        sa_body[0]._rms_norm.weight = mx.random.normal(sa_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+0]) * 0.1 + 1
        sa_body[4]._rms_norm.weight = mx.random.normal(sa_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+1]) * 0.1 + 1
        sa_inner = sa_body[1].inner.inner if hasattr(sa_body[1].inner, "inner") else sa_body[1].inner
        sa_inner.q_proj = mx.random.normal(sa_inner.q_proj.shape, dtype=dtype, key=sub[offs+2]) * 0.05
        sa_inner.kv_proj = mx.random.normal(sa_inner.kv_proj.shape, dtype=dtype, key=sub[offs+3]) * 0.05
        sa_inner._per_dim_scale = mx.random.normal(sa_inner._per_dim_scale.shape, dtype=dtype, key=sub[offs+4]) * 0.1
        sa_inner.sink_key_embeddings = mx.random.normal(sa_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[offs+5]) * 0.05
        sa_inner.sink_value_embeddings = mx.random.normal(sa_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[offs+6]) * 0.05
        sa_body[2].kernel = mx.random.normal(sa_body[2].kernel.shape, dtype=dtype, key=sub[offs+7]) * 0.05

        ca_body = _residual_body_layers(ca_res)
        ca_body[0]._rms_norm.weight = mx.random.normal(ca_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+8]) * 0.1 + 1
        ca_body[4]._rms_norm.weight = mx.random.normal(ca_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+9]) * 0.1 + 1
        ca_inner = ca_body[1].inner.inner if hasattr(ca_body[1].inner, "inner") else ca_body[1].inner
        ca_inner.q_proj = mx.random.normal(ca_inner.q_proj.shape, dtype=dtype, key=sub[offs+10]) * 0.05
        ca_inner.kv_proj = mx.random.normal(ca_inner.kv_proj.shape, dtype=dtype, key=sub[offs+11]) * 0.05
        ca_inner._per_dim_scale = mx.random.normal(ca_inner._per_dim_scale.shape, dtype=dtype, key=sub[offs+12]) * 0.1
        ca_inner.sink_key_embeddings = mx.random.normal(ca_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[offs+14]) * 0.05
        ca_inner.sink_value_embeddings = mx.random.normal(ca_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[offs+15]) * 0.05
        ca_body[2].kernel = mx.random.normal(ca_body[2].kernel.shape, dtype=dtype, key=sub[offs+13]) * 0.05

        ffn_body = _residual_body_layers(ffn_res)
        ffn_body[0]._rms_norm.weight = mx.random.normal(ffn_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+16]) * 0.1 + 1
        ffn_body[5]._rms_norm.weight = mx.random.normal(ffn_body[5]._rms_norm.weight.shape, dtype=dtype, key=sub[offs+17]) * 0.1 + 1
        ffn_body[1].inner._linear.weight = mx.random.normal(ffn_body[1].inner._linear.weight.shape, dtype=dtype, key=sub[offs+18]) * 0.05
        ffn_body[1].inner._linear.bias = mx.random.normal(ffn_body[1].inner._linear.bias.shape, dtype=dtype, key=sub[offs+19]) * 0.01
        ffn_body[3].inner._linear.weight = mx.random.normal(ffn_body[3].inner._linear.weight.shape, dtype=dtype, key=sub[offs+20]) * 0.05
        ffn_body[3].inner._linear.bias = mx.random.normal(ffn_body[3].inner._linear.bias.shape, dtype=dtype, key=sub[offs+21]) * 0.01

    pure = Transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        use_cross_attention=True, cross_attn_source_features=model_dim,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=dtype, param_dtype=dtype,
    )
    _ = pure(x, source=source)

    for li, sl_block in enumerate(slt.layers):
        sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)
        _bridge_self_attn(sa_res, pure.layers[li].self_attn)
        _bridge_cross_attn(ca_res, pure.layers[li].cross_attn)
        _bridge_ffn(ffn_res, pure.layers[li].ffn)

    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)
    sl_y = s.values
    pure_y = pure(x, source=source)

    a, r = tol(dtype, "stack")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="transformer_2_layer_with_cross")


# -----------------------------------------------------------------------------
# Mixed-precision (param fp32 / compute bf16) parity — the production config.
# -----------------------------------------------------------------------------


def _randomize_sl_block(sl_block, sub, *, dtype, has_cross):
    """Randomize every weight of one sl transformer block in place.

    ``sub`` must supply >= 22 split keys. RMSNorm weights are centered at
    1.0 (so the block isn't a trivial identity); projections/kernels get
    small-scale noise. Mirrors the inline randomization in the fp32 tests
    above but is shared by the bf16 tests so the two stay in lock-step.
    """
    sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=has_cross)

    sa_body = _residual_body_layers(sa_res)
    sa_inner = sa_body[1].inner.inner if hasattr(sa_body[1].inner, "inner") else sa_body[1].inner
    sa_body[0]._rms_norm.weight = mx.random.normal(sa_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[0]) * 0.1 + 1
    sa_body[4]._rms_norm.weight = mx.random.normal(sa_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[1]) * 0.1 + 1
    sa_inner.q_proj = mx.random.normal(sa_inner.q_proj.shape, dtype=dtype, key=sub[2]) * 0.05
    sa_inner.kv_proj = mx.random.normal(sa_inner.kv_proj.shape, dtype=dtype, key=sub[3]) * 0.05
    sa_inner._per_dim_scale = mx.random.normal(sa_inner._per_dim_scale.shape, dtype=dtype, key=sub[4]) * 0.1
    if getattr(sa_inner, "sink_key_embeddings", None) is not None:
        sa_inner.sink_key_embeddings = mx.random.normal(sa_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[5]) * 0.05
        sa_inner.sink_value_embeddings = mx.random.normal(sa_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[6]) * 0.05
    sa_body[2].kernel = mx.random.normal(sa_body[2].kernel.shape, dtype=dtype, key=sub[7]) * 0.05

    if has_cross:
        ca_body = _residual_body_layers(ca_res)
        ca_inner = ca_body[1].inner.inner if hasattr(ca_body[1].inner, "inner") else ca_body[1].inner
        ca_body[0]._rms_norm.weight = mx.random.normal(ca_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[8]) * 0.1 + 1
        ca_body[4]._rms_norm.weight = mx.random.normal(ca_body[4]._rms_norm.weight.shape, dtype=dtype, key=sub[9]) * 0.1 + 1
        ca_inner.q_proj = mx.random.normal(ca_inner.q_proj.shape, dtype=dtype, key=sub[10]) * 0.05
        ca_inner.kv_proj = mx.random.normal(ca_inner.kv_proj.shape, dtype=dtype, key=sub[11]) * 0.05
        ca_inner._per_dim_scale = mx.random.normal(ca_inner._per_dim_scale.shape, dtype=dtype, key=sub[12]) * 0.1
        if getattr(ca_inner, "sink_key_embeddings", None) is not None:
            ca_inner.sink_key_embeddings = mx.random.normal(ca_inner.sink_key_embeddings.shape, dtype=dtype, key=sub[20]) * 0.05
            ca_inner.sink_value_embeddings = mx.random.normal(ca_inner.sink_value_embeddings.shape, dtype=dtype, key=sub[21]) * 0.05
        ca_body[2].kernel = mx.random.normal(ca_body[2].kernel.shape, dtype=dtype, key=sub[13]) * 0.05

    ffn_body = _residual_body_layers(ffn_res)
    ffn_body[0]._rms_norm.weight = mx.random.normal(ffn_body[0]._rms_norm.weight.shape, dtype=dtype, key=sub[14]) * 0.1 + 1
    ffn_body[5]._rms_norm.weight = mx.random.normal(ffn_body[5]._rms_norm.weight.shape, dtype=dtype, key=sub[15]) * 0.1 + 1
    ffn_body[1].inner._linear.weight = mx.random.normal(ffn_body[1].inner._linear.weight.shape, dtype=dtype, key=sub[16]) * 0.05
    ffn_body[1].inner._linear.bias = mx.random.normal(ffn_body[1].inner._linear.bias.shape, dtype=dtype, key=sub[17]) * 0.01
    ffn_body[3].inner._linear.weight = mx.random.normal(ffn_body[3].inner._linear.weight.shape, dtype=dtype, key=sub[18]) * 0.05
    ffn_body[3].inner._linear.bias = mx.random.normal(ffn_body[3].inner._linear.bias.shape, dtype=dtype, key=sub[19]) * 0.01


def _bridge_block(sl_block, pure_block, *, has_cross):
    """Bridge one randomized sl block into the matching pure block."""
    sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=has_cross)
    _bridge_self_attn(sa_res, pure_block.self_attn)
    if has_cross:
        _bridge_cross_attn(ca_res, pure_block.cross_attn)
    _bridge_ffn(ffn_res, pure_block.ffn)


def test_transformer_block_bf16_full_seq_parity(rng_key):
    """Single block, self + streaming-cross + FFN, in the production
    mixed-precision config (``param_dtype=fp32``, ``compute_dtype=bf16``),
    full-sequence. The fp32 tests above pin the algorithm; this pins that
    the bf16 *compute path* — every QKV/output projection, SDPA, FFN
    matmul, and RMSNorm running at bf16 with fp32 params — still tracks sl
    within bf16 tolerance.
    """
    param_dtype, compute_dtype = mx.float32, mx.bfloat16
    B, T, model_dim = 1, 6, 32
    num_heads, units_per_head = 4, 8
    ffn_dim = 64
    max_past_horizon = 5
    cross_max_past = 4
    num_sinks = 1
    num_layers = 1
    source_features = model_dim

    slt = _mk_sl_transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        dtype=param_dtype, compute_dtype=compute_dtype,
        use_cross_attention=True, cross_max_past_horizon=cross_max_past,
    )
    x = mx.random.normal((B, T, model_dim), dtype=compute_dtype, key=rng_key) * 0.1
    source = mx.random.normal((B, T, source_features), dtype=compute_dtype,
                              key=mx.random.split(rng_key)[0]) * 0.1
    constants = {"source": _seq(source)}

    # Materialize sl (deferred init runs on first forward).
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)

    # Randomize sl weights as fp32 (param_dtype), bridge into pure.
    sub = mx.random.split(rng_key, 32)
    _randomize_sl_block(slt.layers[0], sub, dtype=param_dtype, has_cross=True)

    pure = Transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        use_cross_attention=True, cross_attn_source_features=source_features,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=compute_dtype, param_dtype=param_dtype,
    )
    _ = pure(x, source=source)  # lazy init of EinsumDense
    _bridge_block(slt.layers[0], pure.layers[0], has_cross=True)

    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=constants)
    sl_y = s.values
    pure_y = pure(x, source=source)

    a, r = tol(mx.bfloat16, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="transformer_block_bf16_full_seq")


def test_transformer_block_bf16_streaming_parity(rng_key):
    """Single block driven one frame at a time in the production
    mixed-precision config. This is the path that actually runs in the
    plugin: streaming *self*-attention (sink-primed ``LocalKVCache``) and
    streaming *cross*-attention (rolling source KV cache) both stepping
    in bf16. The fp32 ``test_local_self_attn_streaming_step_parity`` pins
    self-attn streaming alone; nothing else covered streaming cross-attn
    in bf16 before this.
    """
    param_dtype, compute_dtype = mx.float32, mx.bfloat16
    B, model_dim = 1, 32
    num_heads, units_per_head = 4, 8
    ffn_dim = 64
    max_past_horizon = 5
    cross_max_past = 4
    num_sinks = 1
    num_layers = 1
    source_features = model_dim
    T = 8  # streaming steps

    slt = _mk_sl_transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        dtype=param_dtype, compute_dtype=compute_dtype,
        use_cross_attention=True, cross_max_past_horizon=cross_max_past,
    )
    x = mx.random.normal((B, T, model_dim), dtype=compute_dtype, key=rng_key) * 0.1
    source = mx.random.normal((B, T, source_features), dtype=compute_dtype,
                              key=mx.random.split(rng_key)[0]) * 0.1

    # Materialize sl via a full-sequence pass, then randomize + bridge.
    full_consts = {"source": _seq(source)}
    s = _seq(x)
    for layer in slt.layers:
        s = layer.layer(s, constants=full_consts)
    sub = mx.random.split(rng_key, 32)
    _randomize_sl_block(slt.layers[0], sub, dtype=param_dtype, has_cross=True)

    pure = Transformer(
        num_layers=num_layers, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past_horizon, num_sinks=num_sinks,
        use_cross_attention=True, cross_attn_source_features=source_features,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=compute_dtype, param_dtype=param_dtype,
    )
    _ = pure(x, source=source)  # lazy init of EinsumDense
    _bridge_block(slt.layers[0], pure.layers[0], has_cross=True)

    # sl streaming state (SLTransformer is a SerialCombinatorMixin/Emitting).
    spec = sl.ChannelSpec(shape=(model_dim,), dtype=compute_dtype)
    sl_state = slt.get_initial_state(
        B, spec, constants={"source": _seq(source[:, :1])},
    )
    # pure streaming caches.
    self_caches = pure.make_self_caches()
    cross_caches = pure.make_cross_caches()

    a, r = tol(mx.bfloat16, "stack")
    for t in range(T):
        xt = x[:, t : t + 1, :]
        srct = source[:, t : t + 1, :]
        sl_yt, sl_state, _ = slt.step_with_emits(
            _seq(xt), sl_state, constants={"source": _seq(srct)},
        )
        pure_yt = pure(
            xt, self_caches=self_caches, source=srct, cross_caches=cross_caches,
        )
        assert_close(
            sl_yt.values, pure_yt, atol=a, rtol=r,
            name=f"transformer_block_bf16_streaming[t={t}]",
        )
