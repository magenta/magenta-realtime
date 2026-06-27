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

"""Round-trip tests for ``magenta_rt.nnx.load_weights`` and the
per-subsystem bridge helpers.

For each bridge we:
1. Build a fresh nnx module with random parameters.
2. Run a forward — this gives the *expected* output.
3. Walk the nnx params and emit them as a flat dict in the Linen
   key-naming convention (with the appropriate transposes applied).
4. Build a *second* nnx module from the same constructor (different
   initial random weights).
5. Bridge the flat dict back into the second module via the
   per-subsystem helper.
6. Run the same forward — output should match the expected from step 2.

This validates the bridge correctness without needing the real
multi-GB checkpoint. End-to-end real-weight loading is exercised in
``test_real_checkpoint.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.load_weights import (
    _load_attention_weights, _load_ffn_weights,
    load_decoder_embedder_weights, load_encoder_embedding_weights,
    load_decoder_tail_weights, load_transformer_weights,
)
from magenta_rt.nnx.attention import LocalSelfAttention
from magenta_rt.nnx.depthformer import ScaledEmbedding
from magenta_rt.nnx.spectrostream import (
    ResidualVectorQuantizer,
)
from magenta_rt.nnx.spectrostream.load_weights import load_quantizer_weights
from magenta_rt.nnx.transformer import (
    Encoder, FFN, MultiChannelEmbedding, SelfAttentionBlock, Transformer,
)

from ..conftest import assert_close


# ---------------------------------------------------------------------------
# Helpers — emit nnx params as a Linen-style flat dict
# ---------------------------------------------------------------------------


def _selfattn_to_linen(self_attn_block: SelfAttentionBlock, *, has_sinks: bool) -> dict:
    """Produce a Linen-style nested dict for one self-attention block."""
    attn = self_attn_block.attention
    n_heads = attn.num_heads
    head_dim = attn.units_per_head
    in_dim = attn.q_proj.kernel[...].shape[0]
    kv_in_dim = attn.kv_proj.kernel[...].shape[0]
    kv_dim = n_heads * head_dim

    # Reshape combined kv_proj [in, 2*kv_dim] → [in, 2, n_heads, head_dim]
    # then split into K and V each [in, n_heads, head_dim].
    kv = attn.kv_proj.kernel[...]
    k_flat = kv[:, :kv_dim].reshape(kv_in_dim, n_heads, head_dim)
    v_flat = kv[:, kv_dim:].reshape(kv_in_dim, n_heads, head_dim)
    q = attn.q_proj.kernel[...].reshape(in_dim, n_heads, head_dim)

    inner = {
        "query_projection": {"kernel": q},
        "key_projection": {"kernel": k_flat},
        "value_projection": {"kernel": v_flat},
        "per_dim_scale": attn.per_dim_scale_param[...],
    }
    if has_sinks and attn.sink_key_embeddings is not None:
        inner["sink_key_embeddings"] = attn.sink_key_embeddings[...]
        inner["sink_value_embeddings"] = attn.sink_value_embeddings[...]

    return {
        "pre_norm": {"scale": self_attn_block.pre_norm.scale[...]},
        "post_norm": {"scale": self_attn_block.post_norm.scale[...]},
        "attention": inner,
        "output_projection": {"kernel": attn.output_projection.kernel[...]},
    }


def _ffn_to_linen(ffn: FFN) -> dict:
    return {
        "pre_norm": {"scale": ffn.pre_norm.scale[...]},
        "post_norm": {"scale": ffn.post_norm.scale[...]},
        "ffn_layer1": {
            "kernel": ffn.ffn_layer1.kernel[...],
            "bias": ffn.ffn_layer1.bias[...],
        },
        "ffn_layer2": {
            "kernel": ffn.ffn_layer2.kernel[...],
            "bias": ffn.ffn_layer2.bias[...],
        },
    }


def _selfattn_to_linen_at(self_attn_block: SelfAttentionBlock, index: int, *, has_sinks: bool) -> dict:
    attn = self_attn_block.attention
    n_heads = attn.num_heads
    head_dim = attn.units_per_head
    in_dim = attn.q_proj.kernel[...].shape[1]
    kv_in_dim = attn.kv_proj.kernel[...].shape[1]
    kv_dim = n_heads * head_dim

    kv = attn.kv_proj.kernel[...][index]
    k_flat = kv[:, :kv_dim].reshape(kv_in_dim, n_heads, head_dim)
    v_flat = kv[:, kv_dim:].reshape(kv_in_dim, n_heads, head_dim)
    q = attn.q_proj.kernel[...][index].reshape(in_dim, n_heads, head_dim)

    inner = {
        "query_projection": {"kernel": q},
        "key_projection": {"kernel": k_flat},
        "value_projection": {"kernel": v_flat},
        "per_dim_scale": attn.per_dim_scale_param[...][index],
    }
    if has_sinks and attn.sink_key_embeddings is not None:
        inner["sink_key_embeddings"] = attn.sink_key_embeddings[...][index]
        inner["sink_value_embeddings"] = attn.sink_value_embeddings[...][index]

    return {
        "pre_norm": {"scale": self_attn_block.pre_norm.scale[...][index]},
        "post_norm": {"scale": self_attn_block.post_norm.scale[...][index]},
        "attention": inner,
        "output_projection": {"kernel": attn.output_projection.kernel[...][index]},
    }


def _ffn_to_linen_at(ffn: FFN, index: int) -> dict:
    return {
        "pre_norm": {"scale": ffn.pre_norm.scale[...][index]},
        "post_norm": {"scale": ffn.post_norm.scale[...][index]},
        "ffn_layer1": {
            "kernel": ffn.ffn_layer1.kernel[...][index],
            "bias": ffn.ffn_layer1.bias[...][index],
        },
        "ffn_layer2": {
            "kernel": ffn.ffn_layer2.kernel[...][index],
            "bias": ffn.ffn_layer2.bias[...][index],
        },
    }


def _transformer_to_linen(t: Transformer, *, has_sinks: bool) -> dict:
    out = {}
    is_scanned = "0" not in nnx.to_pure_dict(nnx.state(t))["layers"]
    if is_scanned:
        for i in range(t.num_layers):
            d = {
                "self_attention": _selfattn_to_linen_at(t.layers.self_attn, i, has_sinks=has_sinks),
                "ffn": _ffn_to_linen_at(t.layers.ffn, i),
            }
            if t.layers.cross_attn is not None:
                d["cross_attention"] = _selfattn_to_linen_at(t.layers.cross_attn, i, has_sinks=has_sinks)
            out[f"x_layers_{i}"] = d
    else:
        for i, block in enumerate(t.layers):
            d = {
                "self_attention": _selfattn_to_linen(block.self_attn, has_sinks=has_sinks),
                "ffn": _ffn_to_linen(block.ffn),
            }
            if block.cross_attn is not None:
                d["cross_attention"] = _selfattn_to_linen(block.cross_attn, has_sinks=has_sinks)
            out[f"x_layers_{i}"] = d
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bridge_quantizer_round_trip(rng_key):
    src = ResidualVectorQuantizer(
        num_quantizers=4, num_embeddings=8, embedding_dim=16,
        rngs=nnx.Rngs(0),
    )
    src.embedding = nnx.Param(
        jax.random.normal(rng_key, src.embedding[...].shape) * 0.05
    )
    codes = jnp.array([[[0, 1, 2, 3]]], dtype=jnp.int32)
    expected = src.codes_to_embeddings(codes)

    dst = ResidualVectorQuantizer(
        num_quantizers=4, num_embeddings=8, embedding_dim=16,
        rngs=nnx.Rngs(1),
    )
    load_quantizer_weights(dst, {"embedding": src.embedding[...]})
    got = dst.codes_to_embeddings(codes)
    assert_close(expected, got, atol=1e-6, rtol=1e-6, name="quantizer")


def test_bridge_decoder_embedder_round_trip(rng_key):
    src = ScaledEmbedding(vocab_size=64, dim=16, rngs=nnx.Rngs(0))
    src.embedding.embedding = nnx.Param(
        jax.random.normal(rng_key, src.embedding.embedding[...].shape) * 0.05
    )
    ids = jnp.array([[5, 17, 31]], dtype=jnp.int32)
    expected = src(ids)

    dst = ScaledEmbedding(vocab_size=64, dim=16, rngs=nnx.Rngs(1))
    load_decoder_embedder_weights(
        dst, {"embedding": {"embedding": src.embedding.embedding[...]}},
    )
    got = dst(ids)
    assert_close(expected, got, atol=1e-6, rtol=1e-6, name="decoder_embedder")


def test_bridge_encoder_round_trip(rng_key):
    rngs = nnx.Rngs(0)
    enc_emb = MultiChannelEmbedding(
        dimension=8, num_embeddings_per_channel=[16],
        num_channels=1, num_reserved_embeddings=0,
        round_num_embeddings_to_multiple_of_128=False,
        reduction_fn=jnp.mean, rngs=rngs,
    )
    src = Encoder(
        embedding=enc_emb, embedding_dimension=8, body=None, rngs=rngs,
    )
    sub = jax.random.split(rng_key, 3)
    src.embedding.embedding = nnx.Param(
        jax.random.normal(sub[0], src.embedding.embedding[...].shape) * 0.05
    )
    src.encoder_ln.scale = nnx.Param(
        jax.random.normal(sub[1], src.encoder_ln.scale[...].shape) * 0.1 + 1.0
    )
    src.encoder_ln.bias = nnx.Param(
        jax.random.normal(sub[2], src.encoder_ln.bias[...].shape) * 0.05
    )

    ids = jnp.array([[[3], [7]]], dtype=jnp.int32)
    expected = src(ids)

    dst_emb = MultiChannelEmbedding(
        dimension=8, num_embeddings_per_channel=[16],
        num_channels=1, num_reserved_embeddings=0,
        round_num_embeddings_to_multiple_of_128=False,
        reduction_fn=jnp.mean, rngs=nnx.Rngs(1),
    )
    dst = Encoder(
        embedding=dst_emb, embedding_dimension=8, body=None, rngs=nnx.Rngs(1),
    )
    flat_subdict = {
        "body": {
            "encoder_embedding": {"embedding": src.embedding.embedding[...]},
            "encoder_ln": {
                "scale": src.encoder_ln.scale[...],
                "bias": src.encoder_ln.bias[...],
            },
        },
    }
    load_encoder_embedding_weights(dst, flat_subdict)
    got = dst(ids)
    assert_close(expected, got, atol=1e-5, rtol=1e-5, name="encoder")


def test_bridge_self_attention_round_trip(rng_key):
    src = SelfAttentionBlock(
        model_dim=16, num_heads=2, units_per_head=8,
        max_past_horizon=4, num_sinks=1, rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(rng_key, (1, 5, 16))
    expected = src(x)

    dst = SelfAttentionBlock(
        model_dim=16, num_heads=2, units_per_head=8,
        max_past_horizon=4, num_sinks=1, rngs=nnx.Rngs(1),
    )
    _load_attention_weights(
        dst, _selfattn_to_linen(src, has_sinks=True), has_sinks=True,
    )
    got = dst(x)
    assert_close(expected, got, atol=1e-5, rtol=1e-5, name="self_attention")


def test_bridge_ffn_round_trip(rng_key):
    src = FFN(model_dim=16, hidden_dim=32, rngs=nnx.Rngs(0))
    x = jax.random.normal(rng_key, (1, 5, 16))
    expected = src(x)

    dst = FFN(model_dim=16, hidden_dim=32, rngs=nnx.Rngs(1))
    _load_ffn_weights(dst, _ffn_to_linen(src))
    got = dst(x)
    assert_close(expected, got, atol=1e-5, rtol=1e-5, name="ffn")


def test_bridge_transformer_round_trip(rng_key):
    src = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=4, num_sinks=0,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(rng_key, (1, 5, 16))
    expected = src(x)

    dst = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=4, num_sinks=0,
        rngs=nnx.Rngs(1),
    )
    # Functional JAX state-dict bridging:
    graph_def, abs_state = nnx.split(dst)
    from magenta_rt.nnx.load_weights import load_transformer_state_dict
    load_transformer_state_dict(
        abs_state,
        _transformer_to_linen(src, has_sinks=False),
        has_sinks=False,
    )
    nnx.update(dst, abs_state)
    got = dst(x)
    assert_close(expected, got, atol=1e-4, rtol=1e-4, name="transformer")


def test_bridge_depth_tail_round_trip(rng_key):
    """Bridges depth_body's final_ln + to_logits."""

    rngs = nnx.Rngs(0)

    class _Tail(nnx.Module):
        def __init__(self):
            self.final_ln = nnx.LayerNorm(num_features=8, epsilon=1e-6, use_bias=True, use_scale=True, rngs=rngs)
            self.to_logits = nnx.Linear(8, 16, use_bias=False, rngs=rngs)

        def __call__(self, x):
            return self.to_logits(self.final_ln(x))

    src = _Tail()
    x = jax.random.normal(rng_key, (1, 4, 8))
    expected = src(x)

    dst = _Tail()
    flat = {
        "final_ln": {
            "scale": src.final_ln.scale[...],
            "bias": src.final_ln.bias[...],
        },
        "to_logits": {
            "kernel": src.to_logits.kernel[...],
        },
    }
    load_decoder_tail_weights(dst, flat)
    got = dst(x)
    assert_close(expected, got, atol=1e-5, rtol=1e-5, name="depth_tail")


def test_bridge_decoder_unit_uses_correct_3x3_a_order(rng_key):
    """For the (1,1)-strides transposed Conv2DResidualUnit, Linen's
    body order is conv2d_3x3_a → body[1], conv2d_3x3 → body[3].
    This pins the convention so the bridge doesn't silently swap them.
    """
    from magenta_rt.nnx.spectrostream.model import Conv2DResidualUnit
    from magenta_rt.nnx.spectrostream.load_weights import _load_unit_weights

    rngs = nnx.Rngs(0)
    unit = Conv2DResidualUnit(
        input_channels=8, output_channels=8,
        strides=(1, 1), dilation=(1, 1), transposed=True,
        padding="causal", use_shortcut=False,
        rngs=rngs,
    )
    sub = jax.random.split(rng_key, 4)
    # body[1] kernel and body[3] kernel are distinct random values.
    k_a = jax.random.normal(sub[0], (8, 3, 3, 8)) * 0.1  # OHWI
    k = jax.random.normal(sub[1], (8, 3, 3, 8)) * 0.2

    # Linen-style HWIO: transpose (1,2,3,0) of OHWI.
    linen_subdict = {
        "conv2d_3x3_a": {"conv": {
            "kernel": jnp.transpose(k_a, (1, 2, 3, 0)),
            "bias": jnp.zeros((8,)),
        }},
        "conv2d_3x3": {"conv": {
            "kernel": jnp.transpose(k, (1, 2, 3, 0)),
            "bias": jnp.zeros((8,)),
        }},
    }
    _load_unit_weights(unit, linen_subdict)
    # body[1] should now hold the conv2d_3x3_a kernel; body[3] should hold conv2d_3x3.
    np.testing.assert_array_equal(np.asarray(unit.body[1].kernel[...]),
                                  np.asarray(k_a))
    np.testing.assert_array_equal(np.asarray(unit.body[3].kernel[...]),
                                  np.asarray(k))
