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

"""Checkpoint-free fp32 parity of ``nnx`` and ``mlx_pure`` against the JAX/Linen
ground truth, at the ``tiny`` config. See ``conftest.py`` for the design and the
shared jax runner / random-weight fixture.

These run in CI (no checkpoint required) and are the continuous guarantee that
the ports track ``magenta_rt.jax``; the full-scale, real-magnitude version lives
in the checkpoint-gated ``test_jax_logit_parity.py`` tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import CFG_A, CFG_B, assert_matches_jax, cond_blocks


# ---------------------------------------------------------------------------
# nnx
# ---------------------------------------------------------------------------


def _run_nnx_fp32(ckpt):
    """One fp32 greedy step of the tiny nnx depthformer, loaded from the shared
    Linen file. Replays the depth loop eagerly (rather than ``decoder.step``,
    which runs it inside ``nnx.scan``) so each codebook's pre-soft-cap logits
    can be captured, mirroring ``step``'s depth body exactly."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from magenta_rt.nnx import model as nnx_configs
    from magenta_rt.nnx import depthformer as nnx_df
    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.sample_utils import sample_categorical_with_temperature

    base = nnx_configs.get_model_class("tiny")

    class _TinyF32(base):  # nnx specs are plain classes; subclass to set dtype
        dtype = jnp.float32

    spec = _TinyF32()
    enc_dec = nnx_df.EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    target_cfg = spec.target_tokens_config
    mrt = nnx_model.MagentaRT2Sampler(
        depthformer_model=enc_dec,
        spectrostream=None,
        num_reserved_tokens=target_cfg.num_extra_tokens,
        codebook_size=target_cfg.codebook_size,
        int16_outputs=False,
    )
    mrt.load_checkpoint(ckpt)

    n_in = sum(c.rvq_truncation_level for c in spec.input_configs)
    pos, neg_a, neg_b = cond_blocks(n_in)
    src = jnp.asarray(np.stack([pos, neg_a, neg_b], 0).reshape(3, 1, -1))

    decoder = mrt.depthformer.decoder
    encoded = mrt.depthformer.encoder(src)
    mrt.init_streaming(batch_size=3, rngs=nnx.Rngs(0), codec_streaming=True)

    temporal_inputs = decoder._temporal_input(
        decoder._embed_tokens(decoder.previous_frame[...])
    )
    temporal_out = decoder.temporal(temporal_inputs, source=encoded)

    decoder.depth.soft_reset_caches()
    depth_input = decoder._adapt_depth(temporal_out)
    if decoder.dtype is not None:
        depth_input = depth_input.astype(decoder.dtype)
    depth_logits = []
    key = jax.random.key(0)
    for q in range(decoder.num_active_codebooks):
        depth_out = decoder.depth(depth_input)
        logits = decoder._logits(depth_out)  # pre-soft-cap
        depth_logits.append(np.asarray(logits, np.float32))
        cap = decoder.soft_cap_logits
        capped = jnp.tanh(logits / cap) * cap if cap is not None else logits
        min_v = decoder.num_reserved_tokens + q * decoder.codebook_size
        key, step_key = jax.random.split(key)
        sample_q = sample_categorical_with_temperature(
            capped.astype(jnp.float32), rng_key=step_key,
            temperature=0.0, top_k=1, cfg_scales=[CFG_A, CFG_B], cfg_arity=2,
            valid_range=(min_v, min_v + decoder.codebook_size),
        )
        depth_input = decoder._adapt_depth(
            decoder.embedder(sample_q[..., None]).squeeze(-2)
        )

    return {
        "encoded_source": np.asarray(encoded, np.float32),
        "temporal_outputs": np.asarray(temporal_out, np.float32),
        "depth_logits": depth_logits,
    }


def test_nnx_matches_jax_fp32(jax_ground_truth, tiny_linen_ckpt):
    """nnx depthformer must match the JAX/Linen ground truth at fp32: encoder
    output, temporal output, and every codebook's pre-soft-cap depth logits."""
    pytest.importorskip("sequence_layers.jax")
    nnx_signals = _run_nnx_fp32(tiny_linen_ckpt)
    assert_matches_jax(jax_ground_truth, nnx_signals, label="nnx")


# ---------------------------------------------------------------------------
# mlx_pure
# ---------------------------------------------------------------------------


def _build_loaded_sl_depthformer(linen_params):
    """Build a tiny sl-MLX ``EncoderDecoder``, materialize it, and load the tiny
    Linen depthformer weights into it.

    mlx_pure has no direct Linen loader — production loads via the sl-MLX
    combinator (``mlx_pure.load_weights.load_weights_from_combinator``). The
    shipped sl loader (``magenta_rt.mlx.load_weights.load_weights``) is
    branched-musiccoca-encoder-only and requires soundstream params, neither of
    which a checkpoint-free tiny depthformer export has. So we reuse the same
    per-subsystem sl helpers it does, but drive them directly for the tiny,
    single-channel, depthformer-only case.
    """
    import mlx.core as mx
    import sequence_layers.mlx as sl
    from sequence_layers.mlx import export as slexport
    from magenta_rt.mlx import model as cm
    from magenta_rt.mlx.load_weights import (
        _to_mx, _load_transformer, _load_layer_norm, _load_dense,
    )

    base = cm.get_model_class("tiny")

    class _TinyF32(base):
        compute_dtype = mx.float32
        param_dtype = mx.float32

    spec = _TinyF32()
    n_in = spec.input_num_channels
    ed = spec.depthformer_config().make()

    constants = {
        "classifier_free_guidance_scale_musiccoca": mx.array([CFG_A]),
        "classifier_free_guidance_scale_notes": mx.array([CFG_B]),
        "temperature": mx.array([0.0]),
        "top_k": mx.array([1]),
    }
    slexport._materialize_deferred(
        ed.sampler, batch_size=1,
        input_spec=sl.ShapeDType((n_in,), mx.int32), constants=constants,
    )

    df = linen_params["depthformer"]

    # Encoder: non-branched MultiChannelEmbedding + encoder_ln (last layer).
    enc_body = ed.encoder.body
    enc_body.layers[0].embedding = _to_mx(
        np.asarray(df["encoder"]["body"]["encoder_embedding"]["embedding"])
    )
    _load_layer_norm(enc_body.layers[-1], df["encoder"]["body"]["encoder_ln"])

    # Decoder embedder (Serial[Embedding, Scale]).
    dec = ed.decoder
    embed_layer, _scale = dec.embedder.layers
    embed_layer._embedding.weight = _to_mx(
        np.asarray(df["decoder"]["decoder_embedding"]["embedding"]["embedding"])
    )

    # Temporal + depth transformers (depth layers[0] is the Identity adapter).
    _load_transformer(dec.temporal_body.layers[0], df["decoder"]["temporal_body"]["transformer"])
    _load_transformer(dec.depth_body.layers[1], df["decoder"]["depth_body"]["transformer"])

    # Depth tail: final_ln (layers[2]) + to_logits (layers[3]).
    jax_depth = df["decoder"]["depth_body"]
    _load_layer_norm(dec.depth_body.layers[2], jax_depth["final_ln"])
    _load_dense(dec.depth_body.layers[3], jax_depth["to_logits"])

    mx.eval(ed.parameters())
    return ed


def _run_mlx_pure_fp32(ckpt):
    """One fp32 greedy step of the tiny mlx_pure depthformer, weights bridged in
    via a structurally-matching sl-MLX depthformer loaded from the shared Linen
    file. Captures pre-soft-cap depth logits by spying on ``decoder._logits``."""
    import mlx.core as mx
    from magenta_rt.jax.system import _load_jax_weights
    from magenta_rt.mlx_pure import configs as pure_configs
    from magenta_rt.mlx_pure.load_weights import load_depthformer_weights

    linen_params = _load_jax_weights(ckpt)["params"]
    sl_ed = _build_loaded_sl_depthformer(linen_params)

    base = pure_configs.get_model_class("tiny")

    class _TinyF32(base):
        compute_dtype = mx.float32
        param_dtype = mx.float32

    pspec = _TinyF32()
    enc_dec = pspec.build_decoder()
    load_depthformer_weights(enc_dec, sl_ed)
    mx.eval(enc_dec.parameters())

    n_in = pspec.input_num_channels
    pos, neg_a, neg_b = cond_blocks(n_in)
    src = mx.array(np.stack([pos, neg_a, neg_b], 0).reshape(3, 1, -1), dtype=mx.int32)
    decoder = enc_dec.decoder

    encoded = enc_dec.encode(src)
    state = enc_dec.make_initial_state(batch_size=3, seed=0)
    embedded = decoder._embed_tokens(state.previous_frame)
    temporal_inputs = decoder._temporal_input(embedded)
    temporal_out = decoder.temporal(
        temporal_inputs, source=encoded,
        self_caches=state.temporal.self_caches,
        cross_caches=state.temporal.cross_caches,
    )

    captured = []
    orig_logits = decoder._logits

    def _spy(d):
        out = orig_logits(d)
        captured.append(np.asarray(out.astype(mx.float32)))
        return out

    decoder._logits = _spy
    try:
        state2 = enc_dec.make_initial_state(batch_size=3, seed=0)
        enc_dec.step(
            state2, source_frame=encoded, temperature=0.0, top_k=1,
            cfg_scales=[CFG_A, CFG_B], cfg_arity=2,
        )
    finally:
        decoder._logits = orig_logits

    return {
        "encoded_source": np.asarray(encoded.astype(mx.float32)),
        "temporal_outputs": np.asarray(temporal_out.astype(mx.float32)),
        "depth_logits": captured,
    }


def test_mlx_pure_matches_jax_fp32(jax_ground_truth, tiny_linen_ckpt):
    """mlx_pure depthformer must match the JAX/Linen ground truth at fp32:
    encoder output, temporal output, and every codebook's depth logits."""
    pytest.importorskip("sequence_layers.jax")
    mlx_signals = _run_mlx_pure_fp32(tiny_linen_ckpt)
    assert_matches_jax(jax_ground_truth, mlx_signals, label="mlx_pure")
