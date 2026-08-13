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

"""Real-checkpoint parity test: pure-MLX vs sl on actual production weights.

Gated by ``@pytest.mark.checkpoint`` and a presence check on the
checkpoint file. Auto-skips when the file is absent so this test is
safe to keep enabled in CI without checkpoints.

Strategy:
1. Load the sl-backed system via ``magenta_rt.mlx.load_weights``.
2. Build a structurally-matching pure-MLX system from the same model
   spec.
3. Bridge the parameter tree via :func:`mlx_pure.load_weights.mirror_params`
   (per-leaf shape/name mapping documented in the bridge call).
4. Run identical conditioning tokens through both pipelines.
5. Diff the temporal-decoder output (the highest-leverage signal —
   downstream errors compound from there).

The full bridge mapping is non-trivial because sl uses
underscore-prefixed sub-modules (``_linear``, ``_rms_norm``) that get
renamed to ``linear`` / ``norm`` in pure. The mapping below covers
the temporal-decoder-only path; SpectroStream and depth-decoder
mappings are noted as TODO.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def checkpoint_path():
    p = Path(__file__).resolve().parents[3] / "checkpoints"
    candidates = sorted(p.glob("pianorollbaseline_v1v3*.safetensors")) if p.exists() else []
    if not candidates:
        pytest.skip(f"no v1v3 checkpoint .safetensors in {p}")
    return candidates[0]



@pytest.mark.checkpoint
@pytest.mark.slow
def test_temporal_decoder_real_checkpoint_parity(checkpoint_path, rng_key):
    """Load the production checkpoint into both sl and pure; verify
    that the temporal-decoder embedding step (token → temporal output)
    matches numerically.
    """
    import dataclasses
    import sequence_layers.mlx as sl
    from magenta_rt.mlx import model as sl_model
    from magenta_rt.mlx import system as sl_system
    from magenta_rt.mlx import spectrostream as sl_spectrostream
    from magenta_rt.mlx.load_weights import load_weights as sl_load_weights

    from magenta_rt.mlx_pure import configs as pure_configs
    from .conftest import assert_close, tol

    def _residual_body_layers(residual):
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

    def _bridge_cross_attn(sl_cross_residual, pure_block):
        body = _residual_body_layers(sl_cross_residual)
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

    def _walk_block_chain(sl_block_serial, *, has_cross: bool):
        children = sl_block_serial.layers
        self_attn = children[0]
        cross = children[1] if has_cross else None
        ffn = children[2]
        return self_attn, cross, ffn

    # ---- Detect model name from checkpoint ----
    ckpt_name = checkpoint_path.name.lower()
    if "smallm4air" in ckpt_name:
        model_name = "mrt2_small"
    else:
        model_name = "mrt2_base"

    # ---- Load sl ----
    sl_spec = sl_model.get_model_class(model_name)()
    rvq_truncation_level = sl_spec.spectrostream.rvq_truncation_level

    mrt_system = sl_system.MagentaRT2Sampler.Config(
        depthformer=sl_spec.depthformer_config(),
        spectrostream=sl_spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq_truncation_level, use_unique_codes=False,
        ),
    ).make()

    print(f"Loading SL weights from {checkpoint_path} (model preset: {model_name})...")
    sl_load_weights(mrt_system, str(checkpoint_path), num_input_channels=sl_spec.input_num_channels)

    # ---- Build pure ----
    pure_spec = pure_configs.get_model_class(model_name)()
    pure_enc_dec = pure_spec.build_decoder()
    num_layers = pure_spec.decoder_temporal_size.num_layers

    # ---- Materialize Pure ----
    # mrt2_small's encoder (256-d) and temporal body (1024-d) differ in
    # width, so the cross-attention source is encoder-width, *not*
    # temporal-width. Keep the two dims separate.
    B, T = 1, 5
    model_dim = pure_spec.decoder_temporal_size.model_dims
    source_dim = pure_spec.encoder_size.model_dims
    x = mx.random.normal((B, T, model_dim), key=rng_key)
    source = mx.random.normal((B, T, source_dim), key=mx.random.split(rng_key)[0])

    _ = pure_enc_dec.decoder.temporal(x, source=source)

    # ---- Bridge ----
    sl_temporal = mrt_system.depthformer.decoder.temporal_body
    pure_temporal = pure_enc_dec.decoder.temporal

    slt = sl_temporal.layers[0]
    assert len(slt.layers) == num_layers == len(pure_temporal.layers), (
        f"layer-count mismatch: sl={len(slt.layers)} "
        f"spec={num_layers} pure={len(pure_temporal.layers)}"
    )

    for li in range(num_layers):
        sl_block = slt.layers[li]
        sa_res, ca_res, ffn_res = _walk_block_chain(sl_block, has_cross=True)
        _bridge_self_attn(sa_res, pure_temporal.layers[li].self_attn)
        _bridge_cross_attn(ca_res, pure_temporal.layers[li].cross_attn)
        _bridge_ffn(ffn_res, pure_temporal.layers[li].ffn)

    # ---- Run identical inputs and compare per block ----
    def _seq(values: mx.array) -> sl.Sequence:
        return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))

    # Real weights are bf16 after the sl loader's bf16 conversion, and
    # both pipelines run bf16 compute — so use bf16 tolerances. Failures
    # are collected per stage and asserted at the end so one bad block
    # doesn't mask the rest.
    a, r = tol(mx.bfloat16, "block")
    failures: list[str] = []

    def _check(sl_vals, pure_vals, name):
        try:
            assert_close(sl_vals, pure_vals, atol=a, rtol=r, name=name)
            print(f"  {name} ✓")
        except AssertionError as e:
            print(f"  {name} FAILED:\n{e}")
            failures.append(name)

    # Block 0: check each sub-stage (self-attn / cross-attn / FFN)
    # separately to localize any divergence.
    sa_res, ca_res, ffn_res = _walk_block_chain(slt.layers[0], has_cross=True)

    sl_sa_out = sa_res.layer(_seq(x), constants={"source": _seq(source)})
    pure_sa_out = pure_temporal.layers[0].self_attn(x)
    _check(sl_sa_out.values, pure_sa_out, "block_0_self_attn_parity")

    sl_ca_out = ca_res.layer(sl_sa_out, constants={"source": _seq(source)})
    pure_ca_out = pure_temporal.layers[0].cross_attn(pure_sa_out, source=source)
    _check(sl_ca_out.values, pure_ca_out, "block_0_cross_attn_parity")

    sl_ffn_out = ffn_res.layer(sl_ca_out, constants={"source": _seq(source)})
    pure_ffn_out = pure_temporal.layers[0].ffn(pure_ca_out)
    _check(sl_ffn_out.values, pure_ffn_out, "block_0_ffn_parity")

    # Blocks 1..N-1: whole-block parity, threading the running activation.
    sl_x_seq = sl_ffn_out
    pure_x = pure_ffn_out
    for li in range(1, num_layers):
        sl_x_seq = slt.layers[li].layer(sl_x_seq, constants={"source": _seq(source)})
        pure_x = pure_temporal.layers[li](pure_x, source=source)
        _check(sl_x_seq.values, pure_x, f"block_{li}_parity")

    assert not failures, (
        f"{len(failures)} temporal-decoder parity stage(s) diverged: {failures}"
    )
