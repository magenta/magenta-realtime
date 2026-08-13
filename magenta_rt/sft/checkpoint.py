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

"""Helpers to load pretrained Linen-format safetensors checkpoints into the
NNX / MLX-pure ``EncoderDecoder`` trainers, and to export the trained model
back into Linen-format safetensors for the existing inference paths.

The "loader" side reuses the per-subtree bridges in
``magenta_rt.nnx.load_weights`` / ``magenta_rt.mlx_pure.load_weights``; we
just adapt them to work against a bare ``EncoderDecoder`` (the trainers
build the depthformer directly, not the full ``MagentaRT2Sampler`` system).

The "exporter" side is the inverse: walk the trained model and produce a
flat ``{path/with/slashes: ndarray}`` dict matching the Linen layout the
inference loaders expect, then drop into ``safetensors.numpy.save_file``.
SpectroStream params (which never get trained) are copied through from the
*source* checkpoint when one is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# NNX side
# ---------------------------------------------------------------------------

def load_nnx_depthformer_from_safetensors(model, checkpoint_path: str | Path) -> None:
    """Load Linen-format depthformer weights into an NNX ``EncoderDecoder``.

    The shipped ``magenta_rt.nnx.load_weights.load_from_jax_safetensors``
    expects a full ``MagentaRT2Sampler`` system; we wrap our ``EncoderDecoder``
    in a ``{"depthformer": ...}`` shim so the same bridge can be reused.
    """
    from flax import nnx
    from magenta_rt.nnx.load_weights import _load_unflattened, load_system_state_dict

    # host=True: keep the (fp32) checkpoint in host RAM and cast each leaf to the
    # model's param dtype (e.g. bf16) before it hits the GPU — so a 9.6 GB fp32
    # base checkpoint never co-resides on-device with the bf16 model (16 GB OOM).
    nested = _load_unflattened(checkpoint_path, host=True)
    params = nested.get("params", nested)

    graph_def, abs_state = nnx.split(model)
    shim = {"depthformer": abs_state}
    load_system_state_dict(shim, params)
    nnx.update(model, abs_state)


def export_nnx_to_linen_safetensors(
    model,
    out_path: str | Path,
    *,
    source_checkpoint_path: Optional[str | Path] = None,
) -> int:
    """Export an NNX ``EncoderDecoder``'s params as Linen-format safetensors.

    Walks ``nnx.state(model)`` and renames keys to the Linen layout the
    inference loaders consume (``params/depthformer/...``). When a source
    checkpoint is provided, SpectroStream params + any unaltered keys are
    copied through verbatim so the result is a drop-in replacement.

    Returns the number of tensors written.
    """
    import jax
    from flax import nnx
    from safetensors.numpy import save_file

    # 1. Captured original keys (for SpectroStream + unchanged passthrough).
    base: dict[str, np.ndarray] = {}
    if source_checkpoint_path is not None:
        from safetensors import safe_open
        with safe_open(str(source_checkpoint_path), framework="numpy") as f:
            for k in f.keys():
                base[k] = f.get_tensor(k)

    # 2. Walk the live NNX state, emit Linen-style keys for the depthformer.
    # We collect every parameter-like Variable, including:
    #   * `nnx.Param` (base trainable weights),
    #   * `Frozen` retyped weights (encoder freeze for full-SFT),
    #   * `MRTLoRAParam` subclasses (treated as Param — merge them first via
    #     `magenta_rt.sft.lora.merge_lora_into_base` for a "plain" export).
    from .freeze import Frozen
    state = nnx.state(model, (nnx.Param, Frozen))
    extracted: dict[str, np.ndarray] = {}

    def _put(linen_path: str, leaf) -> None:
        arr = np.asarray(jax.device_get(leaf))
        if arr.dtype == np.dtype("bfloat16"):
            arr = arr.astype(np.float32)
        # safetensors.save_file writes a tensor's raw buffer and assumes it is
        # C-contiguous; a non-contiguous array (e.g. the key/value halves from
        # np.split(kv_proj) reshaped) would be serialized as garbage. Force a
        # contiguous copy so every leaf round-trips exactly.
        extracted[f"params/depthformer/{linen_path}"] = np.ascontiguousarray(arr)

    # ── Encoder ──
    _put("encoder/body/encoder_embedding/embedding",
         state["encoder"]["embedding"]["embedding"])
    _put("encoder/body/encoder_ln/scale",
         state["encoder"]["encoder_ln"]["scale"])
    if "bias" in state["encoder"]["encoder_ln"]:
        _put("encoder/body/encoder_ln/bias",
             state["encoder"]["encoder_ln"]["bias"])

    # ── Decoder ──
    dec = state["decoder"]
    _put("decoder/decoder_embedding/embedding/embedding",
         dec["embedder"]["embedding"]["embedding"])

    if "depth_input_adapter" in dec and dec["depth_input_adapter"] is not None:
        _put("decoder/depth_body/depth_input_adapter/kernel",
             dec["depth_input_adapter"]["kernel"])

    _put("decoder/depth_body/final_ln/scale", dec["final_ln"]["scale"])
    if "bias" in dec["final_ln"]:
        _put("decoder/depth_body/final_ln/bias", dec["final_ln"]["bias"])

    _put("decoder/depth_body/to_logits/kernel", dec["to_logits"]["kernel"])
    if "bias" in dec["to_logits"]:
        _put("decoder/depth_body/to_logits/bias", dec["to_logits"]["bias"])

    # Transformer stacks — `nnx.scan` stores per-layer params with a
    # leading axis (num_layers); Linen uses one subdict per `x_layers_i`.
    _emit_transformer(dec["temporal"], "decoder/temporal_body/transformer", _put,
                      has_sinks=True)
    _emit_transformer(dec["depth"], "decoder/depth_body/transformer", _put,
                      has_sinks=False)

    # 3. Merge: base (passthrough) ← extracted (overrides). New keys from
    # extracted are added (depthformer); base keys outside the depthformer
    # subtree (i.e., soundstream) survive untouched.
    out = dict(base)
    out.update(extracted)
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    save_file(out, str(out_path))
    return len(out)


def _emit_transformer(xformer_state, prefix: str, put, *, has_sinks: bool):
    """Write a stacked Transformer subtree out under `prefix/x_layers_i/...`."""
    layers = xformer_state["layers"]
    sa = layers["self_attn"]["attention"]
    num_layers = sa["q_proj"]["kernel"].shape[0]
    cross = layers.get("cross_attn") if "cross_attn" in layers else None
    ffn = layers["ffn"]

    for i in range(num_layers):
        base_p = f"{prefix}/x_layers_{i}"

        # self-attention
        _emit_attention(sa, f"{base_p}/self_attention", put, i, has_sinks=has_sinks)

        # optional cross-attention
        if cross is not None:
            _emit_attention(cross["attention"], f"{base_p}/cross_attention",
                            put, i, has_sinks=False)

        # ffn
        _put_slice(ffn["ffn_layer1"]["kernel"], i, f"{base_p}/ffn/ffn_layer1/kernel", put)
        if "bias" in ffn["ffn_layer1"]:
            _put_slice(ffn["ffn_layer1"]["bias"], i, f"{base_p}/ffn/ffn_layer1/bias", put)
        _put_slice(ffn["ffn_layer2"]["kernel"], i, f"{base_p}/ffn/ffn_layer2/kernel", put)
        if "bias" in ffn["ffn_layer2"]:
            _put_slice(ffn["ffn_layer2"]["bias"], i, f"{base_p}/ffn/ffn_layer2/bias", put)

        # norms (residual blocks)
        if "pre_norm" in layers["self_attn"]:
            _put_slice(layers["self_attn"]["pre_norm"]["scale"], i,
                       f"{base_p}/self_attention/pre_norm/scale", put)
        if "post_norm" in layers["self_attn"]:
            _put_slice(layers["self_attn"]["post_norm"]["scale"], i,
                       f"{base_p}/self_attention/post_norm/scale", put)
        if cross is not None and "pre_norm" in cross:
            _put_slice(cross["pre_norm"]["scale"], i,
                       f"{base_p}/cross_attention/pre_norm/scale", put)
        if cross is not None and "post_norm" in cross:
            _put_slice(cross["post_norm"]["scale"], i,
                       f"{base_p}/cross_attention/post_norm/scale", put)
        if "pre_norm" in ffn:
            _put_slice(ffn["pre_norm"]["scale"], i, f"{base_p}/ffn/pre_norm/scale", put)
        if "post_norm" in ffn:
            _put_slice(ffn["post_norm"]["scale"], i, f"{base_p}/ffn/post_norm/scale", put)


def _emit_attention(attn_state, prefix: str, put, idx: int, *, has_sinks: bool):
    """Emit Linen-format attention keys under `prefix`.

    Linen layout the loader expects (per layer i, after un-flattening
    slash-keys):

        <prefix>/
          attention/
            query_projection/kernel   [in, n_heads, head_dim]
            key_projection/kernel     [in, n_heads, head_dim]
            value_projection/kernel   [in, n_heads, head_dim]
            per_dim_scale             [head_dim]            (LEAF)
            sink_key_embeddings       (optional)
            sink_value_embeddings     (optional)
          output_projection/kernel    [out, n_heads, head_dim]

    Note that `output_projection` is *sibling* to `attention`, not nested
    inside it. NNX flattens those by storing the output projection on the
    LocalSelfAttention module itself; we have to put it back outside.
    """
    q_kernel = attn_state["q_proj"]["kernel"][idx]      # [in, q_dim]
    kv_kernel = attn_state["kv_proj"]["kernel"][idx]    # [in, 2*kv_dim]
    out_kernel = attn_state["output_projection"]["kernel"][idx]  # [d, n, h]
    n_heads = out_kernel.shape[1]
    head_dim = out_kernel.shape[2]

    in_dim = q_kernel.shape[0]
    put(f"{prefix}/attention/query_projection/kernel",
        q_kernel.reshape(in_dim, n_heads, head_dim))
    k_kernel, v_kernel = np.split(np.asarray(kv_kernel), 2, axis=-1)
    put(f"{prefix}/attention/key_projection/kernel",
        k_kernel.reshape(in_dim, n_heads, head_dim))
    put(f"{prefix}/attention/value_projection/kernel",
        v_kernel.reshape(in_dim, n_heads, head_dim))
    put(f"{prefix}/output_projection/kernel", out_kernel)

    if "per_dim_scale_param" in attn_state:
        _put_slice(attn_state["per_dim_scale_param"], idx,
                   f"{prefix}/attention/per_dim_scale", put)

    # The NNX state always declares sink_{key,value}_embeddings under each
    # attention block — they're zero-sized when num_sinks=0. The loader's
    # `has_sinks` check is propagated through to cross-attention, so we
    # have to emit these regardless of whether the parent transformer
    # actually uses them. Loader-side guards on `"sink_key_embeddings" in attn`
    # filter no-op writes.
    if "sink_key_embeddings" in attn_state and attn_state["sink_key_embeddings"] is not None:
        _put_slice(attn_state["sink_key_embeddings"], idx,
                   f"{prefix}/attention/sink_key_embeddings", put)
        _put_slice(attn_state["sink_value_embeddings"], idx,
                   f"{prefix}/attention/sink_value_embeddings", put)


def _put_slice(stacked_leaf, idx: int, key: str, put) -> None:
    """Pull layer `idx` out of a stacked vmapped leaf and emit it."""
    arr = np.asarray(stacked_leaf)
    put(key, arr[idx])
