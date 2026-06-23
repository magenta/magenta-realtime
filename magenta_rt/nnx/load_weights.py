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

"""Load Linen-style safetensors checkpoints into the nnx tree.

:func:`load_from_jax_safetensors` reads the safetensors file produced
by ``magenta_rt/jax/generate.py`` and bridges per-subsystem into the
matching ``MagentaRT2Sampler`` nnx tree. Constructors handle random
initialization on their own via the standard ``rngs`` plumbing, so
there is no separate ``init_random_params`` entry point.

Layout conversions applied by the bridges below:

* **Conv2D / Conv2DTranspose kernel** — Linen ``[kH, kW, in, out]``
  (HWIO) → nnx ``[out, kH, kW, in]`` (OHWI).
* **Linear / Einsum kernel** — Linen layout matches nnx
  (``[in, out]`` and ``[d, n, h]`` respectively); direct copy.
* **Self-attention QKV** — Linen has separate query / key / value
  projections each ``[in, n_heads, head_dim]``. nnx uses one
  ``q_proj`` ``[in, q_dim]`` plus a combined ``kv_proj``
  ``[in, 2*kv_dim]``.

Two flavors of per-subsystem helpers live here:

* **Module-API** (``_load_attention_weights`` / ``_load_ffn_weights`` /
  ``load_*_weights``) — write into live ``nnx.Module`` instances via the
  ``[...]`` Param protocol. Used by the round-trip parity tests.
* **State-dict** (``load_system_state_dict`` / ``load_transformer_state_dict`` /
  ``_load_*_state_dict*``) — write into the abstract state pytree from
  ``nnx.split``. The production load path uses this flavor so the
  ``nnx.scan``-stacked transformer layers stay contiguous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import safetensors.flax as safetensors_flax
from flax import nnx
import flax.traverse_util as flaxtu


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _set_param(param: nnx.Param, value: jnp.ndarray) -> None:
    """Assign ``value`` to a Variable slot using the nnx [...] protocol."""
    if value.shape != param[...].shape:
        raise ValueError(
            f"shape mismatch: source {value.shape} vs target {param[...].shape}"
        )
    param[...] = value.astype(param[...].dtype)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_unflattened(checkpoint_path: str | Path, *, host: bool = False) -> dict:
    """Read a Linen safetensors file and unflatten to a nested dict.

    ``host=True`` loads the tensors as **numpy** (host RAM) instead of jax
    device arrays. The per-leaf assignment (``_set_param``) then casts each
    fp32 leaf to the target param dtype on the host and transfers only the
    (e.g. bf16) result — so the full fp32 checkpoint never lands on the
    accelerator at once. Essential for big bf16 models (mrt2_base: a 9.6 GB
    fp32 checkpoint would OOM a 16 GB GPU alongside the 4.8 GB bf16 model).
    """
    if host:
        from safetensors.numpy import load_file as _load_numpy
        flat = _load_numpy(str(checkpoint_path))
    else:
        flat = safetensors_flax.load_file(str(checkpoint_path))
    nested = {tuple(k.split("/")): v for k, v in flat.items()}
    return flaxtu.unflatten_dict(nested)


def load_from_jax_safetensors(
    pure_system, checkpoint_path: str | Path, *, host: bool = False
) -> None:
    """Load weights from a Linen safetensors checkpoint into an nnx
    ``MagentaRT2Sampler`` system.

    ``host=True`` reads the tensors into host RAM (numpy) instead of loading the
    whole fp32 file onto the accelerator at once; the per-leaf assignment then
    casts each leaf to the target (e.g. bf16) param dtype on transfer. Essential
    when a second model must co-reside on the GPU — e.g. the periodic
    ``AudioSampleWriter`` sampler alongside a bf16 ``mrt2_base`` training model,
    where an on-device fp32 load (9.8 GB) would OOM a 16 GB card.
    """
    nested = _load_unflattened(checkpoint_path, host=host)
    params = nested.get("params", nested)

    # 1. SpectroStream live in-place bridging. The checkpoint stores
    # the codec subtree at ``params/soundstream/`` — a historical
    # artifact from before the SoundStream → SpectroStream rename. We
    # keep reading the old key since the on-disk format isn't ours
    # to retag.
    if pure_system.spectrostream is not None and "soundstream" in params:
        pure_system.spectrostream.load_state_dict(params["soundstream"])

    # 2. Split and get the state object
    graph_def, abs_state = nnx.split(pure_system)

    # 3. Depthformer-side functional state loading
    load_system_state_dict(abs_state, params)

    # 4. Update the system
    nnx.update(pure_system, abs_state)


# ---------------------------------------------------------------------------
# Module-API bridges (write into live nnx.Module instances)
# ---------------------------------------------------------------------------


def load_encoder_embedding_weights(pure_encoder, encoder_subdict: Mapping) -> None:
    """Pure ``Encoder`` ← Linen encoder body.

    Linen layout (from the smallm4air checkpoint):
        body/encoder_embedding/embedding   [total_vocab, dim]
        body/encoder_ln/scale              [dim]
        body/encoder_ln/bias               [dim]
    """
    body = encoder_subdict["body"]
    _set_param(
        pure_encoder.embedding.embedding,
        jnp.asarray(body["encoder_embedding"]["embedding"]),
    )
    _set_param(
        pure_encoder.encoder_ln.scale,
        jnp.asarray(body["encoder_ln"]["scale"]),
    )
    if pure_encoder.encoder_ln.bias is not None:
        _set_param(
            pure_encoder.encoder_ln.bias,
            jnp.asarray(body["encoder_ln"]["bias"]),
        )


def load_decoder_embedder_weights(pure_embedder, embedder_subdict: Mapping) -> None:
    """Pure ``ScaledEmbedding.embedding.embedding`` ← Linen decoder embedding.

    Linen layout:
        decoder_embedding/embedding/embedding  [vocab, dim]
    """
    table = jnp.asarray(embedder_subdict["embedding"]["embedding"])
    _set_param(pure_embedder.embedding.embedding, table)


def _load_attention_weights(pure_attn_block, attn_subdict: Mapping, *, has_sinks: bool) -> None:
    """Mirror one attention residual into a pure ``SelfAttentionBlock``
    or ``CrossAttentionBlock``.

    Linen layout per block:
        pre_norm/scale
        attention/query_projection/kernel  [in, n_heads, head_dim]
        attention/key_projection/kernel    [src_in, n_heads, head_dim]
        attention/value_projection/kernel  [src_in, n_heads, head_dim]
        attention/per_dim_scale            [head_dim]
        attention/sink_key_embeddings      [num_sinks, n_heads, head_dim]   (optional)
        attention/sink_value_embeddings    [num_sinks, n_heads, head_dim]   (optional)
        output_projection/kernel           [d, n_heads, head_dim]
        post_norm/scale
    """
    _set_param(
        pure_attn_block.pre_norm.scale,
        jnp.asarray(attn_subdict["pre_norm"]["scale"]),
    )
    _set_param(
        pure_attn_block.post_norm.scale,
        jnp.asarray(attn_subdict["post_norm"]["scale"]),
    )

    attn = pure_attn_block.attention
    inner = attn_subdict["attention"]

    q_kernel = jnp.asarray(inner["query_projection"]["kernel"])
    k_kernel = jnp.asarray(inner["key_projection"]["kernel"])
    v_kernel = jnp.asarray(inner["value_projection"]["kernel"])

    in_q, n_heads, head_dim = q_kernel.shape
    in_kv = k_kernel.shape[0]
    q_proj = q_kernel.reshape(in_q, n_heads * head_dim)
    kv_proj = jnp.concatenate(
        [k_kernel.reshape(in_kv, n_heads * head_dim),
         v_kernel.reshape(in_kv, n_heads * head_dim)],
        axis=-1,
    )
    _set_param(attn.q_proj.kernel, q_proj)
    _set_param(attn.kv_proj.kernel, kv_proj)
    _set_param(
        attn.per_dim_scale_param,
        jnp.asarray(inner["per_dim_scale"]),
    )

    if has_sinks and attn.sink_key_embeddings is not None:
        _set_param(
            attn.sink_key_embeddings,
            jnp.asarray(inner["sink_key_embeddings"]),
        )
        _set_param(
            attn.sink_value_embeddings,
            jnp.asarray(inner["sink_value_embeddings"]),
        )

    # Output projection: linen [d, n, h] = nnx [d, n, h] (no transpose).
    out_kernel = jnp.asarray(attn_subdict["output_projection"]["kernel"])
    _set_param(attn.output_projection.kernel, out_kernel)


def _load_ffn_weights(pure_ffn, ffn_subdict: Mapping) -> None:
    """Mirror one FFN residual.

    Linen layout:
        ffn/pre_norm/scale
        ffn/ffn_layer1/kernel   [d, hidden]
        ffn/ffn_layer1/bias     [hidden]
        ffn/ffn_layer2/kernel   [hidden, d]
        ffn/ffn_layer2/bias     [d]
        ffn/post_norm/scale
    """
    _set_param(pure_ffn.pre_norm.scale, jnp.asarray(ffn_subdict["pre_norm"]["scale"]))
    _set_param(pure_ffn.post_norm.scale, jnp.asarray(ffn_subdict["post_norm"]["scale"]))

    l1 = ffn_subdict["ffn_layer1"]
    _set_param(
        pure_ffn.ffn_layer1.kernel,
        jnp.asarray(l1["kernel"]),
    )
    _set_param(pure_ffn.ffn_layer1.bias, jnp.asarray(l1["bias"]))

    l2 = ffn_subdict["ffn_layer2"]
    _set_param(
        pure_ffn.ffn_layer2.kernel,
        jnp.asarray(l2["kernel"]),
    )
    _set_param(pure_ffn.ffn_layer2.bias, jnp.asarray(l2["bias"]))


def load_transformer_weights(pure_xformer, transformer_subdict: Mapping, *, has_sinks: bool) -> None:
    """Mirror ``num_layers`` ``x_layers_N`` blocks from a Linen
    ``transformer`` subdict into a pure :class:`Transformer`.

    Each Linen block has self_attention, optionally cross_attention, and ffn.
    """
    layers = transformer_subdict
    for i, block in enumerate(pure_xformer.layers):
        key = f"x_layers_{i}"
        ld = layers[key]
        _load_attention_weights(block.self_attn, ld["self_attention"], has_sinks=has_sinks)
        if block.cross_attn is not None:
            _load_attention_weights(block.cross_attn, ld["cross_attention"], has_sinks=has_sinks)
        _load_ffn_weights(block.ffn, ld["ffn"])


def load_decoder_tail_weights(pure_decoder, depth_body_subdict: Mapping) -> None:
    """Mirror the post-transformer tail (final_ln + to_logits).

    Linen layout:
        depth_body/final_ln/scale, /bias
        depth_body/to_logits/kernel  [d, vocab]
        depth_body/to_logits/bias    [vocab]
    """
    fln = depth_body_subdict["final_ln"]
    _set_param(pure_decoder.final_ln.scale, jnp.asarray(fln["scale"]))
    if pure_decoder.final_ln.bias is not None:
        _set_param(pure_decoder.final_ln.bias, jnp.asarray(fln["bias"]))

    tl = depth_body_subdict["to_logits"]
    _set_param(
        pure_decoder.to_logits.kernel,
        jnp.asarray(tl["kernel"]),
    )
    if pure_decoder.to_logits.bias is not None and "bias" in tl:
        _set_param(pure_decoder.to_logits.bias, jnp.asarray(tl["bias"]))


# ---------------------------------------------------------------------------
# State-dict bridges (write into the abstract state pytree from nnx.split)
# ---------------------------------------------------------------------------


def load_system_state_dict(state_dict: dict, params_dict: Mapping) -> None:
    """Populate pure JAX variables state dict from Linen safetensors dict."""
    df = params_dict["depthformer"]

    # 1. Encoder embedding — branched (pretrained-MusicCoCa, mrt2) or plain.
    enc_body = df["encoder"]["body"]
    enc_emb = state_dict["depthformer"]["encoder"]["embedding"]
    if "encoder_embedding" in enc_body:
        enc_emb["embedding"][...] = jnp.asarray(
            enc_body["encoder_embedding"]["embedding"]
        )
    else:
        mul = enc_body["layers_1"]["branched_mulan_embedder"]["mulan_embedder"]
        enc_emb["mulan_embedder"]["mulan_dequantizer"]["embedding"][...] = jnp.asarray(
            mul["mulan_dequantizer"]["embedding"]
        )
        enc_emb["mulan_embedder"]["depth_input_adapter"]["kernel"][...] = jnp.asarray(
            mul["depth_input_adapter"]["kernel"]
        )
        reg = enc_body["layers_1"]["branched_regular_embedder"]["regular_embedder"]
        enc_emb["regular_embedder"]["embedding"][...] = jnp.asarray(reg["embedding"])
    state_dict["depthformer"]["encoder"]["encoder_ln"]["scale"][...] = jnp.asarray(
        df["encoder"]["body"]["encoder_ln"]["scale"]
    )
    if "bias" in df["encoder"]["body"]["encoder_ln"] and "bias" in state_dict["depthformer"]["encoder"]["encoder_ln"]:
        state_dict["depthformer"]["encoder"]["encoder_ln"]["bias"][...] = jnp.asarray(
            df["encoder"]["body"]["encoder_ln"]["bias"]
        )

    # 2. Decoder Embedder
    state_dict["depthformer"]["decoder"]["embedder"]["embedding"]["embedding"][...] = jnp.asarray(
        df["decoder"]["decoder_embedding"]["embedding"]["embedding"]
    )

    # 2.5 Depth Input Adapter
    if (
        "depth_input_adapter" in state_dict["depthformer"]["decoder"]
        and "depth_input_adapter" in df["decoder"]["depth_body"]
    ):
        state_dict["depthformer"]["decoder"]["depth_input_adapter"]["kernel"][...] = jnp.asarray(
            df["decoder"]["depth_body"]["depth_input_adapter"]["kernel"]
        )

    # 3. Temporal Transformer
    load_transformer_state_dict(
        state_dict["depthformer"]["decoder"]["temporal"],
        df["decoder"]["temporal_body"]["transformer"],
        has_sinks=True,
    )

    # 4. Depth Transformer
    load_transformer_state_dict(
        state_dict["depthformer"]["decoder"]["depth"],
        df["decoder"]["depth_body"]["transformer"],
        has_sinks=False,
    )

    # 5. Decoder Tail
    dt = df["decoder"]["depth_body"]
    state_dict["depthformer"]["decoder"]["final_ln"]["scale"][...] = jnp.asarray(
        dt["final_ln"]["scale"]
    )
    if "bias" in dt["final_ln"] and "bias" in state_dict["depthformer"]["decoder"]["final_ln"]:
        state_dict["depthformer"]["decoder"]["final_ln"]["bias"][...] = jnp.asarray(
            dt["final_ln"]["bias"]
        )

    state_dict["depthformer"]["decoder"]["to_logits"]["kernel"][...] = jnp.asarray(
        dt["to_logits"]["kernel"]
    )
    if "bias" in dt["to_logits"] and "bias" in state_dict["depthformer"]["decoder"]["to_logits"]:
        state_dict["depthformer"]["decoder"]["to_logits"]["bias"][...] = jnp.asarray(
            dt["to_logits"]["bias"]
        )


def load_transformer_state_dict(transformer_state: dict, transformer_subdict: Mapping, *, has_sinks: bool) -> None:
    layers = transformer_subdict
    is_scanned = "0" not in transformer_state["layers"]

    if is_scanned:
        # layers variable JAX arrays are stacked: assign using slice indexing
        for i in range(len(layers)):
            key = f"x_layers_{i}"
            ld = layers[key]
            _load_attention_state_dict_at(transformer_state["layers"]["self_attn"], ld["self_attention"], i, has_sinks=has_sinks)
            if "cross_attn" in transformer_state["layers"] and transformer_state["layers"]["cross_attn"] is not None:
                _load_attention_state_dict_at(transformer_state["layers"]["cross_attn"], ld["cross_attention"], i, has_sinks=has_sinks)
            _load_ffn_state_dict_at(transformer_state["layers"]["ffn"], ld["ffn"], i)
    else:
        # layers variable is standard Python dict/nnx.List of individual blocks
        for i in range(len(layers)):
            key = f"x_layers_{i}"
            ld = layers[key]
            str_i = str(i)
            _load_attention_state_dict(transformer_state["layers"][str_i]["self_attn"], ld["self_attention"], has_sinks=has_sinks)
            if "cross_attn" in transformer_state["layers"][str_i] and transformer_state["layers"][str_i]["cross_attn"] is not None:
                _load_attention_state_dict(transformer_state["layers"][str_i]["cross_attn"], ld["cross_attention"], has_sinks=has_sinks)
            _load_ffn_state_dict(transformer_state["layers"][str_i]["ffn"], ld["ffn"])


def _load_attention_state_dict(pure_attn: dict, inner_subdict: Mapping, *, has_sinks: bool) -> None:
    pure_attn["pre_norm"]["scale"][...] = jnp.asarray(inner_subdict["pre_norm"]["scale"])
    pure_attn["post_norm"]["scale"][...] = jnp.asarray(inner_subdict["post_norm"]["scale"])

    attn = pure_attn["attention"]
    inner = inner_subdict["attention"]

    q_kernel = jnp.asarray(inner["query_projection"]["kernel"])
    k_kernel = jnp.asarray(inner["key_projection"]["kernel"])
    v_kernel = jnp.asarray(inner["value_projection"]["kernel"])

    in_q, n_heads, head_dim = q_kernel.shape
    in_kv = k_kernel.shape[0]
    q_proj = q_kernel.reshape(in_q, n_heads * head_dim)
    kv_proj = jnp.concatenate(
        [k_kernel.reshape(in_kv, n_heads * head_dim),
         v_kernel.reshape(in_kv, n_heads * head_dim)],
        axis=-1,
    )

    attn["q_proj"]["kernel"][...] = q_proj
    attn["kv_proj"]["kernel"][...] = kv_proj
    attn["per_dim_scale_param"][...] = jnp.asarray(inner["per_dim_scale"])

    if has_sinks and "sink_key_embeddings" in attn:
        attn["sink_key_embeddings"][...] = jnp.asarray(inner["sink_key_embeddings"])
        attn["sink_value_embeddings"][...] = jnp.asarray(inner["sink_value_embeddings"])

    attn["output_projection"]["kernel"][...] = jnp.asarray(inner_subdict["output_projection"]["kernel"])


def _load_attention_state_dict_at(pure_attn: dict, inner_subdict: Mapping, index: int, *, has_sinks: bool) -> None:
    pure_attn["pre_norm"]["scale"][...] = pure_attn["pre_norm"]["scale"][...].at[index].set(
        jnp.asarray(inner_subdict["pre_norm"]["scale"])
    )
    pure_attn["post_norm"]["scale"][...] = pure_attn["post_norm"]["scale"][...].at[index].set(
        jnp.asarray(inner_subdict["post_norm"]["scale"])
    )

    attn = pure_attn["attention"]
    inner = inner_subdict["attention"]

    q_kernel = jnp.asarray(inner["query_projection"]["kernel"])
    k_kernel = jnp.asarray(inner["key_projection"]["kernel"])
    v_kernel = jnp.asarray(inner["value_projection"]["kernel"])

    in_q, n_heads, head_dim = q_kernel.shape
    in_kv = k_kernel.shape[0]
    q_proj = q_kernel.reshape(in_q, n_heads * head_dim)
    kv_proj = jnp.concatenate(
        [k_kernel.reshape(in_kv, n_heads * head_dim),
         v_kernel.reshape(in_kv, n_heads * head_dim)],
        axis=-1,
    )

    attn["q_proj"]["kernel"][...] = attn["q_proj"]["kernel"][...].at[index].set(q_proj)
    attn["kv_proj"]["kernel"][...] = attn["kv_proj"]["kernel"][...].at[index].set(kv_proj)
    attn["per_dim_scale_param"][...] = attn["per_dim_scale_param"][...].at[index].set(
        jnp.asarray(inner["per_dim_scale"])
    )

    if has_sinks and "sink_key_embeddings" in attn:
        attn["sink_key_embeddings"][...] = attn["sink_key_embeddings"][...].at[index].set(
            jnp.asarray(inner["sink_key_embeddings"])
        )
        attn["sink_value_embeddings"][...] = attn["sink_value_embeddings"][...].at[index].set(
            jnp.asarray(inner["sink_value_embeddings"])
        )

    attn["output_projection"]["kernel"][...] = attn["output_projection"]["kernel"][...].at[index].set(
        jnp.asarray(inner_subdict["output_projection"]["kernel"])
    )


def _load_ffn_state_dict(pure_ffn: dict, ffn_subdict: Mapping) -> None:
    pure_ffn["pre_norm"]["scale"][...] = jnp.asarray(ffn_subdict["pre_norm"]["scale"])
    pure_ffn["post_norm"]["scale"][...] = jnp.asarray(ffn_subdict["post_norm"]["scale"])

    l1 = ffn_subdict["ffn_layer1"]
    pure_ffn["ffn_layer1"]["kernel"][...] = jnp.asarray(l1["kernel"])
    pure_ffn["ffn_layer1"]["bias"][...] = jnp.asarray(l1["bias"])

    l2 = ffn_subdict["ffn_layer2"]
    pure_ffn["ffn_layer2"]["kernel"][...] = jnp.asarray(l2["kernel"])
    pure_ffn["ffn_layer2"]["bias"][...] = jnp.asarray(l2["bias"])


def _load_ffn_state_dict_at(pure_ffn: dict, ffn_subdict: Mapping, index: int) -> None:
    pure_ffn["pre_norm"]["scale"][...] = pure_ffn["pre_norm"]["scale"][...].at[index].set(
        jnp.asarray(ffn_subdict["pre_norm"]["scale"])
    )
    pure_ffn["post_norm"]["scale"][...] = pure_ffn["post_norm"]["scale"][...].at[index].set(
        jnp.asarray(ffn_subdict["post_norm"]["scale"])
    )

    l1 = ffn_subdict["ffn_layer1"]
    pure_ffn["ffn_layer1"]["kernel"][...] = pure_ffn["ffn_layer1"]["kernel"][...].at[index].set(
        jnp.asarray(l1["kernel"])
    )
    pure_ffn["ffn_layer1"]["bias"][...] = pure_ffn["ffn_layer1"]["bias"][...].at[index].set(
        jnp.asarray(l1["bias"])
    )

    l2 = ffn_subdict["ffn_layer2"]
    pure_ffn["ffn_layer2"]["kernel"][...] = pure_ffn["ffn_layer2"]["kernel"][...].at[index].set(
        jnp.asarray(l2["kernel"])
    )
    pure_ffn["ffn_layer2"]["bias"][...] = pure_ffn["ffn_layer2"]["bias"][...].at[index].set(
        jnp.asarray(l2["bias"])
    )
