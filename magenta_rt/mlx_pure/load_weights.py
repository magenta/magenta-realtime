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

"""Weight loading for `mlx_pure` modules.

Production callers load a Linen safetensors checkpoint into the
existing sl-backed module tree via :mod:`magenta_rt.mlx.load_weights`,
then mirror the parameters into a structurally-matching `mlx_pure`
tree via :func:`load_weights_from_combinator` (or its narrower siblings
:func:`load_via_bridge` / :func:`mirror_params`). This reuses the
well-tested key-mapping logic in `magenta_rt.mlx.load_weights` and
keeps `mlx_pure`'s own runtime free of any `sequence_layers` imports.

The user is expected to have already constructed the pure module
tree with shapes matching the checkpoint.
"""

from __future__ import annotations

from typing import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .spectrostream.load_weights import (
    load_quantizer_weights,
    load_spectrostream_weights,
    load_spectrostream_decoder_weights,
    load_spectrostream_encoder_weights,
)


# ---------------------------------------------------------------------------
# Flat-leaf parameter mirroring
# ---------------------------------------------------------------------------


def _flat_params(module: nn.Module) -> dict[str, mx.array]:
    return dict(tree_flatten(module.parameters()))


def mirror_params(
    src: nn.Module,
    dst: nn.Module,
    *,
    name_map: Mapping[str, str] | None = None,
    transform: Mapping[str, Callable] | None = None,
    strict: bool = True,
) -> None:
    """Copy parameters from ``src`` into ``dst``.

    By default copies by exact name. Use ``name_map`` to rename
    individual leaf paths (key = name in ``dst``, value = name in
    ``src``); use ``transform`` to apply a per-leaf array transform
    (e.g., transpose).

    Raises if ``strict`` and any ``dst`` leaf has no matching ``src``
    leaf.
    """
    src_flat = _flat_params(src)
    dst_flat = _flat_params(dst)
    name_map = dict(name_map or {})
    transform = dict(transform or {})

    out: dict[str, mx.array] = {}
    missing: list[str] = []
    for dst_name, dst_arr in dst_flat.items():
        src_name = name_map.get(dst_name, dst_name)
        if src_name not in src_flat:
            if strict:
                missing.append(f"{dst_name} (from {src_name})")
            continue
        arr = src_flat[src_name]
        fn = transform.get(dst_name)
        if fn is not None:
            arr = fn(arr)
        if arr.shape != dst_arr.shape:
            raise ValueError(
                f"shape mismatch for '{dst_name}': src {arr.shape} vs dst {dst_arr.shape}"
            )
        out[dst_name] = arr.astype(dst_arr.dtype)

    if missing:
        raise KeyError(f"missing src params for dst leaves: {missing}")

    dst.update(tree_unflatten(list(out.items())))


def load_via_bridge(
    pure_module: nn.Module,
    sl_module: nn.Module,
    *,
    name_map: Mapping[str, str] | None = None,
    transform: Mapping[str, Callable[[mx.array], mx.array]] | None = None,
    strict: bool = True,
) -> None:
    """Bridge-load: copy parameters from an already-loaded sl module
    tree into a structurally-matching pure module tree.

    Typical caller pattern::

        from magenta_rt.mlx.load_weights import load_weights as sl_load
        from magenta_rt.mlx_pure.load_weights import load_via_bridge
        sl_load(sl_system, ckpt_path)
        load_via_bridge(pure_system, sl_system)
    """
    mirror_params(
        sl_module, pure_module,
        name_map=name_map, transform=transform, strict=strict,
    )



# ---------------------------------------------------------------------------
# Random-weight initialization (for demo / untrained equivalence checks)
# ---------------------------------------------------------------------------


def init_random_params(
    module: nn.Module,
    *,
    seed: int = 0,
    std: float = 0.02,
    only_zeros: bool = True,
) -> None:
    """Replace all-zero parameter leaves with normal-distributed samples.

    Pure modules construct their parameter arrays via ``mx.zeros`` so an
    untouched tree produces zero output through downstream nonlinearities.
    For demonstration runs (no real checkpoint), this fills those zeros
    with ``Normal(0, std)`` samples deterministically from ``seed``.

    Args:
        module: any ``nn.Module`` (typically the pure :class:`MagentaRT2Sampler`).
        seed: PRNG seed for reproducibility.
        std: standard deviation of the normal samples.
        only_zeros: when True (default), only overwrite leaves that are
            currently exactly zero. Set to False to randomize every leaf.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(module.parameters()))
    key = mx.random.key(seed)
    new = {}
    for i, (name, arr) in enumerate(sorted(flat.items())):
        sub = mx.random.split(key, len(flat) + 1)[i]
        if only_zeros:
            is_zero = bool(mx.all(arr == 0).item()) if arr.size > 0 else True
            if not is_zero:
                continue
        sample = mx.random.normal(arr.shape, dtype=mx.float32, key=sub) * std
        new[name] = sample.astype(arr.dtype)
    if new:
        module.update(tree_unflatten(list(new.items())))



# ---------------------------------------------------------------------------
# Per-subsystem sl → pure parameter loading helpers
# ---------------------------------------------------------------------------
#
# Each helper takes a target pure module and the matching sl module and
# writes the kernel/bias/scale arrays directly. They're independent of
# one another so callers can mirror only the parts they care about
# (useful for narrow parity tests and for extending to new model specs).


def _get_inner(layer):
    """Unwrap sl's ``Deferred`` / similar wrapper to expose the actual
    parameter-bearing layer (``inner`` attribute).
    """
    if hasattr(layer, "inner") and layer.inner is not None:
        return layer.inner
    return layer


def load_encoder_embedding_weights(pure_encoder, sl_encoder_body) -> None:
    """Pure encoder embedding + ``encoder_ln`` (``LayerNorm``) ← sl's
    encoder.body Serial.

    Two embedding layouts are supported, discriminated by the pure side:

    * **Branched** (``mrt2`` pretrained-MusicCoCa): pure
      ``BranchedEncoderEmbedding``. sl layout is
      ``body.layers[0]`` = ``Parallel[mulan_branch, regular_branch]`` where
      ``mulan_branch`` = ``Serial[crop, Serial[offset, Embedding(mulan_dequantizer),
      sum, Dense(depth_input_adapter)]]`` and ``regular_branch`` =
      ``Serial[crop, MultiChannelEmbedding]``.
    * **Plain** (e.g. tiny): pure ``MultiChannelEmbedding`` ←
      ``body.layers[0]``.
    """
    pure_emb = pure_encoder.embedding
    if hasattr(pure_emb, "mulan_embedder"):
        parallel = sl_encoder_body.layers[0]
        mulan_branch = parallel.layers[0]
        regular_branch = parallel.layers[1]
        # mulan_branch: Serial[crop_Lambda, mulan_embedder_Serial]
        mulan_serial = _get_inner(mulan_branch.layers[1])
        sl_dequant = _get_inner(mulan_serial.layers[1])  # Embedding
        sl_adapter = _get_inner(mulan_serial.layers[3])  # Dense
        pure_emb.mulan_embedder.mulan_dequantizer.weight = sl_dequant._embedding.weight
        pure_emb.mulan_embedder.depth_input_adapter.linear.weight = sl_adapter._linear.weight
        if getattr(sl_adapter._linear, "bias", None) is not None:
            pure_emb.mulan_embedder.depth_input_adapter.linear.bias = sl_adapter._linear.bias
        # regular_branch: Serial[crop_Lambda, MultiChannelEmbedding]
        sl_regular = _get_inner(regular_branch.layers[1])
        pure_emb.regular_embedder.embedding = sl_regular.embedding
    else:
        sl_emb_layer = sl_encoder_body.layers[0]
        pure_emb.embedding = sl_emb_layer.embedding

    sl_enc_ln = None
    for l in sl_encoder_body.layers:
        if type(l).__name__ == "LayerNormalization":
            sl_enc_ln = l
            break
    if sl_enc_ln is not None:
        ln_inner = sl_enc_ln._layer_norm if hasattr(sl_enc_ln, "_layer_norm") else sl_enc_ln
        pure_ln = pure_encoder.encoder_ln
        if hasattr(ln_inner, "weight"):
            pure_ln.weight = ln_inner.weight
        if hasattr(ln_inner, "bias") and getattr(ln_inner, "bias", None) is not None:
            pure_ln.bias = ln_inner.bias


def load_decoder_embedder_weights(pure_embedder, sl_embedder_serial) -> None:
    """Pure ``ScaledEmbedding.embedding.weight`` ← sl ``Serial[Embedding,
    Scale]``'s mx.nn.Embedding row-major matrix at
    ``layers[0]._embedding.weight``.
    """
    sl_emb = sl_embedder_serial.layers[0]
    pure_embedder.embedding.weight = sl_emb._embedding.weight


def _load_attn_residual_weights(sl_residual, pure_attn_block) -> None:
    """Mirror one attention Residual into a pure SelfAttn / CrossAttn
    block. sl body layout:
    ``[RMSNorm(pre), Attention, EinsumDense(output), CheckpointName,
    RMSNorm(post), Dropout]``.
    """
    body = sl_residual.body.layers
    sl_pre = _get_inner(body[0])
    sl_attn = _get_inner(body[1])
    sl_out = _get_inner(body[2])
    sl_post = _get_inner(body[4])

    if hasattr(sl_pre, "_rms_norm") and sl_pre._rms_norm is not None:
        pure_attn_block.pre_norm.weight = sl_pre._rms_norm.weight
    if hasattr(sl_post, "_rms_norm") and sl_post._rms_norm is not None:
        pure_attn_block.post_norm.weight = sl_post._rms_norm.weight

    attn = pure_attn_block.attention
    attn.q_proj = sl_attn.q_proj
    attn.kv_proj = sl_attn.kv_proj
    if hasattr(sl_attn, "_per_dim_scale"):
        attn.per_dim_scale = sl_attn._per_dim_scale
    if hasattr(sl_attn, "sink_key_embeddings") and sl_attn.sink_key_embeddings is not None:
        attn.sink_key_embeddings = sl_attn.sink_key_embeddings
    if hasattr(sl_attn, "sink_value_embeddings") and sl_attn.sink_value_embeddings is not None:
        attn.sink_value_embeddings = sl_attn.sink_value_embeddings

    if hasattr(sl_out, "kernel"):
        attn.output_projection.kernel = sl_out.kernel


def _load_ffn_residual_weights(sl_residual, pure_ffn) -> None:
    """Mirror one FFN Residual into a pure FFN block. sl body:
    ``[RMSNorm(pre), Dense(ffn1), Dropout, Dense(ffn2), CheckpointName,
    RMSNorm(post), Dropout]``.
    """
    body = sl_residual.body.layers
    sl_pre = _get_inner(body[0])
    sl_post = _get_inner(body[5])
    if hasattr(sl_pre, "_rms_norm") and sl_pre._rms_norm is not None:
        pure_ffn.pre_norm.weight = sl_pre._rms_norm.weight
    if hasattr(sl_post, "_rms_norm") and sl_post._rms_norm is not None:
        pure_ffn.post_norm.weight = sl_post._rms_norm.weight

    sl_denses = [l for l in body if "Dense" in type(l).__name__]
    d1 = _get_inner(sl_denses[0])
    d2 = _get_inner(sl_denses[1])
    pure_ffn.ffn_layer1.linear.weight = d1._linear.weight
    pure_ffn.ffn_layer1.linear.bias = d1._linear.bias
    pure_ffn.ffn_layer2.linear.weight = d2._linear.weight
    pure_ffn.ffn_layer2.linear.bias = d2._linear.bias


def load_transformer_weights(pure_xformer, sl_xformer) -> None:
    """Mirror an entire transformer stack (temporal or depth)."""
    for i, pure_blk in enumerate(pure_xformer.layers):
        sl_blk = sl_xformer.layers[i]
        residuals = [l for l in sl_blk.layers if type(l).__name__ == "Residual"]
        if len(residuals) == 3:
            _load_attn_residual_weights(residuals[0], pure_blk.self_attn)
            if pure_blk.cross_attn is not None:
                _load_attn_residual_weights(residuals[1], pure_blk.cross_attn)
            _load_ffn_residual_weights(residuals[2], pure_blk.ffn)
        elif len(residuals) == 2:
            _load_attn_residual_weights(residuals[0], pure_blk.self_attn)
            _load_ffn_residual_weights(residuals[1], pure_blk.ffn)
        else:
            raise ValueError(f"unexpected residual count {len(residuals)}")


def load_decoder_tail_weights(pure_decoder, sl_depth_body) -> None:
    """Mirror the depth_input_adapter, final_ln, and to_logits at the
    tail of the depth body Serial."""
    if pure_decoder.depth_input_adapter is not None and len(sl_depth_body.layers) > 0:
        sl_adapter = _get_inner(sl_depth_body.layers[0])
        pure_decoder.depth_input_adapter.linear.weight = sl_adapter._linear.weight
        if hasattr(sl_adapter, "_linear") and getattr(sl_adapter._linear, "bias", None) is not None:
            pure_decoder.depth_input_adapter.linear.bias = sl_adapter._linear.bias
    if len(sl_depth_body.layers) > 2:
        sl_fln = _get_inner(sl_depth_body.layers[2])
        inner_ln = sl_fln._layer_norm if hasattr(sl_fln, "_layer_norm") else sl_fln
        if inner_ln is not None:
            pure_decoder.final_ln.weight = inner_ln.weight
            if getattr(inner_ln, "bias", None) is not None:
                pure_decoder.final_ln.bias = inner_ln.bias
    if len(sl_depth_body.layers) > 3:
        sl_tl = _get_inner(sl_depth_body.layers[3])
        pure_decoder.to_logits.linear.weight = sl_tl._linear.weight
        if hasattr(sl_tl, "_linear") and getattr(sl_tl._linear, "bias", None) is not None:
            pure_decoder.to_logits.linear.bias = sl_tl._linear.bias


def load_depthformer_weights(pure_df, sl_df) -> None:
    """Top-level depthformer weights loading: encoder embedding + LayerNorm,
    decoder embedder, temporal + depth transformers, depth tail."""
    load_encoder_embedding_weights(pure_df.encoder, sl_df.encoder.body)
    load_decoder_embedder_weights(pure_df.decoder.embedder, sl_df.decoder.embedder)
    load_transformer_weights(pure_df.decoder.temporal, sl_df.decoder.temporal_body.layers[0])
    load_transformer_weights(pure_df.decoder.depth, sl_df.decoder.depth_body.layers[1])
    load_decoder_tail_weights(pure_df.decoder, sl_df.decoder.depth_body)


def load_weights_from_combinator(pure_system, sl_system) -> None:
    """Mirror parameters from a built sequence-layers combinator ``MagentaRT2Sampler``
    into a structurally-matching ``mlx_pure.model.MagentaRT2Sampler``.

    Thin orchestrator over the per-subsystem helpers above; a new
    model spec only needs whichever helpers differ in shape.

    Caller is expected to have already populated ``sl_system``'s
    parameter values (via ``magenta_rt.mlx.load_weights.load_weights``
    for real checkpoints, or :func:`init_random_params` on top of
    ``sl_export._materialize_deferred`` for random-init demos).
    """
    print("Loading weights via custom layout mapping...")
    # ---- Depthformer (the critical path for token parity) ----
    load_encoder_embedding_weights(
        pure_system.depthformer.encoder,
        sl_system.depthformer.encoder.body,
    )
    print("  Mapped Encoder Embedding + LayerNorm.")

    load_decoder_embedder_weights(
        pure_system.depthformer.decoder.embedder,
        sl_system.depthformer.decoder.embedder,
    )
    print("  Mapped Decoder Embedder.")

    load_transformer_weights(
        pure_system.depthformer.decoder.temporal,
        sl_system.depthformer.decoder.temporal_body.layers[0],
    )
    print("  Mapped Depthformer temporal transformer.")

    load_transformer_weights(
        pure_system.depthformer.decoder.depth,
        sl_system.depthformer.decoder.depth_body.layers[1],
    )
    print("  Mapped Depthformer depth transformer.")

    load_decoder_tail_weights(
        pure_system.depthformer.decoder,
        sl_system.depthformer.decoder.depth_body,
    )
    print("  Mapped depth_input_adapter / final_ln / to_logits.")

    # ---- SpectroStream codec (only if the pure system carries one) ----
    # main renamed the sl attribute ``soundstream`` -> ``spectrostream``.
    if getattr(pure_system, "spectrostream", None) is not None:
        sl_ss = getattr(sl_system, "spectrostream", None)
        if sl_ss is None:
            sl_ss = sl_system.soundstream  # back-compat
        load_quantizer_weights(pure_system.spectrostream.quantizer, sl_ss.quantizer)
        print("  Mapped quantizer embeddings.")
        load_spectrostream_decoder_weights(
            pure_system.spectrostream.decoder, sl_ss.decoder,
        )
        print("  Mapped SpectroStream decoder.")
        load_spectrostream_encoder_weights(
            pure_system.spectrostream.encoder, sl_ss.encoder, sl_ss.config,
        )
        print("  Mapped SpectroStream encoder.")
    print("Custom layout weight loading complete.")


def _build_loaded_sl_sampler(checkpoint_path: str | Path, model_name: str):
    """Build a structurally-matching sl ``MagentaRT2Sampler`` and load the
    Linen safetensors checkpoint into it.

    Builds the sl sampler exactly as :class:`magenta_rt.mlx.system.MagentaRT2System`
    does, then loads the checkpoint via the well-tested
    :func:`magenta_rt.mlx.load_weights.load_weights`. Shared by the full-system
    and depthformer-only mirroring entry points below.
    """
    from magenta_rt.mlx import model as combinator_model
    from magenta_rt.mlx import spectrostream as combinator_ss
    from magenta_rt.mlx.system import MagentaRT2Sampler
    from magenta_rt.mlx.load_weights import load_weights as combinator_load_weights

    spec = combinator_model.get_model_class(model_name)()
    sl_sampler = MagentaRT2Sampler.Config(
        depthformer=spec.depthformer_config(),
        spectrostream=combinator_ss.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=spec.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        ),
        int16_outputs=False,
    ).make()
    combinator_load_weights(
        sl_sampler, str(checkpoint_path),
        num_input_channels=spec.input_num_channels,
    )
    return sl_sampler


def load_from_safetensors(
    pure_system: nn.Module,
    checkpoint_path: str | Path,
    *,
    model_name: str = "mrt2_small",
) -> None:
    """Load a standard Linen safetensors checkpoint into the pure module
    hierarchy by way of a structurally-matching sl ``MagentaRT2Sampler``.

    Builds the sl sampler, loads the checkpoint into it, then mirrors the
    parameters (depthformer **and** SpectroStream codec) into ``pure_system``.
    """
    sl_sampler = _build_loaded_sl_sampler(checkpoint_path, model_name)
    load_weights_from_combinator(pure_system, sl_sampler)


def load_sft_depthformer_from_safetensors(
    pure_depthformer: nn.Module,
    checkpoint_path: str | Path,
    *,
    model_name: str = "mrt2_small",
) -> None:
    """Load a pretrained depthformer into an SFT-trainable ``EncoderDecoder``.

    This is the depthformer-only counterpart to :func:`load_from_safetensors`:
    the SFT trainer's model is ``spec.build_decoder()`` (an
    :class:`~magenta_rt.mlx_pure.depthformer.EncoderDecoder`, no codec), so this
    builds the sl sampler, loads the real checkpoint into it, then mirrors only
    the depthformer weights — encoder embedding + LayerNorm, decoder embedder,
    temporal + depth transformers, and the depth tail. The SpectroStream codec
    is skipped (training never needs it; audio sampling builds a separate full
    system). Mirrors :func:`magenta_rt.sft.checkpoint.load_nnx_depthformer_from_safetensors`
    on the NNX side.
    """
    sl_sampler = _build_loaded_sl_sampler(checkpoint_path, model_name)
    load_depthformer_weights(pure_depthformer, sl_sampler.depthformer)


# ---------------------------------------------------------------------------
# DIRECT Linen → pure loader (no sl bridge, no fp32 sl model in memory)
# ---------------------------------------------------------------------------
#
# The sl-bridge path is a *composition* of two transforms:
#
#   1. ``magenta_rt.mlx.load_weights.load_weights``  (Linen flat dict → sl
#      module attributes) — this is where ALL the non-identity reshapes live:
#        * Dense kernels are transposed: pure/sl ``linear.weight`` = Linen
#          ``kernel.T`` (``[in, out]`` → ``[out, in]``).
#        * Attention Q/K/V kernels ``[in, heads, uph]`` are reshaped to
#          ``[in, heads*uph]``; K and V are concatenated into a single
#          ``kv_proj`` ``[in, 2*heads*uph]`` (matching the pure
#          ``LocalSelfAttention.kv_proj`` / cross-attn ``QueryAndKeyValue``
#          layout).
#        * The attention output projection (EinsumDense ``...nh,dnh->...d``)
#          kernel is copied verbatim (NO transpose) — ``[d, heads, uph]``.
#        * Norm scales / biases are copied verbatim.
#        * The branched encoder embedding reads three Linen keys (mulan
#          dequantizer embedding, mulan depth_input_adapter kernel, regular
#          embedder embedding).
#   2. ``load_depthformer_weights`` (sl attr → pure param) — pure-side mirror,
#      which is **pure pass-through identity** for every leaf (it just copies
#      ``sl_attn.q_proj`` → ``pure.attention.q_proj`` etc.).
#
# Because step 2 is identity, the direct Linen → pure transform IS step 1's
# transform written straight into the pure module tree. We walk the pure
# module structure (the same attribute paths ``load_depthformer_weights``
# uses) and assign arrays read directly from the Linen flat dict, applying
# step 1's reshapes inline. No sl model is ever constructed, so the peak is
# ~model size instead of ~model + fp32-sl.
#
# Every assignment casts to the destination param's existing dtype
# (``arr.astype(dst.dtype)``), matching the sl-bridge ``mirror_params`` cast
# so the result is bit-identical (bf16 in, bf16 out).


class _LazyLinen:
    """Lazy nested view over a Linen safetensors file (``safe_open`` handle).

    The direct loader navigates the checkpoint purely by key indexing
    (``d["params"]["depthformer"][...]``) and ``in`` checks — it never
    iterates keys (layer loops are driven by the *pure* module's
    ``enumerate``). So this only needs ``__getitem__`` (subtree or leaf) and
    ``__contains__``.

    Crucially each leaf tensor is read from disk **on access** via
    ``get_tensor`` and not retained, so the full fp32 checkpoint is never
    resident at once: peak ≈ the bf16 module + one fp32 tensor (~hundreds of
    MB), not module + the whole ~10 GB fp32 dict. This is what lets
    ``mrt2_base`` load under ~6 GB instead of thrashing a 16 GB Mac.
    """

    def __init__(self, handle, keys, prefix=()):
        self._h = handle
        self._keys = keys           # set[str] of all flat "a/b/c" keys
        self._prefix = prefix       # tuple path to this subtree

    def __contains__(self, key) -> bool:
        flat = "/".join(self._prefix + (key,))
        return flat in self._keys or any(
            k.startswith(flat + "/") for k in self._keys
        )

    def __getitem__(self, key):
        path = self._prefix + (key,)
        flat = "/".join(path)
        if flat in self._keys:                       # leaf → read one tensor
            return self._h.get_tensor(flat)
        if any(k.startswith(flat + "/") for k in self._keys):  # subtree
            return _LazyLinen(self._h, self._keys, path)
        raise KeyError(flat)


def _np_load_linen(checkpoint_path):
    """Lazy nested view of the Linen safetensors (numpy, no JAX/flax import).

    Returns a :class:`_LazyLinen` backed by a ``safe_open`` handle; tensors
    are read per-leaf on access (memory-safe for ``mrt2_base``), not all at
    once. ``safetensors.numpy`` is never used, so no device arrays and no
    ~10 GB fp32 dict are materialized.
    """
    from safetensors import safe_open

    handle = safe_open(str(checkpoint_path), framework="numpy")
    return _LazyLinen(handle, set(handle.keys()))


def _reshape_proj(kernel):
    """Attention projection kernel ``[in, heads, uph]`` → ``[in, heads*uph]``."""
    import numpy as np

    in_dim = kernel.shape[0]
    out_dim = int(np.prod(kernel.shape[1:]))
    return kernel.reshape(in_dim, out_dim)


def _set(dst_owner, attr, arr):
    """Assign ``arr`` (numpy) to ``dst_owner.<attr>``, casting to the dst
    param's existing dtype (matches the sl-bridge ``astype(dst.dtype)`` so
    results are bit-identical). Raises on shape mismatch.
    """
    cur = getattr(dst_owner, attr)
    new = mx.array(np.asarray(arr))
    if cur is not None and tuple(new.shape) != tuple(cur.shape):
        raise ValueError(
            f"shape mismatch for {type(dst_owner).__name__}.{attr}: "
            f"src {tuple(new.shape)} vs dst {tuple(cur.shape)}"
        )
    if cur is not None:
        new = new.astype(cur.dtype)
    # Materialize NOW so the fp32 source (the just-read Linen tensor) is freed
    # before the next leaf is read. Without this, MLX laziness keeps every
    # fp32 source alive in the graph until the load's final mx.eval — i.e. the
    # whole ~10 GB fp32 checkpoint stays resident, defeating the per-tensor
    # streaming and thrashing a 16 GB Mac on mrt2_base. With it, the load peak
    # is ~the bf16 module + one fp32 tensor.
    mx.eval(new)
    setattr(dst_owner, attr, new)


def _direct_load_attention(pure_attn, jax_attn) -> None:
    """Pure ``LocalSelfAttention`` / ``StreamingCrossAttention`` ← Linen
    attention params. Composes the QKV reshape+concat from
    ``mlx.load_weights._load_attention_weights`` with the identity mirror in
    ``mlx_pure.load_weights._load_attn_residual_weights``.
    """
    if "query_key_value_projection" in jax_attn:  # combined QKV
        qkv = np.asarray(jax_attn["query_key_value_projection"]["kernel"])
        q_kernel, k_kernel, v_kernel = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    elif "key_value_projection" in jax_attn:  # separate Q, combined KV
        q_kernel = np.asarray(jax_attn["query_projection"]["kernel"])
        kv = np.asarray(jax_attn["key_value_projection"]["kernel"])
        k_kernel, v_kernel = kv[:, 0], kv[:, 1]
    elif "shared_key_value_projection" in jax_attn:
        q_kernel = np.asarray(jax_attn["query_projection"]["kernel"])
        shared = np.asarray(jax_attn["shared_key_value_projection"]["kernel"])
        k_kernel = v_kernel = shared
    else:  # separate Q/K/V
        q_kernel = np.asarray(jax_attn["query_projection"]["kernel"])
        k_kernel = np.asarray(jax_attn["key_projection"]["kernel"])
        v_kernel = np.asarray(jax_attn["value_projection"]["kernel"])

    q_flat = _reshape_proj(q_kernel)
    k_flat = _reshape_proj(k_kernel)
    v_flat = _reshape_proj(v_kernel)

    # Pure attention always stores q_proj + a combined kv_proj.
    _set(pure_attn, "q_proj", q_flat)
    _set(pure_attn, "kv_proj", np.concatenate([k_flat, v_flat], axis=-1))

    if "per_dim_scale" in jax_attn:
        _set(pure_attn, "per_dim_scale", np.asarray(jax_attn["per_dim_scale"]))
    if "sink_key_embeddings" in jax_attn:
        _set(pure_attn, "sink_key_embeddings",
             np.asarray(jax_attn["sink_key_embeddings"]))
    if "sink_value_embeddings" in jax_attn:
        _set(pure_attn, "sink_value_embeddings",
             np.asarray(jax_attn["sink_value_embeddings"]))


def _direct_load_attn_block(pure_block, jax_params) -> None:
    """One self/cross attention block: pre/post RMS, attention, output proj.

    ``jax_params`` is the Linen ``self_attention`` / ``cross_attention``
    subtree. Mirrors ``mlx.load_weights._load_attn_residual`` (the source of
    the EinsumDense ``output_projection`` verbatim-copy and the RMS scales)
    composed with the identity pure mirror.
    """
    _set(pure_block.pre_norm, "weight", np.asarray(jax_params["pre_norm"]["scale"]))
    _set(pure_block.post_norm, "weight", np.asarray(jax_params["post_norm"]["scale"]))
    _direct_load_attention(pure_block.attention, jax_params["attention"])
    # EinsumDense output projection: kernel copied verbatim ([d, heads, uph]).
    _set(pure_block.attention.output_projection, "kernel",
         np.asarray(jax_params["output_projection"]["kernel"]))


def _direct_load_dense(pure_dense, jax_params) -> None:
    """Pure ``Dense`` (wraps ``nn.Linear``) ← Linen Dense. The kernel is
    TRANSPOSED (``[in, out]`` → ``[out, in]``), matching
    ``mlx.load_weights._load_dense``.

    A Linen Dense may carry a ``bias`` even when the pure ``Dense`` was built
    ``bias=False`` (e.g. ``to_logits``): the sl-bridge mirror assigns the bias
    onto ``linear`` unconditionally in that case, so to stay bit-identical we
    create the attribute here too. The bias inherits the kernel's dtype (the
    sl ``convert_to_bf16`` pass casts it alongside the weight).
    """
    kernel = np.asarray(jax_params["kernel"])
    _set(pure_dense.linear, "weight", kernel.T)
    if "bias" in jax_params:
        bias = mx.array(np.asarray(jax_params["bias"]))
        cur = getattr(pure_dense.linear, "bias", None)
        # Match the sl-bridge dtype: the weight's dtype after the bf16 pass.
        bias = bias.astype(pure_dense.linear.weight.dtype if cur is None else cur.dtype)
        pure_dense.linear.bias = bias


def _direct_load_ffn(pure_ffn, jax_params) -> None:
    """FFN block: pre/post RMS + two transposed Dense layers."""
    _set(pure_ffn.pre_norm, "weight", np.asarray(jax_params["pre_norm"]["scale"]))
    _set(pure_ffn.post_norm, "weight", np.asarray(jax_params["post_norm"]["scale"]))
    _direct_load_dense(pure_ffn.ffn_layer1, jax_params["ffn_layer1"])
    _direct_load_dense(pure_ffn.ffn_layer2, jax_params["ffn_layer2"])


def _direct_load_transformer(pure_xformer, jax_xformer) -> None:
    """Stack of transformer blocks ← Linen ``x_layers_{i}`` subtrees."""
    for i, pure_blk in enumerate(pure_xformer.layers):
        jax_blk = jax_xformer[f"x_layers_{i}"]
        _direct_load_attn_block(pure_blk.self_attn, jax_blk["self_attention"])
        if pure_blk.cross_attn is not None and "cross_attention" in jax_blk:
            _direct_load_attn_block(pure_blk.cross_attn, jax_blk["cross_attention"])
        _direct_load_ffn(pure_blk.ffn, jax_blk["ffn"])


def _direct_load_encoder_embedding(pure_encoder, jax_encoder_body) -> None:
    """Pure encoder embedding + ``encoder_ln`` ← Linen ``encoder.body``.

    Composes ``mlx.load_weights.load_weights``'s branched-encoder block
    (three Linen keys) with the identity pure mirror.
    """
    pure_emb = pure_encoder.embedding
    if "layers_1" in jax_encoder_body:  # branched (pretrained-MusicCoCa)
        jax_branched = jax_encoder_body["layers_1"]
        jax_mulan = jax_branched["branched_mulan_embedder"]["mulan_embedder"]
        jax_regular = (
            jax_branched["branched_regular_embedder"]["regular_embedder"]
        )
        if not hasattr(pure_emb, "mulan_embedder"):
            raise ValueError(
                "checkpoint has a branched encoder embedding but the pure "
                "module is plain (no mulan_embedder)"
            )
        _set(pure_emb.mulan_embedder.mulan_dequantizer, "weight",
             np.asarray(jax_mulan["mulan_dequantizer"]["embedding"]))
        _direct_load_dense(
            pure_emb.mulan_embedder.depth_input_adapter,
            jax_mulan["depth_input_adapter"],
        )
        _set(pure_emb.regular_embedder, "embedding",
             np.asarray(jax_regular["embedding"]))
    else:  # plain MultiChannelEmbedding
        if hasattr(pure_emb, "mulan_embedder"):
            raise ValueError(
                "pure module has a branched encoder embedding but the "
                "checkpoint is plain"
            )
        # The plain embedding table lives at body.layers[0] on the sl side;
        # the Linen key for a non-branched MultiChannelEmbedding is the
        # embedder under layers_1 (Logging occupies layers_0). Resolve it.
        jax_plain = None
        for v in jax_encoder_body.values():
            if isinstance(v, dict) and "embedding" in v:
                jax_plain = v
                break
        if jax_plain is None:
            raise ValueError("could not locate plain encoder embedding key")
        _set(pure_emb, "embedding", np.asarray(jax_plain["embedding"]))

    # encoder_ln (LayerNorm: scale + bias).
    jax_ln = jax_encoder_body["encoder_ln"]
    _set(pure_encoder.encoder_ln, "weight", np.asarray(jax_ln["scale"]))
    if "bias" in jax_ln and getattr(pure_encoder.encoder_ln, "bias", None) is not None:
        _set(pure_encoder.encoder_ln, "bias", np.asarray(jax_ln["bias"]))


def _match_sl_bf16_dtypes(pure_depthformer: nn.Module) -> None:
    """Set the pure depthformer's param dtypes to exactly what the sl-bridge
    path produces after its ``convert_to_bf16`` pass: **bf16 for every leaf
    except** the attention ``per_dim_scale`` arrays, which stay fp32.

    On the sl side ``per_dim_scale`` lives at ``inner._per_dim_scale`` (a
    private, unregistered attribute) so ``mlx.load_weights.convert_to_bf16``'s
    ``module.parameters()`` walk never reaches it — it remains fp32, and the
    pure mirror copies it verbatim. Reproducing that one blind spot here is
    what makes the direct load bit-identical to the bridge. We mutate the
    destination dtypes *before* copying so the per-leaf ``astype(dst.dtype)``
    in :func:`_set` lands on the right type.
    """
    import mlx.utils

    def _is_per_dim(path: str) -> bool:
        return path.endswith("per_dim_scale")

    flat = dict(mlx.utils.tree_flatten(pure_depthformer.parameters()))
    new = {}
    for path, arr in flat.items():
        if _is_per_dim(path):
            target = mx.float32
        else:
            target = mx.bfloat16
        if arr.dtype != target:
            new[path] = arr.astype(target)
    if new:
        pure_depthformer.update(mlx.utils.tree_unflatten(list(new.items())))
    mx.eval(pure_depthformer.parameters())


def load_depthformer_from_safetensors_direct(
    pure_depthformer: nn.Module,
    checkpoint_path: str | Path,
    *,
    match_sl_bf16: bool = True,
) -> None:
    """DIRECT Linen → pure depthformer loader (no sl bridge).

    Reads the Linen safetensors flat dict and writes every depthformer
    parameter into ``pure_depthformer`` (an
    :class:`mlx_pure.depthformer.EncoderDecoder`) by applying the same
    composed transform the sl-bridge path produces — without ever
    constructing an sl model. Casts every leaf to the destination param's
    existing dtype (``arr.astype(dst.dtype)``), matching the sl-bridge
    ``mirror_params`` cast so the result is **bit-identical**.

    ``match_sl_bf16`` (default True): before copying, set the destination
    dtypes to exactly what the sl-bridge produces — bf16 everywhere except
    ``per_dim_scale`` (fp32), via :func:`_match_sl_bf16_dtypes`. This makes a
    freshly-built fp32 module bit-identical to the sl-bridge output (which is
    the standard inference / SFT dtype layout). Pass ``False`` to keep the
    module's existing dtypes untouched (e.g. an all-bf16 build, or a pure-fp32
    parity check).

    Peaks at ~model size (one Linen numpy copy + the pure module), not
    ~model + fp32-sl. This is the memory-safe path for ``mrt2_base`` (2.4 B).
    """
    if match_sl_bf16:
        _match_sl_bf16_dtypes(pure_depthformer)
    nested = _np_load_linen(checkpoint_path)
    jax_df = nested["params"]["depthformer"]
    pure_dec = pure_depthformer.decoder

    # Encoder embedding + LayerNorm.
    _direct_load_encoder_embedding(
        pure_depthformer.encoder, jax_df["encoder"]["body"],
    )

    # Decoder token embedder (scaled Embedding) — verbatim copy.
    jax_emb = jax_df["decoder"]["decoder_embedding"]["embedding"]["embedding"]
    _set(pure_dec.embedder.embedding, "weight", np.asarray(jax_emb))

    # Temporal + depth transformers.
    _direct_load_transformer(
        pure_dec.temporal,
        jax_df["decoder"]["temporal_body"]["transformer"],
    )
    _direct_load_transformer(
        pure_dec.depth,
        jax_df["decoder"]["depth_body"]["transformer"],
    )

    # Depth tail: depth_input_adapter (optional), final_ln, to_logits.
    jax_depth = jax_df["decoder"]["depth_body"]
    if pure_dec.depth_input_adapter is not None and "depth_input_adapter" in jax_depth:
        _direct_load_dense(pure_dec.depth_input_adapter,
                           jax_depth["depth_input_adapter"])
    jax_fln = jax_depth["final_ln"]
    _set(pure_dec.final_ln, "weight", np.asarray(jax_fln["scale"]))
    if "bias" in jax_fln and getattr(pure_dec.final_ln, "bias", None) is not None:
        _set(pure_dec.final_ln, "bias", np.asarray(jax_fln["bias"]))
    _direct_load_dense(pure_dec.to_logits, jax_depth["to_logits"])

    mx.eval(pure_depthformer.parameters())


def _read_soundstream_params(checkpoint_path: str | Path):
    """Stream ONLY the ``params/soundstream/*`` subtree of the Linen checkpoint
    into a nested numpy dict — the value :func:`magenta_rt.mlx.spectrostream`
    ``.load_weights.load_spectrostream_weights`` expects as ``soundstream_params``.

    Reproduces ``_load_jax_params(path)['params']['soundstream']`` (slash-split,
    unflattened) but reads only the codec tensors via ``safe_open.get_tensor``,
    so the ~9.6 GB depthformer keys are never touched. The codec is tens of MB.
    """
    from safetensors import safe_open

    handle = safe_open(str(checkpoint_path), framework="numpy")
    prefix = "params/soundstream/"
    nested: dict = {}
    for k in handle.keys():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split("/")
        node = nested
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = handle.get_tensor(k)
    return nested


def _build_loaded_sl_spectrostream(checkpoint_path: str | Path, model_name: str):
    """Build ONLY a standalone sl ``SpectroStream`` and load its codec weights.

    The codec twin of :func:`_build_loaded_sl_sampler`: it builds the same sl
    ``SpectroStream`` the full sampler holds (deferred conv layers — cheap until
    materialized) and loads the quantizer + decoder weights via the well-tested
    :func:`magenta_rt.mlx.spectrostream.load_weights.load_spectrostream_weights`,
    fed only the streamed ``soundstream`` params. Crucially **no depthformer is
    ever built**, so the ~9.6 GB fp32 sl depthformer twin that
    ``_build_loaded_sl_sampler`` allocates (just to extract the codec) is gone —
    the codec-load peak drops to ~tens of MB on top of the resident bf16 pure
    depthformer.

    The returned sl ``SpectroStream`` exposes ``.quantizer`` / ``.decoder`` /
    ``.encoder`` / ``.config`` exactly like the one inside the sampler, so the
    pure codec mirror loaders consume it unchanged. (The encoder is loaded only
    when a sibling ``encoder.safetensors`` exists — same as the sampler path;
    the pure encoder loader self-materializes the sl encoder either way.)
    """
    from magenta_rt.mlx import model as combinator_model
    from magenta_rt.mlx import spectrostream as combinator_ss
    from magenta_rt.mlx.spectrostream.load_weights import load_spectrostream_weights

    spec = combinator_model.get_model_class(model_name)()
    sl_ss = combinator_ss.SpectroStream(
        combinator_ss.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=spec.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        )
    )
    load_spectrostream_weights(
        sl_ss, str(checkpoint_path),
        soundstream_params=_read_soundstream_params(checkpoint_path),
    )
    return sl_ss


def load_from_safetensors_direct(
    pure_module: nn.Module,
    checkpoint_path: str | Path,
    *,
    model_name: str = "mrt2_small",
    depthformer_only: bool = False,
    match_sl_bf16: bool = True,
) -> None:
    """DIRECT loader entry point (no sl bridge for the depthformer).

    * ``depthformer_only=True`` — ``pure_module`` is an
      :class:`mlx_pure.depthformer.EncoderDecoder` (the SFT-trainable model);
      load only the depthformer via the direct path.
    * ``depthformer_only=False`` — ``pure_module`` is a full
      :class:`mlx_pure.model.MagentaRT2Sampler`; load the depthformer directly
      and the SpectroStream codec via the existing sl-bridge helper. (The
      codec is small relative to the depthformer; the bridge mirror for it is
      cheap and already well-tested. The memory hog the direct path eliminates
      is the depthformer's fp32 sl twin.)

    ``model_name`` is accepted for signature symmetry with
    :func:`load_from_safetensors` / :func:`load_sft_depthformer_from_safetensors`;
    the direct depthformer load reads structure off ``pure_module`` and the
    Linen keys, so it does not need the sl spec.
    """
    if depthformer_only:
        load_depthformer_from_safetensors_direct(
            pure_module, checkpoint_path, match_sl_bf16=match_sl_bf16,
        )
        return

    load_depthformer_from_safetensors_direct(
        pure_module.depthformer, checkpoint_path, match_sl_bf16=match_sl_bf16,
    )
    # Codec via a standalone sl SpectroStream (NO fp32 sl depthformer twin).
    if getattr(pure_module, "spectrostream", None) is not None:
        sl_ss = _build_loaded_sl_spectrostream(checkpoint_path, model_name)
        load_quantizer_weights(pure_module.spectrostream.quantizer, sl_ss.quantizer)
        load_spectrostream_decoder_weights(
            pure_module.spectrostream.decoder, sl_ss.decoder,
        )
        load_spectrostream_encoder_weights(
            pure_module.spectrostream.encoder, sl_ss.encoder, sl_ss.config,
        )
        mx.eval(pure_module.spectrostream.parameters())
