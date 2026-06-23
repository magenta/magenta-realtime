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

"""SpectroStream-specific bridge helpers (Linen safetensors → nnx).

Source of truth is the flat dict produced by
``safetensors.flax.load_file``; the helpers here walk that dict
(prefix-stripped to ``spectrostream/...``) and write into the
corresponding nnx modules.

* **Conv2D / Conv2DTranspose kernel** — Linen ``[kH, kW, in, out]``
  → nnx ``[out, kH, kW, in]``. The bridge applies the same
  ``transpose(k, (3, 0, 1, 2))`` for both.

Orchestrated by
:func:`magenta_rt.nnx.load_weights.load_from_jax_safetensors`.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
from einops import rearrange

from ..load_weights import _set_param


def _conv2d_kernel_hwio_to_ohwi(k: jnp.ndarray) -> jnp.ndarray:
    """Linen conv kernel ``[kH, kW, in, out]`` → nnx ``[out, kH, kW, in]``."""
    return rearrange(k, "h w i o -> o h w i")


def load_quantizer_weights(pure_q, quantizer_subdict: Mapping) -> None:
    """``ResidualVectorQuantizer.embedding`` ← Linen RVQ embedding.

    Linen layout:
        spectrostream/quantizer/embedding   [num_q, num_emb, dim]
    """
    table = jnp.asarray(quantizer_subdict["embedding"])
    _set_param(pure_q.embedding, table)


def _load_conv2d_weights(pure_conv, conv_subdict: Mapping) -> None:
    """Linen conv (HWIO) → nnx Conv2D / Conv2DTranspose (OHWI).

    Handles lazy-init Conv2D / Conv2DTranspose kernels (where
    ``in_features=None`` was passed at construction time) — uses the
    safetensors HWIO kernel's I axis to compute the right
    in_features and forces the kernel into existence before the copy.
    """
    inner = conv_subdict["conv"]
    linen_kernel = jnp.asarray(inner["kernel"])
    in_channels = linen_kernel.shape[2]  # HWIO → I axis is position 2.
    # ensure_initialized takes the *post-groups* in count for forward
    # Conv2D; for Conv2DTranspose it takes in_features directly.
    if hasattr(pure_conv, "groups"):
        pure_conv.ensure_initialized(in_channels * pure_conv.groups)
    else:
        pure_conv.ensure_initialized(in_channels)
    _set_param(
        pure_conv.kernel,
        _conv2d_kernel_hwio_to_ohwi(linen_kernel),
    )
    if pure_conv.bias is not None and "bias" in inner:
        _set_param(pure_conv.bias, jnp.asarray(inner["bias"]))


def _load_input_residual_weights(pure_il, il_subdict: Mapping) -> None:
    """Linen ``input_layer`` → pure :class:`_DecoderInputResidual`.

    Linen sub-tree:
        input_layer/conv1x1_first/conv/{kernel,bias}        — body
        input_layer/shortcut_layer/conv1x1_b1/conv/{kernel,bias}
        input_layer/shortcut_layer/conv1x1_b2/conv/{kernel,bias}
    """
    _load_conv2d_weights(pure_il.body_conv, il_subdict["conv1x1_first"])
    _load_conv2d_weights(pure_il.shortcut_conv1, il_subdict["shortcut_layer"]["conv1x1_b1"])
    _load_conv2d_weights(pure_il.shortcut_conv2, il_subdict["shortcut_layer"]["conv1x1_b2"])


def _load_unit_weights(pure_unit, unit_subdict: Mapping) -> None:
    """One transposed-decoder ``Conv2DResidualUnit``.

    Linen sub-tree (when the unit has shortcut + transpose):
        conv2dtranspose_KxL/conv/{kernel,bias}   — body[1] (the transpose conv)
        conv2d_3x3/conv/{kernel,bias}            — body[3] (post-transpose 3×3)
        shortcut_layer/conv1x1/conv/{kernel,bias} — shortcut Conv2D (if present)

    For the (1, 1)-strides initial unit (transposed=True, strides=(1,1)),
    Linen has only conv2d_3x3 and conv2d_3x3_a — no transpose conv —
    handled separately by ``_load_unit_weights``.
    """
    # Identify the transpose conv key (varies by stride: conv2dtranspose_4x3,
    # conv2dtranspose_3x6, etc.).
    t_keys = [k for k in unit_subdict if k.startswith("conv2dtranspose")]
    if t_keys:
        # Strides != (1,1) transposed unit. Linen body order:
        #   body[1] = conv2dtranspose_KxL  (first conv after first act)
        #   body[3] = conv2d_3x3           (second conv after second act)
        _load_conv2d_weights(pure_unit.body[1], unit_subdict[t_keys[0]])
        _load_conv2d_weights(pure_unit.body[3], unit_subdict["conv2d_3x3"])
    else:
        # Strides=(1,1) transposed unit. Linen body order is the
        # OPPOSITE of the no-transpose case (see
        # ``magenta_rt/jax/spectrostream.py:142-191``):
        #   body[1] = conv2d_3x3_a   (first conv)
        #   body[3] = conv2d_3x3     (second conv)
        _load_conv2d_weights(pure_unit.body[1], unit_subdict["conv2d_3x3_a"])
        _load_conv2d_weights(pure_unit.body[3], unit_subdict["conv2d_3x3"])

    if pure_unit.shortcut is not None and "shortcut_layer" in unit_subdict:
        # Find the Conv2D in the shortcut list (others are AveragePooling
        # / Upsample which have no parameters). The kernel may be lazily
        # allocated (in_features=None) so we identify Conv2D by its
        # ``filters`` attribute rather than ``kernel is not None``.
        from ..conv import Conv2D
        for layer in pure_unit.shortcut:
            if isinstance(layer, Conv2D):
                _load_conv2d_weights(layer, unit_subdict["shortcut_layer"]["conv1x1"])
                break


def load_decoder_weights(pure_dec, decoder_subdict: Mapping) -> None:
    """Linen ``spectrostream/decoder/...`` → pure :class:`SpectroStreamDecoder`.

    Layout (smallm4air checkpoint):
        decoder/input_layer/...                       → _input_residual
        decoder/input_layers_residual_unit/...        → _ungrouped.layers[0]
        decoder/decoder_0/...                         → _ungrouped.layers[1]
                                                        (the (1,1)-strides
                                                        bottleneck-equivalent)
        decoder/decoder_1..decoder_6/...              → grouped child layers
        decoder/output_layer/base_conv_last/...       → final base_conv_last
    """
    _load_input_residual_weights(pure_dec._input_residual, decoder_subdict["input_layer"])

    # _ungrouped.layers[0] is the input_layers_residual_unit
    # (transposed=True, strides=(1,1), no shortcut). Same Linen body
    # order as `_load_unit_weights`'s no-transpose-conv branch:
    #   body[1] = conv2d_3x3_a (first), body[3] = conv2d_3x3 (second).
    iru = decoder_subdict["input_layers_residual_unit"]
    iru_pure = pure_dec._ungrouped.layers[0]
    _load_conv2d_weights(iru_pure.body[1], iru["conv2d_3x3_a"])
    _load_conv2d_weights(iru_pure.body[3], iru["conv2d_3x3"])

    # Decoder body. With channel_splits, decoder_0 lives in _ungrouped and
    # decoder_1..6 in _grouped.inner.layers.
    dec_keys = sorted(
        k for k in decoder_subdict if k.startswith("decoder_") and k != "decoder"
    )
    if pure_dec._grouped is not None:
        # Channel-splits flow: decoder_0 in ungrouped[1], rest in grouped.
        if dec_keys:
            _load_unit_weights(
                pure_dec._ungrouped.layers[1], decoder_subdict[dec_keys[0]],
            )
            grouped_layers = pure_dec._grouped.inner.layers
            for i, k in enumerate(dec_keys[1:]):
                _load_unit_weights(grouped_layers[i], decoder_subdict[k])
            # The last layer in grouped is the base_conv_last (Conv2D
            # appended after the residual chain). Linen path:
            # decoder/output_layer/base_conv_last
            base_last = decoder_subdict["output_layer"]["base_conv_last"]
            # base_conv_last is the LAST element of the grouped inner list.
            _load_conv2d_weights(grouped_layers[-1], base_last)
    else:
        # No channel-splits: everything in ungrouped.
        ungrouped_layers = pure_dec._ungrouped.layers
        for i, k in enumerate(dec_keys, start=1):
            _load_unit_weights(ungrouped_layers[i], decoder_subdict[k])
        base_last = decoder_subdict["output_layer"]["base_conv_last"]
        _load_conv2d_weights(ungrouped_layers[-1], base_last)


def _load_encoder_unit_weights(pure_unit, unit_subdict: Mapping) -> None:
    """One forward (non-transposed) encoder ``Conv2DResidualUnit``.

    For ``transposed=False`` (see ``model.py:245-264``) the body is
    ``[act, conv2d_3x3, act, resample]`` so the parameter-bearing convs
    sit at body[1] (the ``conv2d_3x3``) and body[3] (the strided resample
    ``conv2d_KxL_a``). This is the OPPOSITE assignment to the decoder's
    transposed unit (whose resample/transpose conv is body[1]).

    Linen sub-tree (when the unit has a shortcut):
        conv2d_3x3/conv/{kernel,bias}        — body[1] (the 3×3)
        conv2d_KxL_a/conv/{kernel,bias}      — body[3] (the strided resample)
        shortcut_layer/conv1x1/conv/{kernel,bias} — shortcut Conv2D (if present)
    """
    # The 3×3 conv key is exactly "conv2d_3x3"; the resample conv key ends
    # in "_a" (e.g. conv2d_3x4_a, conv2d_4x3_a).
    a_keys = [k for k in unit_subdict if k.endswith("_a")]
    assert a_keys, f"no _a resample conv in {list(unit_subdict)}"
    _load_conv2d_weights(pure_unit.body[1], unit_subdict["conv2d_3x3"])
    _load_conv2d_weights(pure_unit.body[3], unit_subdict[a_keys[0]])

    if pure_unit.shortcut is not None and "shortcut_layer" in unit_subdict:
        from ..conv import Conv2D
        for layer in pure_unit.shortcut:
            if isinstance(layer, Conv2D):
                _load_conv2d_weights(
                    layer, unit_subdict["shortcut_layer"]["conv1x1"],
                )
                break


def _load_output_convs_weights(pure_oc, oc_subdict: Mapping) -> None:
    """Linen ``output_convs`` → pure :class:`_OutputConvsResidual`.

    Linen sub-tree:
        output_convs/conv1x1_last/conv/{kernel,bias}              — body_conv
        output_convs/shortcut_layer/conv1x1_b1/conv/{kernel,bias} — shortcut_conv1
        output_convs/shortcut_layer/conv1x1_b2/conv/{kernel,bias} — shortcut_conv2
    """
    _load_conv2d_weights(pure_oc.body_conv, oc_subdict["conv1x1_last"])
    _load_conv2d_weights(
        pure_oc.shortcut_conv1, oc_subdict["shortcut_layer"]["conv1x1_b1"],
    )
    _load_conv2d_weights(
        pure_oc.shortcut_conv2, oc_subdict["shortcut_layer"]["conv1x1_b2"],
    )


def load_encoder_weights(pure_enc, encoder_subdict: Mapping) -> None:
    """Linen ``params/encoder/...`` → pure :class:`SpectroStreamEncoder`.

    Layout (mrt2 codec, channel_splits=2 / channel_recombo_block=-2≡6):
        encoder/base_conv_first/...        → _prefix[.inner].layers[0]
        encoder/encoder_0..encoder_5/...   → _prefix[.inner].layers[1..6]
        encoder/encoder_6/...              → _post.layers[0]
        encoder/bottleneck/...             → _post.layers[1]
        encoder/output_convs/...           → _output_convs

    ``_prefix`` is a :class:`ParallelChannels` (unwrapped via ``.inner``)
    when channel_splits is set, or a plain ``nnx.Sequential`` otherwise —
    mirroring how ``load_decoder_weights`` handles grouped/ungrouped.
    """
    from ..conv import ParallelChannels

    prefix = pure_enc._prefix
    if isinstance(prefix, ParallelChannels):
        prefix_layers = prefix.inner.layers
    else:
        prefix_layers = prefix.layers

    # base_conv_first → first layer of the prefix.
    _load_conv2d_weights(prefix_layers[0], encoder_subdict["base_conv_first"])

    # The encoder residual units split across _prefix (the early levels) and
    # _post (the recombo level + bottleneck). Walk the Linen encoder_N keys
    # in order, filling the prefix's residual units first, then _post.
    enc_keys = sorted(
        (k for k in encoder_subdict if k.startswith("encoder_")),
        key=lambda k: int(k.split("_")[1]),
    )
    pure_units = list(prefix_layers[1:])
    if pure_enc._post is not None:
        # _post = [encoder residual units..., bottleneck]; the bottleneck is
        # loaded separately below, so only the leading units take encoder_N.
        pure_units += list(pure_enc._post.layers[:-1])
    assert len(pure_units) == len(enc_keys), (
        f"encoder unit count mismatch: {len(pure_units)} pure vs "
        f"{len(enc_keys)} Linen ({enc_keys})"
    )
    for pure_u, k in zip(pure_units, enc_keys):
        _load_encoder_unit_weights(pure_u, encoder_subdict[k])

    # bottleneck → last residual unit of _post.
    _load_encoder_unit_weights(
        pure_enc._post.layers[-1], encoder_subdict["bottleneck"],
    )

    # output_convs.
    _load_output_convs_weights(pure_enc._output_convs, encoder_subdict["output_convs"])


def load_spectrostream_weights(pure_ss, spectrostream_subdict: Mapping) -> None:
    """Top-level SpectroStream bridge: quantizer + decoder + encoder.

    The mrt2 depthformer checkpoint's ``soundstream`` subtree carries only
    the quantizer + decoder; the encoder ships separately as a standalone
    Linen safetensors (``resources/spectrostream/encoder.safetensors``,
    resolvable via :func:`magenta_rt.paths.resolve_encoder_weights`). When
    the encoder is absent from ``spectrostream_subdict`` we load it from
    that resource so ``waveform_to_codes`` runs on real (not random) weights.
    Without this the encoder stays randomly initialised and produces garbage
    codes — which silently corrupts SFT dataset exports
    (``mrt sft export --backend nnx``).
    """
    if pure_ss.quantizer is not None and "quantizer" in spectrostream_subdict:
        load_quantizer_weights(pure_ss.quantizer, spectrostream_subdict["quantizer"])
    if "decoder" in spectrostream_subdict:
        load_decoder_weights(pure_ss.decoder, spectrostream_subdict["decoder"])

    encoder = getattr(pure_ss, "encoder", None)
    if encoder is None:
        return
    if "encoder" in spectrostream_subdict:
        load_encoder_weights(encoder, spectrostream_subdict["encoder"])
        return

    # Encoder not in the checkpoint subtree — fall back to the shared
    # standalone resource. Mirror the mlx backend: warn and skip (don't
    # crash) if the resource file is missing.
    import os

    from ... import paths

    encoder_path = str(paths.resolve_encoder_weights())
    if not os.path.exists(encoder_path):
        print(
            f"  [nnx spectrostream] encoder weights not found at "
            f"{encoder_path}; encoder left at init (waveform_to_codes "
            f"will produce garbage)."
        )
        return
    enc_tree = _read_linen_encoder(encoder_path)
    load_encoder_weights(encoder, enc_tree)


def _read_linen_encoder(path: str) -> Mapping:
    """Read the standalone Linen encoder safetensors → ``params/encoder``
    nested dict of numpy arrays.

    Uses the raw numpy safetensors reader (not ``safetensors.flax``) — the
    JAX Metal backend errors on ``safetensors.flax.load_file`` with
    ``default_memory_space is not supported``. Tensors are materialised
    inside the ``with`` block (they're invalid once it closes).
    """
    from safetensors import safe_open

    nested: dict = {}
    with safe_open(path, framework="numpy") as handle:
        for k in handle.keys():
            parts = k.split("/")
            node = nested
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = handle.get_tensor(k)
    return nested["params"]["encoder"]
