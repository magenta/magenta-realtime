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

"""SpectroStream parameter loading helpers: copy weights from a built
sl ``SpectroStream`` (or a slice of one) into the corresponding pure
``SpectroStream`` modules (encoder, decoder, RVQ).

Each public helper is independent and can be called in isolation
(useful for narrow parity tests and for extending to new SpectroStream
specs). The orchestrator that wires them into a full system is
:func:`magenta_rt.mlx_pure.load_weights.load_weights_from_combinator`.
"""

from __future__ import annotations

import mlx.core as mx


def _get_inner(layer):
    """Unwrap sl's ``Deferred`` / similar wrapper to expose the actual
    parameter-bearing layer (``inner`` attribute).
    """
    if hasattr(layer, "inner") and layer.inner is not None:
        return layer.inner
    return layer


def load_quantizer_weights(pure_q, sl_q) -> None:
    """``ResidualVectorQuantizer.embedding`` ← sl's RVQ embedding."""
    pure_q.embedding = sl_q.embedding


def _load_decoder_input_residual_weights(pure_il, sl_il) -> None:
    """sl input_layer (Residual) → pure ``_DecoderInputResidual``."""
    il_conv = _get_inner(sl_il.body.layers[0].layers[0])
    pure_il.body_conv.kernel = il_conv.kernel
    pure_il.body_conv.bias = il_conv.bias
    il_sc1 = _get_inner(sl_il.shortcut.layers[0].layers[0])
    pure_il.shortcut_conv1.kernel = il_sc1.kernel
    pure_il.shortcut_conv1.bias = il_sc1.bias
    il_sc2 = _get_inner(sl_il.shortcut.layers[2].layers[0])
    pure_il.shortcut_conv2.kernel = il_sc2.kernel
    pure_il.shortcut_conv2.bias = il_sc2.bias


def _load_decoder_unit_weights(pure_unit, sl_unit) -> None:
    """One decoder ``Conv2DResidualUnit`` (transposed)."""
    sl_conv_t = _get_inner(sl_unit.body.layers[0].layers[-1])
    pure_unit.body[1].kernel = sl_conv_t.kernel
    pure_unit.body[1].bias = sl_conv_t.bias
    sl_conv = _get_inner(sl_unit.body.layers[1].layers[-1])
    pure_unit.body[3].kernel = sl_conv.kernel
    pure_unit.body[3].bias = sl_conv.bias
    if pure_unit.shortcut is not None:
        for layer in pure_unit.shortcut:
            if type(layer).__name__ == "Conv2D":
                sc_conv = _get_inner(sl_unit.shortcut.layers[0].layers[0])
                layer.kernel = sc_conv.kernel
                layer.bias = sc_conv.bias
                break


def _load_encoder_unit_weights(pure_unit, sl_unit) -> None:
    """One encoder ``Conv2DResidualUnit`` (forward / non-transposed)."""
    sl_3x3 = _get_inner(sl_unit.body.layers[0].layers[-1])
    pure_unit.body[1].kernel = sl_3x3.kernel
    pure_unit.body[1].bias = sl_3x3.bias
    sl_resample = _get_inner(sl_unit.body.layers[1].layers[-1])
    pure_unit.body[3].kernel = sl_resample.kernel
    pure_unit.body[3].bias = sl_resample.bias
    if pure_unit.shortcut is not None:
        for layer in pure_unit.shortcut:
            if type(layer).__name__ == "Conv2D":
                for sl_layer in sl_unit.shortcut.layers:
                    inner = _get_inner(sl_layer)
                    if type(inner).__name__ == "Conv2D":
                        layer.kernel = inner.kernel
                        layer.bias = inner.bias
                        break
                    if hasattr(sl_layer, "layers"):
                        for sub in sl_layer.layers:
                            inner2 = _get_inner(sub)
                            if type(inner2).__name__ == "Conv2D":
                                layer.kernel = inner2.kernel
                                layer.bias = inner2.bias
                                break
                break


def load_spectrostream_decoder_weights(pure_dec, sl_dec) -> None:
    """sl SpectroStream ``decoder`` Serial → pure ``SpectroStreamDecoder``."""
    _load_decoder_input_residual_weights(pure_dec._input_residual, sl_dec.layers[1])
    sl_iru = sl_dec.layers[3]
    pure_iru = pure_dec._ungrouped.layers[0]
    iru_c1 = _get_inner(sl_iru.body.layers[0].layers[1])
    pure_iru.body[1].kernel = iru_c1.kernel
    pure_iru.body[1].bias = iru_c1.bias
    iru_c2 = _get_inner(sl_iru.body.layers[1].layers[1])
    pure_iru.body[3].kernel = iru_c2.kernel
    pure_iru.body[3].bias = iru_c2.bias

    _load_decoder_unit_weights(pure_dec._ungrouped.layers[1], sl_dec.layers[4])

    pc_child = sl_dec.layers[5].child
    for i in range(6):
        _load_decoder_unit_weights(pure_dec._grouped.inner.layers[i], pc_child.layers[i])

    sl_out = pc_child.layers[6]
    pure_out = pure_dec._grouped.inner.layers[7]
    out_conv = _get_inner(sl_out.layers[1].layers[0])
    pure_out.kernel = out_conv.kernel
    pure_out.bias = out_conv.bias


def load_spectrostream_encoder_weights(pure_enc, sl_enc, sl_ss_config) -> None:
    """sl SpectroStream ``encoder`` Serial → pure ``SpectroStreamEncoder``.

    Materializes sl's ``DeferredConv2D`` layers via a 2-frame dummy
    forward first (matches the pattern in
    ``magenta_rt.mlx.load_weights.load_weights``).
    """
    try:
        import sequence_layers.mlx as _sl
        _dummy = _sl.Sequence(
            mx.zeros((1, 2, sl_ss_config.num_bins, sl_ss_config.num_channels),
                     dtype=mx.float32),
            mx.ones((1, 2), dtype=mx.bool_),
        )
        sl_enc.layer(_dummy)
    except Exception:
        pass

    sl_pc = None
    for layer in sl_enc.layers:
        if type(layer).__name__ == "ParallelChannels":
            sl_pc = layer
            break

    if sl_pc is None:
        sl_base_conv = _get_inner(sl_enc.layers[0].layers[0])
        pure_base_conv = pure_enc._prefix.layers[0] if pure_enc._prefix is not None else pure_enc._post.layers[0]
        pure_base_conv.kernel = sl_base_conv.kernel
        pure_base_conv.bias = sl_base_conv.bias
        sl_units = [l for l in sl_enc.layers if type(l).__name__ == "Residual"]
        pure_units = []
        if pure_enc._prefix is not None:
            pure_units += list(pure_enc._prefix.layers[1:])
        if pure_enc._post is not None:
            pure_units += list(pure_enc._post.layers)
        sl_output_residual = sl_units.pop()
        for sl_u, pure_u in zip(sl_units, pure_units):
            _load_encoder_unit_weights(pure_u, sl_u)
    else:
        pc_layers = [l for l in sl_pc.child.layers if type(l).__name__ != "Delay"]
        sl_base_serial = pc_layers[0]
        sl_base_conv = _get_inner(sl_base_serial.layers[0])
        pure_prefix_layers = list(pure_enc._prefix.inner.layers)
        pure_prefix_layers[0].kernel = sl_base_conv.kernel
        pure_prefix_layers[0].bias = sl_base_conv.bias
        sl_prefix_units = [l for l in pc_layers[1:] if type(l).__name__ == "Residual"]
        for i, sl_u in enumerate(sl_prefix_units, start=1):
            _load_encoder_unit_weights(pure_prefix_layers[i], sl_u)

        sl_post_layers = []
        seen_pc = False
        for l in sl_enc.layers:
            if l is sl_pc:
                seen_pc = True
                continue
            if not seen_pc:
                continue
            if type(l).__name__ == "Residual":
                sl_post_layers.append(l)
        sl_output_residual = sl_post_layers.pop()
        for sl_u, pure_u in zip(sl_post_layers, pure_enc._post.layers):
            _load_encoder_unit_weights(pure_u, sl_u)

    sl_oc_body = _get_inner(sl_output_residual.body.layers[0].layers[0])
    pure_enc._output_convs.body_conv.kernel = sl_oc_body.kernel
    pure_enc._output_convs.body_conv.bias = sl_oc_body.bias
    sl_oc_sc1 = _get_inner(sl_output_residual.shortcut.layers[1].layers[0])
    pure_enc._output_convs.shortcut_conv1.kernel = sl_oc_sc1.kernel
    pure_enc._output_convs.shortcut_conv1.bias = sl_oc_sc1.bias
    sl_oc_sc2 = _get_inner(sl_output_residual.shortcut.layers[3].layers[0])
    pure_enc._output_convs.shortcut_conv2.kernel = sl_oc_sc2.kernel
    pure_enc._output_convs.shortcut_conv2.bias = sl_oc_sc2.bias


def load_spectrostream_weights(pure_ss, sl_ss) -> None:
    """Top-level SpectroStream weights loading: quantizer + decoder + encoder."""
    load_quantizer_weights(pure_ss.quantizer, sl_ss.quantizer)
    load_spectrostream_decoder_weights(pure_ss.decoder, sl_ss.decoder)
    load_spectrostream_encoder_weights(pure_ss.encoder, sl_ss.encoder, sl_ss.config)


# ---------------------------------------------------------------------------
# Direct Linen -> pure loader (NO sequence_layers bridge).
#
# The standalone SpectroStream Linen safetensors (resources/spectrostream/
# {encoder,decoder}.safetensors) carry the codec ENTIRELY (encoder, decoder,
# quantizer); no main checkpoint is needed. The conv kernel transforms here
# reproduce exactly what ``magenta_rt.mlx.spectrostream.load_weights`` applies
# when building the sl module — and since the sl -> pure copy above is the
# identity for conv kernels/biases, applying those transforms inline and
# assigning straight into the pure modules is bit-identical to the sl bridge.
#
# Conv kernel layouts:
#   * Conv2D:          Linen [kH, kW, in, out]      -> pure [out, kH, kW, in]
#   * Conv2DTranspose: Linen [kH, kW, in, out], flipped along (kH, kW) FIRST
#                      (MLX conv_transpose2d does not flip; JAX does)
#                      -> pure [out, kH, kW, in]
# ---------------------------------------------------------------------------

import numpy as _np


def _to_pure(arr, dst_dtype):
    """numpy array -> mx.array cast to the pure param dtype."""
    return mx.array(_np.ascontiguousarray(arr)).astype(dst_dtype)


def _conv2d_kernel(kernel):
    """Linen Conv2D kernel [kH, kW, in, out] -> pure [out, kH, kW, in]."""
    return _np.transpose(_np.asarray(kernel), (3, 0, 1, 2))


def _conv2d_transpose_kernel(kernel):
    """Linen ConvTranspose kernel [kH, kW, in, out] -> pure [out, kH, kW, in].

    Flip along both spatial axes first (MLX's conv_transpose2d does not flip
    the kernel, JAX's does — so we pre-flip), then reorder.
    """
    kernel = _np.asarray(kernel)[::-1, ::-1, :, :]
    return _np.transpose(kernel, (3, 0, 1, 2))


def _read_linen_safetensors(path):
    """Read a Linen safetensors into a nested dict of numpy arrays.

    Splits the slash-joined keys (e.g. ``params/encoder/encoder_0/...``) into a
    nested dict mirroring the JAX param tree.
    """
    from safetensors import safe_open

    nested: dict = {}
    with safe_open(str(path), framework="numpy") as handle:
        for k in handle.keys():
            parts = k.split("/")
            node = nested
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = handle.get_tensor(k)
    return nested


def _jax_conv(block, conv_name):
    """Fetch a conv's params from a Linen block, unwrapping the extra
    ``conv/`` nesting level (``block[conv_name]['conv']``).
    """
    p = block[conv_name]
    if "conv" in p:
        p = p["conv"]
    return p


def _assign_conv2d(pure_conv, jax_params, dst_dtype):
    pure_conv.in_features = _np.asarray(jax_params["kernel"]).shape[2]
    pure_conv.kernel = _to_pure(_conv2d_kernel(jax_params["kernel"]), dst_dtype)
    if "bias" in jax_params and pure_conv.bias is not None:
        pure_conv.bias = _to_pure(jax_params["bias"], dst_dtype)


def _assign_conv2d_transpose(pure_conv, jax_params, dst_dtype):
    pure_conv.in_features = _np.asarray(jax_params["kernel"]).shape[2]
    pure_conv.kernel = _to_pure(
        _conv2d_transpose_kernel(jax_params["kernel"]), dst_dtype
    )
    if "bias" in jax_params and pure_conv.bias is not None:
        pure_conv.bias = _to_pure(jax_params["bias"], dst_dtype)


def _find_block_conv_keys(block, *, transpose):
    """Pick out the (resample, 3x3, shortcut) conv keys from a Linen block.

    Encoder blocks: a ``conv2d_3x3`` and a resample conv whose key contains
    ``_a`` (e.g. ``conv2d_3x4_a``). Decoder blocks: a ``conv2d_3x3`` and a
    ``conv2dtranspose_*`` resample conv. ``shortcut_layer`` is optional.
    """
    conv_3x3_key = resample_key = shortcut_key = None
    for k in block:
        if "shortcut" in k:
            shortcut_key = k
        elif transpose and "conv2dtranspose" in k:
            resample_key = k
        elif (not transpose) and k.endswith("_a"):
            resample_key = k
        elif "conv2d_3x3" in k:
            conv_3x3_key = k
    return conv_3x3_key, resample_key, shortcut_key


def _shortcut_conv_key(jax_shortcut):
    for k in jax_shortcut:
        if "conv1x1" in k:
            return k
    raise AssertionError(f"no conv1x1 in shortcut: {list(jax_shortcut)}")


def _pure_shortcut_conv(pure_unit):
    """Return the Conv2D inside a pure residual unit's shortcut list (or None)."""
    if pure_unit.shortcut is None:
        return None
    for layer in pure_unit.shortcut:
        if type(layer).__name__ == "Conv2D":
            return layer
    return None


def _load_encoder_unit_from_linen(pure_unit, jax_block, dst_dtype):
    """Forward (non-transposed) encoder residual unit. body: [act, 3x3,
    act, resample]; convs at body[1] and body[3]."""
    conv_3x3_key, resample_key, shortcut_key = _find_block_conv_keys(
        jax_block, transpose=False
    )
    assert conv_3x3_key is not None, f"no conv2d_3x3 in {list(jax_block)}"
    assert resample_key is not None, f"no _a resample conv in {list(jax_block)}"
    _assign_conv2d(pure_unit.body[1], _jax_conv(jax_block, conv_3x3_key), dst_dtype)
    _assign_conv2d(pure_unit.body[3], _jax_conv(jax_block, resample_key), dst_dtype)
    if shortcut_key is not None:
        sc = _pure_shortcut_conv(pure_unit)
        assert sc is not None, "pure encoder unit has a shortcut conv slot"
        jax_sc = jax_block[shortcut_key]
        _assign_conv2d(sc, _jax_conv(jax_sc, _shortcut_conv_key(jax_sc)), dst_dtype)


def _load_decoder_unit_from_linen(pure_unit, jax_block, dst_dtype):
    """Transposed decoder residual unit. body: [act, transpose, act, 3x3];
    transpose at body[1], conv at body[3]."""
    conv_3x3_key, transpose_key, shortcut_key = _find_block_conv_keys(
        jax_block, transpose=True
    )
    assert transpose_key is not None, f"no conv2dtranspose in {list(jax_block)}"
    assert conv_3x3_key is not None, f"no conv2d_3x3 in {list(jax_block)}"
    _assign_conv2d_transpose(
        pure_unit.body[1], _jax_conv(jax_block, transpose_key), dst_dtype
    )
    _assign_conv2d(pure_unit.body[3], _jax_conv(jax_block, conv_3x3_key), dst_dtype)
    if shortcut_key is not None:
        sc = _pure_shortcut_conv(pure_unit)
        assert sc is not None, "pure decoder unit has a shortcut conv slot"
        jax_sc = jax_block[shortcut_key]
        _assign_conv2d(sc, _jax_conv(jax_sc, _shortcut_conv_key(jax_sc)), dst_dtype)


def load_quantizer_from_linen(pure_q, jax_quantizer, dst_dtype) -> None:
    """``ResidualVectorQuantizer.embedding`` <- Linen quantizer embedding."""
    pure_q.embedding = _to_pure(jax_quantizer["embedding"], dst_dtype)


def load_spectrostream_encoder_from_linen(pure_enc, jax_encoder, dst_dtype) -> None:
    """Linen ``params/encoder`` -> pure ``SpectroStreamEncoder``.

    The pure encoder (channel_splits=2 config) is:
      ``_prefix.inner`` = [base_conv (Conv2D), encoder_0..encoder_5 units]
      ``_post``         = [encoder_6 unit, bottleneck unit]
      ``_output_convs`` = body_conv + shortcut_conv1/2 (the output_convs block)
    """
    prefix_layers = pure_enc._prefix.inner.layers
    # base_conv_first -> prefix.inner.layers[0]
    _assign_conv2d(
        prefix_layers[0], _jax_conv(jax_encoder, "base_conv_first"), dst_dtype
    )
    # encoder_0..encoder_5 -> prefix.inner.layers[1..6]
    for i in range(6):
        _load_encoder_unit_from_linen(
            prefix_layers[i + 1], jax_encoder[f"encoder_{i}"], dst_dtype
        )
    # encoder_6 -> _post.layers[0]; bottleneck -> _post.layers[1]
    post_layers = pure_enc._post.layers
    _load_encoder_unit_from_linen(post_layers[0], jax_encoder["encoder_6"], dst_dtype)
    _load_encoder_unit_from_linen(post_layers[1], jax_encoder["bottleneck"], dst_dtype)
    # output_convs: conv1x1_last -> body_conv; shortcut conv1x1_b1/b2.
    jax_oc = jax_encoder["output_convs"]
    _assign_conv2d(
        pure_enc._output_convs.body_conv,
        _jax_conv(jax_oc, "conv1x1_last"),
        dst_dtype,
    )
    jax_oc_sc = jax_oc["shortcut_layer"]
    _assign_conv2d(
        pure_enc._output_convs.shortcut_conv1,
        _jax_conv(jax_oc_sc, "conv1x1_b1"),
        dst_dtype,
    )
    _assign_conv2d(
        pure_enc._output_convs.shortcut_conv2,
        _jax_conv(jax_oc_sc, "conv1x1_b2"),
        dst_dtype,
    )


def load_spectrostream_decoder_from_linen(pure_dec, jax_decoder, dst_dtype) -> None:
    """Linen ``params/decoder`` -> pure ``SpectroStreamDecoder``.

    The pure decoder (channel_splits=2 config) is:
      ``_input_residual``   = body_conv + shortcut_conv1/2 (input_layer)
      ``_ungrouped.layers`` = [input_layers_residual_unit, decoder_0 unit]
      ``_grouped.inner``    = [decoder_1..decoder_6 units, act, base_conv_last]
    """
    # input_layer: conv1x1_first -> body_conv; shortcut conv1x1_b1/b2.
    jax_il = jax_decoder["input_layer"]
    _assign_conv2d(
        pure_dec._input_residual.body_conv,
        _jax_conv(jax_il, "conv1x1_first"),
        dst_dtype,
    )
    jax_il_sc = jax_il["shortcut_layer"]
    _assign_conv2d(
        pure_dec._input_residual.shortcut_conv1,
        _jax_conv(jax_il_sc, "conv1x1_b1"),
        dst_dtype,
    )
    _assign_conv2d(
        pure_dec._input_residual.shortcut_conv2,
        _jax_conv(jax_il_sc, "conv1x1_b2"),
        dst_dtype,
    )
    # input_layers_residual_unit -> _ungrouped.layers[0]. This is a forward
    # residual unit (no transpose): body [act, conv2d_3x3_a, act, conv2d_3x3]
    # with convs at body[1] (conv2d_3x3_a) and body[3] (conv2d_3x3).
    jax_iru = jax_decoder["input_layers_residual_unit"]
    iru = pure_dec._ungrouped.layers[0]
    _assign_conv2d(iru.body[1], _jax_conv(jax_iru, "conv2d_3x3_a"), dst_dtype)
    _assign_conv2d(iru.body[3], _jax_conv(jax_iru, "conv2d_3x3"), dst_dtype)
    # decoder_0 -> _ungrouped.layers[1] (transposed unit).
    _load_decoder_unit_from_linen(
        pure_dec._ungrouped.layers[1], jax_decoder["decoder_0"], dst_dtype
    )
    # decoder_1..decoder_6 -> _grouped.inner.layers[0..5] (transposed units).
    grouped_layers = pure_dec._grouped.inner.layers
    for i in range(6):
        _load_decoder_unit_from_linen(
            grouped_layers[i], jax_decoder[f"decoder_{i + 1}"], dst_dtype
        )
    # output_layer: base_conv_last -> the final Conv2D in _grouped.inner.
    out_conv = grouped_layers[-1]
    assert type(out_conv).__name__ == "Conv2D", (
        f"expected final grouped layer to be Conv2D, got {type(out_conv).__name__}"
    )
    _assign_conv2d(
        out_conv,
        _jax_conv(jax_decoder["output_layer"], "base_conv_last"),
        dst_dtype,
    )


def load_spectrostream_from_linen(
    pure_ss, *, encoder_path=None, decoder_path=None
) -> None:
    """Load a pure ``SpectroStream`` ENTIRELY from the standalone Linen
    safetensors — no sl bridge, no depthformer, no main checkpoint.

    Args:
      pure_ss: a freshly built pure ``SpectroStream`` (e.g. from
        ``spec.build_spectrostream()``).
      encoder_path: Linen encoder safetensors (``params/encoder/*``). Defaults
        to ``magenta_rt.paths.resolve_encoder_weights()``.
      decoder_path: Linen decoder safetensors (``params/decoder/*`` +
        ``params/quantizer/embedding``). Defaults to
        ``magenta_rt.paths.resolve_decoder_weights()``.

    Populates the quantizer, decoder, and encoder. Arrays are cast to the pure
    quantizer's existing param dtype (the build default, fp32) so the result is
    bit-identical to the sl-bridge codec path.
    """
    from magenta_rt import paths

    if encoder_path is None:
        encoder_path = paths.resolve_encoder_weights()
    if decoder_path is None:
        decoder_path = paths.resolve_decoder_weights()

    # The build default codec dtype (fp32) — matches the sl-bridge result.
    dst_dtype = pure_ss.quantizer.embedding.dtype

    enc_tree = _read_linen_safetensors(encoder_path)["params"]["encoder"]
    dec_root = _read_linen_safetensors(decoder_path)["params"]
    dec_tree = dec_root["decoder"]
    quant_tree = dec_root["quantizer"]

    load_quantizer_from_linen(pure_ss.quantizer, quant_tree, dst_dtype)
    load_spectrostream_decoder_from_linen(pure_ss.decoder, dec_tree, dst_dtype)
    load_spectrostream_encoder_from_linen(pure_ss.encoder, enc_tree, dst_dtype)
