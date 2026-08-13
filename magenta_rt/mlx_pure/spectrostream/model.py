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

"""SpectroStream codec — pure-MLX.

* :class:`ResidualVectorQuantizer` — codes ↔ embeddings table lookup
  and nearest-neighbor encoding. Inference-only (no training EMA).
* :class:`Conv2DResidualUnit` — residual block used by encoder /
  decoder.
* :class:`SpectroStreamEncoder` / :class:`SpectroStreamDecoder` —
  Conv2D-only stacks (no STFT wrapping).
* :class:`SpectroStream` — top-level wrapper providing
  :meth:`codes_to_waveform` and :meth:`waveform_to_codes` (the latter
  goes through the encoder + STFT).

Streaming is per-frame: every internal Conv2D / Conv2DTranspose
carries its own left-context cache via :class:`Conv2DCache`, the
InverseSTFT has an :class:`OverlapAddCache`, and the decoder's
lookahead is tracked inline as an integer countdown
(``_lookahead_remaining``).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence as _Seq, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from audiotree import AudioTree

from ..conv import (
    AveragePooling2D, Conv2D, Conv2DTranspose, ParallelChannels, Upsample2D,
)
from ..signal import STFT, InverseSTFT, hann_window, inverse_stft_window_fn

# SpectroStream operating sample rate (the codec assumes 48 kHz input).
SAMPLE_RATE = 48_000


class ResidualVectorQuantizer(nn.Module):
    """RVQ codebook table.

    The embedding tensor has shape
    ``[num_quantizers, num_embeddings, embedding_dim]``. Inference paths:

    * :meth:`codes_to_embeddings` — sums looked-up embeddings across the
      RVQ levels (matches ``magenta_rt.mlx.spectrostream.RVQ.codes_to_embeddings``
      with ``use_gather=True``).
    * :meth:`embeddings_to_codes` — nearest-neighbor encoding, returning
      one index per RVQ level. Output may use the "unique-codes" layout
      (``code_q + q * num_embeddings``).
    """

    def __init__(
        self,
        *,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        use_unique_codes: bool = False,
        truncation_level: Optional[int] = None,
        encoded_truncation_level: Optional[int] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_unique_codes = use_unique_codes
        self.truncation_level = truncation_level
        self.encoded_truncation_level = encoded_truncation_level
        self.param_dtype = param_dtype

        self.embedding = mx.zeros(
            (num_quantizers, num_embeddings, embedding_dim), dtype=param_dtype
        )

    @property
    def num_expected_input_codes(self) -> int:
        return self.truncation_level if self.truncation_level is not None else self.num_quantizers

    @property
    def num_expected_output_codes(self) -> int:
        return self.encoded_truncation_level if self.encoded_truncation_level is not None else self.num_quantizers

    # ------------------------------------------------------------------
    # Codes → embeddings
    # ------------------------------------------------------------------

    def codes_to_embeddings(self, codes: mx.array) -> mx.array:
        """``codes`` shape ``[B, T, num_input_codebooks]`` int32 → ``[B, T, embedding_dim]``."""
        if codes.ndim != 3:
            raise ValueError(f"expected 3D input, got {codes.shape=}")
        if codes.dtype not in (mx.int32, mx.uint32):
            raise ValueError(f"expected int32/uint32, got {codes.dtype=}")
        codes = codes.astype(mx.int32)

        if self.use_unique_codes:
            codes = codes % self.num_embeddings

        num_input = codes.shape[-1]
        if num_input > self.num_expected_input_codes:
            raise ValueError(
                f"got {num_input} input codebooks, expected ≤ {self.num_expected_input_codes}"
            )

        if num_input == 0:
            return mx.zeros(
                codes.shape[:2] + (self.embedding_dim,), self.embedding.dtype,
            )
        # Vectorized per-quantizer lookup via advanced indexing:
        # ``self.embedding[:num_input]`` is ``[Q, num_emb, dim]``;
        # ``codes`` is ``[B, T, Q]``. We want, for each (B, T, q),
        # ``embedding[q, codes[B, T, q], :]``, then sum over q.
        emb_used = self.embedding[:num_input]               # [Q, E, D]
        codes_q_first = mx.transpose(codes, (2, 0, 1))      # [Q, B, T]
        q_idx = mx.arange(num_input, dtype=mx.int32)[:, None, None]  # [Q,1,1]
        per_q = emb_used[q_idx, codes_q_first]              # [Q, B, T, D]
        return mx.sum(per_q, axis=0)                        # [B, T, D]

    # ------------------------------------------------------------------
    # Embeddings → codes
    # ------------------------------------------------------------------

    def embeddings_to_codes(
        self, inputs: mx.array, num_quantizers: Optional[int] = None
    ) -> mx.array:
        """Greedy residual encoding. Input ``[B, T, embedding_dim]``."""
        Q = num_quantizers if num_quantizers is not None else self.num_expected_output_codes
        residual = inputs
        codes = []
        for q in range(Q):
            cb = self.embedding[q]  # [num_emb, dim]
            distances = (
                mx.sum(residual ** 2, axis=-1, keepdims=True)
                - 2.0 * (residual @ cb.T)
                + mx.sum(cb ** 2, axis=-1)
            )
            code_q = mx.argmin(distances, axis=-1)
            quantized = mx.take(cb, code_q, axis=0)
            residual = residual - quantized
            codes.append(code_q)
        out = mx.stack(codes, axis=-1)
        if self.use_unique_codes:
            offsets = mx.arange(Q) * self.num_embeddings
            out = out + offsets
        return out

    def step_codes_to_embeddings(self, codes: mx.array, cache=None) -> mx.array:
        del cache
        return self.codes_to_embeddings(codes)

    def step_embeddings_to_codes(
        self, inputs: mx.array, cache=None, num_quantizers: Optional[int] = None
    ) -> mx.array:
        del cache
        return self.embeddings_to_codes(inputs, num_quantizers=num_quantizers)


# -----------------------------------------------------------------------------
# Helper modules for the SpectroStream encoder/decoder.
# -----------------------------------------------------------------------------


def _to_pair(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def _ss_conv2d_paddings(
    padding: str, kernel_size: tuple[int, int], strides: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """Compute (time_padding, spatial_padding) for the SpectroStream conv2d
    helper convention.

    Mirrors ``magenta_rt.mlx.spectrostream.conv2d``: when the user
    requests ``padding='causal'`` the underlying Conv2D uses
    ``time_padding='semicausal'`` and an explicit symmetric spatial pad.
    """
    pad_freq = max((kernel_size[1] - 1) * dilation[1] + 1 - strides[1], 0)
    spatial_pad = (pad_freq // 2, pad_freq - pad_freq // 2)
    time_pad = "semicausal" if padding == "causal" else padding
    return time_pad, spatial_pad


def _ss_conv2d(
    *, in_features: Optional[int], filters: int, kernel_size: tuple[int, int],
    strides: tuple[int, int], padding: str, dilation: tuple[int, int],
    param_dtype: mx.Dtype, compute_dtype: mx.Dtype,
) -> Conv2D:
    """Build a Conv2D with the SpectroStream-style padding convention.

    Pass ``in_features=None`` to defer kernel allocation until the
    first forward pass (used by the channel-split decoder section).
    """
    time_pad, spatial_pad = _ss_conv2d_paddings(padding, kernel_size, strides, dilation)
    return Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=kernel_size, strides=strides, dilation_rate=dilation,
        time_padding=time_pad, spatial_padding=spatial_pad,
        use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
    )


class Conv2DResidualUnit(nn.Module):
    """Residual block: act → conv2d (transpose) → act → conv2d.

    Mirrors ``magenta_rt.mlx.spectrostream.conv2d_residual_unit`` —
    including the SpectroStream-specific padding convention via
    :func:`_ss_conv2d_paddings`.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        strides: tuple[int, int],
        dilation: tuple[int, int],
        transposed: bool,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        padding: str = "causal",
        use_shortcut: bool = True,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.float32,
        deferred_in: bool = False,
    ):
        super().__init__()
        strides = _to_pair(strides)
        dilation = _to_pair(dilation)
        self.strides = strides
        self.dilation = dilation
        self.transposed = transposed
        self.padding = padding
        self.use_shortcut = use_shortcut
        self.activation_fn = activation_fn

        resample_kernel_size = (max(3, 2 * strides[0]), max(3, 2 * strides[1]))

        # When ``deferred_in`` is True, the unit's first param-bearing
        # layer (and its shortcut's first conv) defers kernel allocation
        # to runtime so it can pick up the per-group channel count from
        # a wrapping :class:`ParallelChannels`. Subsequent inner layers
        # use eager init because their channel counts come from the
        # preceding layer's filter count, which is unaffected by the
        # split.
        first_in = None if deferred_in else input_channels

        # ----- Body layers -----
        body_layers: list[nn.Module] = []
        if transposed:
            filters = output_channels
            if strides == (1, 1):
                body_layers.append(activation_fn)
                body_layers.append(_ss_conv2d(
                    in_features=first_in, filters=filters,
                    kernel_size=(3, 3), strides=(1, 1),
                    padding=padding, dilation=(1, 1),
                    param_dtype=param_dtype, compute_dtype=compute_dtype,
                ))
                inner_in = filters
            else:
                body_layers.append(activation_fn)
                body_layers.append(Conv2DTranspose(
                    in_features=first_in, filters=filters,
                    kernel_size=resample_kernel_size, strides=strides,
                    time_padding=padding, spatial_padding="same",
                    use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
                ))
                inner_in = filters
        else:
            filters = input_channels
            inner_in = input_channels

        # Body 3×3 conv (always present). After the first layer the
        # channel counts are known, so use eager init.
        body_in_3x3 = inner_in if (transposed or not deferred_in) else None
        body_layers.append(activation_fn)
        body_layers.append(_ss_conv2d(
            in_features=body_in_3x3, filters=filters,
            kernel_size=(3, 3), strides=(1, 1),
            padding=padding, dilation=dilation,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        ))

        if not transposed:
            body_layers.append(activation_fn)
            body_layers.append(_ss_conv2d(
                in_features=filters, filters=output_channels,
                kernel_size=resample_kernel_size, strides=strides,
                padding=padding, dilation=(1, 1),
                param_dtype=param_dtype, compute_dtype=compute_dtype,
            ))
        self.body = body_layers

        # ----- Shortcut layers -----
        if use_shortcut:
            shortcut: list[nn.Module] = []
            sc_in = input_channels
            if strides != (1, 1) and not transposed:
                shortcut.append(AveragePooling2D(
                    pool_size=strides, strides=strides,
                    time_padding="semicausal" if padding == "causal" else padding,
                    spatial_padding="valid",
                ))
            if input_channels != output_channels:
                shortcut.append(_ss_conv2d(
                    in_features=None if deferred_in else sc_in,
                    filters=output_channels,
                    kernel_size=(1, 1), strides=(1, 1),
                    padding="causal", dilation=(1, 1),
                    param_dtype=param_dtype, compute_dtype=compute_dtype,
                ))
                sc_in = output_channels
            if strides != (1, 1) and transposed:
                shortcut.append(Upsample2D(rate=strides))
            self.shortcut = shortcut
        else:
            self.shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        body_out = x
        for layer in self.body:
            body_out = layer(body_out)
        if self.shortcut is None:
            return body_out
        sc = x
        for layer in self.shortcut:
            sc = layer(sc)
        return body_out + sc

    def step(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        """Streaming step for Conv2DResidualUnit."""
        if cache is not None:
            body_cache = getattr(cache, "body", None)
            sc_cache = getattr(cache, "shortcut", None)
        else:
            body_cache = sc_cache = None

        body_out = x
        if body_cache is None:
            body_cache = [None] * len(self.body)
        for i, layer in enumerate(self.body):
            if hasattr(layer, "step"):
                body_out = layer.step(body_out, body_cache[i])
            else:
                body_out = layer(body_out)

        if self.shortcut is None:
            return body_out

        sc = x
        if sc_cache is None:
            sc_cache = [None] * len(self.shortcut)
        for i, layer in enumerate(self.shortcut):
            if hasattr(layer, "step"):
                sc = layer.step(sc, sc_cache[i])
            else:
                sc = layer(sc)

        return body_out + sc


class _OutputConvsResidual(nn.Module):
    """The 1×1 conv stack at the end of the encoder.

    Body: 1×1 Conv (proj to num_features).
    Shortcut: Conv1×1 (proj to bottleneck) → activation → Conv1×1 (proj
    to num_features) → activation. Output is body + shortcut.
    """

    def __init__(
        self, *, input_channels: int, bottleneck_channels: int, output_channels: int,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        param_dtype: mx.Dtype = mx.float32, compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.body_conv = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.shortcut_act1 = activation_fn
        self.shortcut_conv1 = Conv2D(
            in_features=input_channels, filters=bottleneck_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.shortcut_act2 = activation_fn
        self.shortcut_conv2 = Conv2D(
            in_features=bottleneck_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )

    def __call__(self, x: mx.array) -> mx.array:
        body = self.body_conv(x)
        sc = self.shortcut_act1(x)
        sc = self.shortcut_conv1(sc)
        sc = self.shortcut_act2(sc)
        sc = self.shortcut_conv2(sc)
        return body + sc


class _DecoderInputResidual(nn.Module):
    """The 1×1 conv stack at the start of the decoder (mirror of the
    encoder's _OutputConvsResidual).
    """

    def __init__(
        self, *, input_channels: int, output_channels: int,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        param_dtype: mx.Dtype = mx.float32, compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.body_conv = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.shortcut_conv1 = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.shortcut_act = activation_fn
        self.shortcut_conv2 = Conv2D(
            in_features=output_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, compute_dtype=compute_dtype,
        )

    def __call__(self, x: mx.array) -> mx.array:
        body = self.body_conv(x)
        sc = self.shortcut_conv1(x)
        sc = self.shortcut_act(sc)
        sc = self.shortcut_conv2(sc)
        return body + sc


# -----------------------------------------------------------------------------
# Encoder / Decoder networks (no STFT wrapping)
# -----------------------------------------------------------------------------


class SpectroStreamEncoder(nn.Module):
    """SpectroStream encoder Conv2D stack (post-STFT to bottleneck features).

    Input shape: ``[B, T, num_input_bins, num_channels]`` complex-as-float.
    Output shape: ``[B, T_out, num_output_features]``.

    Mirrors ``magenta_rt.mlx.spectrostream.spectrostream_encoder_config``
    for ``channel_splits ∈ {None, 2}`` and ``lookahead == 0``.
    """

    def __init__(
        self,
        *,
        base_conv_depth: int,
        base_conv_size: Union[int, tuple[int, int]],
        ratios: _Seq[tuple[int, int]],
        mults: _Seq[Union[int, float]],
        dilations: Optional[Union[_Seq[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        num_input_bins: int = 160,
        num_input_channels: int = 4,
        num_output_features: int = 64,
        causal: bool = True,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if isinstance(base_conv_size, int):
            base_conv_size = (base_conv_size, base_conv_size)
        if dilations is None:
            dilations = (1, 1)
        if isinstance(dilations[0], int):
            dilations = (dilations,) * len(ratios)

        padding = "causal" if causal else "same"
        num_blocks = len(ratios) + 1
        if channel_splits is not None:
            channel_recombo_block %= num_blocks
            if num_input_channels % channel_splits != 0:
                raise ValueError(
                    f"num_input_channels {num_input_channels} not divisible by "
                    f"channel_splits {channel_splits}"
                )

        # When channel_splits is in use, the prefix runs inside a
        # ``ParallelChannels(num_groups=channel_splits)`` wrapper, so the
        # FIRST layer of the prefix (base_conv) sees per-group input at
        # runtime: ``num_input_channels // channel_splits``.
        # Subsequent layers in the prefix use their normal channel
        # counts (the conv ``filters`` arg dominates after the first
        # mismatch is resolved by the first layer). Post-PC layers see
        # the concat'd output, so the FIRST post-PC layer's input
        # channels = ``last_prefix_output * channel_splits``; later
        # post-PC layers use their normal counts.
        prefix_first_in = (
            num_input_channels // channel_splits if channel_splits else num_input_channels
        )

        # base_conv follows the SpectroStream-style padding rule (semicausal
        # time + symmetric explicit spatial pad), matching sl's
        # ``spectrostream.conv2d`` helper. Using a plain Conv2D with
        # ``spatial_padding=padding`` would put a (K-1, 0) split on the
        # frequency axis; sl uses ((K-1)//2, K-1-(K-1)//2).
        base_conv = _ss_conv2d(
            in_features=prefix_first_in, filters=base_conv_depth,
            kernel_size=base_conv_size, strides=(1, 1),
            padding=padding, dilation=(1, 1),
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )

        # Build prefix and post-PC layer lists. base_conv goes into the
        # prefix unless recombo_block == 0 (entire pipeline is post-PC,
        # which the production config does not use but we support).
        prefix: list[nn.Module] = []
        post: list[nn.Module] = []
        if channel_splits and channel_recombo_block == 0:
            base_conv = _ss_conv2d(
                in_features=num_input_channels, filters=base_conv_depth,
                kernel_size=base_conv_size, strides=(1, 1),
                padding=padding, dilation=(1, 1),
                param_dtype=param_dtype, compute_dtype=compute_dtype,
            )
            post.append(base_conv)
        else:
            prefix.append(base_conv)

        input_channels = base_conv_depth
        output_channels = base_conv_depth
        curr_num_bins = num_input_bins
        for level_index, (strides_i, dilation_i, mult) in enumerate(
            zip(ratios, dilations, mults)
        ):
            output_channels = int(np.round(output_channels * mult))
            curr_num_bins //= strides_i[1]
            in_for_block = input_channels
            if channel_splits and channel_recombo_block == level_index:
                # First post-PC block: input is doubled by the concat.
                in_for_block = input_channels * channel_splits
            block = Conv2DResidualUnit(
                input_channels=in_for_block, output_channels=output_channels,
                strides=strides_i, dilation=dilation_i, transposed=False,
                activation_fn=activation_fn, padding=padding,
                use_shortcut=is_resnet,
                param_dtype=param_dtype, compute_dtype=compute_dtype,
            )
            if channel_splits and level_index < channel_recombo_block:
                prefix.append(block)
            else:
                post.append(block)
            input_channels = output_channels

        # Bottleneck (always at the end; no further mult). The
        # bottleneck is the (num_blocks - 1)th block in sl's
        # numbering — it goes into the prefix only when
        # ``channel_splits`` is set AND the recombo block is *after*
        # the bottleneck (impossible: max valid recombo is num_blocks-1).
        # So in practice it always goes to ``post``; the only
        # subtlety is doubling its input when recombo == num_blocks-1.
        in_for_bottleneck = input_channels
        if channel_splits and channel_recombo_block == num_blocks - 1:
            in_for_bottleneck = input_channels * channel_splits
        bottleneck = Conv2DResidualUnit(
            input_channels=in_for_bottleneck, output_channels=output_channels,
            strides=(1, 1), dilation=(1, 1), transposed=False,
            activation_fn=activation_fn, padding=padding,
            use_shortcut=is_resnet,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        post.append(bottleneck)

        # Wire up: prefix → PC(prefix) → post.
        if channel_splits and prefix:
            self._prefix = ParallelChannels(
                inner=_Sequential(prefix), num_groups=channel_splits,
            )
        else:
            self._prefix = _Sequential(prefix) if prefix else None
        self._post = _Sequential(post) if post else None

        flat_channels = curr_num_bins * output_channels
        self._flat_pre_channels = flat_channels
        self._output_convs = _OutputConvsResidual(
            input_channels=flat_channels,
            bottleneck_channels=output_channels,
            output_channels=num_output_features,
            activation_fn=activation_fn,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.num_output_features = num_output_features

    def __call__(self, x: mx.array) -> mx.array:
        if self._prefix is not None:
            x = self._prefix(x)
        if self._post is not None:
            x = self._post(x)
        # x: [B, T, S, C]. Flatten S × C into a single channel dim.
        B, T, S, C = x.shape
        x = x.reshape(B, T, 1, S * C)
        x = self._output_convs(x)  # [B, T, 1, num_output_features]
        x = x.reshape(B, T, -1)
        return x

    def step(self, x: mx.array, cache=None) -> mx.array:
        del cache
        return self(x)


class _Sequential(nn.Module):
    """Plain sequential composition — needed because ``nn.Module`` wraps
    its sub-modules so they get registered for parameter traversal.
    """

    def __init__(self, layers: _Seq[nn.Module]):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class SpectroStreamDecoder(nn.Module):
    """SpectroStream decoder Conv2D stack (bottleneck features → STFT
    domain). Mirror image of :class:`SpectroStreamEncoder`.

    Input: ``[B, T, num_features]``. Output:
    ``[B, T_out, num_output_bins, num_output_channels]``.
    """

    def __init__(
        self,
        *,
        base_conv_depth: int,
        base_conv_size: Union[int, tuple[int, int]],
        ratios: _Seq[tuple[int, int]],
        mults: _Seq[Union[int, float]],
        dilations: Optional[Union[_Seq[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        num_input_features: int = 64,
        num_output_bins: int = 160,
        num_output_channels: int = 4,
        causal: bool = True,
        decoder_lookahead: int = 0,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if isinstance(base_conv_size, int):
            base_conv_size = (base_conv_size, base_conv_size)
        if dilations is None:
            dilations = (1, 1)
        if isinstance(dilations[0], int):
            dilations = (dilations,) * len(ratios)
        padding = "causal" if causal else "same"

        total_time_stride = int(np.prod([r[0] for r in ratios]))
        total_freq_stride = int(np.prod([r[1] for r in ratios]))
        input_bins = num_output_bins // total_freq_stride
        output_channels = int(base_conv_depth * np.prod(mults))
        proj_filters = input_bins * output_channels
        self._lookahead_length = int(decoder_lookahead * total_time_stride)

        num_blocks = len(ratios) + 1
        if channel_splits is not None:
            channel_recombo_block %= num_blocks
            if num_output_channels % channel_splits != 0:
                raise ValueError(
                    f"num_output_channels {num_output_channels} not divisible by "
                    f"channel_splits {channel_splits}"
                )

        # Input residual: 1×1 convs on [B, T, 1, num_features]. Always
        # runs before the channel split (full feature dim).
        self._input_residual = _DecoderInputResidual(
            input_channels=num_input_features,
            output_channels=proj_filters,
            activation_fn=activation_fn,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self._input_bins = input_bins

        # Build the residual-unit chain. With ``channel_splits`` the
        # split happens *during* the reverse iteration: layers built
        # before the transition (the "ungrouped" prefix) run on full
        # channels; layers after run inside ``ParallelChannels`` and
        # use deferred Conv2D (in_features=None) so kernels allocate
        # for the per-group channel count at first forward.
        ungrouped: list[nn.Module] = []
        grouped: list[nn.Module] = []
        in_split = False  # True once we've crossed the recombo block

        def _maybe_in_features(ch: int) -> Optional[int]:
            return None if (channel_splits and in_split) else ch

        # Initial unit at (1, 1) strides.
        in_for_first = output_channels
        if channel_splits and channel_recombo_block == num_blocks - 1:
            output_channels *= channel_splits
            in_for_first = output_channels
        first_unit = Conv2DResidualUnit(
            input_channels=in_for_first, output_channels=output_channels,
            strides=(1, 1), dilation=(1, 1), transposed=True,
            activation_fn=activation_fn, padding=padding,
            use_shortcut=is_resnet,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        ungrouped.append(first_unit)
        input_channels = output_channels
        if channel_splits and channel_recombo_block == num_blocks - 1:
            in_split = True
            output_channels //= channel_splits

        for level_index, (strides_i, dilation_i, mult) in enumerate(
            zip(ratios[::-1], dilations[::-1], mults[::-1])
        ):
            output_channels = int(np.round(output_channels / mult))

            transition_here = (channel_splits and channel_recombo_block == num_blocks - 2 - level_index)
            if transition_here:
                output_channels *= channel_splits

            # Build unit
            unit = Conv2DResidualUnit(
                input_channels=input_channels, output_channels=output_channels,
                strides=strides_i, dilation=dilation_i, transposed=True,
                activation_fn=activation_fn, padding=padding,
                use_shortcut=is_resnet,
                param_dtype=param_dtype, compute_dtype=compute_dtype,
                deferred_in=in_split,
            )

            input_channels = output_channels

            (grouped if in_split else ungrouped).append(unit)

            if transition_here:
                in_split = True
                output_channels //= channel_splits

        # Final activation + base conv (always inside the split when one exists).
        per_group_out_channels = (
            num_output_channels // channel_splits if channel_splits else num_output_channels
        )
        grouped_or_un = grouped if (channel_splits and in_split) else ungrouped
        grouped_or_un.append(activation_fn)
        # base_conv_last follows the SpectroStream-style padding rule
        # (matches ``magenta_rt.mlx.spectrostream.conv2d``): time uses
        # ``semicausal`` for the causal mode; spatial pad is the
        # symmetric explicit split. Using the same _ss_conv2d helper as
        # the per-block convs keeps this consistent.
        grouped_or_un.append(_ss_conv2d(
            in_features=None if (channel_splits and in_split) else input_channels,
            filters=per_group_out_channels,
            kernel_size=base_conv_size, strides=(1, 1),
            padding=padding, dilation=(1, 1),
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        ))

        if channel_splits and grouped:
            self._ungrouped = _Sequential(ungrouped) if ungrouped else None
            self._grouped = ParallelChannels(
                inner=_Sequential(grouped), num_groups=channel_splits,
            )
        else:
            self._ungrouped = _Sequential(ungrouped) if ungrouped else None
            self._grouped = None
        self.num_output_bins = num_output_bins
        self.num_output_channels = num_output_channels
        # Streaming-mode bookkeeping: number of lookahead frames still
        # to skip across calls (matches sl's
        # ``Lookahead.step`` countdown semantics).
        self._streaming = False
        self._lookahead_remaining = 0

    def __call__(self, x: mx.array) -> mx.array:
        B, T, F = x.shape
        x = x.reshape(B, T, 1, F)
        x = self._input_residual(x)            # [B, T, 1, proj_filters]
        x = x.reshape(B, T, self._input_bins, -1)
        if self._ungrouped is not None:
            x = self._ungrouped(x)
        if self._grouped is not None:
            x = self._grouped(x)
        # Lookahead: drop the first ``decoder_lookahead*total_time_stride``
        # output STFT frames and shorten the sequence (matches sl's
        # ``Lookahead.layer`` with preserve_length_in_layer=False).
        # In streaming mode the drop is spread across calls via
        # ``_lookahead_remaining``.
        if self._lookahead_length > 0:
            if self._streaming:
                drop = min(self._lookahead_remaining, x.shape[1])
                if drop > 0:
                    x = x[:, drop:]
                    self._lookahead_remaining -= drop
            else:
                x = x[:, self._lookahead_length :]
        return x

    def step(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        """Streaming step for SpectroStreamDecoder."""
        B, T, F = x.shape
        x = x.reshape(B, T, 1, F)

        if cache is not None:
            ir_cache = getattr(cache, "input_residual", None)
            un_cache = getattr(cache, "ungrouped", None)
            gr_cache = getattr(cache, "grouped", None)
        else:
            ir_cache = un_cache = gr_cache = None

        if hasattr(self._input_residual, "step"):
            x = self._input_residual.step(x, ir_cache)
        else:
            x = self._input_residual(x)

        x = x.reshape(B, T, self._input_bins, -1)

        if self._ungrouped is not None:
            x = self._ungrouped.step(x, un_cache)

        if self._grouped is not None:
            x = self._grouped.step(x, gr_cache)

        return x


# -----------------------------------------------------------------------------
# STFT / InverseSTFT wrappers (mirroring spectrostream_stft_config)
# -----------------------------------------------------------------------------


class SpectroStreamSTFT(nn.Module):
    """STFT + complex-as-float bitcast, channel-tiling, DC bin removal.

    Mirrors ``spectrostream_stft_config``: forward STFT, then bitcast the
    complex64 output to a 2× larger float32 channel dim (real, imag),
    drop the DC bin (or the Nyquist bin if ``keep_dc=True``), and tile
    across audio channels if needed.
    """

    def __init__(
        self,
        *,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        time_padding: str,
        keep_dc: bool,
        num_channels: int,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(f"num_channels must be even, got {num_channels}")
        self.num_audio_channels = num_channels // 2
        self.fft_length = fft_length
        self.keep_dc = keep_dc
        self.compute_dtype = compute_dtype
        self._stft = STFT(
            frame_length=frame_length, frame_step=frame_step,
            fft_length=fft_length, window_fn=hann_window,
            time_padding=time_padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: channel-major [B, num_audio_channels, T]; promote bare mono
        # [B, T] by inserting the channel axis.
        if x.ndim == 2:
            x = x[:, None, :]
        v = self._stft(x)  # complex [B, F, num_freqs, C_audio]
        if v.shape[3] == 1 and self.num_audio_channels > 1:
            v = mx.tile(v, (1, 1, 1, self.num_audio_channels))
        # Bitcast complex64 -> 2x float32 along the channel axis.
        v_real = mx.stack([v.real, v.imag], axis=-1)  # [..., 2]
        v = v_real.reshape(v.shape[:-1] + (v.shape[-1] * 2,))
        # Drop the DC bin (or the Nyquist if keep_dc).
        v = v[:, :, :-1] if self.keep_dc else v[:, :, 1:]
        return v.astype(self.compute_dtype)

    def step(self, x: mx.array, cache=None) -> mx.array:
        del cache
        return self(x)


class SpectroStreamInverseSTFT(nn.Module):
    """Inverse of :class:`SpectroStreamSTFT`."""

    def __init__(
        self,
        *,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        causal: bool,
        keep_dc: bool,
        num_bins: int,
        num_channels: int,
        compute_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.num_channels = num_channels
        self.keep_dc = keep_dc
        self.fft_length = fft_length
        self.compute_dtype = compute_dtype
        self._istft = InverseSTFT(
            frame_length=frame_length, frame_step=frame_step,
            fft_length=fft_length,
            window_fn=inverse_stft_window_fn(frame_step, hann_window),
            time_padding="causal" if causal else "same",
        )

    def __call__(self, v: mx.array) -> mx.array:
        v = v.astype(mx.float32)
        # v: [B, T, num_bins, num_channels]. Pad back the dropped bin.
        channel_padding = (0, 1) if self.keep_dc else (1, 0)
        v = mx.pad(v, [(0, 0), (0, 0), channel_padding, (0, 0)])
        # Bitcast 2*float32 -> complex64 along the last axis.
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // 2, 2))
        v = (v[..., 0] + 1j * v[..., 1]).astype(mx.complex64)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)
            v = v[..., None]
        out = self._istft(v)  # [B, C_audio, T]
        if out.shape[1] == 1:
            out = out.squeeze(1)  # mono -> [B, T]
        return out

    def step(self, v: mx.array, cache=None) -> mx.array:
        """Streaming inverse-STFT.

        When ``cache`` is provided (an :class:`mlx_pure.cache.OverlapAddCache`),
        each call expects ``T=1`` STFT frame and emits ``frame_step``
        samples using the streaming overlap-add path. Otherwise this
        delegates to non-streaming :meth:`__call__` on the chunk.
        """
        if cache is None:
            return self(v)
        v = v.astype(mx.float32)
        channel_padding = (0, 1) if self.keep_dc else (1, 0)
        v = mx.pad(v, [(0, 0), (0, 0), channel_padding, (0, 0)])
        v = v.reshape(v.shape[:-1] + (v.shape[-1] // 2, 2))
        v = (v[..., 0] + 1j * v[..., 1]).astype(mx.complex64)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)[..., None]
        out = self._istft.step(v, cache)  # [B, C_audio, T]
        if out.shape[1] == 1:
            out = out.squeeze(1)  # mono -> [B, T]
        return out


# -----------------------------------------------------------------------------
# Top-level SpectroStream
# -----------------------------------------------------------------------------


class SpectroStream(nn.Module):
    """SpectroStream codec: STFT, encoder, RVQ quantizer, decoder, InverseSTFT.

    Mirrors ``magenta_rt.mlx.spectrostream.SpectroStream`` for the locked
    feature subset (no embedding_normalizer, no mock decoder, no
    Lookahead/Delay). Provides:

    * :meth:`waveform_to_codes` — audio → STFT → encoder → quantize → codes.
    * :meth:`codes_to_waveform` — codes → unquantize → decoder → InverseSTFT → audio.
    """

    def __init__(
        self,
        *,
        # STFT.
        stft_frame_length: int,
        stft_frame_step: int,
        stft_fft_length: int,
        # Encoder/decoder shape parameters.
        ratios: _Seq[tuple[int, int]],
        mults: _Seq[Union[int, float]],
        dilations: Optional[Union[_Seq[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[mx.array], mx.array] = nn.elu,
        num_bins: int,
        num_channels: int,
        num_features: int,
        causal: bool = True,
        encoder_base_conv_depth: int,
        encoder_base_conv_size: Union[int, tuple[int, int]],
        decoder_base_conv_depth: int,
        decoder_base_conv_size: Union[int, tuple[int, int]],
        # Quantizer.
        quantizer: Optional[ResidualVectorQuantizer] = None,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype = mx.float32,
        keep_dc: bool = False,
        decoder_lookahead: int = 0,
    ):
        super().__init__()
        # STFT and encoder. The shipping config (sl
        # ``spectrostream_stft_config``) uses ``reverse_causal`` for the
        # forward STFT regardless of ``causal``: framing is anchored at
        # the past, so the only padding is on the right (zero-pads the
        # last frame). The InverseSTFT below uses 'causal' or 'same'
        # following ``causal``.
        self.stft = SpectroStreamSTFT(
            frame_length=stft_frame_length, frame_step=stft_frame_step,
            fft_length=stft_fft_length, time_padding="reverse_causal",
            keep_dc=keep_dc, num_channels=num_channels,
            compute_dtype=compute_dtype,
        )
        self.encoder = SpectroStreamEncoder(
            base_conv_depth=encoder_base_conv_depth,
            base_conv_size=encoder_base_conv_size,
            ratios=ratios, mults=mults, dilations=dilations,
            channel_splits=channel_splits,
            channel_recombo_block=channel_recombo_block,
            is_resnet=is_resnet, activation_fn=activation_fn,
            num_input_bins=num_bins, num_input_channels=num_channels,
            num_output_features=num_features, causal=causal,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        # Decoder and InverseSTFT.
        self.decoder = SpectroStreamDecoder(
            base_conv_depth=decoder_base_conv_depth,
            base_conv_size=decoder_base_conv_size,
            ratios=ratios, mults=mults, dilations=dilations,
            channel_splits=channel_splits,
            channel_recombo_block=channel_recombo_block,
            is_resnet=is_resnet, activation_fn=activation_fn,
            num_input_features=num_features,
            num_output_bins=num_bins, num_output_channels=num_channels,
            causal=causal,
            decoder_lookahead=decoder_lookahead,
            param_dtype=param_dtype, compute_dtype=compute_dtype,
        )
        self.inverse_stft = SpectroStreamInverseSTFT(
            frame_length=stft_frame_length, frame_step=stft_frame_step,
            fft_length=stft_fft_length,
            causal=causal, keep_dc=keep_dc,
            num_bins=num_bins, num_channels=num_channels,
            compute_dtype=compute_dtype,
        )
        self.quantizer = quantizer

    def waveform_to_embeddings(self, audio: mx.array) -> mx.array:
        return self.encoder(self.stft(audio))

    def embeddings_to_waveform(self, embeddings: mx.array) -> mx.array:
        return self.inverse_stft(self.decoder(embeddings))

    def waveform_to_codes(self, audio: "mx.array | AudioTree") -> mx.array:
        """Encode a 48 kHz ``[B, C, T]`` waveform into RVQ codes.

        ``audio`` may be a raw mx/numpy array or an ``AudioTree`` (unwrapped to
        its waveform; its ``sample_rate`` must match the codec's ``SAMPLE_RATE``).
        """
        if isinstance(audio, AudioTree):
            if audio.sample_rate != SAMPLE_RATE:
                raise ValueError(
                    f"AudioTree sample_rate {audio.sample_rate} != codec rate "
                    f"{SAMPLE_RATE}; resample before waveform_to_codes."
                )
            audio = mx.array(np.asarray(audio.waveform))
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        return self.quantizer.embeddings_to_codes(self.waveform_to_embeddings(audio))

    def codes_to_waveform(self, codes: mx.array) -> mx.array:
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        embeddings = self.quantizer.codes_to_embeddings(codes)
        return self.embeddings_to_waveform(embeddings)

    # ------------------------------------------------------------------
    # Streaming forwards
    # ------------------------------------------------------------------

    def enable_streaming(self) -> None:
        """Switch the codec into streaming mode.

        Flips every :class:`Conv2D` / :class:`Conv2DTranspose` in the
        encoder/decoder into streaming-step mode (each instance keeps
        its own ``Conv2DCache`` left-context buffer), arms the
        decoder's lookahead countdown, and allocates an
        :class:`mlx_pure.cache.OverlapAddCache` for the InverseSTFT.
        Subsequent :meth:`codes_to_waveform` / :meth:`step_codes_to_waveform`
        calls then maintain state across calls so concatenated per-step
        audio matches a single non-streaming forward on the joined
        codes (sl convention).
        """
        from ..cache import OverlapAddCache
        from ..conv import enable_streaming as _enable
        _enable(self.encoder)
        _enable(self.decoder)
        self.decoder._streaming = True
        self.decoder._lookahead_remaining = self.decoder._lookahead_length
        self._streaming = True
        self._istft_cache = OverlapAddCache()

    def disable_streaming(self) -> None:
        """Restore non-streaming codec behaviour and drop all caches."""
        from ..conv import disable_streaming as _disable
        _disable(self.encoder)
        _disable(self.decoder)
        self.decoder._streaming = False
        self.decoder._lookahead_remaining = 0
        self._streaming = False
        self._istft_cache = None

    def reset_caches(self) -> None:
        """Zero every streaming cache in place — the conv left-context
        buffers in the encoder/decoder and the InverseSTFT overlap
        buffer — keeping them allocated and the codec streaming-armed,
        and drain the lookahead countdown to 0.

        Unlike :meth:`disable_streaming` (which drops the caches to
        ``None``) this preserves their shapes and dtypes. Used to
        neutralize a warmup-allocated codec state so a shipped streaming
        snapshot carries no generation content.
        """
        from ..conv import reset_streaming_caches
        reset_streaming_caches(self.encoder)
        reset_streaming_caches(self.decoder)
        if self._istft_cache is not None and self._istft_cache.buffer is not None:
            self._istft_cache.buffer = mx.zeros_like(self._istft_cache.buffer)
        self.decoder._lookahead_remaining = 0

    def step_codes_to_waveform(self, codes: mx.array) -> mx.array:
        """Streaming ``codes_to_waveform`` for one chunk.

        Requires :meth:`enable_streaming` to have been called first;
        each conv layer holds its own left-context cache and the
        InverseSTFT uses a shared OverlapAddCache.
        """
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        embeddings = self.quantizer.codes_to_embeddings(codes)
        decoded = self.decoder(embeddings)
        if getattr(self, "_streaming", False) and self._istft_cache is not None:
            return self.inverse_stft.step(decoded, cache=self._istft_cache)
        return self.inverse_stft(decoded)

    def step_waveform_to_codes(self, audio: mx.array) -> mx.array:
        return self.waveform_to_codes(audio)
