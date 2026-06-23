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

"""SpectroStream codec.

* :class:`ResidualVectorQuantizer` — codes ↔ embeddings table lookup
  + nearest-neighbor encoding (inference-only).
* :class:`Conv2DResidualUnit` — residual block used by encoder / decoder.
* :class:`SpectroStreamEncoder` / :class:`SpectroStreamDecoder` — Conv2D
  stacks (no STFT wrapping).
* :class:`SpectroStreamSTFT` / :class:`SpectroStreamInverseSTFT` —
  forward / inverse STFT with the SpectroStream bitcast convention.
* :class:`SpectroStream` — top-level wrapper. ``init_cache`` /
  ``remove_cache`` arm the streaming pipeline (per-conv left context,
  decoder lookahead countdown, InverseSTFT overlap buffer).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from ..conv import (
    AveragePooling2D, Conv2D, Conv2DTranspose, ParallelChannels, Upsample2D,
    remove_cache as _remove_cache,
)
from ..signal import STFT, InverseSTFT, hann_window, inverse_stft_window_fn


class LookaheadState(nnx.Variable):
    pass


class ResidualVectorQuantizer(nnx.Module):
    """RVQ codebook table.

    Embedding tensor shape ``[num_quantizers, num_embeddings, embedding_dim]``.
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
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_unique_codes = use_unique_codes
        self.truncation_level = truncation_level
        self.encoded_truncation_level = encoded_truncation_level
        self.param_dtype = param_dtype

        # Codebook init: match the ``default_embed_init`` shape rule —
        # each ``[E, D]`` codebook slab gets variance-scaling-normal
        # init with fan_in = D.
        init_fn = nnx.initializers.variance_scaling(
            1.0, "fan_in", "normal", in_axis=-1, out_axis=(0, 1),
        )
        self.embedding = nnx.Param(
            init_fn(
                rngs.params(),
                (num_quantizers, num_embeddings, embedding_dim),
                param_dtype,
            )
        )

    @property
    def num_expected_input_codes(self) -> int:
        return (
            self.truncation_level
            if self.truncation_level is not None
            else self.num_quantizers
        )

    @property
    def num_expected_output_codes(self) -> int:
        return (
            self.encoded_truncation_level
            if self.encoded_truncation_level is not None
            else self.num_quantizers
        )

    def codes_to_embeddings(self, codes: jnp.ndarray) -> jnp.ndarray:
        """``[B, T, num_input_codebooks]`` int → ``[B, T, embedding_dim]``."""
        if codes.ndim != 3:
            raise ValueError(f"expected 3D input, got {codes.shape=}")
        if codes.dtype not in (jnp.int32, jnp.uint32):
            raise ValueError(f"expected int32/uint32, got {codes.dtype=}")
        codes = codes.astype(jnp.int32)
        if self.use_unique_codes:
            codes = codes % self.num_embeddings
        num_input = codes.shape[-1]
        if num_input > self.num_expected_input_codes:
            raise ValueError(
                f"got {num_input} input codebooks, "
                f"expected ≤ {self.num_expected_input_codes}"
            )
        embedding = self.embedding[...]
        if num_input == 0:
            return jnp.zeros(
                codes.shape[:2] + (self.embedding_dim,), embedding.dtype,
            )
        # Vectorized per-quantizer lookup via advanced indexing:
        # ``embedding[:num_input]`` is ``[Q, num_emb, dim]``;
        # ``codes`` is ``[B, T, Q]``. We want, for each (B, T, q),
        # ``embedding[q, codes[B, T, q], :]``, then sum over q.
        emb_used = embedding[:num_input]                      # [Q, E, D]
        codes_q_first = jnp.moveaxis(codes, -1, 0)            # [Q, B, T]
        q_idx = jnp.arange(num_input)[:, None, None]          # [Q, 1, 1]
        per_q = emb_used[q_idx, codes_q_first]                # [Q, B, T, D]
        return jnp.sum(per_q, axis=0)                         # [B, T, D]

    def embeddings_to_codes(
        self, inputs: jnp.ndarray, num_quantizers: Optional[int] = None,
    ) -> jnp.ndarray:
        """Greedy residual encoding. Input ``[B, T, embedding_dim]``."""
        Q = num_quantizers if num_quantizers is not None else self.num_expected_output_codes
        residual = inputs
        embedding = self.embedding[...]
        codes = []
        for q in range(Q):
            cb = embedding[q]
            distances = (
                jnp.sum(residual ** 2, axis=-1, keepdims=True)
                - 2.0 * (residual @ cb.T)
                + jnp.sum(cb ** 2, axis=-1)
            )
            code_q = jnp.argmin(distances, axis=-1)
            quantized = jnp.take(cb, code_q, axis=0)
            residual = residual - quantized
            codes.append(code_q)
        out = jnp.stack(codes, axis=-1)
        if self.use_unique_codes:
            offsets = jnp.arange(Q) * self.num_embeddings
            out = out + offsets
        return out


def _to_pair(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def _ss_conv2d_paddings(
    padding: str, kernel_size: tuple[int, int], strides: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """``(time_padding, spatial_padding)`` for the SpectroStream helper convention."""
    pad_freq = max((kernel_size[1] - 1) * dilation[1] + 1 - strides[1], 0)
    spatial_pad = (pad_freq // 2, pad_freq - pad_freq // 2)
    time_pad = "semicausal" if padding == "causal" else padding
    return time_pad, spatial_pad


def _ss_conv2d(
    *, in_features: Optional[int], filters: int, kernel_size: tuple[int, int],
    strides: tuple[int, int], padding: str, dilation: tuple[int, int],
    param_dtype: jnp.dtype, dtype: jnp.dtype, rngs: nnx.Rngs,
) -> Conv2D:
    """Build a Conv2D with the SpectroStream-style padding convention."""
    time_pad, spatial_pad = _ss_conv2d_paddings(padding, kernel_size, strides, dilation)
    return Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=kernel_size, strides=strides, dilation_rate=dilation,
        time_padding=time_pad, spatial_padding=spatial_pad,
        use_bias=True, param_dtype=param_dtype, dtype=dtype,
        rngs=rngs,
    )


class Conv2DResidualUnit(nnx.Module):
    """Residual block: act → conv2d (transposed) → act → conv2d."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        strides: tuple[int, int],
        dilation: tuple[int, int],
        transposed: bool,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
        padding: str = "causal",
        use_shortcut: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        strides = _to_pair(strides)
        dilation = _to_pair(dilation)
        self.strides = strides
        self.dilation = dilation
        self.transposed = transposed
        self.padding = padding
        self.use_shortcut = use_shortcut
        self.activation_fn = activation_fn

        resample_kernel_size = (max(3, 2 * strides[0]), max(3, 2 * strides[1]))

        body_layers: list[nnx.Module] = []
        if transposed:
            filters = output_channels
            if strides == (1, 1):
                body_layers.append(activation_fn)
                body_layers.append(_ss_conv2d(
                    in_features=input_channels, filters=filters,
                    kernel_size=(3, 3), strides=(1, 1),
                    padding=padding, dilation=(1, 1),
                    param_dtype=param_dtype, dtype=dtype, rngs=rngs,
                ))
                inner_in = filters
            else:
                body_layers.append(activation_fn)
                body_layers.append(Conv2DTranspose(
                    in_features=input_channels, filters=filters,
                    kernel_size=resample_kernel_size, strides=strides,
                    time_padding=padding, spatial_padding="same",
                    use_bias=True, param_dtype=param_dtype, dtype=dtype,
                    rngs=rngs,
                ))
                inner_in = filters
        else:
            filters = input_channels
            inner_in = input_channels

        body_layers.append(activation_fn)
        body_layers.append(_ss_conv2d(
            in_features=inner_in, filters=filters,
            kernel_size=(3, 3), strides=(1, 1),
            padding=padding, dilation=dilation,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        ))

        if not transposed:
            body_layers.append(activation_fn)
            body_layers.append(_ss_conv2d(
                in_features=filters, filters=output_channels,
                kernel_size=resample_kernel_size, strides=strides,
                padding=padding, dilation=(1, 1),
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
            ))
        self.body = nnx.List(body_layers)

        # Shortcut.
        if use_shortcut:
            shortcut: list[nnx.Module] = []
            sc_in = input_channels
            if strides != (1, 1) and not transposed:
                shortcut.append(AveragePooling2D(
                    pool_size=strides, strides=strides,
                    time_padding="semicausal" if padding == "causal" else padding,
                    spatial_padding="valid",
                ))
            if input_channels != output_channels:
                shortcut.append(_ss_conv2d(
                    in_features=sc_in,
                    filters=output_channels,
                    kernel_size=(1, 1), strides=(1, 1),
                    padding="causal", dilation=(1, 1),
                    param_dtype=param_dtype, dtype=dtype, rngs=rngs,
                ))
                sc_in = output_channels
            if strides != (1, 1) and transposed:
                shortcut.append(Upsample2D(rate=strides))
            self.shortcut = nnx.List(shortcut)
        else:
            self.shortcut = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        body_out = x
        for layer in self.body:
            body_out = layer(body_out)
        if self.shortcut is None:
            return body_out
        sc = x
        for layer in self.shortcut:
            sc = layer(sc)
        return body_out + sc


class _OutputConvsResidual(nnx.Module):
    """1×1 conv stack at the end of the encoder."""

    def __init__(
        self, *, input_channels: int, bottleneck_channels: int, output_channels: int,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
        param_dtype: jnp.dtype = jnp.float32, dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.body_conv = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )
        self.shortcut_act1 = activation_fn
        self.shortcut_conv1 = Conv2D(
            in_features=input_channels, filters=bottleneck_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )
        self.shortcut_act2 = activation_fn
        self.shortcut_conv2 = Conv2D(
            in_features=bottleneck_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        body = self.body_conv(x)
        sc = self.shortcut_act1(x)
        sc = self.shortcut_conv1(sc)
        sc = self.shortcut_act2(sc)
        sc = self.shortcut_conv2(sc)
        return body + sc


class _DecoderInputResidual(nnx.Module):
    """1×1 conv stack at the start of the decoder."""

    def __init__(
        self, *, input_channels: int, output_channels: int,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
        param_dtype: jnp.dtype = jnp.float32, dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.body_conv = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )
        self.shortcut_conv1 = Conv2D(
            in_features=input_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )
        self.shortcut_act = activation_fn
        self.shortcut_conv2 = Conv2D(
            in_features=output_channels, filters=output_channels,
            kernel_size=(1, 1), strides=(1, 1),
            time_padding="semicausal", spatial_padding=(0, 0),
            use_bias=True, param_dtype=param_dtype, dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        body = self.body_conv(x)
        sc = self.shortcut_conv1(x)
        sc = self.shortcut_act(sc)
        sc = self.shortcut_conv2(sc)
        return body + sc

    def init_cache(self, *, batch: int, spatial: int, dtype=jnp.float32) -> None:
        self.body_conv.init_cache(batch=batch, spatial=spatial, dtype=dtype)
        self.shortcut_conv1.init_cache(batch=batch, spatial=spatial, dtype=dtype)
        self.shortcut_conv2.init_cache(batch=batch, spatial=spatial, dtype=dtype)


class SpectroStreamEncoder(nnx.Module):
    """SpectroStream encoder Conv2D stack (post-STFT to bottleneck features)."""

    def __init__(
        self,
        *,
        base_conv_depth: int,
        base_conv_size: Union[int, tuple[int, int]],
        ratios: Sequence[tuple[int, int]],
        mults: Sequence[Union[int, float]],
        dilations: Optional[Union[Sequence[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
        num_input_bins: int = 160,
        num_input_channels: int = 4,
        num_output_features: int = 64,
        causal: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
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

        prefix_first_in = (
            num_input_channels // channel_splits if channel_splits else num_input_channels
        )

        base_conv = _ss_conv2d(
            in_features=prefix_first_in, filters=base_conv_depth,
            kernel_size=base_conv_size, strides=(1, 1),
            padding=padding, dilation=(1, 1),
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )

        prefix: list[nnx.Module] = []
        post: list[nnx.Module] = []
        if channel_splits and channel_recombo_block == 0:
            base_conv = _ss_conv2d(
                in_features=num_input_channels, filters=base_conv_depth,
                kernel_size=base_conv_size, strides=(1, 1),
                padding=padding, dilation=(1, 1),
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
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
                in_for_block = input_channels * channel_splits
            block = Conv2DResidualUnit(
                input_channels=in_for_block, output_channels=output_channels,
                strides=strides_i, dilation=dilation_i, transposed=False,
                activation_fn=activation_fn, padding=padding,
                use_shortcut=is_resnet,
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
            )
            if channel_splits and level_index < channel_recombo_block:
                prefix.append(block)
            else:
                post.append(block)
            input_channels = output_channels

        in_for_bottleneck = input_channels
        if channel_splits and channel_recombo_block == num_blocks - 1:
            in_for_bottleneck = input_channels * channel_splits
        bottleneck = Conv2DResidualUnit(
            input_channels=in_for_bottleneck, output_channels=output_channels,
            strides=(1, 1), dilation=(1, 1), transposed=False,
            activation_fn=activation_fn, padding=padding,
            use_shortcut=is_resnet,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        post.append(bottleneck)

        if channel_splits and prefix:
            self._prefix = ParallelChannels(
                inner=nnx.Sequential(*prefix), num_groups=channel_splits,
            )
        else:
            self._prefix = nnx.Sequential(*prefix) if prefix else None
        self._post = nnx.Sequential(*post) if post else None

        flat_channels = curr_num_bins * output_channels
        self._flat_pre_channels = flat_channels
        # The output-convs shortcut bottleneck is the flattened channel count
        # (curr_num_bins * output_channels), NOT output_channels alone — the
        # real codec's ``output_convs/shortcut_layer/conv1x1_b1`` projects
        # flat_channels → flat_channels (confirmed by the Linen encoder
        # weights: conv1x1_b1 is [1, 1, flat, flat], conv1x1_b2 [1, 1, flat,
        # num_features]). Building it at output_channels left the encoder
        # un-loadable (shape mismatch on conv1x1_b1).
        self._output_convs = _OutputConvsResidual(
            input_channels=flat_channels,
            bottleneck_channels=flat_channels,
            output_channels=num_output_features,
            activation_fn=activation_fn,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.num_output_features = num_output_features

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self._prefix is not None:
            x = self._prefix(x)
        if self._post is not None:
            x = self._post(x)
        x = rearrange(x, "b t s c -> b t 1 (s c)")
        x = self._output_convs(x)
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)
        return x


class SpectroStreamDecoder(nnx.Module):
    """SpectroStream decoder Conv2D stack (mirror image of the encoder).

    Streaming lifecycle behavior:
    * ``streaming`` boolean attribute serves as the global dispatch flag
      (set recursively via ``model.set_attributes(streaming=True)``).
    * ``_lookahead_remaining`` tracks the transient leading output frames zeroed
      out during streaming initialization. Tracked dynamically as an ``nnx.Variable``
      subclass (``LookaheadState``) so it participates in JAX tracing, using a
      broadcasted mask to keep the step output shapes constant under JIT.
    """

    def __init__(
        self,
        *,
        base_conv_depth: int,
        base_conv_size: Union[int, tuple[int, int]],
        ratios: Sequence[tuple[int, int]],
        mults: Sequence[Union[int, float]],
        dilations: Optional[Union[Sequence[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
        num_input_features: int = 64,
        num_output_bins: int = 160,
        num_output_channels: int = 4,
        causal: bool = True,
        decoder_lookahead: int = 0,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
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

        self._input_residual = _DecoderInputResidual(
            input_channels=num_input_features,
            output_channels=proj_filters,
            activation_fn=activation_fn,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self._input_bins = input_bins

        ungrouped: list[nnx.Module] = []
        grouped: list[nnx.Module] = []
        in_split = False

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
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
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
            transition_here = (
                channel_splits and channel_recombo_block == num_blocks - 2 - level_index
            )
            if transition_here:
                output_channels *= channel_splits
            actual_in = (
                input_channels // channel_splits
                if (channel_splits and in_split and len(grouped) == 0)
                else input_channels
            )
            unit = Conv2DResidualUnit(
                input_channels=actual_in, output_channels=output_channels,
                strides=strides_i, dilation=dilation_i, transposed=True,
                activation_fn=activation_fn, padding=padding,
                use_shortcut=is_resnet,
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
            )
            input_channels = output_channels
            (grouped if in_split else ungrouped).append(unit)
            if transition_here:
                in_split = True
                output_channels //= channel_splits

        per_group_out_channels = (
            num_output_channels // channel_splits if channel_splits else num_output_channels
        )
        grouped_or_un = grouped if (channel_splits and in_split) else ungrouped
        actual_final_in = (
            input_channels // channel_splits
            if (channel_splits and in_split and len(grouped) == 0)
            else input_channels
        )
        grouped_or_un.append(activation_fn)
        grouped_or_un.append(_ss_conv2d(
            in_features=actual_final_in,
            filters=per_group_out_channels,
            kernel_size=base_conv_size, strides=(1, 1),
            padding=padding, dilation=(1, 1),
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        ))

        if channel_splits and grouped:
            self._ungrouped = nnx.Sequential(*ungrouped) if ungrouped else None
            self._grouped = ParallelChannels(
                inner=nnx.Sequential(*grouped), num_groups=channel_splits,
            )
        else:
            self._ungrouped = nnx.Sequential(*ungrouped) if ungrouped else None
            self._grouped = None
        self.num_output_bins = num_output_bins
        self.num_output_channels = num_output_channels

        self.streaming: bool = False
        self._lookahead_remaining = LookaheadState(jnp.array(0, dtype=jnp.int32))

    def init_cache(self, *, batch: int = 1, dtype=jnp.float32) -> None:
        self._lookahead_remaining[...] = self._lookahead_length

        # The pre-loop input residual operates on the un-binned input
        # (``[B, T, 1, F]``) before ``SpectroStreamDecoder.__call__``
        # reshapes the channel dim into ``self._input_bins`` spatial
        # rows, so its convs see ``spatial=1``. The ungrouped/grouped
        # iteration below would otherwise miss this block entirely.
        self._input_residual.init_cache(batch=batch, spatial=1, dtype=dtype)

        curr_spatial = self._input_bins

        if self._ungrouped is not None:
            for layer in self._ungrouped.layers:
                if isinstance(layer, Conv2DResidualUnit):
                    if layer.shortcut is not None:
                        for sc_layer in layer.shortcut:
                            if isinstance(sc_layer, Conv2D):
                                sc_layer.init_cache(batch=batch, spatial=curr_spatial, dtype=dtype)
                    _, sw = layer.strides
                    curr_spatial = curr_spatial * sw
                    if hasattr(layer.body[1], "init_cache"):
                        layer.body[1].init_cache(batch=batch, spatial=curr_spatial, dtype=dtype)
                    if hasattr(layer.body[3], "init_cache"):
                        layer.body[3].init_cache(batch=batch, spatial=curr_spatial, dtype=dtype)
                elif isinstance(layer, Conv2D):
                    layer.init_cache(batch=batch, spatial=curr_spatial, dtype=dtype)

        if self._grouped is not None:
            g_batch = self._grouped.num_groups * batch
            for layer in self._grouped.inner.layers:
                if isinstance(layer, Conv2DResidualUnit):
                    if layer.shortcut is not None:
                        for sc_layer in layer.shortcut:
                            if isinstance(sc_layer, Conv2D):
                                sc_layer.init_cache(batch=g_batch, spatial=curr_spatial, dtype=dtype)
                    _, sw = layer.strides
                    curr_spatial = curr_spatial * sw
                    if hasattr(layer.body[1], "init_cache"):
                        layer.body[1].init_cache(batch=g_batch, spatial=curr_spatial, dtype=dtype)
                    if hasattr(layer.body[3], "init_cache"):
                        layer.body[3].init_cache(batch=g_batch, spatial=curr_spatial, dtype=dtype)
                elif isinstance(layer, Conv2D):
                    layer.init_cache(batch=g_batch, spatial=curr_spatial, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(x, "b t f -> b t 1 f")
        x = self._input_residual(x)
        x = rearrange(x, "b t 1 (bins c) -> b t bins c", bins=self._input_bins)
        if self._ungrouped is not None:
            x = self._ungrouped(x)
        if self._grouped is not None:
            x = self._grouped(x)
        if self._lookahead_length > 0:
            if self.streaming:
                drop = jnp.minimum(self._lookahead_remaining[...], x.shape[1])
                # Ensure drop is integer for arange comparison
                if drop.dtype != jnp.int32:
                    drop = drop.astype(jnp.int32)
                mask = jnp.arange(x.shape[1])[None, :, None, None] >= drop
                x = x * mask
                self._lookahead_remaining[...] = self._lookahead_remaining[...] - drop
            else:
                x = x[:, self._lookahead_length:]
        return x


class SpectroStreamSTFT(nnx.Module):
    """STFT + complex-as-float bitcast, channel-tiling, DC bin removal."""

    def __init__(
        self,
        *,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        time_padding: str,
        keep_dc: bool,
        num_channels: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        if num_channels % 2 != 0:
            raise ValueError(f"num_channels must be even, got {num_channels}")
        self.num_audio_channels = num_channels // 2
        self.fft_length = fft_length
        self.keep_dc = keep_dc
        self.dtype = dtype
        self._stft = STFT(
            frame_length=frame_length, frame_step=frame_step,
            fft_length=fft_length, window_fn=hann_window,
            time_padding=time_padding,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Audio arrives channel-major [B, C_audio, T]; promote bare mono
        # [B, T] by inserting the channel axis.
        if x.ndim == 2:
            x = x[:, None, :]
        v = self._stft(x)  # complex [B, F, num_freqs, C_audio]
        if v.shape[3] == 1 and self.num_audio_channels > 1:
            v = jnp.tile(v, (1, 1, 1, self.num_audio_channels))
        # Bitcast complex64 -> 2x float32 along channel axis.
        v_real = jnp.stack([v.real, v.imag], axis=-1)
        v = rearrange(v_real, "... c two -> ... (c two)")
        v = v[:, :, :-1] if self.keep_dc else v[:, :, 1:]
        return v.astype(self.dtype)


class SpectroStreamInverseSTFT(nnx.Module):
    """Inverse of :class:`SpectroStreamSTFT` (full-seq + streaming step)."""

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
        dtype: jnp.dtype = jnp.float32,
    ):
        self.num_bins = num_bins
        self.num_channels = num_channels
        self.keep_dc = keep_dc
        self.fft_length = fft_length
        self.dtype = dtype
        self._istft = InverseSTFT(
            frame_length=frame_length, frame_step=frame_step,
            fft_length=fft_length,
            window_fn=inverse_stft_window_fn(frame_step, hann_window),
            time_padding="causal" if causal else "same",
        )

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        self._istft.init_cache(
            batch=batch, num_channels=self.num_channels // 2, dtype=dtype,
        )

    def _bitcast(self, v: jnp.ndarray) -> jnp.ndarray:
        v = v.astype(jnp.float32)
        channel_padding = (0, 1) if self.keep_dc else (1, 0)
        v = jnp.pad(v, [(0, 0), (0, 0), channel_padding, (0, 0)])
        v = rearrange(v, "... (c two) -> ... c two", two=2)
        v = (v[..., 0] + 1j * v[..., 1]).astype(jnp.complex64)
        if v.shape[-1] == 1:
            v = v.squeeze(-1)[..., None]
        return v

    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        v = self._bitcast(v)
        if self._istft.streaming:
            out = self._istft.step(v)  # [B, C_audio, T]
        else:
            out = self._istft(v)  # [B, C_audio, T]
        if out.shape[1] == 1:
            out = out.squeeze(1)  # mono -> [B, T]
        return out


class SpectroStream(nnx.Module):
    """SpectroStream codec: STFT, encoder, RVQ quantizer, decoder, InverseSTFT."""

    def __init__(
        self,
        *,
        sample_rate: int,
        # STFT.
        stft_frame_length: int,
        stft_frame_step: int,
        stft_fft_length: int,
        # Encoder/decoder shape parameters.
        ratios: Sequence[tuple[int, int]],
        mults: Sequence[Union[int, float]],
        dilations: Optional[Union[Sequence[tuple[int, int]], tuple[int, int]]] = None,
        channel_splits: Optional[int] = None,
        channel_recombo_block: int = -1,
        is_resnet: bool = True,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nnx.elu,
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
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        keep_dc: bool = False,
        decoder_lookahead: int = 0,
        rngs: nnx.Rngs = None,
    ):
        self.sample_rate = sample_rate
        self.stft = SpectroStreamSTFT(
            frame_length=stft_frame_length, frame_step=stft_frame_step,
            fft_length=stft_fft_length, time_padding="reverse_causal",
            keep_dc=keep_dc, num_channels=num_channels,
            dtype=dtype,
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
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
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
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.inverse_stft = SpectroStreamInverseSTFT(
            frame_length=stft_frame_length, frame_step=stft_frame_step,
            fft_length=stft_fft_length,
            causal=causal, keep_dc=keep_dc,
            num_bins=num_bins, num_channels=num_channels,
            dtype=dtype,
        )
        self.quantizer = quantizer

    def load_state_dict(self, state_dict: dict) -> None:
        """Populate all internal parameters from a nested state dictionary."""
        from .load_weights import load_spectrostream_weights
        load_spectrostream_weights(self, state_dict)

    # --- non-streaming API -------------------------------------------------

    def waveform_to_embeddings(self, audio: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(self.stft(audio))

    def embeddings_to_waveform(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        return self.inverse_stft(self.decoder(embeddings))

    def waveform_to_codes(self, audio: jnp.ndarray) -> jnp.ndarray:
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        return self.quantizer.embeddings_to_codes(self.waveform_to_embeddings(audio))

    def codes_to_waveform(self, codes: jnp.ndarray) -> jnp.ndarray:
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        embeddings = self.quantizer.codes_to_embeddings(codes)
        return self.embeddings_to_waveform(embeddings)

    # --- streaming lifecycle ----------------------------------------------

    def init_cache(self, *, batch: int = 1, dtype=jnp.float32) -> None:
        """Arm the codec for streaming. Call after
        ``model.set_attributes(streaming=True)``.
        """
        self.decoder.init_cache(batch=batch, dtype=dtype)
        self.inverse_stft.init_cache(batch=batch, dtype=dtype)

    def remove_cache(self) -> None:
        _remove_cache(self)
        self.decoder._lookahead_remaining[...] = 0

    def step_codes_to_waveform(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Streaming ``codes_to_waveform`` for one chunk."""
        if self.quantizer is None:
            raise RuntimeError("quantizer not configured")
        embeddings = self.quantizer.codes_to_embeddings(codes)
        decoded = self.decoder(embeddings)
        return self.inverse_stft(decoded)

    def step_waveform_to_codes(self, audio: jnp.ndarray) -> jnp.ndarray:
        return self.waveform_to_codes(audio)
