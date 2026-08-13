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

"""2D convolution wrappers for the SpectroStream codec.

Mirrors ``sequence_layers.mlx.convolution2d`` for the locked subset
(the modes the SpectroStream config actually uses). Kernel layout
matches sl: ``[filters, kH, kW, in_channels // groups]`` so the
parameter tree lines up for checkpoint loading via
:mod:`mlx_pure.load_weights`.

Padding modes supported: ``'valid'``, ``'same'``, ``'causal'``,
``'semicausal'``, ``'reverse_causal'``. The Magenta-RT shipping
SpectroStream uses ``'semicausal'`` (time) + explicit spatial padding.

Each layer exposes a streaming :meth:`step` interface alongside the
full-sequence :meth:`__call__`. With a :class:`Conv2DCache` threaded
through, ``Conv2D.step`` and ``Conv2DTranspose.step`` are stateful:

* ``Conv2D``: holds a left-context ring buffer of size
  ``effective_kernel_t - 1`` along the time axis. For
  ``time_padding='causal'`` (any stride_t) and ``'semicausal'``
  (stride_t=1), concatenated streaming output is bit-equivalent to a
  single non-streaming call on the concatenated input.
* ``Conv2DTranspose``: holds a ``kernel_t - stride_t`` overlap buffer
  for ``time_padding='causal'``; per-step output length is
  ``T_in * stride_t``. With the SpectroStream / sl trim convention
  (drop ``kernel_t - stride_t`` from the *right*), concatenated
  streaming outputs match the non-streaming trim sample-for-sample
  on the leading ``T_full * stride_t`` window.

When ``cache`` is ``None``, ``step`` falls back to a stateless
chunk-based forward (per-chunk edge artifacts present, matching
``MagentaRT2Sampler.step``'s default mode). The other helper layers
(``AveragePooling2D``, ``Upsample2D``, ``ParallelChannels``) are
stateless in time so their ``step`` is identical to ``__call__``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence as _Seq, Union

import mlx.core as mx
import mlx.nn as nn


class Conv2DCache:
    """Streaming left-context buffer for :class:`Conv2D` / related layers.

    ``buffer`` holds the trailing input context (or overlap, for the
    transposed conv) carried across :meth:`step` calls; it starts
    unallocated and is lazily allocated on the first streaming step.
    """

    def __init__(self) -> None:
        self.buffer: Optional[mx.array] = None

    def empty(self) -> bool:
        return self.buffer is None

    def reset(self) -> None:
        """Zero the left-context buffer in place, keeping it allocated.

        Unlike dropping the cache (``= None``), this preserves the
        buffer's shape/dtype so a warmup-allocated codec state can be
        neutralized without re-deriving conv buffer shapes.
        """
        if self.buffer is not None:
            self.buffer = mx.zeros_like(self.buffer)

    @property
    def state(self):
        return (self.buffer,)

    @state.setter
    def state(self, v):
        (self.buffer,) = v


def enable_streaming(module: nn.Module) -> None:
    """Recursively flip every :class:`Conv2D` / :class:`Conv2DTranspose`
    in ``module`` into streaming mode and reset their internal
    left-context caches.
    """
    for m in module.modules():
        if isinstance(m, (Conv2D, Conv2DTranspose)):
            m._streaming = True
            m._streaming_cache = None


def disable_streaming(module: nn.Module) -> None:
    """Restore non-streaming behaviour and drop streaming caches."""
    for m in module.modules():
        if isinstance(m, (Conv2D, Conv2DTranspose)):
            m._streaming = False
            m._streaming_cache = None


def reset_streaming_caches(module: nn.Module) -> None:
    """Zero every :class:`Conv2D` / :class:`Conv2DTranspose` streaming
    left-context buffer in ``module`` in place, keeping the buffers
    allocated and the layers streaming-armed.

    Unlike :func:`disable_streaming` (which drops the caches to ``None``)
    this leaves the cache shapes/dtypes intact — used to neutralize a
    warmup-allocated codec state so the shipped snapshot carries no
    generation content.
    """
    for m in module.modules():
        if isinstance(m, (Conv2D, Conv2DTranspose)):
            if m._streaming_cache is not None:
                m._streaming_cache.reset()


def _normalize_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def _effective_kernel_size(kernel_size: int, dilation_rate: int) -> int:
    return (kernel_size - 1) * dilation_rate + 1


def _explicit_padding(padding, kernel_size: int, stride: int, dilation_rate: int) -> tuple[int, int]:
    """Compute (pad_left, pad_right) for one axis."""
    if not isinstance(padding, str):
        return tuple(padding)
    ek = _effective_kernel_size(kernel_size, dilation_rate)
    if padding in ("causal", "causal_valid"):
        return (ek - 1, 0)
    if padding == "semicausal":
        pad_left = max(ek - stride, 0)
        return (pad_left, ek - 1 - pad_left)
    if padding in ("reverse_causal", "reverse_causal_valid"):
        return (0, ek - 1)
    if padding == "same":
        pad = ek - 1
        return (pad // 2, pad - pad // 2)
    if padding == "valid":
        return (0, 0)
    raise ValueError(f"unsupported padding mode: {padding!r}")


class Conv2D(nn.Module):
    """2D convolution with separate time + spatial padding.

    Input shape: ``[B, T, S, C_in]`` where T is the time axis and S is
    the spatial (frequency) axis. Output: ``[B, T_out, S_out, C_out]``.
    """

    def __init__(
        self,
        *,
        in_features: Optional[int] = None,
        filters: int,
        kernel_size: Union[int, _Seq[int]] = (1, 1),
        strides: Union[int, _Seq[int]] = (1, 1),
        dilation_rate: Union[int, _Seq[int]] = (1, 1),
        time_padding: str = "valid",
        spatial_padding: Union[str, tuple[int, int]] = "same",
        groups: int = 1,
        use_bias: bool = True,
        activation: Optional[Callable[[mx.array], mx.array]] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.filters = filters
        self.kernel_size = _normalize_2tuple(kernel_size)
        self.strides = _normalize_2tuple(strides)
        self.dilation_rate = _normalize_2tuple(dilation_rate)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.groups = groups
        self.use_bias = use_bias
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        # Streaming-mode flag (toggled by enable_streaming/disable_streaming
        # on the enclosing module tree). When True, ``__call__`` routes
        # through ``step`` with a per-instance Conv2DCache so left-context
        # is preserved across calls.
        self._streaming = False
        self._streaming_cache: Optional[Conv2DCache] = None

        # Kernel layout matches sl: [filters, kH, kW, in/groups]. When
        # ``in_features`` is None the kernel is allocated lazily on the
        # first ``__call__`` based on the actual input channel count
        # (matches sl's deferred Conv2D, needed for ``channel_splits``).
        kh, kw = self.kernel_size
        if in_features is not None:
            self.kernel = mx.zeros((filters, kh, kw, in_features // groups), dtype=param_dtype)
        else:
            self.kernel = None  # deferred
        if use_bias:
            self.bias = mx.zeros((filters,), dtype=param_dtype)
        else:
            self.bias = None

    def _ensure_initialized(self, in_features: int) -> None:
        if self.kernel is not None:
            return
        kh, kw = self.kernel_size
        self.in_features = in_features
        self.kernel = mx.zeros(
            (self.filters, kh, kw, in_features // self.groups), dtype=self.param_dtype,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self._streaming:
            if self._streaming_cache is None:
                self._streaming_cache = Conv2DCache()
            return self.step(x, self._streaming_cache)
        # x: [B, T, S, C_in]
        self._ensure_initialized(x.shape[-1])
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        x = x.astype(dtype)
        time_pad = _explicit_padding(self.time_padding, self.kernel_size[0], self.strides[0], self.dilation_rate[0])
        spatial_pad = _explicit_padding(self.spatial_padding, self.kernel_size[1], self.strides[1], self.dilation_rate[1])
        if any(time_pad) or any(spatial_pad):
            x = mx.pad(x, [(0, 0), time_pad, spatial_pad, (0, 0)])
        y = mx.conv_general(
            x,
            self.kernel.astype(dtype),
            stride=self.strides,
            padding=((0, 0), (0, 0)),
            kernel_dilation=self.dilation_rate,
            input_dilation=(1, 1),
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def step(self, x: mx.array, cache: Optional[Conv2DCache] = None) -> mx.array:
        """Streaming forward.

        With a :class:`Conv2DCache` the layer maintains a left-context
        ring buffer along the time axis so that concatenating ``step``
        outputs across chunks matches a single non-streaming call on the
        concatenated input — exactly, when ``time_padding`` has zero
        right-pad (``'causal'``, or ``'semicausal'`` with stride_t=1)
        and ``T_in`` is a multiple of stride_t.

        Without a cache, falls back to the non-streaming forward on the
        chunk; per-chunk edge artifacts are not corrected.
        """
        if cache is None:
            return self(x)
        self._ensure_initialized(x.shape[-1])
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        x = x.astype(dtype)

        kt, ks = self.kernel_size
        st, ss = self.strides
        dt, ds = self.dilation_rate
        time_pad = _explicit_padding(self.time_padding, kt, st, dt)
        spatial_pad = _explicit_padding(self.spatial_padding, ks, ss, ds)
        pad_left, pad_right = time_pad
        if pad_right > 0:
            raise NotImplementedError(
                f"Conv2D.step: stateful streaming requires zero right-pad "
                f"(time_padding={self.time_padding!r}, stride_t={st} gives "
                f"pad_right={pad_right}). Drop the cache to fall back to "
                f"chunk-based step."
            )
        if x.shape[1] % st != 0:
            raise ValueError(
                f"Conv2D.step: T_in ({x.shape[1]}) must be a multiple of "
                f"stride_t ({st})"
            )

        if cache.buffer is None:
            shape = (x.shape[0], pad_left) + tuple(x.shape[2:])
            cache.buffer = mx.zeros(shape, dtype=x.dtype)
        else:
            cache.buffer = cache.buffer.astype(x.dtype)

        combined = mx.concatenate([cache.buffer, x], axis=1)
        if any(spatial_pad):
            padded = mx.pad(combined, [(0, 0), (0, 0), spatial_pad, (0, 0)])
        else:
            padded = combined
        y = mx.conv_general(
            padded,
            self.kernel.astype(dtype),
            stride=self.strides,
            padding=((0, 0), (0, 0)),
            kernel_dilation=self.dilation_rate,
            input_dilation=(1, 1),
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        # Save trailing pad_left input frames as the new left context.
        if pad_left > 0:
            cache.buffer = combined[:, -pad_left:]
        return y


class Conv2DTranspose(nn.Module):
    """2D transposed convolution. Input ``[B, T, S, C_in]`` →
    ``[B, T*stride_t, S*stride_s, C_out]``. Time padding uses the same
    mode strings as :class:`Conv2D`.
    """

    def __init__(
        self,
        *,
        in_features: Optional[int] = None,
        filters: int,
        kernel_size: Union[int, _Seq[int]],
        strides: Union[int, _Seq[int]],
        time_padding: str = "same",
        spatial_padding: Union[str, tuple[int, int]] = "same",
        use_bias: bool = True,
        activation: Optional[Callable[[mx.array], mx.array]] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if time_padding not in ("same", "causal"):
            raise NotImplementedError(
                f"Conv2DTranspose: only 'same' and 'causal' time_padding supported; got {time_padding!r}"
            )
        self.in_features = in_features
        self.filters = filters
        self.kernel_size = _normalize_2tuple(kernel_size)
        self.strides = _normalize_2tuple(strides)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.use_bias = use_bias
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self._streaming = False
        self._streaming_cache: Optional[Conv2DCache] = None

        kh, kw = self.kernel_size
        # Kernel layout matches sl: [filters, kH, kW, in_features].
        if in_features is not None:
            self.kernel = mx.zeros((filters, kh, kw, in_features), dtype=param_dtype)
        else:
            self.kernel = None  # deferred
        if use_bias:
            self.bias = mx.zeros((filters,), dtype=param_dtype)
        else:
            self.bias = None

    def _ensure_initialized(self, in_features: int) -> None:
        if self.kernel is not None:
            return
        kh, kw = self.kernel_size
        self.in_features = in_features
        self.kernel = mx.zeros((self.filters, kh, kw, in_features), dtype=self.param_dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if self._streaming:
            if self._streaming_cache is None:
                self._streaming_cache = Conv2DCache()
            return self.step(x, self._streaming_cache)
        self._ensure_initialized(x.shape[-1])
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        x = x.astype(dtype)
        kh, kw = self.kernel_size
        sh, sw = self.strides
        # Use mx.conv_transpose with the [filters, kH, kW, in] kernel layout.
        # mx.conv_transpose accepts padding='valid' / 'same' or explicit pads.
        y = mx.conv_transpose2d(
            x,
            self.kernel.astype(dtype),
            stride=self.strides,
            padding=(0, 0),
        )
        # Trim to match sl's `time_padding` and `spatial_padding`.
        # Convention (matches sequence_layers/mlx/convolution.py
        # `_transpose_conv_output_trim`): for ``causal`` we trim the
        # right (future) end so that output[t] only depends on past
        # input frames; for ``same`` we split the trim half-and-half.
        if self.time_padding == "same":
            time_trim_left = (kh - sh) // 2
            time_trim_right = kh - sh - time_trim_left
        elif self.time_padding == "causal":
            time_trim_left = 0
            time_trim_right = max(0, kh - sh)
        else:
            time_trim_left = time_trim_right = 0

        if isinstance(self.spatial_padding, str):
            if self.spatial_padding == "same":
                sp_trim_left = (kw - sw) // 2
                sp_trim_right = kw - sw - sp_trim_left
            elif self.spatial_padding == "causal":
                sp_trim_left = 0
                sp_trim_right = max(0, kw - sw)
            else:
                sp_trim_left = sp_trim_right = 0
        else:
            sp_trim_left, sp_trim_right = self.spatial_padding

        if time_trim_left or time_trim_right:
            T = y.shape[1]
            y = y[:, time_trim_left : T - time_trim_right]
        if sp_trim_left or sp_trim_right:
            S = y.shape[2]
            y = y[:, :, sp_trim_left : S - sp_trim_right]

        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def step(self, x: mx.array, cache: Optional[Conv2DCache] = None) -> mx.array:
        """Streaming transposed forward with overlap-add cache.

        Implements the standard streaming-transposed-conv pattern: each
        ``conv_transpose`` output of length ``T_in*S + (K-S)`` overlaps
        the previous step's output by ``K-S`` samples; the cache holds
        that overlap (pre-bias, pre-activation). Per-step output length
        is ``T_in * stride_t``.

        Note: for ``time_padding='causal'``, the *first* step's output
        contains a ``K-S``-sample warmup transient that the
        non-streaming forward trims away. After that, concatenated
        per-step outputs match the non-streaming forward shifted by
        ``K-S`` samples (i.e., ``streaming_concat[K-S:]`` equals
        ``non_streaming[:len(streaming_concat) - (K-S)]``).
        """
        if cache is None:
            return self(x)
        if self.time_padding != "causal":
            raise NotImplementedError(
                f"Conv2DTranspose.step: streaming only supports "
                f"time_padding='causal' (got {self.time_padding!r})"
            )
        self._ensure_initialized(x.shape[-1])
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        x = x.astype(dtype)
        kh, kw = self.kernel_size
        sh, sw = self.strides
        T_in = x.shape[1]
        T_emit = T_in * sh
        overlap = max(kh - sh, 0)

        raw = mx.conv_transpose2d(
            x, self.kernel.astype(dtype), stride=self.strides, padding=(0, 0),
        )
        # Spatial trim (mirrors non-streaming __call__).
        if isinstance(self.spatial_padding, str):
            if self.spatial_padding == "same":
                sp_trim_left = (kw - sw) // 2
                sp_trim_right = kw - sw - sp_trim_left
            elif self.spatial_padding == "causal":
                sp_trim_left = 0
                sp_trim_right = max(0, kw - sw)
            else:
                sp_trim_left = sp_trim_right = 0
        else:
            sp_trim_left, sp_trim_right = self.spatial_padding
        if sp_trim_left or sp_trim_right:
            S_total = raw.shape[2]
            raw = raw[:, :, sp_trim_left : S_total - sp_trim_right]

        if cache.buffer is None:
            cache.buffer = mx.zeros(
                (raw.shape[0], overlap) + tuple(raw.shape[2:]), dtype=dtype,
            )
        else:
            cache.buffer = cache.buffer.astype(dtype)

        if overlap > 0:
            head = raw[:, :overlap] + cache.buffer
            merged = mx.concatenate([head, raw[:, overlap:]], axis=1)
        else:
            merged = raw
        emit = merged[:, :T_emit]
        if self.bias is not None:
            emit = emit + self.bias.astype(dtype)
        if self.activation is not None:
            emit = self.activation(emit)
        if overlap > 0:
            cache.buffer = merged[:, T_emit:]
        return emit


class AveragePooling2D(nn.Module):
    """2D average pooling. Time padding uses the :class:`Conv2D` modes."""

    def __init__(
        self,
        *,
        pool_size: Union[int, _Seq[int]],
        strides: Optional[Union[int, _Seq[int]]] = None,
        time_padding: str = "valid",
        spatial_padding: Union[str, tuple[int, int]] = "valid",
    ):
        super().__init__()
        self.pool_size = _normalize_2tuple(pool_size)
        self.strides = _normalize_2tuple(strides) if strides is not None else self.pool_size
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding

    def __call__(self, x: mx.array) -> mx.array:
        time_pad = _explicit_padding(self.time_padding, self.pool_size[0], self.strides[0], 1)
        spatial_pad = _explicit_padding(self.spatial_padding, self.pool_size[1], self.strides[1], 1)
        if any(time_pad) or any(spatial_pad):
            x = mx.pad(x, [(0, 0), time_pad, spatial_pad, (0, 0)])
        # Implement avg pool via conv with constant kernel.
        ph, pw = self.pool_size
        C = x.shape[-1]
        kernel = mx.full((C, ph, pw, 1), 1.0 / (ph * pw), dtype=x.dtype)
        y = mx.conv_general(
            x, kernel,
            stride=self.strides, padding=((0, 0), (0, 0)),
            kernel_dilation=(1, 1), input_dilation=(1, 1),
            groups=C,
        )
        return y

    def step(self, x: mx.array, cache: Optional[Conv2DCache] = None) -> mx.array:
        del cache
        return self(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor 2D upsampling by integer ``rate``."""

    def __init__(self, rate: Union[int, _Seq[int]]):
        super().__init__()
        self.rate = _normalize_2tuple(rate)

    def __call__(self, x: mx.array) -> mx.array:
        rh, rw = self.rate
        # Repeat along the time and spatial axes.
        x = mx.repeat(x, rh, axis=1)
        x = mx.repeat(x, rw, axis=2)
        return x

    def step(self, x: mx.array, cache: Optional[Conv2DCache] = None) -> mx.array:
        del cache
        return self(x)


class ParallelChannels(nn.Module):
    """Split the channel axis into ``num_groups`` groups, run a shared
    ``inner`` module on each group independently, then concat back
    along the channel axis.

    Mirrors ``sequence_layers.mlx.ParallelChannels`` with
    ``combination=CONCAT``. The inner module is *one* shared module
    (no separate parameters per group) — sl behaviour.

    Implementation: groups are reshaped into the *batch* axis, the
    inner module runs **once** on the bigger batch, then the output
    is reshaped back. Equivalent to looping over groups but lets each
    inner streaming-cache slot hold its own per-(group, batch) state
    naturally — no swap-in / swap-out, no per-group bookkeeping list.
    Mirrors the rewrite in :class:`magenta_rt.nnx.conv.ParallelChannels`.
    """

    def __init__(self, *, inner: nn.Module, num_groups: int):
        super().__init__()
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self.inner = inner
        self.num_groups = num_groups

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, S, C]. Split last axis into (num_groups, per_group)
        # then move the groups into the batch axis.
        B, T, S, C = x.shape
        if C % self.num_groups != 0:
            raise ValueError(
                f"channel dim {C} not divisible by num_groups {self.num_groups}"
            )
        per_group = C // self.num_groups
        x = x.reshape(B, T, S, self.num_groups, per_group)
        x = mx.transpose(x, (3, 0, 1, 2, 4))           # [G, B, T, S, per_group]
        x = x.reshape(self.num_groups * B, T, S, per_group)

        y = self.inner(x)  # [G*B, T_out, S_out, out_per_group]

        out_per_group = y.shape[-1]
        T_out, S_out = y.shape[1], y.shape[2]
        y = y.reshape(self.num_groups, B, T_out, S_out, out_per_group)
        y = mx.transpose(y, (1, 2, 3, 0, 4))           # [B, T_out, S_out, G, out_per_group]
        return y.reshape(B, T_out, S_out, self.num_groups * out_per_group)

    def step(self, x: mx.array, cache: Optional[list[Any]] = None) -> mx.array:
        """Streaming step. ``cache`` is accepted for API parity with the
        old per-group cache-list interface but is now ignored — the
        inner module's own streaming caches (allocated with batch =
        ``num_groups * B``) hold all the per-group state.
        """
        del cache
        return self(x)
