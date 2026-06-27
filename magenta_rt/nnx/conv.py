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

Each :class:`Conv2D` / :class:`Conv2DTranspose` carries a
``cached_left: nnx.Cache | None`` slot for its left-context buffer,
allocated by :meth:`init_cache` and used when ``streaming=True``.

Kernel layout is ``[filters, kH, kW, in_channels // groups]``;
:func:`jax.lax.conv_general_dilated` is invoked with
``dimension_numbers=('NHWC', 'OHWI', 'NHWC')`` so safetensors weights
flow in 1:1.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence as _Seq, Union

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import lax


_DimNums = ("NHWC", "OHWI", "NHWC")


# OHWI kernel layout: fan_in lives on the last axis, fan_out on axis 0.
# This is the sl-equivalent of ``lecun_normal()`` for a HWIO kernel —
# variance-scaling with ``mode='fan_in'`` over the same set of axes.
_default_kernel_init = nnx.initializers.lecun_normal(in_axis=-1, out_axis=0)


# Streaming API note: there's no module-level ``enable_streaming`` /
# ``disable_streaming`` helper — the idiomatic nnx way is method-style
# on the model: ``model.set_attributes(streaming=True)`` + ``model.init_cache(batch, dtype)``,
# then ``model.set_attributes(streaming=False)`` + ``model.remove_cache()``.
# Top-level containers (Transformer, Encoder, SpectroStream, MagentaRT2Sampler)
# implement ``init_cache(...)`` / ``remove_cache()`` by walking
# ``nnx.iter_modules(self)``; the :func:`remove_cache` helper below is
# the implementation those methods use.


def remove_cache(module: nnx.Module) -> None:
    """Clear every ``nnx.Cache`` slot in ``module``'s subtree back to
    ``nnx.data(None)``. Used by top-level container ``remove_cache()``
    methods.
    """
    for _path, m in nnx.iter_modules(module):
        if m is module:
            continue
        if hasattr(m, "remove_cache") and callable(getattr(m, "remove_cache")):
            m.remove_cache()
        else:
            for name in list(vars(m)):
                v = getattr(m, name, None)
                if isinstance(v, nnx.Cache):
                    setattr(m, name, nnx.data(None))


def _normalize_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def _effective_kernel_size(kernel_size: int, dilation_rate: int) -> int:
    return (kernel_size - 1) * dilation_rate + 1


def _explicit_padding(padding, kernel_size: int, stride: int, dilation_rate: int) -> tuple[int, int]:
    """``(pad_left, pad_right)`` for one axis."""
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


class Conv2D(nnx.Module):
    """2D convolution with separate time + spatial padding.

    Input ``[B, T, S, C_in]`` → output ``[B, T_out, S_out, C_out]``.
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
        activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
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
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.streaming: bool = False
        self.cached_left: nnx.Cache | None = nnx.data(None)
        self._rngs = rngs.fork() if rngs is not None else nnx.data(None)

        kh, kw = self.kernel_size
        if in_features is not None:
            shape = (filters, kh, kw, in_features // groups)
            self.kernel = nnx.Param(
                _default_kernel_init(rngs.params(), shape, param_dtype)
            )
        else:
            self.kernel = nnx.data(None)
        if use_bias:
            self.bias = nnx.Param(
                jnp.zeros((filters,), dtype=param_dtype)
            )
        else:
            self.bias = None

    def ensure_initialized(self, in_features: int) -> None:
        if self.kernel is not None:
            return
        kh, kw = self.kernel_size
        self.in_features = in_features
        shape = (self.filters, kh, kw, in_features // self.groups)
        self.kernel = nnx.Param(
            _default_kernel_init(self._rngs.params(), shape, self.param_dtype)
        )

    def init_cache(
        self,
        *,
        batch: int,
        spatial: int,
        in_features: Optional[int] = None,
        dtype=jnp.float32,
    ) -> None:
        """Allocate the streaming left-context buffer.

        Walks the time-padding mode to figure out how much left context
        to retain. ``in_features`` defaults to the kernel's known input
        channels (after lazy init) or the explicit value if the kernel
        is still deferred.
        """
        kt = self.kernel_size[0]
        st = self.strides[0]
        dt = self.dilation_rate[0]
        pad_left, _ = _explicit_padding(self.time_padding, kt, st, dt)
        if in_features is None:
            if self.kernel is None:
                raise RuntimeError(
                    "Conv2D.init_cache: in_features must be provided when "
                    "the kernel hasn't been initialized yet."
                )
            in_features = self.kernel[...].shape[3] * self.groups
        self.cached_left = nnx.Cache(
            jnp.zeros((batch, pad_left, spatial, in_features), dtype=dtype)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.streaming:
            return self.step(x)
        self.ensure_initialized(x.shape[-1])
        dtype = self.dtype if self.dtype is not None else x.dtype
        x = x.astype(dtype)
        time_pad = _explicit_padding(self.time_padding, self.kernel_size[0],
                                     self.strides[0], self.dilation_rate[0])
        spatial_pad = _explicit_padding(self.spatial_padding, self.kernel_size[1],
                                        self.strides[1], self.dilation_rate[1])
        if any(time_pad) or any(spatial_pad):
            x = jnp.pad(x, [(0, 0), time_pad, spatial_pad, (0, 0)])
        y = lax.conv_general_dilated(
            x,
            self.kernel[...].astype(dtype),
            window_strides=self.strides,
            padding=((0, 0), (0, 0)),
            rhs_dilation=self.dilation_rate,
            dimension_numbers=_DimNums,
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias[...].astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Streaming forward with the left-context cache."""
        self.ensure_initialized(x.shape[-1])
        dtype = self.dtype if self.dtype is not None else x.dtype
        x = x.astype(dtype)

        kt, ks = self.kernel_size
        st, ss = self.strides
        dt, ds = self.dilation_rate
        time_pad = _explicit_padding(self.time_padding, kt, st, dt)
        spatial_pad = _explicit_padding(self.spatial_padding, ks, ss, ds)
        pad_left, pad_right = time_pad
        if pad_right > 0:
            raise NotImplementedError(
                f"Conv2D.step: streaming requires zero right-pad "
                f"(time_padding={self.time_padding!r}, stride_t={st} gives "
                f"pad_right={pad_right})."
            )
        if x.shape[1] % st != 0:
            raise ValueError(
                f"Conv2D.step: T_in ({x.shape[1]}) must be a multiple of "
                f"stride_t ({st})"
            )

        if self.cached_left is None:
            shape = (x.shape[0], pad_left, x.shape[2], x.shape[3])
            self.cached_left = nnx.Cache(jnp.zeros(shape, dtype=x.dtype))

        buf = self.cached_left[...].astype(x.dtype)
        combined = jnp.concatenate([buf, x], axis=1)
        if any(spatial_pad):
            padded = jnp.pad(combined, [(0, 0), (0, 0), spatial_pad, (0, 0)])
        else:
            padded = combined
        y = lax.conv_general_dilated(
            padded,
            self.kernel[...].astype(dtype),
            window_strides=self.strides,
            padding=((0, 0), (0, 0)),
            rhs_dilation=self.dilation_rate,
            dimension_numbers=_DimNums,
            feature_group_count=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias[...].astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        if pad_left > 0:
            self.cached_left[...] = combined[:, -pad_left:]
        return y


class Conv2DTranspose(nnx.Module):
    """2D transposed convolution.

    ``[B, T, S, C_in]`` → ``[B, T*stride_t, S*stride_s, C_out]``. Time
    padding modes ``'same'`` and ``'causal'`` only.
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
        activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        if time_padding not in ("same", "causal"):
            raise NotImplementedError(
                f"Conv2DTranspose: only 'same' and 'causal' time_padding "
                f"supported; got {time_padding!r}"
            )
        self.in_features = in_features
        self.filters = filters
        self.kernel_size = _normalize_2tuple(kernel_size)
        self.strides = _normalize_2tuple(strides)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.use_bias = use_bias
        self.activation = activation
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.streaming = False
        self.cached_left: nnx.Cache | None = nnx.data(None)
        self._rngs = rngs.fork() if rngs is not None else nnx.data(None)

        kh, kw = self.kernel_size
        if in_features is not None:
            shape = (filters, kh, kw, in_features)
            self.kernel = nnx.Param(
                _default_kernel_init(rngs.params(), shape, param_dtype)
            )
        else:
            self.kernel = nnx.data(None)
        if use_bias:
            self.bias = nnx.Param(
                jnp.zeros((filters,), dtype=param_dtype)
            )
        else:
            self.bias = None

    def ensure_initialized(self, in_features: int) -> None:
        if self.kernel is not None:
            return
        kh, kw = self.kernel_size
        self.in_features = in_features
        shape = (self.filters, kh, kw, in_features)
        self.kernel = nnx.Param(
            _default_kernel_init(self._rngs.params(), shape, self.param_dtype)
        )

    def init_cache(
        self,
        *,
        batch: int,
        spatial: int,
        dtype=jnp.float32,
    ) -> None:
        """Eagerly allocate the streaming transposed overlap buffer."""
        kh, kw = self.kernel_size
        sh, sw = self.strides
        overlap = max(kh - sh, 0)
        self.cached_left = nnx.Cache(
            jnp.zeros((batch, overlap, spatial, self.filters), dtype=dtype)
        )

    def _explicit_transpose_padding(self, padding: str, kernel_size: int,
                                    stride: int, dilation_rate: int) -> tuple[int, int]:
        """sl-jax ``transpose_conv_explicit_padding`` recipe.

        Mirrors :func:`sequence_layers.jax.convolution.transpose_conv_explicit_padding`
        verbatim so the safetensors weights see the same numerical
        operation as the Linen tree they were trained with.
        """
        ek = (kernel_size - 1) * dilation_rate + 1
        if padding == "valid":
            pad_amount = ek + stride - 2 + max(ek - stride, 0)
            pad_left = ek - 1
            pad_right = pad_amount - pad_left
        elif padding == "causal":
            pad_amount = ek + stride - 2
            pad_left = ek - 1
            pad_right = pad_amount - pad_left
        elif padding == "same":
            pad_amount = ek + stride - 2
            if stride > ek - 1:
                pad_left = ek - 1
            else:
                pad_left = math.ceil(pad_amount / 2)
            pad_right = pad_amount - pad_left
        else:
            raise ValueError(f"unsupported transpose padding: {padding}")
        return pad_left, pad_right

    def _conv_transpose(self, x: jnp.ndarray, dtype, *,
                        time_pad=None, spatial_pad=None) -> jnp.ndarray:
        """Implements transposed convolution as an input-dilated forward
        ``conv_general_dilated`` (sl-jax convention).

        Equivalent to ``lax.conv_transpose`` mathematically but uses the
        exact same primitive sl-jax does, with the same explicit padding
        semantics — required for bit-exact parity against the Linen
        checkpoint.
        """
        if time_pad is None:
            time_pad = self._explicit_transpose_padding(
                self.time_padding, self.kernel_size[0],
                self.strides[0], 1,
            )
        if spatial_pad is None:
            if isinstance(self.spatial_padding, str):
                spatial_pad = self._explicit_transpose_padding(
                    self.spatial_padding, self.kernel_size[1],
                    self.strides[1], 1,
                )
            else:
                spatial_pad = tuple(self.spatial_padding)
        return lax.conv_general_dilated(
            x,
            self.kernel[...].astype(dtype),
            window_strides=(1, 1),
            padding=(time_pad, spatial_pad),
            lhs_dilation=self.strides,
            rhs_dilation=(1, 1),
            dimension_numbers=_DimNums,
            feature_group_count=1,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.streaming:
            return self.step(x)
        self.ensure_initialized(x.shape[-1])
        dtype = self.dtype if self.dtype is not None else x.dtype
        x = x.astype(dtype)

        y = self._conv_transpose(x, dtype)

        if self.bias is not None:
            y = y + self.bias[...].astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Streaming transposed forward.

        For ``time_padding='causal'`` the streaming step uses
        ``pad_left=ek-1, pad_right=0`` (instead of the non-streaming
        ``pad_right=stride-1``) and stashes the trailing ``ek-stride``
        output samples as the overlap-add buffer for the next call.
        Concatenated streaming output then matches the non-streaming
        forward shifted by ``ek-stride`` warmup samples.
        """
        if self.time_padding != "causal":
            raise NotImplementedError(
                f"Conv2DTranspose.step: streaming only supports "
                f"time_padding='causal' (got {self.time_padding!r})"
            )
        self.ensure_initialized(x.shape[-1])
        dtype = self.dtype if self.dtype is not None else x.dtype
        x = x.astype(dtype)
        kh, _kw = self.kernel_size
        sh, _sw = self.strides
        T_in = x.shape[1]
        T_emit = T_in * sh
        overlap = max(kh - sh, 0)

        # sl-jax convention: streaming step uses VALID time-padding
        # (pad_left = ek-1, pad_right = ek-1 + max(ek-stride, 0)) so the
        # raw output covers ``T_in*stride + (ek-stride)`` samples; the
        # trailing ``ek-stride`` are stashed as the overlap-add buffer
        # for the next step.
        time_pad = self._explicit_transpose_padding(
            "valid", self.kernel_size[0], self.strides[0], 1,
        )
        if isinstance(self.spatial_padding, str):
            spatial_pad = self._explicit_transpose_padding(
                self.spatial_padding, self.kernel_size[1],
                self.strides[1], 1,
            )
        else:
            spatial_pad = tuple(self.spatial_padding)
        raw = self._conv_transpose(x, dtype, time_pad=time_pad,
                                   spatial_pad=spatial_pad)

        if self.cached_left is None:
            self.cached_left = nnx.Cache(
                jnp.zeros(
                    (raw.shape[0], overlap, raw.shape[2], raw.shape[3]),
                    dtype=dtype,
                )
            )
        buf = self.cached_left[...].astype(dtype)

        if overlap > 0:
            head = raw[:, :overlap] + buf
            merged = jnp.concatenate([head, raw[:, overlap:]], axis=1)
        else:
            merged = raw
        emit = merged[:, :T_emit]
        if self.bias is not None:
            emit = emit + self.bias[...].astype(dtype)
        if self.activation is not None:
            emit = self.activation(emit)
        if overlap > 0:
            self.cached_left[...] = merged[:, T_emit:]
        return emit


class AveragePooling2D(nnx.Module):
    """2D average pooling. Time padding uses the :class:`Conv2D` modes.

    Stateless in time so :meth:`step` is identical to :meth:`__call__`.
    """

    def __init__(
        self,
        *,
        pool_size: Union[int, _Seq[int]],
        strides: Optional[Union[int, _Seq[int]]] = None,
        time_padding: str = "valid",
        spatial_padding: Union[str, tuple[int, int]] = "valid",
    ):
        self.pool_size = _normalize_2tuple(pool_size)
        self.strides = _normalize_2tuple(strides) if strides is not None else self.pool_size
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        time_pad = _explicit_padding(self.time_padding, self.pool_size[0], self.strides[0], 1)
        spatial_pad = _explicit_padding(self.spatial_padding, self.pool_size[1], self.strides[1], 1)
        if any(time_pad) or any(spatial_pad):
            x = jnp.pad(x, [(0, 0), time_pad, spatial_pad, (0, 0)])
        ph, pw = self.pool_size
        # Implement avg-pool via a depthwise convolution with constant kernel.
        C = x.shape[-1]
        kernel = jnp.full((C, ph, pw, 1), 1.0 / (ph * pw), dtype=x.dtype)
        y = lax.conv_general_dilated(
            x, kernel,
            window_strides=self.strides,
            padding=((0, 0), (0, 0)),
            dimension_numbers=_DimNums,
            feature_group_count=C,
        )
        return y

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        return self(x)


class Upsample2D(nnx.Module):
    """Nearest-neighbor 2D upsampling by integer ``rate``."""

    def __init__(self, rate: Union[int, _Seq[int]]):
        self.rate = _normalize_2tuple(rate)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        rh, rw = self.rate
        x = jnp.repeat(x, rh, axis=1)
        x = jnp.repeat(x, rw, axis=2)
        return x

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        return self(x)


class ParallelChannels(nnx.Module):
    """Split the channel axis into ``num_groups`` groups, run a shared
    ``inner`` module on each group independently, then concat back
    along the channel axis.

    Implementation: the original sl convention loops over groups with
    a shared ``inner`` and swaps the inner's cache state in / out per
    group. That works in eager mode but breaks under ``nnx.jit``
    because the cache state has to round-trip through the trace.

    Instead we reshape the groups into the *batch* axis, run the
    inner module **once** on the bigger batch, then reshape back.
    Each row of the inner's batch dim holds its own streaming state
    naturally — no swap-in / swap-out, no per-group bookkeeping
    list, and the inner's ``nnx.Cache`` slots are allocated once at
    ``num_groups * B`` batch and reused across calls.
    """

    def __init__(self, *, inner: nnx.Module, num_groups: int):
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self.inner = inner
        self.num_groups = num_groups

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, S, C]. Split C into (G, per_group) and fold G into batch.
        B, C = x.shape[0], x.shape[-1]
        if C % self.num_groups != 0:
            raise ValueError(
                f"channel dim {C} not divisible by num_groups {self.num_groups}"
            )
        x = rearrange(x, "b t s (g c) -> (g b) t s c", g=self.num_groups)
        y = self.inner(x)
        return rearrange(y, "(g b) t s c -> b t s (g c)", g=self.num_groups, b=B)

    def step(self, x: jnp.ndarray) -> jnp.ndarray:
        return self(x)
