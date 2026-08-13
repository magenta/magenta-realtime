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

"""Custom leaf layers.

:class:`Dense`, :class:`EinsumDense`, and :class:`LayerNorm` live here:

* ``Dense`` wraps :class:`mlx.nn.Linear` with the sl-style ``compute_dtype``
  cast, optional activation, and an *in-place* ``to_quantized`` that the
  custom ``mlx_pure.quantize`` framework hooks into (the built-in
  ``nn.Linear.to_quantized`` returns a new :class:`nn.QuantizedLinear` —
  a different shape than the surrounding GPTQ scaffolding wants).
* ``EinsumDense`` has no built-in equivalent: the attention output
  projection (``"...nh,dnh->...d"``) needs the einsum form plus a
  custom ``to_quantized`` path that flattens the last two axes into a
  single ``quantized_matmul`` input.
* ``LayerNorm`` / ``RMSNorm`` are thin :class:`mlx.nn` subclasses that
  fix the output dtype to match sl's normalization layers. With an fp32
  scale and bf16 input, the bare ``mlx.nn`` norms promote their output
  to fp32; sl forces it back to the input (compute) dtype. ``LayerNorm``
  additionally upcasts to fp32 *before* normalizing (sl's
  ``reductions_in_at_least_fp32=True``); ``RMSNorm`` does not (sl's
  ``RMSNormalization`` has no such upcast).

Embedding used to live here too; it was a thin wrapper over
``mlx.nn.Embedding`` and has been removed.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence as _Seq

import mlx.core as mx
import mlx.nn as nn


class LayerNorm(nn.LayerNorm):
    """``mlx.nn.LayerNorm`` that matches sl's ``LayerNormalization``.

    sl's builtin path does
    ``self._layer_norm(x.astype(fp32)).astype(original_dtype)`` when
    ``reductions_in_at_least_fp32=True`` — the default, and the value
    used for both ``encoder_ln`` and ``final_ln``. Calling
    ``nn.LayerNorm`` directly on bf16 input (a) skips that fp32 upcast,
    so the affine transform rounds at bf16, and (b) with an fp32 scale,
    *promotes* the output to fp32 instead of returning bf16. Both
    diverge from sl (visible as logit drift through ``final_ln`` →
    ``to_logits``). Subclassing keeps ``.weight`` / ``.bias`` as direct
    attributes so the checkpoint bridge is unchanged.
    """

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(orig_dtype)


class RMSNorm(nn.RMSNorm):
    """``mlx.nn.RMSNorm`` that matches sl's ``RMSNormalization``.

    sl's builtin path is ``self._rms_norm(x.values).astype(x.dtype)`` —
    no fp32 upcast (unlike ``LayerNormalization``), but it *does* force
    the result back to the input dtype. With an fp32 scale and bf16
    input, bare ``nn.RMSNorm`` promotes its output to fp32; left
    unchecked, that fp32 stream then propagates through the residual
    add and silently runs the rest of the block at fp32 instead of
    bf16 — diverging from sl. This subclass restores the input dtype
    and nothing else.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x).astype(x.dtype)


class Dense(nn.Module):
    """Linear layer with optional activation and compute-dtype cast.

    Mirrors ``sequence_layers.mlx.Dense``: weights stored as
    ``[out_features, in_features]`` (mlx-native), bias as
    ``[out_features]``. The forward pass casts the input to
    ``compute_dtype`` (default fp32) before the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: Optional[Callable[[mx.array], mx.array]] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight = self.linear.weight.astype(param_dtype)
        if bias:
            self.linear.bias = self.linear.bias.astype(param_dtype)
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype

    def __call__(self, x: mx.array) -> mx.array:
        dtype = self.compute_dtype if self.compute_dtype is not None else self.param_dtype
        if getattr(self, "_quantized", False):
            v = x.astype(dtype)
            y = mx.quantized_matmul(
                v, self.q_weight, scales=self.q_scales, biases=self.q_biases,
                transpose=True, group_size=self._q_group_size, bits=self._q_bits,
            )
            if self.linear.bias is not None if hasattr(self.linear, "bias") else False:
                y = y + self.linear.bias.astype(dtype)
            elif self._q_bias is not None:
                y = y + self._q_bias.astype(dtype)
            if self.activation is not None:
                y = self.activation(y)
            return y
        y = self.linear(x.astype(dtype))
        if self.activation is not None:
            y = self.activation(y)
        return y

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine") -> "Dense":
        """In-place int4 / int8 quantization of the kernel.

        Mirrors ``sequence_layers.mlx.dense.Dense.to_quantized``: the
        ``[out, in]`` weight is quantized via ``mx.quantize`` and the
        forward pass switches to ``mx.quantized_matmul``. The bias (if
        any) is preserved in floating-point.
        """
        if getattr(self, "_quantized", False):
            return self
        weight = self.linear.weight  # [out, in]
        if weight.shape[-1] % group_size != 0:
            return self  # silently skip if shape isn't compatible (matches sl)
        self.q_weight, self.q_scales, self.q_biases = mx.quantize(
            weight, group_size=group_size, bits=bits,
        )
        self._q_group_size = group_size
        self._q_bits = bits
        # Preserve bias separately so it survives parameter traversal.
        self._q_bias = self.linear.bias if hasattr(self.linear, "bias") else None
        # Drop the original (now-redundant) full-precision weight.
        self.linear = nn.Identity()  # placeholder; nothing else references it
        self._quantized = True
        return self


class EinsumDense(nn.Module):
    """Einstein-summation dense layer.

    Equation form: ``"...ab,bc->...ac"`` etc., matching
    ``sequence_layers.mlx.EinsumDense``. ``output_shape`` is the channel
    output (excluding batch/time and ``...``); ``None`` means "infer
    from input". ``bias_axes`` is a string of output axes that get bias.

    Weights are created lazily on the first call once the input shape
    is known.
    """

    def __init__(
        self,
        equation: str,
        output_shape: _Seq[Optional[int]],
        *,
        bias_axes: str = "",
        activation: Optional[Callable[[mx.array], mx.array]] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.equation = equation
        self.output_shape_spec = tuple(output_shape)
        self.bias_axes = bias_axes
        self.activation = activation
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self.kernel: Optional[mx.array] = None
        self.bias: Optional[mx.array] = None
        self._initialized = False

    @staticmethod
    def _parse(equation: str) -> tuple[str, str, str]:
        if "->" not in equation:
            raise ValueError(f"invalid einsum equation: {equation}")
        left, output_spec = equation.split("->")
        input_spec, kernel_spec = left.split(",")
        if not input_spec.startswith("...") or not output_spec.startswith("..."):
            raise ValueError("equation must be of the form '...X,Y->...Z'")
        return input_spec, kernel_spec, output_spec

    def _resolve_shapes(self, channel_shape: _Seq[int]):
        in_spec, kernel_spec, out_spec = self._parse(self.equation)
        in_axes, out_axes = in_spec[3:], out_spec[3:]
        if len(in_axes) != len(channel_shape):
            raise ValueError(f"input rank mismatch: {in_axes} vs {channel_shape}")
        in_dims = {a: channel_shape[i] for i, a in enumerate(in_axes)}
        out_shape = list(self.output_shape_spec)
        if len(out_axes) != len(out_shape):
            raise ValueError(f"output rank mismatch: {out_axes} vs {out_shape}")
        for i, a in enumerate(out_axes):
            if out_shape[i] is None:
                out_shape[i] = in_dims[a]
        out_dims = {a: out_shape[i] for i, a in enumerate(out_axes)}
        kernel_shape = []
        for a in kernel_spec:
            if a in in_dims:
                kernel_shape.append(in_dims[a])
            elif a in out_dims:
                kernel_shape.append(out_dims[a])
            else:
                raise ValueError(f"weight axis '{a}' not in input or output spec")

        bias_shape = None
        if self.bias_axes:
            first = min(out_axes.find(c) for c in self.bias_axes)
            bias_shape = tuple(
                out_dims[c] if c in self.bias_axes else 1 for c in out_axes[first:]
            )
        return tuple(out_shape), tuple(kernel_shape), bias_shape

    def _ensure_initialized(self, x: mx.array):
        if self._initialized:
            return
        # If the kernel was already populated externally (e.g. via the
        # weight-loading bridge), don't overwrite it with zeros.
        if self.kernel is not None:
            self._initialized = True
            return
        # x is [B, T, *channel_shape] — drop batch and time.
        channel_shape = x.shape[2:]
        _, kernel_shape, bias_shape = self._resolve_shapes(channel_shape)
        self.kernel = mx.zeros(kernel_shape, dtype=self.param_dtype)
        if bias_shape is not None:
            self.bias = mx.zeros(bias_shape, dtype=self.param_dtype)
        self._initialized = True

    def __call__(self, x: mx.array) -> mx.array:
        if getattr(self, "_quantized", False):
            return self._call_quantized(x)
        self._ensure_initialized(x)
        dtype = self.compute_dtype if self.compute_dtype is not None else self.param_dtype

        # IMPORTANT: match sl's mixed-precision behaviour exactly. sl casts
        # ONLY the input ``v`` to ``compute_dtype`` and leaves ``kernel``
        # and ``bias`` in ``param_dtype`` (typically fp32). MLX promotes
        # during the einsum so the reduction is done at the higher
        # precision; downcasting the kernel/bias to bf16 here produces a
        # different (lossier) result and breaks real-checkpoint parity.
        y = mx.einsum(self.equation, x.astype(dtype), self.kernel)
        if self.bias is not None:
            y = y + self.bias
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _call_quantized(self, x: mx.array) -> mx.array:
        """Forward pass for the quantized ``'...nh,dnh->...d'`` path.

        Mirrors ``sequence_layers.mlx.dense.EinsumDense.to_quantized``:
        flatten the last two input axes, run ``mx.quantized_matmul``
        with ``transpose=True``, then re-add bias and activation.
        """
        dtype = self.compute_dtype if self.compute_dtype is not None else self.param_dtype
        n, h = self._q_n, self._q_h
        v = x.astype(dtype)
        v_2d = v.reshape(*v.shape[:-2], n * h)
        y = mx.quantized_matmul(
            v_2d, self.q_weight, scales=self.q_scales, biases=self.q_biases,
            transpose=True, group_size=self._q_group_size, bits=self._q_bits,
        )
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine") -> "EinsumDense":
        """In-place int4 / int8 quantization for the
        ``'...nh,dnh->...d'`` einsum (the attention output projection
        layout). Returns ``self`` unchanged for any other equation or
        if the kernel hasn't been initialized.
        """
        if getattr(self, "_quantized", False):
            return self
        if self.kernel is None or self.equation != "...nh,dnh->...d":
            return self
        d, n, h = self.kernel.shape
        if (n * h) % group_size != 0:
            return self
        kernel_2d = self.kernel.reshape(d, n * h)
        self.q_weight, self.q_scales, self.q_biases = mx.quantize(
            kernel_2d, group_size=group_size, bits=bits,
        )
        self._q_group_size = group_size
        self._q_bits = bits
        self._q_n, self._q_h = n, h
        self.kernel = None  # drop the full-precision kernel
        self._quantized = True
        return self
