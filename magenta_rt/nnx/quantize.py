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

"""Weight-only int8 quantization for the NNX depthformer (streaming inference).

Streaming generation is batch-1 / one frame per step, so each matmul reads a
whole weight matrix to multiply a tiny activation — it is **weight-bandwidth
bound**. Storing weights as int8 (half the bytes of bf16) therefore ~halves the
dominant cost: a microbenchmark of the streaming shape on an RTX 4080 SUPER
shows ~1.67x for weight-only int8 (XLA fuses the int8 read + dequant into the
GEMM), with full int8xint8 only marginally faster — so this keeps activations in
bf16 (quality-safe, no activation quantization), matching the weight-only scheme
the MLX/`.mlxfn` apps ship.

:class:`QuantizedLinear` is a drop-in for ``nnx.Linear`` (per-output-channel
symmetric int8). :func:`quantize_in_place` swaps every ``nnx.Linear`` in a module
tree for one — including the ``base`` Linear inside a ``LoRAAdapter`` wrapper, so
it composes with LoRA (int8 base + bf16 adapter delta, QLoRA-style inference).
Leading axes (the ``nnx.scan``-stacked transformer layers) are handled by
quantizing over the ``in`` axis (-2); ``nnx.scan`` slices the layer axis away
before ``__call__``, so the forward always sees a 2-D ``[in, out]`` weight.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx


class QuantizedLinear(nnx.Module):
    """Weight-only int8 replacement for an ``nnx.Linear``.

    Captures the base layer's kernel as per-output-channel symmetric int8
    (``qweight`` int8 ``[..., in, out]`` + ``scale`` ``[..., out]``) and keeps
    its bias / dtypes. The forward dequantizes in the compute dtype and lets XLA
    fuse the int8 read into the matmul. Numerically a quantization approximation
    of the base (≈ int8 round-off on the weights); activations are untouched.
    """

    def __init__(self, base: nnx.Linear):
        # Quantize on the HOST (numpy): upcasting the full nnx.scan-stacked
        # kernel to fp32 on-device is a multi-GB transient that OOMs beside the
        # resident model (e.g. a [20, 3072, 8192] FFN kernel → 2 GB). Doing it in
        # host RAM uploads only the int8 weights (half the bytes) + scales.
        kf = np.asarray(base.kernel[...]).astype(np.float32)   # (..., in, out)
        amax = np.max(np.abs(kf), axis=-2, keepdims=True)      # (..., 1, out)
        scale = np.maximum(amax, 1e-8) / 127.0                 # (..., 1, out)
        q = np.round(kf / scale).clip(-127, 127).astype(np.int8)  # (..., in, out)
        self.qweight = nnx.Param(jnp.asarray(q))
        self.scale = nnx.Param(
            jnp.asarray(scale.squeeze(-2)).astype(base.param_dtype))  # (..., out)
        self.bias = base.bias                          # nnx.Param or None
        self.dtype = base.dtype                         # compute dtype
        self.param_dtype = base.param_dtype

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        cdt = self.dtype if self.dtype is not None else x.dtype
        # Dequantize in the compute dtype: w = qweight * scale. XLA fuses the
        # int8 load + convert into the GEMM, so only int8 bytes are read.
        w = self.qweight[...].astype(cdt) * self.scale[...][..., None, :].astype(cdt)
        y = jnp.matmul(x.astype(cdt), w)
        if self.bias is not None:
            y = y + self.bias[...].astype(cdt)
        return y


def quantize_in_place(model: nnx.Module, *, bits: int = 8) -> int:
    """Replace every ``nnx.Linear`` under ``model`` with a weight-only int8
    :class:`QuantizedLinear` (in place). Walks into nested modules, including a
    ``LoRAAdapter``'s ``base`` Linear, so the int8 base composes with a bf16 LoRA
    adapter. Returns the number of Linears quantized.

    ``bits`` is accepted for parity with the MLX exporter's API but only 8 is
    implemented here (the value the apps ship and the one the streaming-shape
    benchmark favours); other values raise.
    """
    if bits != 8:
        raise ValueError(f"only bits=8 is implemented for the NNX path, got {bits}")

    count = 0

    def _walk(node: nnx.Module) -> None:
        nonlocal count
        for attr in list(vars(node)):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr)
            if isinstance(child, nnx.Linear):
                setattr(node, attr, QuantizedLinear(child))
                count += 1
            elif isinstance(child, nnx.Module) and not isinstance(child, QuantizedLinear):
                _walk(child)

    _walk(model)
    return count
