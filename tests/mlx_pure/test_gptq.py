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

"""GPTQ tests for ``mlx_pure.quantize``.

* Identity-Hessian path is bit-equal to ``quantize_in_place`` output
  on a Dense and an EinsumDense, since GPTQ degenerates to plain
  nearest-rounding when ``H = I``.
* Real-Hessian path: full pipeline runs, output is finite and not
  trivially zero, and the packed weight tensors have the expected
  ``mx.quantized_matmul`` shapes.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from magenta_rt.mlx_pure.layers import Dense, EinsumDense
from magenta_rt.mlx_pure.quantize import (
    gptq_calibrate_and_quantize, quantize_in_place,
)


class _DenseModel(nn.Module):
    def __init__(self, in_f=64, out_f=128):
        super().__init__()
        self.lin = Dense(in_f, out_f, bias=False, param_dtype=mx.float32)
        # Fill with reproducible non-zero weights.
        self.lin.linear.weight = (
            mx.random.normal(self.lin.linear.weight.shape, key=mx.random.key(1))
            * 0.05
        )

    def __call__(self, x):
        return self.lin(x)


class _EinsumModel(nn.Module):
    """``...nh,dnh->...d`` — the attention output-projection layout."""

    def __init__(self, n=4, h=16, d=128):
        super().__init__()
        self.proj = EinsumDense(
            equation="...nh,dnh->...d",
            output_shape=(d,),
            bias_axes="",
            param_dtype=mx.float32,
        )
        # Force lazy init by running a dummy forward.
        dummy = mx.zeros((1, 1, n, h))
        self.proj(dummy)
        self.proj.kernel = (
            mx.random.normal(self.proj.kernel.shape, key=mx.random.key(2))
            * 0.05
        )

    def __call__(self, x):
        return self.proj(x)


def _calibrate(model, x):
    def _fn(m):
        for _ in range(3):
            m(x)
    return _fn


def test_gptq_identity_hessian_matches_naive_quantize_dense():
    """With H=I, GPTQ should degenerate to nearest-rounding and produce
    bit-identical output to ``quantize_in_place``."""
    x = mx.random.normal((4, 32, 64), key=mx.random.key(7))

    naive = _DenseModel()
    quantize_in_place(naive, group_size=32, bits=4)
    naive_out = naive(x)

    gptq = _DenseModel()
    gptq_calibrate_and_quantize(
        gptq, _calibrate(gptq, x),
        group_size=32, bits=4,
        debug_identity_hessian=True,
    )
    gptq_out = gptq(x)

    np.testing.assert_array_equal(np.array(naive_out), np.array(gptq_out))


def test_gptq_identity_hessian_matches_naive_quantize_einsum():
    """Same identity-Hessian smoke test for the EinsumDense path."""
    x = mx.random.normal((2, 16, 4, 16), key=mx.random.key(11))

    naive = _EinsumModel()
    quantize_in_place(naive, group_size=32, bits=4)
    naive_out = naive(x)

    gptq = _EinsumModel()
    gptq_calibrate_and_quantize(
        gptq, _calibrate(gptq, x),
        group_size=32, bits=4,
        debug_identity_hessian=True,
    )
    gptq_out = gptq(x)

    np.testing.assert_array_equal(np.array(naive_out), np.array(gptq_out))


def test_gptq_full_pipeline_smoke_dense():
    """Real Hessian: pipeline runs, output is finite and non-zero,
    and the layer's q_weight has the expected mx.quantize shape."""
    x = mx.random.normal((4, 32, 64), key=mx.random.key(13))
    model = _DenseModel()
    paths = gptq_calibrate_and_quantize(
        model, _calibrate(model, x), group_size=32, bits=4,
    )
    assert paths == ["lin"]
    assert getattr(model.lin, "_quantized", False)
    # mx.quantize for [out=128, in=64] with group_size=32 bits=4:
    # q_weight is [128, 64*4//32] = [128, 8], dtype uint32.
    assert model.lin.q_weight.shape == (128, 8)
    assert model.lin.q_weight.dtype == mx.uint32
    out = model(x)
    arr = np.array(out)
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() > 0


def test_gptq_full_pipeline_smoke_einsum():
    x = mx.random.normal((2, 16, 4, 16), key=mx.random.key(17))
    model = _EinsumModel()
    paths = gptq_calibrate_and_quantize(
        model, _calibrate(model, x), group_size=32, bits=4,
    )
    assert paths == ["proj"]
    assert getattr(model.proj, "_quantized", False)
    # kernel was [d=128, n=4, h=16] → flatten to [128, 64], pack to [128, 8].
    assert model.proj.q_weight.shape == (128, 8)
    out = model(x)
    arr = np.array(out)
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() > 0


def test_gptq_skip_callback():
    """``skip=`` leaves a layer at full precision."""
    x = mx.random.normal((4, 32, 64), key=mx.random.key(19))
    model = _DenseModel()
    paths = gptq_calibrate_and_quantize(
        model, _calibrate(model, x),
        group_size=32, bits=4,
        skip=lambda p, m: p == "lin",
    )
    assert paths == []
    assert not getattr(model.lin, "_quantized", False)


def test_gptq_skip_doesnt_leak_class_swap_on_skipped_layers():
    """Skipped layers must end up with their original ``__class__``,
    not the ``_GPTQWrapped`` subclass the activation-capture context
    manager swaps in. Regression check for a leak that only manifests
    when at least one layer is quantized and another is skipped.
    """

    class _TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Dense(64, 64, bias=False, param_dtype=mx.float32)
            self.b = Dense(64, 32, bias=False, param_dtype=mx.float32)
            self.a.linear.weight = mx.random.normal(
                self.a.linear.weight.shape, key=mx.random.key(0),
            ) * 0.05
            self.b.linear.weight = mx.random.normal(
                self.b.linear.weight.shape, key=mx.random.key(1),
            ) * 0.05

        def __call__(self, x):
            return self.b(self.a(x))

    model = _TwoLayer()
    pre_a_cls = type(model.a)
    pre_b_cls = type(model.b)
    x = mx.random.normal((4, 32, 64), key=mx.random.key(19))
    paths = gptq_calibrate_and_quantize(
        model, _calibrate(model, x),
        group_size=32, bits=4,
        skip=lambda p, m: p == "a",
    )
    assert paths == ["b"]
    assert getattr(model.b, "_quantized", False)
    assert not getattr(model.a, "_quantized", False)
    # Skipped layer must not have been left with a swapped class.
    assert type(model.a) is pre_a_cls
    assert type(model.b) is pre_b_cls
    assert not hasattr(model.a, "_gptq_path")
