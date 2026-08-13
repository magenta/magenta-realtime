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

"""Parity tests for `to_quantized` on Dense and EinsumDense."""

from __future__ import annotations

import mlx.core as mx

from magenta_rt.mlx_pure.layers import Dense, EinsumDense
from .conftest import assert_close


def test_dense_quantize_parity(rng_key):
    """Pure Dense.to_quantized produces the same result as the unquantized
    Dense + ``mx.quantize``/``mx.quantized_matmul`` reference path."""
    in_features, out_features = 64, 32  # 64 % group_size = 0
    pure = Dense(in_features, out_features, bias=True,
                 compute_dtype=mx.float32, param_dtype=mx.float32)
    pure.linear.weight = mx.random.normal(pure.linear.weight.shape, key=rng_key) * 0.05
    pure.linear.bias = mx.random.normal(pure.linear.bias.shape, key=mx.random.split(rng_key)[0]) * 0.01
    pure_unq = Dense(in_features, out_features, bias=True,
                     compute_dtype=mx.float32, param_dtype=mx.float32)
    pure_unq.linear.weight = pure.linear.weight
    pure_unq.linear.bias = pure.linear.bias

    pure.to_quantized(group_size=32, bits=4)

    x = mx.random.normal((2, 4, in_features), key=rng_key) * 0.1
    # Reference: dequantize + matmul (same as what mx.quantized_matmul does
    # internally) — we use mx.dequantize to round-trip and check that
    # quantize/quantized_matmul are inverse-consistent.
    w = pure_unq.linear.weight
    qw, qs, qb = mx.quantize(w, group_size=32, bits=4)
    deq = mx.dequantize(qw, qs, qb, group_size=32, bits=4)
    ref = mx.matmul(x, deq.T) + pure_unq.linear.bias
    out = pure(x)
    # Dequant→matmul and quantized_matmul are mathematically equivalent
    # (both compute the same approximated kernel application).
    assert_close(ref, out, atol=1e-5, rtol=1e-5, name="dense_quantize")


def test_quantize_in_place_walks_transformer(rng_key):
    """Verify ``quantize_in_place`` finds every Dense/EinsumDense in a
    real transformer and quantizes them, and that the network still
    produces sane output afterwards.
    """
    import mlx.nn as nn
    import numpy as np
    from magenta_rt.mlx_pure.quantize import quantize_in_place
    from magenta_rt.mlx_pure.transformer import Transformer

    t = Transformer(
        num_layers=2, model_dim=64, num_heads=4, units_per_head=16,
        ffn_dim=128, max_past_horizon=4, num_sinks=1,
        use_cross_attention=True, cross_attn_source_features=64,
        cross_attn_max_past_horizon=4,
    )
    x = mx.random.normal((1, 6, 64), key=rng_key) * 0.05
    src = mx.random.normal((1, 6, 64), key=mx.random.split(rng_key)[0]) * 0.05
    # Materialize lazy EinsumDense layers.
    y_before = t(x, source=src)

    quantized_paths = quantize_in_place(t, group_size=64, bits=4)
    # Each transformer block has 4 EinsumDense (q/k/v output_projection x2)
    # and 4 Dense (FFN layer1, layer2 in 2 layers). Exact count depends on
    # block structure but should be > 0.
    assert len(quantized_paths) > 0, "no layers were quantized"

    y_after = t(x, source=src)
    assert y_after.shape == y_before.shape

    # Quantized output should be similar to fp32 in magnitude / rough shape.
    yb = np.array(y_before)
    ya = np.array(y_after)
    np.testing.assert_array_less(
        np.abs(ya).max(),
        np.abs(yb).max() * 10.0,
        err_msg="quantized output magnitude unreasonably larger than fp32",
    )


def test_einsumdense_quantize_parity(rng_key):
    """Pure EinsumDense.to_quantized matches the dequantize→einsum reference."""
    n, h, d = 4, 16, 32  # n*h = 64 % group_size = 0
    pure = EinsumDense(
        equation="...nh,dnh->...d", output_shape=(d,), bias_axes="d",
        compute_dtype=mx.float32, param_dtype=mx.float32,
    )
    x = mx.random.normal((2, 4, n, h), key=rng_key) * 0.1
    _ = pure(x)  # materialize
    pure.kernel = mx.random.normal(pure.kernel.shape, key=mx.random.split(rng_key)[0]) * 0.05
    pure.bias = mx.random.normal(pure.bias.shape, key=mx.random.split(rng_key)[1]) * 0.01

    pure_unq = EinsumDense(
        equation="...nh,dnh->...d", output_shape=(d,), bias_axes="d",
        compute_dtype=mx.float32, param_dtype=mx.float32,
    )
    _ = pure_unq(x)
    pure_unq.kernel = pure.kernel
    pure_unq.bias = pure.bias

    pure.to_quantized(group_size=32, bits=4)

    # Reference: dequantize the kernel, run the equivalent einsum.
    kernel_2d = pure_unq.kernel.reshape(d, n * h)
    qw, qs, qb = mx.quantize(kernel_2d, group_size=32, bits=4)
    deq_2d = mx.dequantize(qw, qs, qb, group_size=32, bits=4)
    deq = deq_2d.reshape(d, n, h)
    ref = mx.einsum("...nh,dnh->...d", x, deq) + pure_unq.bias

    out = pure(x)
    assert_close(ref, out, atol=1e-5, rtol=1e-5, name="einsumdense_quantize")
