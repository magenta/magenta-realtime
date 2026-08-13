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

"""Parity tests for `mlx_pure.layers` vs `sequence_layers.mlx`."""

from __future__ import annotations

import mlx.core as mx
import pytest
import sequence_layers.mlx as sl

from magenta_rt.mlx_pure import layers
from .conftest import assert_close, tol


def _sl_make(cfg):
    return cfg.make(backend="mlx")


def _make_seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


# -----------------------------------------------------------------------------
# Dense
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("use_bias", [False, True])
def test_dense_parity(use_bias, rng_key):
    dtype = mx.float32
    in_features, out_features = 16, 24
    sl_cfg = sl.Dense.Config(
        features=out_features,
        use_bias=use_bias,
        param_dtype=dtype,
        compute_dtype=dtype,
    )
    sl_layer = _sl_make(sl_cfg)
    x = mx.random.normal((2, 5, in_features), dtype=dtype, key=rng_key)
    sample = _make_seq(x)
    _ = sl_layer.layer(sample)  # materialize deferred weights

    pure = layers.Dense(
        in_features,
        out_features,
        bias=use_bias,
        compute_dtype=dtype,
        param_dtype=dtype,
    )
    # Bridge weights via direct attribute access (sl uses underscore-prefixed
    # children which are excluded from `parameters()`).
    pure.linear.weight = sl_layer.inner._linear.weight
    if use_bias:
        pure.linear.bias = sl_layer.inner._linear.bias

    y_sl = sl_layer.layer(sample).values
    y_pure = pure(x)

    a, r = tol(dtype, "leaf")
    assert_close(y_sl, y_pure, atol=a, rtol=r, name="dense")


# -----------------------------------------------------------------------------
# EinsumDense
# -----------------------------------------------------------------------------


def test_einsumdense_parity(rng_key):
    sl_layer = _sl_make(
        sl.EinsumDense.Config(
            equation="...nh,dnh->...d",
            output_shape=(48,),
            bias_axes="d",
            param_dtype=mx.float32,
            compute_dtype=mx.float32,
        )
    )
    n, h = 4, 8
    x = mx.random.normal((2, 6, n, h), dtype=mx.float32, key=rng_key)
    sample = _make_seq(x)
    _ = sl_layer.layer(sample)

    pure = layers.EinsumDense(
        equation="...nh,dnh->...d",
        output_shape=(48,),
        bias_axes="d",
        compute_dtype=mx.float32,
        param_dtype=mx.float32,
    )
    _ = pure(x)  # materialize
    pure.kernel = sl_layer.kernel
    pure.bias = sl_layer.bias

    y_sl = sl_layer.layer(sample).values
    y_pure = pure(x)

    a, r = tol(mx.float32, "leaf")
    assert_close(y_sl, y_pure, atol=a, rtol=r, name="einsumdense")
