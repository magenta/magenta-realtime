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

"""Parity tests for Conv2D / Conv2DTranspose / Upsample2D."""

from __future__ import annotations

import mlx.core as mx
import pytest
import sequence_layers.mlx as sl

from magenta_rt.mlx_pure.conv import (
    AveragePooling2D, Conv2D, Conv2DTranspose, Upsample2D,
)
from .conftest import assert_close, tol


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


@pytest.mark.parametrize("time_padding", ["causal", "semicausal", "valid", "same"])
@pytest.mark.parametrize("strides", [(1, 1), (2, 1), (1, 2)])
def test_conv2d_parity(time_padding, strides, rng_key):
    in_features, filters = 4, 6
    sl_layer = sl.Conv2D.Config(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        time_padding=time_padding,
        spatial_padding="same",
        groups=1,
        use_bias=True,
        param_dtype=mx.float32,
        compute_dtype=mx.float32,
    ).make(backend="mlx")

    B, T, S = 2, 8, 7
    x = mx.random.normal((B, T, S, in_features), key=rng_key) * 0.1
    sample = _seq(x)
    _ = sl_layer.layer(sample)
    sub = mx.random.split(rng_key, 2)
    sl_layer.inner.kernel = mx.random.normal(sl_layer.inner.kernel.shape, key=sub[0]) * 0.1
    sl_layer.inner.bias = mx.random.normal(sl_layer.inner.bias.shape, key=sub[1]) * 0.05

    pure = Conv2D(
        in_features=in_features, filters=filters, kernel_size=(3, 3),
        strides=strides, time_padding=time_padding, spatial_padding="same",
        groups=1, use_bias=True, param_dtype=mx.float32, compute_dtype=mx.float32,
    )
    pure.kernel = sl_layer.inner.kernel
    pure.bias = sl_layer.inner.bias

    sl_y = sl_layer.layer(sample).values
    pure_y = pure(x)
    a, r = tol(mx.float32, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r,
                 name=f"conv2d_{time_padding}_strides={strides}")


@pytest.mark.parametrize("strides", [(2, 2), (4, 1), (1, 4)])
def test_conv2d_transpose_parity(strides, rng_key):
    in_features, filters = 4, 6
    sl_layer = sl.Conv2DTranspose.Config(
        filters=filters,
        kernel_size=(2 * strides[0], 2 * strides[1]),
        strides=strides,
        time_padding="same",
        spatial_padding="same",
        use_bias=True,
        param_dtype=mx.float32,
        compute_dtype=mx.float32,
    ).make(backend="mlx")

    B, T, S = 2, 4, 5
    x = mx.random.normal((B, T, S, in_features), key=rng_key) * 0.1
    sample = _seq(x)
    _ = sl_layer.layer(sample)
    sub = mx.random.split(rng_key, 2)
    sl_layer.inner.kernel = mx.random.normal(sl_layer.inner.kernel.shape, key=sub[0]) * 0.1
    sl_layer.inner.bias = mx.random.normal(sl_layer.inner.bias.shape, key=sub[1]) * 0.05

    pure = Conv2DTranspose(
        in_features=in_features, filters=filters,
        kernel_size=(2 * strides[0], 2 * strides[1]),
        strides=strides, time_padding="same", spatial_padding="same",
        use_bias=True, param_dtype=mx.float32, compute_dtype=mx.float32,
    )
    pure.kernel = sl_layer.inner.kernel
    pure.bias = sl_layer.inner.bias

    sl_y = sl_layer.layer(sample).values
    pure_y = pure(x)
    a, r = tol(mx.float32, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r,
                 name=f"conv2d_t_strides={strides}")


def test_upsample2d_parity(rng_key):
    sl_layer = sl.Upsample2D.Config(rate=(2, 3)).make(backend="mlx")
    pure = Upsample2D(rate=(2, 3))
    x = mx.random.normal((2, 4, 5, 6), key=rng_key) * 0.1
    sl_y = sl_layer.layer(_seq(x)).values
    pure_y = pure(x)
    a, r = tol(mx.float32, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="upsample2d")


@pytest.mark.parametrize(
    "in_features,filters,groups",
    [(4, 8, 2), (4, 4, 4), (6, 6, 3)],  # grouped, depthwise (g==in==out), grouped
)
def test_conv2d_grouped_parity(in_features, filters, groups, rng_key):
    """Grouped / depthwise conv (``groups > 1``). ``AveragePooling2D`` and the
    production codec rely on grouping; only ``groups=1`` was covered before.
    Kernel layout ``[filters, kH, kW, in//groups]`` is shared by sl and pure,
    so the bridge is a direct copy."""
    sl_layer = sl.Conv2D.Config(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        time_padding="causal",
        spatial_padding="same",
        groups=groups,
        use_bias=True,
        param_dtype=mx.float32,
        compute_dtype=mx.float32,
    ).make(backend="mlx")

    B, T, S = 2, 8, 7
    x = mx.random.normal((B, T, S, in_features), key=rng_key) * 0.1
    sample = _seq(x)
    _ = sl_layer.layer(sample)
    sub = mx.random.split(rng_key, 2)
    sl_layer.inner.kernel = mx.random.normal(sl_layer.inner.kernel.shape, key=sub[0]) * 0.1
    sl_layer.inner.bias = mx.random.normal(sl_layer.inner.bias.shape, key=sub[1]) * 0.05

    pure = Conv2D(
        in_features=in_features, filters=filters, kernel_size=(3, 3),
        strides=(1, 1), time_padding="causal", spatial_padding="same",
        groups=groups, use_bias=True, param_dtype=mx.float32, compute_dtype=mx.float32,
    )
    pure.kernel = sl_layer.inner.kernel
    pure.bias = sl_layer.inner.bias

    sl_y = sl_layer.layer(sample).values
    pure_y = pure(x)
    a, r = tol(mx.float32, "block")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name=f"conv2d_grouped_g{groups}")
