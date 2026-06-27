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

"""Parity tests for ``magenta_rt.nnx.conv`` vs flax.linen reference.

Covers ``Conv2D`` (multiple paddings × strides), ``Conv2DTranspose``,
``Upsample2D``, ``AveragePooling2D``, and a smoke test for
``ParallelChannels`` (concat behaviour).
"""

from __future__ import annotations

import flax.linen as linen
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from magenta_rt.nnx.conv import (
    AveragePooling2D, Conv2D, Conv2DTranspose, ParallelChannels, Upsample2D,
)

from .conftest import assert_close


# -----------------------------------------------------------------------------
# Conv2D
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("time_padding", ["valid", "same", "causal"])
def test_conv2d_matches_linen_valid_paddings(time_padding, rng_key):
    """Conv2D with ``spatial_padding='valid'`` and a varying time-pad
    matches flax.linen's standard Conv at the same explicit padding.
    """
    in_features, filters, kt, ks = 4, 6, 3, 1
    pure = Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=(kt, ks), strides=(1, 1),
        time_padding=time_padding, spatial_padding="valid",
        use_bias=True, rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 3)
    x = jax.random.normal(sub[0], (1, 12, 5, in_features))
    pure.kernel = nnx.Param(jax.random.normal(sub[1], pure.kernel[...].shape) * 0.1)
    pure.bias = nnx.Param(jax.random.normal(sub[2], pure.bias[...].shape) * 0.1)

    y_pure = pure(x)

    # Reference: linen.Conv with explicit padding.
    if time_padding == "valid":
        time_pad = (0, 0)
    elif time_padding == "same":
        pad = max(0, kt - 1)
        time_pad = (pad // 2, pad - pad // 2)
    elif time_padding == "causal":
        time_pad = (kt - 1, 0)

    ref = linen.Conv(
        features=filters, kernel_size=(kt, ks), strides=(1, 1),
        padding=(time_pad, (0, 0)), use_bias=True,
    )
    # linen kernel shape: (kt, ks, in_features, filters) = HWIO
    # pure kernel: [filters, kt, ks, in_features] = OHWI
    linen_kernel = jnp.transpose(pure.kernel[...], (1, 2, 3, 0))
    ref_params = {"params": {"kernel": linen_kernel, "bias": pure.bias[...]}}
    y_ref = ref.apply(ref_params, x)
    assert_close(y_ref, y_pure, atol=1e-5, rtol=1e-5, name=f"conv2d_{time_padding}")


@pytest.mark.parametrize(
    "in_features,filters,groups",
    [(4, 8, 2), (4, 4, 4), (6, 6, 3)],  # grouped, depthwise (g==in==out), grouped
)
def test_conv2d_grouped_matches_linen(in_features, filters, groups, rng_key):
    """Grouped / depthwise conv (``groups > 1``) vs linen ``feature_group_count``.
    ``AveragePooling2D`` and the production codec rely on grouping; only
    ``groups=1`` was covered before."""
    kt, ks = 3, 1
    pure = Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=(kt, ks), strides=(1, 1),
        time_padding="causal", spatial_padding="valid",
        groups=groups, use_bias=True, rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 3)
    x = jax.random.normal(sub[0], (1, 12, 5, in_features))
    pure.kernel = nnx.Param(jax.random.normal(sub[1], pure.kernel[...].shape) * 0.1)
    pure.bias = nnx.Param(jax.random.normal(sub[2], pure.bias[...].shape) * 0.1)

    y_pure = pure(x)

    ref = linen.Conv(
        features=filters, kernel_size=(kt, ks), strides=(1, 1),
        padding=((kt - 1, 0), (0, 0)),  # causal
        feature_group_count=groups, use_bias=True,
    )
    # pure kernel [filters, kt, ks, in//groups] (OHWI) -> linen (kt, ks, in//groups, filters) HWIO
    linen_kernel = jnp.transpose(pure.kernel[...], (1, 2, 3, 0))
    ref_params = {"params": {"kernel": linen_kernel, "bias": pure.bias[...]}}
    y_ref = ref.apply(ref_params, x)
    assert_close(y_ref, y_pure, atol=1e-5, rtol=1e-5, name=f"conv2d_grouped_g{groups}")


def test_conv2d_with_strides(rng_key):
    in_features, filters = 4, 6
    pure = Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=(3, 1), strides=(2, 1),
        time_padding="causal", spatial_padding="valid",
        use_bias=False, rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 2)
    x = jax.random.normal(sub[0], (1, 16, 5, in_features))
    pure.kernel = nnx.Param(jax.random.normal(sub[1], pure.kernel[...].shape) * 0.1)

    ref = linen.Conv(
        features=filters, kernel_size=(3, 1), strides=(2, 1),
        padding=((2, 0), (0, 0)), use_bias=False,
    )
    linen_kernel = jnp.transpose(pure.kernel[...], (1, 2, 3, 0))
    y_pure = pure(x)
    y_ref = ref.apply({"params": {"kernel": linen_kernel}}, x)
    assert_close(y_ref, y_pure, atol=1e-5, rtol=1e-5, name="conv2d_strided")


# -----------------------------------------------------------------------------
# Conv2DTranspose
# -----------------------------------------------------------------------------


def test_conv2d_transpose_doubles_time_length(rng_key):
    in_features, filters = 4, 4
    pure = Conv2DTranspose(
        in_features=in_features, filters=filters,
        kernel_size=(2, 1), strides=(2, 1),
        time_padding="causal", spatial_padding="valid",
        use_bias=False, rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 2)
    x = jax.random.normal(sub[0], (1, 8, 4, in_features))
    pure.kernel = nnx.Param(jax.random.normal(sub[1], pure.kernel[...].shape) * 0.1)
    y = pure(x)
    # T_out = T_in * stride_t = 16. Spatial unchanged.
    assert y.shape == (1, 16, 4, filters)


# -----------------------------------------------------------------------------
# AveragePooling2D / Upsample2D
# -----------------------------------------------------------------------------


def test_average_pooling_2d_simple():
    pool = AveragePooling2D(pool_size=(2, 1), strides=(2, 1), time_padding="valid")
    # Input is constant 4 → output should be 4 everywhere.
    x = jnp.full((1, 8, 4, 3), 4.0)
    y = pool(x)
    assert y.shape == (1, 4, 4, 3)
    assert jnp.allclose(y, 4.0)


def test_upsample_2d_repeats():
    up = Upsample2D(rate=(2, 1))
    x = jnp.arange(8).astype(jnp.float32).reshape(1, 4, 2, 1)
    y = up(x)
    assert y.shape == (1, 8, 2, 1)
    # Each row repeated once along time.
    assert jnp.array_equal(y[0, 0:2, :, 0], jnp.broadcast_to(x[0, 0:1, :, 0], (2, 2)))


# -----------------------------------------------------------------------------
# ParallelChannels
# -----------------------------------------------------------------------------


def test_parallel_channels_concat_smoke(rng_key):
    """Without streaming, ParallelChannels(num_groups=2, inner=Conv2D)
    splits → applies inner → concats. Smoke check that the shape and
    per-group output align with running the inner directly on each
    half.
    """
    in_features = 8
    filters = 6
    inner = Conv2D(
        in_features=in_features // 2, filters=filters,
        kernel_size=(1, 1), strides=(1, 1),
        time_padding="valid", spatial_padding="valid",
        use_bias=False, rngs=nnx.Rngs(0),
    )
    sub = jax.random.split(rng_key, 2)
    inner.kernel = nnx.Param(jax.random.normal(sub[0], inner.kernel[...].shape) * 0.1)

    pc = ParallelChannels(inner=inner, num_groups=2)
    x = jax.random.normal(sub[1], (1, 4, 3, in_features))
    y = pc(x)
    # Output shape: [B, T, S, num_groups * filters]
    assert y.shape == (1, 4, 3, 2 * filters)

    # Manual reference: inner on each half, concatenated.
    a = inner(x[..., : in_features // 2])
    b = inner(x[..., in_features // 2:])
    expected = jnp.concatenate([a, b], axis=-1)
    assert_close(y, expected, atol=1e-6, rtol=1e-6, name="parallel_channels")
