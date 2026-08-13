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

"""Stateful streaming Conv2D / Conv2DTranspose parity tests.

Verifies that calling ``step`` chunk-by-chunk with a
:class:`Conv2DCache` produces output bit-equivalent (Conv2D) or
overlap-add-equivalent (Conv2DTranspose) to a single non-streaming
forward on the concatenated input.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from magenta_rt.mlx_pure.conv import Conv2D, Conv2DCache, Conv2DTranspose


def _randomize_conv(layer, scale=0.1, seed=0):
    """Replace zero-init kernel/bias with reproducible normals."""
    if layer.kernel is None:
        # Force lazy init by running a dummy forward.
        dummy = mx.zeros((1, layer.kernel_size[0], layer.kernel_size[1], 4))
        layer(dummy)
    key = mx.random.key(seed)
    sub_k, sub_b = mx.random.split(key, 2)
    layer.kernel = mx.random.normal(layer.kernel.shape, key=sub_k) * scale
    if layer.bias is not None:
        layer.bias = mx.random.normal(layer.bias.shape, key=sub_b) * scale


@pytest.mark.parametrize("time_padding,stride_t,kernel_t", [
    ("causal", 1, 3),
    ("causal", 1, 5),
    ("causal", 2, 4),
    ("semicausal", 1, 3),
    ("semicausal", 1, 5),
])
def test_conv2d_streaming_matches_full(time_padding, stride_t, kernel_t):
    B, T_full, S_in, C_in, C_out = 1, 8, 6, 4, 5
    chunk = stride_t  # one stride's worth per step
    layer = Conv2D(
        in_features=C_in, filters=C_out,
        kernel_size=(kernel_t, 3), strides=(stride_t, 1),
        time_padding=time_padding, spatial_padding="same",
        param_dtype=mx.float32,
    )
    _randomize_conv(layer, seed=1)
    # The semi-causal modes used by SpectroStream pair with explicit
    # spatial padding; with 'same' here, both modes behave consistently
    # on the time axis.

    x = mx.random.normal((B, T_full, S_in, C_in), key=mx.random.key(2))
    full = layer(x)

    cache = Conv2DCache()
    streamed = []
    for i in range(0, T_full, chunk):
        out = layer.step(x[:, i : i + chunk], cache)
        streamed.append(out)
    streamed = mx.concatenate(streamed, axis=1)

    np.testing.assert_allclose(
        np.array(streamed), np.array(full), atol=1e-5, rtol=1e-5,
    )


def test_conv2d_step_without_cache_chunk_independent():
    """When cache is None, step behaves like __call__ on the chunk."""
    layer = Conv2D(
        in_features=4, filters=5,
        kernel_size=(3, 3), strides=(1, 1),
        time_padding="causal", spatial_padding="same",
        param_dtype=mx.float32,
    )
    _randomize_conv(layer, seed=3)
    x = mx.random.normal((1, 4, 6, 4), key=mx.random.key(4))
    a = layer.step(x, cache=None)
    b = layer(x)
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_conv2d_step_rejects_right_pad():
    """Streaming with cache requires zero right-pad."""
    layer = Conv2D(
        in_features=4, filters=5,
        kernel_size=(4, 3), strides=(2, 1),
        time_padding="semicausal",  # stride_t=2 → pad_right > 0
        spatial_padding="same",
        param_dtype=mx.float32,
    )
    _randomize_conv(layer, seed=5)
    cache = Conv2DCache()
    x = mx.random.normal((1, 2, 6, 4), key=mx.random.key(6))
    with pytest.raises(NotImplementedError):
        layer.step(x, cache)


def test_conv2d_transpose_streaming_matches_full_after_warmup():
    """Streaming Conv2DTranspose with overlap-add matches non-streaming
    after dropping the K-S warmup samples."""
    B, T_full, S_in, C_in, C_out = 1, 6, 4, 3, 4
    kernel_t, stride_t = 4, 2
    layer = Conv2DTranspose(
        in_features=C_in, filters=C_out,
        kernel_size=(kernel_t, 3), strides=(stride_t, 1),
        time_padding="causal", spatial_padding="same",
        param_dtype=mx.float32,
    )
    _randomize_conv(layer, seed=7)
    x = mx.random.normal((B, T_full, S_in, C_in), key=mx.random.key(8))
    full = layer(x)

    cache = Conv2DCache()
    streamed = []
    for i in range(0, T_full, 1):
        streamed.append(layer.step(x[:, i : i + 1], cache))
    streamed = mx.concatenate(streamed, axis=1)

    streamed_np = np.array(streamed)
    full_np = np.array(full)
    # With trim direction set to drop K-S from the RIGHT (the
    # SpectroStream / sl convention), concatenated streaming output
    # equals the non-streaming trimmed output sample-for-sample for
    # the leading T_full*stride_t samples.
    cmp_len = full_np.shape[1]
    np.testing.assert_allclose(
        streamed_np[:, : cmp_len],
        full_np[:, : cmp_len],
        atol=1e-5, rtol=1e-5,
    )
