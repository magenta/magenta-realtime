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

"""Streaming-cache tests for ``magenta_rt.nnx.conv``.

The user-facing streaming workflow is the bare nnx primitive set:

  model.set_attributes(streaming=True)   # flip the flag everywhere
  model.init_cache(batch=B, dtype=...)   # alloc attention caches
  ...                                    # streaming forward
  model.set_attributes(streaming=False)
  model.remove_cache()                  # clear conv left-context etc.

These tests pin both the per-Conv behaviour and the
``model.remove_cache`` recursive walk on a ``ParallelChannels`` tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from magenta_rt.nnx.conv import (
    Conv2D, Conv2DTranspose, ParallelChannels, remove_cache,
)


def _rand_conv(filters=4, in_features=4, kt=3, st=1, time_padding="causal", *, seed=0):
    conv = Conv2D(
        in_features=in_features, filters=filters,
        kernel_size=(kt, 1), strides=(st, 1),
        time_padding=time_padding, spatial_padding="valid",
        use_bias=True, rngs=nnx.Rngs(seed),
    )
    sub = jax.random.split(jax.random.key(seed + 1), 2)
    conv.kernel = nnx.Param(jax.random.normal(sub[0], conv.kernel[...].shape) * 0.1)
    conv.bias = nnx.Param(jax.random.normal(sub[1], conv.bias[...].shape) * 0.05)
    return conv


def _rand_conv_t(filters=4, in_features=4, kt=4, st=2, *, seed=0):
    conv = Conv2DTranspose(
        in_features=in_features, filters=filters,
        kernel_size=(kt, 1), strides=(st, 1),
        time_padding="causal", spatial_padding="valid",
        use_bias=False, rngs=nnx.Rngs(seed),
    )
    sub = jax.random.split(jax.random.key(seed + 1), 1)
    conv.kernel = nnx.Param(jax.random.normal(sub[0], conv.kernel[...].shape) * 0.1)
    return conv


# -----------------------------------------------------------------------------
# Conv2D causal streaming
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("st", [1, 2])
def test_conv2d_causal_streaming_concat_matches_full(st):
    """Concatenated ``Conv2D`` streaming chunks match a single
    non-streaming forward on the joined input.
    """
    full = _rand_conv(kt=3, st=st, time_padding="causal", seed=42)
    stream = _rand_conv(kt=3, st=st, time_padding="causal", seed=42)
    # Mirror weights so both convs are identical.
    stream.kernel = nnx.Param(full.kernel[...])
    stream.bias = nnx.Param(full.bias[...])

    x = jax.random.normal(jax.random.key(7), (1, 12, 5, 4))

    y_full = full(x)

    stream.set_attributes(streaming=True, raise_if_not_found=False)
    chunks = []
    chunk_size = 4
    for t in range(0, x.shape[1], chunk_size):
        chunks.append(stream(x[:, t: t + chunk_size]))
    y_stream = jnp.concatenate(chunks, axis=1)

    assert y_full.shape == y_stream.shape
    import numpy as np
    np.testing.assert_allclose(np.asarray(y_full), np.asarray(y_stream),
                               atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------------
# Conv2DTranspose causal streaming
# -----------------------------------------------------------------------------


def test_conv2d_transpose_causal_streaming_overlaps():
    """Streaming Conv2DTranspose: concatenated chunks ``streaming[K-S:]``
    align with ``non_streaming[: -??]``. We just check shape + a finite
    output here; the bit-exact streaming-concat match is exercised
    end-to-end by the SpectroStream streaming tests.
    """
    full = _rand_conv_t(kt=4, st=2, seed=11)
    stream = _rand_conv_t(kt=4, st=2, seed=11)
    stream.kernel = nnx.Param(full.kernel[...])

    x = jax.random.normal(jax.random.key(19), (1, 8, 4, 4))

    y_full = full(x)

    stream.set_attributes(streaming=True, raise_if_not_found=False)
    chunks = []
    for t in range(0, x.shape[1], 2):
        chunks.append(stream(x[:, t: t + 2]))
    y_stream = jnp.concatenate(chunks, axis=1)

    # Output length: T_in * stride_t = 16. Same for both.
    assert y_full.shape == (1, 16, 4, 4)
    assert y_stream.shape == (1, 16, 4, 4)
    # Streaming output is finite and not trivially zero.
    assert jnp.all(jnp.isfinite(y_stream))
    assert jnp.max(jnp.abs(y_stream)) > 0


# -----------------------------------------------------------------------------
# Streaming API: bare nnx primitives (set_attributes / remove_cache)
# -----------------------------------------------------------------------------


def test_set_attributes_streaming_flips_recursively():
    """``model.set_attributes(streaming=True)`` flips the bit on every
    submodule that defines ``streaming``. The bare nnx primitive is
    the recommended public API; the ``remove_cache`` walk is the
    counterpart for clearing cache state between sessions.
    """
    inner = _rand_conv(in_features=2, filters=3, seed=0)
    pc = ParallelChannels(inner=inner, num_groups=2)
    assert inner.streaming is False

    pc.set_attributes(streaming=True, raise_if_not_found=False)
    assert inner.streaming is True

    # eval(streaming=False) flips it back.
    pc.eval(streaming=False)
    assert inner.streaming is False


def test_streaming_cache_survives_nnx_split_merge():
    """``nnx.Cache`` slots round-trip cleanly through ``nnx.split`` /
    ``nnx.merge`` — sanity-check that streaming state is captured by
    the standard nnx state machinery rather than being smuggled in
    plain Python attributes.
    """
    conv = _rand_conv(in_features=2, filters=2, seed=0)
    conv.set_attributes(streaming=True, raise_if_not_found=False)
    x = jax.random.normal(jax.random.key(7), (1, 4, 3, 2))
    _ = conv(x)
    # Grab the cache slot value before split.
    before = conv.cached_left[...]

    graphdef, state = nnx.split(conv)
    rebuilt = nnx.merge(graphdef, state)
    after = rebuilt.cached_left[...]
    assert before.shape == after.shape
    import numpy as np
    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_remove_cache_clears_streaming_state():
    """``remove_cache(model)`` walks the subtree and nulls out every
    ``nnx.Cache`` slot. ParallelChannels no longer has a per-group
    Python cache list (the inner Conv2D's ``cached_left`` is shared
    across all groups via the batch-stack reshape), so the only
    streaming state to clear is the inner conv's cache slot.
    """
    inner = _rand_conv(in_features=2, filters=3, seed=0)
    pc = ParallelChannels(inner=inner, num_groups=2)
    pc.set_attributes(streaming=True, raise_if_not_found=False)

    x = jax.random.normal(jax.random.key(0), (1, 4, 5, 4))
    _ = pc(x)
    assert inner.cached_left is not None

    remove_cache(pc)
    assert inner.cached_left is None
    # Streaming flag is independent — flipped separately.
    assert inner.streaming is True
    pc.set_attributes(streaming=False, raise_if_not_found=False)
    assert inner.streaming is False
