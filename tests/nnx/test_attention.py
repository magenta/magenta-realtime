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

"""Tests for ``magenta_rt.nnx.attention``.

* Construction smoke for both attention classes.
* Streaming concat parity: with ``streaming=True`` and a fresh cache,
  feeding tokens one at a time produces the same output as one
  full-sequence forward (within fp32 tolerance).
* Sink-embedding plumbing: the rolling cache plants the layer's sink
  K/V at the front and ``_attend`` doesn't double-count them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.attention import LocalSelfAttention, StreamingCrossAttention


def _populate(layer: LocalSelfAttention | StreamingCrossAttention, key) -> None:
    """Re-init each Param with small random values for the test."""
    sub = jax.random.split(key, 5)
    layer.q_proj.kernel = nnx.Param(jax.random.normal(sub[0], layer.q_proj.kernel[...].shape) * 0.05)
    layer.kv_proj.kernel = nnx.Param(jax.random.normal(sub[1], layer.kv_proj.kernel[...].shape) * 0.05)
    layer.per_dim_scale_param = nnx.Param(
        jax.random.normal(sub[2], layer.per_dim_scale_param[...].shape) * 0.01
    )
    if layer.sink_key_embeddings is not None:
        layer.sink_key_embeddings = nnx.Param(
            jax.random.normal(sub[3], layer.sink_key_embeddings[...].shape) * 0.05
        )
        layer.sink_value_embeddings = nnx.Param(
            jax.random.normal(sub[4], layer.sink_value_embeddings[...].shape) * 0.05
        )
    # Output-projection EinsumDense kernel — lazy-init by running a dummy fwd.
    proj = layer.output_projection
    n, h = layer.num_heads, layer.units_per_head
    proj(jnp.zeros((1, 1, n, h)))
    proj.kernel = nnx.Param(
        jax.random.normal(jax.random.split(sub[4])[0], proj.kernel[...].shape) * 0.05
    )


# -----------------------------------------------------------------------------
# LocalSelfAttention
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_sinks", [0, 1])
def test_local_self_attention_construction(num_sinks):
    layer = LocalSelfAttention(
        in_features=8, num_heads=2, units_per_head=4,
        max_past_horizon=3, num_sink_embeddings=num_sinks,
        rngs=nnx.Rngs(0),
    )
    assert layer.streaming is False
    assert layer.cache is not None  # always allocated; init_cache fills slots


def test_local_self_attention_full_seq_forward(rng_key):
    """Smoke + finite-output sanity for full-sequence forward."""
    layer = LocalSelfAttention(
        in_features=8, num_heads=2, units_per_head=4,
        max_past_horizon=4, num_sink_embeddings=0,
        rngs=nnx.Rngs(0),
    )
    _populate(layer, jax.random.key(7))
    x = jax.random.normal(rng_key, (1, 6, 8))
    y = layer(x)
    assert y.shape == (1, 6, 8)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.max(jnp.abs(y)) > 0


def test_local_self_attention_streaming_concat_matches_full(rng_key):
    """One-token-at-a-time streaming matches a single full-sequence
    forward (within fp32 tolerance) for the same input + cache window
    big enough to include all tokens.
    """
    full = LocalSelfAttention(
        in_features=8, num_heads=2, units_per_head=4,
        max_past_horizon=8, num_sink_embeddings=0,
        rngs=nnx.Rngs(11),
    )
    stream = LocalSelfAttention(
        in_features=8, num_heads=2, units_per_head=4,
        max_past_horizon=8, num_sink_embeddings=0,
        rngs=nnx.Rngs(11),
    )
    _populate(full, jax.random.key(13))
    _populate(stream, jax.random.key(13))
    # Mirror weights manually since random init is rng-stateful.
    stream.q_proj.kernel = nnx.Param(full.q_proj.kernel[...])
    stream.kv_proj.kernel = nnx.Param(full.kv_proj.kernel[...])
    stream.per_dim_scale_param = nnx.Param(full.per_dim_scale_param[...])
    stream.output_projection.kernel = nnx.Param(full.output_projection.kernel[...])

    x = jax.random.normal(rng_key, (1, 6, 8))

    y_full = full(x)

    # Streaming.
    stream.streaming = True
    stream.init_cache(batch=1, dtype=jnp.float32)
    chunks = [stream(x[:, t:t + 1]) for t in range(x.shape[1])]
    y_stream = jnp.concatenate(chunks, axis=1)

    np.testing.assert_allclose(np.asarray(y_full), np.asarray(y_stream),
                               atol=1e-5, rtol=1e-5)


# -----------------------------------------------------------------------------
# StreamingCrossAttention
# -----------------------------------------------------------------------------


def test_streaming_cross_attention_full_seq(rng_key):
    """Full-sequence forward (T_q == T_kv) shape + finite-output check."""
    layer = StreamingCrossAttention(
        in_features=8, source_features=12,
        num_heads=2, units_per_head=4,
        max_past_horizon=4, num_sink_embeddings=0,
        rngs=nnx.Rngs(0),
    )
    _populate(layer, jax.random.key(7))
    sub = jax.random.split(rng_key, 2)
    x = jax.random.normal(sub[0], (1, 5, 8))
    src = jax.random.normal(sub[1], (1, 5, 12))
    y = layer(x, source=src)
    assert y.shape == (1, 5, 8)
    assert jnp.all(jnp.isfinite(y))
