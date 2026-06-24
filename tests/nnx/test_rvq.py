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

"""Tests for ``magenta_rt.nnx.spectrostream.model.ResidualVectorQuantizer``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.spectrostream import ResidualVectorQuantizer


def _populated_rvq(num_q=4, num_emb=8, dim=16, *, seed=0, **kw):
    rvq = ResidualVectorQuantizer(
        num_quantizers=num_q,
        num_embeddings=num_emb,
        embedding_dim=dim,
        rngs=nnx.Rngs(seed),
        **kw,
    )
    rvq.embedding = nnx.Param(
        jax.random.normal(jax.random.key(seed + 1), rvq.embedding[...].shape) * 0.05
    )
    return rvq


def test_rvq_codes_to_embeddings_sums_levels():
    """``codes_to_embeddings`` returns the per-level lookup sum."""
    rvq = _populated_rvq()
    codes = jnp.array([[[0, 1, 2, 3]]], dtype=jnp.int32)  # [B=1, T=1, Q=4]
    out = rvq.codes_to_embeddings(codes)
    assert out.shape == (1, 1, rvq.embedding_dim)
    expected = (
        rvq.embedding[...][0, 0]
        + rvq.embedding[...][1, 1]
        + rvq.embedding[...][2, 2]
        + rvq.embedding[...][3, 3]
    )
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.asarray(expected),
                               atol=1e-6, rtol=1e-6)


def test_rvq_embeddings_to_codes_round_trips():
    """For a clean code → embedding → re-encode round trip, codes match."""
    rvq = _populated_rvq()
    codes = jnp.array([[[0, 1, 2, 3]]], dtype=jnp.int32)
    emb = rvq.codes_to_embeddings(codes)
    out_codes = rvq.embeddings_to_codes(emb)
    np.testing.assert_array_equal(np.asarray(out_codes), np.asarray(codes))


def test_rvq_truncation_level_caps_input_codebooks():
    rvq = _populated_rvq(truncation_level=2)
    too_many = jnp.zeros((1, 1, 4), dtype=jnp.int32)
    with pytest.raises(ValueError):
        rvq.codes_to_embeddings(too_many)


def test_rvq_unique_codes_offset():
    """``use_unique_codes=True`` adds ``q * num_emb`` to each output code."""
    rvq = _populated_rvq(use_unique_codes=True)
    codes = jnp.array([[[0, 1, 2, 3]]], dtype=jnp.int32)
    emb = rvq.codes_to_embeddings(codes)
    # The output codes should be 0, 1+8, 2+16, 3+24 if the input
    # round-trips identically.
    out = rvq.embeddings_to_codes(emb)
    expected = jnp.array([[[0 + 0, 1 + 8, 2 + 16, 3 + 24]]], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
