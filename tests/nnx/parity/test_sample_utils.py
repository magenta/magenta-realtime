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

"""Tests for ``magenta_rt.nnx.sample_utils``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from magenta_rt.nnx.sample_utils import sample_categorical_with_temperature


def test_sampling_returns_valid_range():
    """``valid_range`` masks out logits outside the configured codebook
    interval — sampled tokens always land inside.
    """
    B, T, V = 2, 1, 50
    logits = jax.random.normal(jax.random.key(7), (B, T, V), dtype=jnp.float32)
    out = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(0),
        temperature=1.0, valid_range=(10, 20),
    )
    assert out.shape == (B, T)
    arr = np.asarray(out)
    assert np.all((arr >= 10) & (arr < 20))


def test_sampling_argmax_at_temperature_zero():
    """``temperature=0`` short-circuits to argmax (gumbel-noise scaled
    by 0 → tie-break by logits)."""
    B, T, V = 1, 1, 10
    logits = jnp.zeros((B, T, V)).at[..., 7].set(100.0)
    out = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(0), temperature=0.0,
    )
    assert int(out[0, 0]) == 7


def test_cfg_replicates_sample_to_full_batch():
    """With ``cfg_arity=1`` and a 2-element batch, the sampled output
    is repeated to match the original batch (sl convention)."""
    B, T, V = 2, 1, 16
    logits = jax.random.normal(jax.random.key(11), (B, T, V), dtype=jnp.float32)
    out = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(0),
        temperature=1.0, cfg_scales=[3.0], cfg_arity=1,
    )
    # Output shape matches input batch (post-CFG repeat).
    assert out.shape == (B, T)
    assert int(out[0, 0]) == int(out[1, 0])


def test_top_p_restricts_sampling_to_nucleus():
    """``top_p`` (nucleus) masks the low-probability tail: every sample lands
    in the smallest set of top tokens whose softmax mass reaches ``top_p``."""
    V = 12
    # Tokens 3 and 7 hold essentially all the mass (softmax ~0.73 / ~0.27); the
    # rest are ~0. With top_p=0.9 the nucleus is exactly {3, 7}.
    base = jnp.full((V,), -10.0).at[3].set(5.0).at[7].set(4.0)
    logits = jnp.broadcast_to(base, (512, 1, V))
    s = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(0), temperature=1.0, top_p=0.9,
    )
    sampled = set(np.unique(np.asarray(s)).tolist())
    assert sampled == {3, 7}, f"top_p must sample exactly the nucleus, got {sampled}"
    # Negative control: without top_p, the tail is reachable at high temperature.
    s2 = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(1), temperature=20.0, top_p=None,
    )
    assert len(set(np.unique(np.asarray(s2)).tolist())) > 2


def test_cfg_arity_two_combination_formula():
    """cfg_arity=2: combined logits are ``full + Σ scale_i·(full − partial_i)``
    over the ``[full, partial1, partial2]`` batch, and the sample is repeated
    back across the arity."""
    full = jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    p1 = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    p2 = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    logits = jnp.stack([full, p1, p2])[:, None, :]  # [3, 1, V]
    scales = [0.5, 0.25]
    expected = full + scales[0] * (full - p1) + scales[1] * (full - p2)
    out = sample_categorical_with_temperature(
        logits, rng_key=jax.random.key(0), temperature=0.0,
        cfg_scales=scales, cfg_arity=2,
    )
    assert out.shape == (3, 1)  # repeated across arity (orig batch 1 × 3)
    assert int(out[0, 0]) == int(jnp.argmax(expected))
    assert int(out[0, 0]) == int(out[1, 0]) == int(out[2, 0])
