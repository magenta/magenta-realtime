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

"""Sampling helpers for the depthformer's autoregressive generation.

CFG batch layout: ``[fully_cond, partial_cond1, partial_cond2, ...]``
with size ``original_batch_size * (cfg_arity + 1)``; new logits are
``orig + sum_i scale_i * (orig - partial_i)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp


def _large_neg(dtype) -> jnp.ndarray:
    if dtype == jnp.float32:
        return jnp.array(-3.4e38, dtype=dtype)
    if dtype == jnp.bfloat16 or dtype == jnp.float16:
        return jnp.array(-65504.0, dtype=dtype)
    return jnp.array(-1e9, dtype=dtype)


def sample_categorical_with_temperature(
    logits: jnp.ndarray,
    *,
    rng_key: jax.Array,
    temperature: float | jnp.ndarray,
    top_k: Optional[int | jnp.ndarray] = None,
    top_p: Optional[float | jnp.ndarray] = None,
    cfg_scales: Optional[Sequence[float | jnp.ndarray]] = None,
    cfg_arity: int = 0,
    valid_range: Optional[tuple[int, int]] = None,
) -> jnp.ndarray:
    """Categorical sampling via the Gumbel-Max trick.

    ``rng_key`` is a single :func:`jax.random.key` shared across the
    batch (caller advances it between calls via ``jax.random.split``).
    """
    temperature = jnp.asarray(temperature, dtype=logits.dtype)
    if top_k is not None:
        top_k = jnp.asarray(top_k, dtype=jnp.int32)
    if top_p is not None:
        top_p = jnp.asarray(top_p, dtype=jnp.float32)

    if cfg_scales:
        arity = cfg_arity + 1
        B = logits.shape[0]
        if B % arity != 0:
            raise ValueError(f"batch {B} must be divisible by cfg arity {arity}")
        if temperature.ndim == 1:
            temperature = temperature[::arity]
        if top_k is not None and top_k.ndim == 1:
            top_k = top_k[::arity]
        if top_p is not None and top_p.ndim == 1:
            top_p = top_p[::arity]
        full = logits[::arity]
        out = full
        for i, scale_i in enumerate(cfg_scales, start=1):
            scale_i = jnp.asarray(scale_i, dtype=logits.dtype)
            partial = logits[i::arity]
            out = out + scale_i * (full - partial)
        logits = out

    if temperature.ndim == 1:
        temperature = temperature[..., None, None]
    if top_k is not None:
        if top_k.ndim == 1:
            top_k = top_k[..., None, None]
        elif top_k.ndim == 0:
            top_k = top_k[None, None, None]
    if top_p is not None:
        if top_p.ndim == 1:
            top_p = top_p[..., None, None]
        elif top_p.ndim == 0:
            top_p = top_p[None, None, None]

    gumbel = jax.random.gumbel(rng_key, logits.shape, dtype=logits.dtype)

    if valid_range is not None:
        idx = jnp.arange(logits.shape[-1])
        in_range = (idx >= valid_range[0]) & (idx < valid_range[1])
        logits = jnp.where(in_range, logits, _large_neg(logits.dtype))

    if top_k is not None:
        k = jnp.clip(top_k, 1, logits.shape[-1])
        sorted_logits = jnp.sort(logits, axis=-1)
        # take_along_axis with negative index needs the index converted.
        kth = jnp.take_along_axis(sorted_logits, sorted_logits.shape[-1] - k, axis=-1)
        logits = jnp.where(logits >= kth, logits, _large_neg(logits.dtype))

    if top_p is not None:
        sorted_desc = jnp.sort(logits, axis=-1)[..., ::-1]
        cum = jnp.cumsum(jax.nn.softmax(sorted_desc, axis=-1), axis=-1)
        cutoff = jnp.sum((cum < top_p).astype(jnp.int32), axis=-1, keepdims=True)
        cutoff = jnp.minimum(cutoff, logits.shape[-1] - 1)
        thresh = jnp.take_along_axis(sorted_desc, cutoff, axis=-1)
        logits = jnp.where(logits >= thresh, logits, _large_neg(logits.dtype))

    logits = logits + gumbel * temperature
    sample = jnp.argmax(logits, axis=-1)

    if cfg_scales:
        arity = cfg_arity + 1
        sample = jnp.repeat(sample, repeats=arity, axis=0)

    return sample
