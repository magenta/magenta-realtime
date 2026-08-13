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

Mirrors ``magenta_rt.mlx.depthformer._sample_categorical_with_temperature``
for the locked feature surface. The CFG layout matches sl's convention:
the batch is structured ``[fully_cond, partial_cond1, partial_cond2, ...]``
with size ``original_batch_size * (cfg_arity + 1)``; new logits are
``orig + sum_i scale_i * (orig - partial_i)``.
"""

from __future__ import annotations

from typing import Optional, Sequence as _Seq

import mlx.core as mx


def _large_neg(dtype) -> mx.array:
    if dtype == mx.float32:
        return mx.array(-3.4e38, dtype=dtype)
    if dtype == mx.bfloat16 or dtype == mx.float16:
        return mx.array(-65504.0, dtype=dtype)
    return mx.array(-1e9, dtype=dtype)


def sample_categorical_with_temperature(
    logits: mx.array,
    *,
    rng_key: mx.array,
    temperature: float | mx.array,
    top_k: Optional[int | mx.array] = None,
    top_p: Optional[float | mx.array] = None,
    cfg_scales: Optional[_Seq[float | mx.array]] = None,
    cfg_arity: int = 0,
    valid_range: Optional[tuple[int, int]] = None,
) -> mx.array:
    """Categorical sampling via the Gumbel-Max trick.

    Args:
        logits: ``[B, T, V]`` (or ``[B, V]``).
        rng_key: per-batch RNG (``[B]`` of ``mx.random.key``).
        temperature: scalar or per-batch.
        top_k / top_p: optional truncation.
        cfg_scales: when provided, classifier-free guidance is applied.
            ``cfg_arity`` is the number of partial-conditional copies
            per group; the batch must be ``original_B * (cfg_arity + 1)``.
        valid_range: ``(low, high)`` interval of allowed indices; logits
            outside are masked out.

    Returns:
        Sampled token indices, shape ``logits.shape[:-1]``. With CFG the
        result is repeated to match the original batch size.
    """
    temperature = mx.array(temperature, logits.dtype)
    if top_k is not None:
        top_k = mx.array(top_k, mx.int32)
    if top_p is not None:
        top_p = mx.array(top_p, mx.float32)

    if cfg_scales:
        arity = cfg_arity + 1
        B = logits.shape[0]
        if B % arity != 0:
            raise ValueError(f"batch {B} must be divisible by cfg arity {arity}")
        rng_key = rng_key[::arity]
        if temperature.ndim == 1:
            temperature = temperature[::arity]
        if top_k is not None and top_k.ndim == 1:
            top_k = top_k[::arity]
        if top_p is not None and top_p.ndim == 1:
            top_p = top_p[::arity]
        # Apply CFG: new = full + sum_i scale_i * (full - partial_i).
        full = logits[::arity]
        out = full
        for i, scale_i in enumerate(cfg_scales, start=1):
            scale_i = mx.array(scale_i, dtype=logits.dtype)
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

    gumbel = mx.random.gumbel(logits.shape, key=rng_key[0]).astype(logits.dtype)

    if valid_range is not None:
        idx = mx.arange(logits.shape[-1])
        in_range = (idx >= valid_range[0]) & (idx < valid_range[1])
        logits = mx.where(in_range, logits, _large_neg(logits.dtype))

    if top_k is not None:
        k = mx.clip(top_k, 1, logits.shape[-1])
        sorted_logits = mx.sort(logits, axis=-1)
        kth = mx.take_along_axis(sorted_logits, -k, axis=-1)
        logits = mx.where(logits >= kth, logits, _large_neg(logits.dtype))

    if top_p is not None:
        sorted_desc = mx.sort(logits, axis=-1)[..., ::-1]
        cum = mx.cumsum(mx.softmax(sorted_desc, axis=-1), axis=-1)
        cutoff = mx.sum((cum < top_p).astype(mx.int32), axis=-1, keepdims=True)
        cutoff = mx.minimum(cutoff, logits.shape[-1] - 1)
        thresh = mx.take_along_axis(sorted_desc, cutoff, axis=-1)
        logits = mx.where(logits >= thresh, logits, _large_neg(logits.dtype))

    logits = logits + gumbel * temperature
    # ``mx.argmax`` yields uint32; force int32 so the sampled tokens
    # match the int32 SOS frame from ``make_initial_state`` and the
    # int32 ``forced_tokens`` path. Without this the depthformer's
    # ``previous_frame`` flips dtype after the first step, which makes
    # the exported ``.mlxfn`` non-iterable (its traced input dtype no
    # longer matches the dtype it returns).
    sample = mx.argmax(logits, axis=-1).astype(mx.int32)

    if cfg_scales:
        arity = cfg_arity + 1
        sample = mx.repeat(sample, repeats=arity, axis=0)

    return sample
