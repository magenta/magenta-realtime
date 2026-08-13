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

"""Common helpers ported from `mlx_lm.models.base`."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
) -> mx.array:
    """Causal (and optionally sliding-window) attention mask.

    Returns a boolean mask of shape `[N, offset + N]` where True means
    "query may attend to this key".
    """
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(
    h: mx.array,
    cache: Any | None = None,
    *,
    window_size: Optional[int] = None,
    return_array: bool = False,
):
    """Build a causal mask for queries `h` against the cache's past.

    Mirrors `mlx_lm.models.base.create_attention_mask`. When the cache
    has a `make_mask` method, that is preferred (used for sink slots).
    """
    N = h.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size is not None and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"
