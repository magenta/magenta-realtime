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

"""KV / overlap-add cache modules.

Each class is an ``nnx.Module`` whose array state is held in
``nnx.Cache`` slots, populated by :meth:`init_cache`. Consumer
modules (attention, InverseSTFT) compose one as a child and walk
the tree to call :meth:`init_cache` from a top-level container.

* :class:`LocalKVCache` — sliding-window KV with reserved sink slots
  at the front. Used by ``LocalSelfAttention`` and (without sinks)
  by ``StreamingCrossAttention``.
* :class:`OverlapAddCache` — running overlap buffer for streaming
  ``InverseSTFT``. Used by the SpectroStream decoder.
"""

from __future__ import annotations

import jax.numpy as jnp
from einops import repeat
from flax import nnx


class LocalKVCache(nnx.Module):
    """Sliding-window KV cache with reserved sink slots at the front.

    Layout: ``[B, n_kv_heads, num_sinks + window_size, head_dim]``.
    The first ``num_sinks`` slots are reserved for the layer's learned
    sink-embedding K/V (set via :meth:`prime_sinks`). The rolling
    window holds up to ``window_size`` of the most recent tokens
    (including the current step).

    Constructor takes the shape parameters; the actual array slots are
    allocated by :meth:`init_cache`.
    """

    def __init__(self, *, window_size: int, num_sinks: int = 0):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        self.window_size = window_size
        self.num_sinks = num_sinks

        # Lazy slots — populated by init_cache.
        self.cached_key: nnx.Cache | None = nnx.data(None)
        self.cached_value: nnx.Cache | None = nnx.data(None)
        # Total tokens written (excluding sinks). Used for absolute
        # position tracking in masks.
        self.cache_index: nnx.Cache | None = nnx.data(None)

    def init_cache(
        self,
        *,
        batch: int,
        n_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        dtype=jnp.float32,
    ) -> None:
        """Allocate the rolling-window slots and (zero) sink slots.

        Sinks are initialized to zero; callers populate them via
        :meth:`prime_sinks` before the first :meth:`update_and_fetch`.
        """
        total = self.num_sinks + self.window_size
        self.cached_key = nnx.Cache(
            jnp.zeros((batch, n_kv_heads, total, k_head_dim), dtype=dtype)
        )
        self.cached_value = nnx.Cache(
            jnp.zeros((batch, n_kv_heads, total, v_head_dim), dtype=dtype)
        )
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    def remove_cache(self) -> None:
        self.cached_key = nnx.data(None)
        self.cached_value = nnx.data(None)
        self.cache_index = nnx.data(None)

    def soft_reset(self) -> None:
        """Reset the rolling-window state in place without changing
        slot types or shapes — sets ``cache_index`` to 0 so future
        ``update_and_fetch`` calls overwrite stale K/V from the
        current logical position 0. Safe to call inside ``nnx.jit`` /
        ``nnx.scan``.
        """
        if self.cache_index is not None:
            self.cache_index[...] = jnp.zeros_like(self.cache_index[...])

    @property
    def initialized(self) -> bool:
        return self.cached_key is not None

    def prime_sinks(self, sink_keys: jnp.ndarray, sink_values: jnp.ndarray) -> None:
        """Install learned sink K/V (shape ``[num_sinks, n_heads, head_dim]``)
        by planting them at the front of the rolling buffer.

        The sink K/V are NOT also stashed in separate cache slots: doing so
        aliased the attention's ``sink_*_embeddings`` *parameter* buffers into
        the streaming-state pytree, which blocks donating that state (XLA
        rejects a buffer that is both a non-donated param input and a donated
        state input). The planted copies below are fresh buffers (``.at[].set``),
        so they don't alias. ``init_cache`` must have been called.
        """
        if self.num_sinks == 0:
            return
        if not self.initialized:
            raise RuntimeError("init_cache() must be called before prime_sinks().")
        if sink_keys.shape[0] != self.num_sinks:
            raise ValueError(
                f"sink_keys leading dim {sink_keys.shape[0]} != "
                f"num_sinks {self.num_sinks}"
            )
        # Plant them at the front of the rolling buffer, broadcast over
        # batch. The first ``num_sinks`` slots along the time axis are
        # reserved for sinks; ``init_cache`` already zeroed them.
        ck = self.cached_key[...]
        cv = self.cached_value[...]
        B = ck.shape[0]
        sk = repeat(sink_keys.astype(ck.dtype), "s h d -> b h s d", b=B)
        sv = repeat(sink_values.astype(cv.dtype), "s h d -> b h s d", b=B)
        self.cached_key[...] = ck.at[:, :, :self.num_sinks, :].set(sk)
        self.cached_value[...] = cv.at[:, :, :self.num_sinks, :].set(sv)

    def update_and_fetch(self, keys: jnp.ndarray, values: jnp.ndarray):
        """Append ``[B, n_kv_heads, S, head_dim]`` to the rolling window.

        Returns the full ``[B, n_kv_heads, num_sinks + window_size, head_dim]``
        cache; consumers use :meth:`make_mask` to restrict attention to
        the populated region (matches ``nnx.MultiHeadAttention``).
        """
        if not self.initialized:
            raise RuntimeError(
                "LocalKVCache.init_cache() must be called before update_and_fetch()"
            )
        S = keys.shape[2]
        offset = self.cache_index[...]

        # Compute the destination slot for each of the S new tokens
        # once, then scatter all of them in a single ``.at[].set`` —
        # one fused write per K and V cache instead of S sequential
        # ``lax.dynamic_update_slice`` calls.
        slots = self.num_sinks + (offset + jnp.arange(S, dtype=jnp.int32)) % self.window_size
        ck = self.cached_key[...].at[:, :, slots, :].set(keys.astype(self.cached_key[...].dtype))
        cv = self.cached_value[...].at[:, :, slots, :].set(values.astype(self.cached_value[...].dtype))
        self.cached_key[...] = ck
        self.cached_value[...] = cv
        self.cache_index[...] = offset + S
        return ck, cv

    def make_mask(self, N: int) -> jnp.ndarray:
        """Boolean mask of shape ``[N, num_sinks + window_size]``.

        Sinks are always visible; for non-sink slots the mask is True
        only for slots whose stored logical token is <= the
        corresponding query's logical position.
        """
        offset = self.cache_index[...]
        wsz = jnp.minimum(offset, self.window_size)

        if N == 0:
            return jnp.ones((0, self.num_sinks + self.window_size), dtype=jnp.bool_)

        first_q = offset - N

        # For each physical slot s in [0, window_size), the stored
        # logical index is the largest t in [offset-wsz, offset) with
        # t % window_size == s.
        slots = jnp.arange(self.window_size, dtype=jnp.int32)
        base = offset - wsz
        logical = base + (slots - base) % self.window_size
        logical = jnp.where(logical >= offset, logical - self.window_size, logical)
        # Mark un-filled slots as +inf so they're never visible.
        logical_safe = jnp.where(slots < wsz, logical, jnp.iinfo(jnp.int32).max)

        q_idx = jnp.arange(N, dtype=jnp.int32)[:, None] + first_q  # [N, 1]
        causal_past = q_idx >= logical_safe[None, :]  # [N, window_size]
        sinks = jnp.ones((N, self.num_sinks), dtype=jnp.bool_)
        return jnp.concatenate([sinks, causal_past], axis=-1)


class OverlapAddCache(nnx.Module):
    """Buffer state for streaming overlap-add (used by streaming
    :class:`InverseSTFT`).

    Stores the running sum of the most recent
    ``frame_length - frame_step`` samples that haven't been emitted
    yet. The buffer is allocated by :meth:`init_cache` and grown via
    :meth:`add_overlap`.
    """

    def __init__(self):
        self.buffer: nnx.Cache | None = nnx.data(None)

    def init_cache(
        self,
        *,
        batch: int,
        overlap: int,
        rest_shape: tuple,
        dtype=jnp.float32,
    ) -> None:
        self.buffer = nnx.Cache(
            jnp.zeros((batch, overlap) + tuple(rest_shape), dtype=dtype)
        )

    def remove_cache(self) -> None:
        self.buffer = nnx.data(None)

    @property
    def initialized(self) -> bool:
        return self.buffer is not None
