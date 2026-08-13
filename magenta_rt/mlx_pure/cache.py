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

"""KV-cache objects (mlx-lm style).

Each attention layer takes an optional `cache` arg whose
`update_and_fetch(keys, values) -> (keys, values)` method is the central
contract. Mirrors `mlx_lm.models.cache`.

* :class:`LocalKVCache` — sliding-window + reserved "sink" slots at
  the front (the sink keys/values are layer parameters, written once
  at first use). Used by ``LocalSelfAttention`` and (without sinks)
  by ``StreamingCrossAttention``.
* :class:`OverlapAddCache` — running overlap buffer for the streaming
  ``InverseSTFT``.
* :class:`KVCache` — plain append-only KV, copied near-verbatim from
  mlx-lm. Not used by any shipping module; provided so external
  consumers that already speak the mlx-lm cache protocol can plug in.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from .base import create_causal_mask


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self) -> bool:
        return False

    def size(self) -> int:
        return 0

    @property
    def nbytes(self) -> int:
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class KVCache(_BaseCache):
    """Plain append-only KV cache.

    Identical in contract to `mlx_lm.models.cache.KVCache`: keys/values
    have shape ``[B, n_kv_heads, S, head_dim]``; ``update_and_fetch``
    writes the new step at ``offset`` and returns the populated prefix.
    """

    step = 256

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self) -> int:
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return None, None
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return (self.keys[..., : self.offset, :], self.values[..., : self.offset, :])

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N: int, *, return_array: bool = False, window_size: Optional[int] = None):
        if N == 1 and not return_array:
            return None
        return create_causal_mask(N, offset=self.offset, window_size=window_size)

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class LocalKVCache(_BaseCache):
    """Sliding-window KV cache with reserved sink slots at the front.

    The first ``num_sinks`` slots are reserved for the layer's learned
    sink-embedding K/V (set via :meth:`prime_sinks`). The rolling
    window holds up to ``max_past`` of the most recent tokens.

    ``offset`` (total tokens written) is stored as an ``mx.array``
    scalar so ``mx.exporter`` traces it as runtime state — under
    tracing the slot-write index is computed with ``mx.put_along_axis``
    rather than baked in at trace time. Eager use is unaffected;
    ``offset.item()`` resolves the same value lazily.

    :meth:`update_and_fetch` returns the full ``[..., num_sinks +
    max_past, ...]`` slice. Slots beyond the current window are masked
    out by :meth:`make_mask`, which also returns a fixed-shape
    ``[N, num_sinks + max_past]`` boolean array.
    """

    def __init__(
        self,
        window_size: int,
        num_sinks: int = 0,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        # `max_past` retained as alias for backward compat in tests.
        self.max_past = window_size
        self.num_sinks = num_sinks
        # Filled lazily on first update_and_fetch when shapes are known.
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        # Total tokens written (excluding sinks). Stored as a Python ``int``
        # in eager mode for fast in-place slot writes; the export wrapper
        # promotes this to an ``mx.array`` scalar before tracing so
        # ``mx.exporter`` sees the offset as runtime state. Both paths
        # produce the same fixed-shape return from ``update_and_fetch``.
        self.offset: Any = 0
        # Whether sinks have been primed. Python bool — set once at
        # streaming-arming time, before tracing.
        self._sinks_primed = num_sinks == 0
        self._sink_keys: Optional[mx.array] = None
        self._sink_values: Optional[mx.array] = None

    # Back-compat alias: callers that read ``self.window_size`` get the
    # currently-filled portion of the rolling buffer.
    @property
    def window_size(self) -> int:
        off = int(self.offset.item()) if isinstance(self.offset, mx.array) else int(self.offset)
        return min(off, self.max_past)

    def prime_sinks(self, sink_keys: mx.array, sink_values: mx.array) -> None:
        """Install learned sink K/V (shape ``[num_sinks, n_heads, head_dim]``)."""
        if self.num_sinks == 0:
            return
        if sink_keys.shape[0] != self.num_sinks:
            raise ValueError(
                f"sink_keys leading dim {sink_keys.shape[0]} != "
                f"num_sinks {self.num_sinks}"
            )
        self._sink_keys = sink_keys
        self._sink_values = sink_values
        self._sinks_primed = True

    def _allocate(self, batch: int, n_kv_heads: int, k_head_dim: int,
                  v_head_dim: int, dtype):
        total_len = self.num_sinks + self.max_past
        self.keys = mx.zeros((batch, n_kv_heads, total_len, k_head_dim), dtype=dtype)
        self.values = mx.zeros((batch, n_kv_heads, total_len, v_head_dim), dtype=dtype)
        if self.num_sinks and self._sink_keys is not None:
            sk = self._sink_keys.astype(dtype)
            sv = self._sink_values.astype(dtype)
            sk = mx.transpose(sk, (1, 0, 2))[None]
            sv = mx.transpose(sv, (1, 0, 2))[None]
            self.keys[..., : self.num_sinks, :] = mx.broadcast_to(
                sk, (batch, n_kv_heads, self.num_sinks, k_head_dim)
            )
            self.values[..., : self.num_sinks, :] = mx.broadcast_to(
                sv, (batch, n_kv_heads, self.num_sinks, v_head_dim)
            )

    def init_cache(
        self, *, batch: int, n_kv_heads: int, k_head_dim: int,
        v_head_dim: int, dtype,
    ) -> None:
        """Eagerly allocate the rolling-window key/value buffers as zeros.

        This is the same allocation ``update_and_fetch`` does lazily on
        its first call — exposed as a public entry point so a streaming
        state can be prepared *without* running a warmup step (which
        would leave generation content baked into the buffers). Sinks,
        if any, must be primed via :meth:`prime_sinks` first; ``_allocate``
        plants them into the reserved front slots. The result is a
        fully-allocated, content-neutral cache (zeros + sink slots,
        ``offset`` at 0).
        """
        self._allocate(batch, n_kv_heads, k_head_dim, v_head_dim, dtype)
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if not self._sinks_primed:
            raise RuntimeError(
                "LocalKVCache.prime_sinks() must be called before update_and_fetch()"
            )
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        if self.keys is None:
            self._allocate(B, n_kv_heads, k_head_dim, v_head_dim, keys.dtype)

        if isinstance(self.offset, mx.array):
            # Traced path (used during ``mx.exporter`` export): slot
            # indices must be ``mx.array`` so they survive tracing. Use
            # ``mx.put_along_axis`` per token. ``put_along_axis``
            # broadcasts a scalar-shape index against the value tensor,
            # so we skip the explicit broadcast_to and just reshape the
            # slot scalar to rank-4.
            offset = self.offset
            max_past = mx.array(self.max_past, dtype=mx.int32)
            num_sinks = mx.array(self.num_sinks, dtype=mx.int32)
            for i in range(S):
                slot = (num_sinks + (offset + i) % max_past).reshape((1, 1, 1, 1))
                new_k = keys[..., i:i + 1, :].astype(self.keys.dtype)
                new_v = values[..., i:i + 1, :].astype(self.values.dtype)
                self.keys = mx.put_along_axis(self.keys, slot, new_k, axis=2)
                self.values = mx.put_along_axis(self.values, slot, new_v, axis=2)
            self.offset = self.offset + S
        else:
            # Eager path: Python int offset → fast in-place index
            # assignment plus a tight slice return so attention only
            # processes the currently-filled window.
            off = int(self.offset)
            for i in range(S):
                slot = self.num_sinks + (off + i) % self.max_past
                self.keys[..., slot:slot + 1, :] = keys[..., i:i + 1, :]
                self.values[..., slot:slot + 1, :] = values[..., i:i + 1, :]
            self.offset = off + S
            end = self.num_sinks + min(self.offset, self.max_past)
            return self.keys[..., :end, :], self.values[..., :end, :]
        # Traced path: return the full fixed-shape buffer. Each
        # put_along_axis already produced a fresh array (no in-place
        # mutation under tracing), so no defensive ``[...]`` slice is
        # needed.
        return self.keys, self.values

    def make_mask(self, N: int, *, return_array: bool = False,
                  window_size: Optional[int] = None):
        """Boolean mask. Shape depends on the cache's mode:

        * **Eager** (Python-int offset): ``[N, num_sinks +
          min(offset, max_past)]`` — tight to the currently-filled
          window so attention only sees populated slots.
        * **Traced** (``mx.array`` offset, post export-promotion):
          ``[N, num_sinks + max_past]`` (fixed shape). Slots beyond
          ``offset`` are forced invisible by setting their effective
          logical position to int32-max.
        """
        del return_array, window_size

        if isinstance(self.offset, mx.array):
            return self._make_mask_traced(N)
        return self._make_mask_eager(N)

    def _make_mask_eager(self, N: int) -> mx.array:
        off = int(self.offset)
        wsz = min(off, self.max_past)
        cols = self.num_sinks + wsz
        if N == 0:
            return mx.ones((0, cols), dtype=mx.bool_)
        first_q = off - N
        # For each physical slot j in [0, wsz), the logical token index
        # it holds is the smallest ``t in [off-wsz, off)`` with
        # ``t % max_past == j``.
        logical_for_slot: list[int] = []
        base = off - wsz
        for s in range(wsz):
            t = base + ((s - base) % self.max_past)
            if t >= off:
                t -= self.max_past
            logical_for_slot.append(t)
        log = mx.array(logical_for_slot, dtype=mx.int32)         # [wsz]
        q_idx = mx.arange(first_q, first_q + N, dtype=mx.int32)[:, None]
        causal_past = q_idx >= log[None, :]                       # [N, wsz]
        sinks = mx.ones((N, self.num_sinks), dtype=mx.bool_)
        return mx.concatenate([sinks, causal_past], axis=-1)

    def _make_mask_traced(self, N: int) -> mx.array:
        max_past = self.max_past
        cols = self.num_sinks + max_past
        if N == 0:
            return mx.ones((0, cols), dtype=mx.bool_)
        offset = self.offset.astype(mx.int32)
        max_past_arr = mx.array(max_past, dtype=mx.int32)
        wsz = mx.minimum(offset, max_past_arr)
        first_q = offset - N

        slots = mx.arange(max_past, dtype=mx.int32)              # [max_past]
        base = offset - wsz
        logical = base + (slots - base) % max_past_arr
        logical = mx.where(logical >= offset, logical - max_past_arr, logical)
        # Mark un-filled slots (s >= wsz) as int32-max so they're never visible.
        invalid = mx.array(2 ** 30, dtype=mx.int32)
        logical_safe = mx.where(slots < wsz, logical, invalid)

        q_idx = mx.arange(N, dtype=mx.int32)[:, None] + first_q  # [N, 1]
        causal_past = q_idx >= logical_safe[None, :]              # [N, max_past]
        sinks = mx.ones((N, self.num_sinks), dtype=mx.bool_)
        return mx.concatenate([sinks, causal_past], axis=-1)

    @property
    def state(self):
        return (self.keys, self.values, self.offset)

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class OverlapAddCache(_BaseCache):
    """Buffer state for streaming overlap-add (used by streaming
    :class:`InverseSTFT`).

    Stores the running sum of the most recent ``frame_length - frame_step``
    samples that haven't been emitted yet. Each call appends the next
    output frame's overlap region.
    """

    def __init__(self):
        self.buffer: mx.array | None = None  # [B, *lead, overlap] (time-last)

    def reset(self, shape: tuple, dtype) -> None:
        self.buffer = mx.zeros(tuple(shape), dtype=dtype)

    @property
    def state(self):
        return (self.buffer,)

    @state.setter
    def state(self, v):
        (self.buffer,) = v

    def empty(self) -> bool:
        return self.buffer is None

    @property
    def nbytes(self) -> int:
        return 0 if self.buffer is None else self.buffer.nbytes
