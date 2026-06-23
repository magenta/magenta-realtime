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

"""Attention layers for Magenta-RT (flax.nnx).

Cache state lives on the attention module itself as a child
:class:`magenta_rt.nnx.cache.LocalKVCache` and is dispatched by a
``streaming`` boolean flag (set via
``model.set_attributes(streaming=True)``). Pre-allocate via
:meth:`init_cache(batch, dtype)`.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from .cache import LocalKVCache


# Matches ``flax.linen.linear.default_embed_init`` — used for both
# regular embeddings and the optional sink K/V embeddings (sl applies
# the same init).
_default_embed_init = nnx.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0,
)


def _query_scale(per_dim_scale: jnp.ndarray, units_per_head: int, dtype) -> jnp.ndarray:
    r_softplus_0 = 1.442695041
    base = r_softplus_0 * (1.0 / math.sqrt(units_per_head))
    softplus = jnp.log1p(jnp.exp(per_dim_scale.astype(dtype)))
    return base * softplus


def _attend_with_sinks(
    q, k, v, mask, per_dim_scale, sink_keys, sink_values,
    dropout_rate: float = 0.0,
    rng: Optional[jax.Array] = None,
):
    """Shared attention body: per-dim Q scale, optional learned sinks
    prepended to K/V, then SDPA (or manual einsums if dropout is active).
    """
    compute_dtype = jnp.promote_types(jnp.promote_types(q.dtype, k.dtype), v.dtype)
    q = q.astype(compute_dtype)
    k = k.astype(compute_dtype)
    v = v.astype(compute_dtype)

    B, _T_q, N, H = q.shape
    scale_vec = _query_scale(per_dim_scale, H, compute_dtype)
    q = q * scale_vec

    if sink_keys is not None:
        num_sinks = sink_keys.shape[0]
        sink_k = (sink_keys.astype(q.dtype) / scale_vec)[None]   # [1, S, N, H]
        sink_v = sink_values.astype(v.dtype)[None]               # [1, S, N, H]
        sink_k_b = jnp.broadcast_to(sink_k, (B, num_sinks, N, H))
        sink_v_b = jnp.broadcast_to(sink_v, (B, num_sinks, N, H))
        k = jnp.concatenate([sink_k_b, k], axis=1)
        v = jnp.concatenate([sink_v_b, v], axis=1)
        if mask is not None and mask.dtype == jnp.bool_:
            sink_mask = jnp.ones(
                (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
                dtype=jnp.bool_,
            )
            mask = jnp.concatenate([sink_mask, mask], axis=-1)

    if dropout_rate > 0.0 and rng is not None:
        # Manual attention with dropout because JAX dot_product_attention
        # doesn't support dropout in this JAX version.
        logits = jnp.einsum("b q n h, b k n h -> b n q k", q, k)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        # Compute softmax in fp32 for numerical stability
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(compute_dtype)
        
        keep_prob = 1.0 - dropout_rate
        mask_drop = jax.random.bernoulli(rng, p=keep_prob, shape=probs.shape)
        probs = jnp.where(mask_drop, probs / keep_prob, 0.0)
        
        return jnp.einsum("b n q k, b k n h -> b q n h", probs, v)
    else:
        return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=1.0)


class LocalSelfAttention(nnx.Module):
    """Multi-headed local self-attention with sliding window + optional sinks.

    Streaming dispatch: when ``self.streaming=True`` (set via
    ``model.set_attributes(streaming=True)``), ``__call__`` routes
    through the on-module :class:`LocalKVCache`. Call
    :meth:`init_cache(batch, dtype)` first.
    """

    def __init__(
        self,
        *,
        in_features: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        max_future_horizon: int = 0,
        num_kv_heads: Optional[int] = None,
        use_bias: bool = False,
        per_dim_scale: bool = True,
        attention_logits_soft_cap: Optional[float] = None,
        num_sink_embeddings: int = 0,
        use_sink_scalars: bool = False,
        use_kv_cache_ringbuffer: bool = False,
        use_rope: bool = False,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        model_dimension: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        if num_kv_heads not in (None, num_heads):
            raise NotImplementedError("nnx: GQA (num_kv_heads != num_heads) not supported")
        if max_future_horizon != 0:
            raise NotImplementedError("nnx: max_future_horizon must be 0 (causal)")
        if use_bias:
            raise NotImplementedError("nnx: attention bias not supported")
        if not per_dim_scale:
            raise NotImplementedError("nnx: per_dim_scale=True only")
        if attention_logits_soft_cap is not None:
            raise NotImplementedError("nnx: attention_logits_soft_cap=None only")
        if use_sink_scalars:
            raise NotImplementedError("nnx: use_sink_scalars=False only")
        if use_kv_cache_ringbuffer:
            raise NotImplementedError("nnx: ringbuffer KV cache not supported")
        if use_rope:
            raise NotImplementedError("nnx: RoPE not supported (NoPE only)")
        if max_past_horizon < 0:
            raise NotImplementedError("nnx: max_past_horizon must be >= 0")

        self.in_features = in_features
        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.max_past_horizon = max_past_horizon
        self.num_sink_embeddings = num_sink_embeddings
        self.param_dtype = param_dtype
        self.model_dimension = model_dimension if model_dimension is not None else in_features

        q_dim = num_heads * units_per_head
        kv_dim = num_heads * units_per_head

        self.q_proj = nnx.Linear(
            in_features, q_dim, use_bias=False,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.kv_proj = nnx.Linear(
            in_features, 2 * kv_dim, use_bias=False,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.per_dim_scale_param = nnx.Param(
            jnp.zeros((units_per_head,), dtype=param_dtype)
        )

        if num_sink_embeddings > 0:
            sink_shape = (num_sink_embeddings, num_heads, units_per_head)
            self.sink_key_embeddings = nnx.Param(
                _default_embed_init(rngs.params(), sink_shape, param_dtype)
            )
            self.sink_value_embeddings = nnx.Param(
                _default_embed_init(rngs.params(), sink_shape, param_dtype)
            )
        else:
            self.sink_key_embeddings = None
            self.sink_value_embeddings = None

        self.output_projection = nnx.Einsum(
            einsum_str="...nh,dnh->...d",
            kernel_shape=(self.model_dimension, num_heads, units_per_head),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.attention_dropout_prob = attention_dropout_prob
        self.deterministic = False
        if rngs is not None and attention_dropout_prob > 0:
            self.dropout_rng = rngs.dropout.fork()
        else:
            self.dropout_rng = None

        # Streaming state.
        self.streaming: bool = False
        self.cache = LocalKVCache(
            window_size=max_past_horizon + 1,
            num_sinks=num_sink_embeddings,
        )

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        self.cache.init_cache(
            batch=batch,
            n_kv_heads=self.num_heads,
            k_head_dim=self.units_per_head,
            v_head_dim=self.units_per_head,
            dtype=dtype,
        )
        if self.num_sink_embeddings > 0:
            self.cache.prime_sinks(
                self.sink_key_embeddings[...], self.sink_value_embeddings[...],
            )

    def _project_qkv(self, x: jnp.ndarray):
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        kv_dim = self.num_heads * self.units_per_head
        k = kv[..., :kv_dim]
        val = kv[..., kv_dim:]
        split = "b t (h d) -> b t h d"
        q = rearrange(q, split, h=self.num_heads)
        k = rearrange(k, split, h=self.num_heads)
        val = rearrange(val, split, h=self.num_heads)
        return q, k, val

    def _attend(self, queries, keys, values, mask, dropout_rate=0.0, rng=None):
        return _attend_with_sinks(
            q=queries, k=keys, v=values, mask=mask,
            per_dim_scale=self.per_dim_scale_param[...],
            sink_keys=(
                self.sink_key_embeddings[...]
                if self.sink_key_embeddings is not None else None
            ),
            sink_values=(
                self.sink_value_embeddings[...]
                if self.sink_value_embeddings is not None else None
            ),
            dropout_rate=dropout_rate,
            rng=rng,
        )

    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        dropout_rate = self.attention_dropout_prob
        is_training = not self.deterministic
        rng = self.dropout_rng() if (is_training and self.dropout_rng is not None) else None

        if not self.streaming:
            valid_mask: Optional[jnp.ndarray] = None
            if T > 1:
                row = jnp.arange(T)[:, None]
                col = jnp.arange(T)[None, :]
                banded = (col <= row) & (col >= row - self.max_past_horizon)
                valid_mask = banded.reshape(1, 1, T, T)
            return self.output_projection(self._attend(q, k, v, valid_mask, dropout_rate=dropout_rate, rng=rng))

        # Streaming step.
        if not self.cache.initialized:
            raise RuntimeError(
                "init_cache() must be called before streaming forward "
                "(or use magenta_rt.nnx.enable_streaming(model))"
            )

        k_t = rearrange(k, "b t h d -> b h t d")
        v_t = rearrange(v, "b t h d -> b h t d")
        full_k_t, full_v_t = self.cache.update_and_fetch(k_t, v_t)
        full_k = rearrange(full_k_t, "b h t d -> b t h d")
        full_v = rearrange(full_v_t, "b h t d -> b t h d")

        if mask is None:
            mask = self.cache.make_mask(T)

        # The cache stores sinks at the front; if we also have sink
        # embeddings on this layer, drop the cache's sink rows so
        # _attend's own sink-prepend doesn't double-count.
        if self.cache.num_sinks > 0 and self.sink_key_embeddings is not None:
            full_k = full_k[:, self.cache.num_sinks:]
            full_v = full_v[:, self.cache.num_sinks:]
            mask = mask[:, self.cache.num_sinks:] if mask is not None else None

        if mask is not None and mask.ndim == 2:
            mask = mask[None, None, :, :]
            mask = jnp.broadcast_to(mask, (B, 1, mask.shape[2], mask.shape[3]))

        return self.output_projection(self._attend(q, full_k, full_v, mask, dropout_rate=dropout_rate, rng=rng))


class StreamingCrossAttention(nnx.Module):
    """Streaming cross-attention with sliding-window source K/V cache."""

    def __init__(
        self,
        *,
        in_features: int,
        source_features: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        num_sink_embeddings: int = 0,
        num_kv_heads: Optional[int] = None,
        use_bias: bool = False,
        per_dim_scale: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        model_dimension: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        if num_kv_heads not in (None, num_heads):
            raise NotImplementedError("nnx: GQA not supported")
        if use_bias:
            raise NotImplementedError("nnx: cross-attn bias not supported")
        if not per_dim_scale:
            raise NotImplementedError("nnx: per_dim_scale=True only")
        if max_past_horizon < 1:
            raise ValueError(f"max_past_horizon must be >= 1, got {max_past_horizon}")

        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.max_past_horizon = max_past_horizon
        self.num_sink_embeddings = num_sink_embeddings
        self.param_dtype = param_dtype
        self.model_dimension = model_dimension if model_dimension is not None else in_features

        qkv_dim = num_heads * units_per_head
        self.q_proj = nnx.Linear(
            in_features, qkv_dim, use_bias=False,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.kv_proj = nnx.Linear(
            source_features, 2 * qkv_dim, use_bias=False,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.per_dim_scale_param = nnx.Param(
            jnp.zeros((units_per_head,), dtype=param_dtype)
        )

        if num_sink_embeddings > 0:
            sink_shape = (num_sink_embeddings, num_heads, units_per_head)
            self.sink_key_embeddings = nnx.Param(
                _default_embed_init(rngs.params(), sink_shape, param_dtype)
            )
            self.sink_value_embeddings = nnx.Param(
                _default_embed_init(rngs.params(), sink_shape, param_dtype)
            )
        else:
            self.sink_key_embeddings = None
            self.sink_value_embeddings = None

        self.output_projection = nnx.Einsum(
            einsum_str="...nh,dnh->...d",
            kernel_shape=(self.model_dimension, num_heads, units_per_head),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.attention_dropout_prob = attention_dropout_prob
        self.deterministic = False
        if rngs is not None and attention_dropout_prob > 0:
            self.dropout_rng = rngs.dropout.fork()
        else:
            self.dropout_rng = None

        self.streaming: bool = False
        self.cross_cache = LocalKVCache(
            window_size=max_past_horizon + 1,
            num_sinks=0,
        )

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        self.cross_cache.init_cache(
            batch=batch,
            n_kv_heads=self.num_heads,
            k_head_dim=self.units_per_head,
            v_head_dim=self.units_per_head,
            dtype=dtype,
        )

    def _project_q(self, x: jnp.ndarray) -> jnp.ndarray:
        q = self.q_proj(x)
        return rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)

    def _project_kv(self, source: jnp.ndarray):
        kv = self.kv_proj(source)
        kv_dim = self.num_heads * self.units_per_head
        split = "b t (h d) -> b t h d"
        k = rearrange(kv[..., :kv_dim], split, h=self.num_heads)
        v = rearrange(kv[..., kv_dim:], split, h=self.num_heads)
        return k, v

    def __call__(self, x: jnp.ndarray, *, source: jnp.ndarray) -> jnp.ndarray:
        q = self._project_q(x)
        k_full, v_full = self._project_kv(source)

        if not self.streaming:
            B = x.shape[0]
            T_q = q.shape[1]
            T_kv = k_full.shape[1]
            if T_q != T_kv:
                raise ValueError(
                    f"non-streaming cross-attn requires T_q == T_kv, got "
                    f"{T_q} vs {T_kv}"
                )
            row = jnp.arange(T_q)[:, None]
            col = jnp.arange(T_kv)[None, :]
            banded = (col <= row) & (col >= row - self.max_past_horizon)
            mask = jnp.broadcast_to(
                banded.reshape(1, 1, T_q, T_kv), (B, 1, T_q, T_kv),
            )
            dropout_rate = self.attention_dropout_prob
            is_training = not self.deterministic
            rng = self.dropout_rng() if (is_training and self.dropout_rng is not None) else None
            return self._attend_and_project(q, k_full, v_full, mask, dropout_rate=dropout_rate, rng=rng)

        if not self.cross_cache.initialized:
            raise RuntimeError(
                "init_cache() must be called before streaming forward"
            )

        k_t = rearrange(k_full, "b t h d -> b h t d")
        v_t = rearrange(v_full, "b t h d -> b h t d")
        full_k_t, full_v_t = self.cross_cache.update_and_fetch(k_t, v_t)
        full_k = rearrange(full_k_t, "b h t d -> b t h d")
        full_v = rearrange(full_v_t, "b h t d -> b t h d")

        dropout_rate = self.attention_dropout_prob
        is_training = not self.deterministic
        rng = self.dropout_rng() if (is_training and self.dropout_rng is not None) else None

        # Mask out the cache slots that haven't been filled yet. The
        # rolling window is fixed-size (``max_past_horizon + 1``) but
        # early in a stream only the first few slots hold real source
        # K/V — the rest are zeros. An all-ones mask would let the
        # query attend to those zero slots and corrupt the output (the
        # self-attention path already guards against this via
        # ``cache.make_mask``). ``cross_cache`` has ``num_sinks=0``, so
        # ``make_mask`` returns ``[T_q, window_size]`` aligned with
        # ``full_k``; ``_attend_with_sinks`` extends it for the
        # separately-prepended sink K/V.
        T_q = q.shape[1]
        mask = self.cross_cache.make_mask(T_q)  # [T_q, window_size] bool
        mask = jnp.broadcast_to(
            mask[None, None, :, :], (q.shape[0], 1, T_q, mask.shape[1]),
        )
        return self._attend_and_project(q, full_k, full_v, mask, dropout_rate=dropout_rate, rng=rng)

    def _attend_and_project(self, q, k, v, mask, dropout_rate=0.0, rng=None) -> jnp.ndarray:
        ctx = _attend_with_sinks(
            q=q, k=k, v=v, mask=mask,
            per_dim_scale=self.per_dim_scale_param[...],
            sink_keys=(
                self.sink_key_embeddings[...]
                if self.sink_key_embeddings is not None else None
            ),
            sink_values=(
                self.sink_value_embeddings[...]
                if self.sink_value_embeddings is not None else None
            ),
            dropout_rate=dropout_rate,
            rng=rng,
        )
        return self.output_projection(ctx)
