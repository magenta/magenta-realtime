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

"""Attention layers for Magenta-RT (mlx-lm style).

``LocalSelfAttention`` and ``StreamingCrossAttention`` cover the
shipping subset of the sl.attention surface. Each constructor guards
unsupported feature combinations with ``NotImplementedError`` so a
future config can't silently fall back to a different code path.

The output projection is an :class:`mlx_pure.layers.EinsumDense` with
equation ``"...nh,dnh->...d"`` so the weight layout matches sl 1:1
for checkpoint loading via :mod:`mlx_pure.load_weights`.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import LocalKVCache
from .layers import EinsumDense


def _query_scale(per_dim_scale: mx.array, units_per_head: int, dtype) -> mx.array:
    """Compute the per-dim scale vector matching sl's `_query_scale_vector`.

    With ``per_dim_scale`` enabled (the locked case), scaling is
    ``r_softplus_0 * (1/sqrt(uph)) * softplus(per_dim_scale)`` where
    ``r_softplus_0 = 1.442695041``.
    """
    r_softplus_0 = 1.442695041
    base = r_softplus_0 * (1.0 / math.sqrt(units_per_head))
    softplus = mx.log1p(mx.exp(per_dim_scale.astype(dtype)))
    return base * softplus


def _banded_causal_mask(T_q: int, T_kv: int, max_past_horizon: int) -> mx.array:
    """Local-causal mask: row i attends to cols in [i - max_past, i]."""
    row = mx.arange(T_q)[:, None]
    col = mx.arange(T_kv)[None, :]
    return ((col <= row) & (col >= row - max_past_horizon)).reshape(1, 1, T_q, T_kv)


def _apply_proj(layer: nn.Module, x: mx.array, prefix: str, dtype) -> mx.array:
    """Apply ``layer.{prefix}_proj`` (full-precision) or the quantized
    counterpart at ``{prefix}_proj_{q,scales,biases}``, depending on
    whether ``layer._qkv_quantized`` is set.
    """
    v = x.astype(dtype)
    if getattr(layer, "_qkv_quantized", False):
        return mx.quantized_matmul(
            v,
            getattr(layer, f"{prefix}_proj_q"),
            scales=getattr(layer, f"{prefix}_proj_scales"),
            biases=getattr(layer, f"{prefix}_proj_biases"),
            transpose=True,
            group_size=layer._q_group_size,
            bits=layer._q_bits,
        )
    return mx.matmul(v, getattr(layer, f"{prefix}_proj").astype(dtype))


def _quantize_qkv_in_place(layer: nn.Module, group_size: int, bits: int) -> bool:
    """In-place int4 / int8 quantization of ``q_proj`` and ``kv_proj``.

    Returns True on success, False if the shape is incompatible with
    ``group_size`` (caller treats this as a no-op skip).

    ``mx.quantize`` groups along the last axis. Our kernels are stored
    ``[in, out]`` (sl-native); we transpose to ``[out, in]`` so the
    group axis is the input feature axis, matching Dense's convention.
    """
    if getattr(layer, "_qkv_quantized", False):
        return True
    if layer.q_proj.shape[0] % group_size != 0 or layer.kv_proj.shape[0] % group_size != 0:
        return False
    layer.q_proj_q, layer.q_proj_scales, layer.q_proj_biases = mx.quantize(
        mx.transpose(layer.q_proj, (1, 0)), group_size=group_size, bits=bits,
    )
    layer.kv_proj_q, layer.kv_proj_scales, layer.kv_proj_biases = mx.quantize(
        mx.transpose(layer.kv_proj, (1, 0)), group_size=group_size, bits=bits,
    )
    layer._q_group_size = group_size
    layer._q_bits = bits
    layer._qkv_quantized = True
    # Drop the full-precision kernels so they don't ship in the exported state.
    layer.q_proj = mx.zeros((0,), dtype=layer.param_dtype)
    layer.kv_proj = mx.zeros((0,), dtype=layer.param_dtype)
    return True


def _prepend_sinks(
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array],
    sink_k: mx.array,
    sink_v: mx.array,
    num_heads: int,
    units_per_head: int,
) -> tuple[mx.array, mx.array, Optional[mx.array]]:
    """Concatenate sink K/V rows to the front of ``k`` / ``v``.

    Args:
      k, v: ``[B, n_heads, T_kv, h]``.
      mask: ``[B, 1, T_q, T_kv]`` boolean or None; sinks are always-on
        so we extend it with a True column.
      sink_k, sink_v: ``[num_sinks, n_heads, h]``. ``sink_k`` must
        already be pre-divided by the per-dim query scale so that
        SDPA's scaled queries reproduce JAX's unscaled-sink logits.
    """
    B = k.shape[0]
    num_sinks = sink_k.shape[0]
    sink_k_b = mx.broadcast_to(
        mx.transpose(sink_k, (1, 0, 2))[None],
        (B, num_heads, num_sinks, units_per_head),
    )
    sink_v_b = mx.broadcast_to(
        mx.transpose(sink_v, (1, 0, 2))[None],
        (B, num_heads, num_sinks, units_per_head),
    )
    k = mx.concatenate([sink_k_b, k], axis=2)
    v = mx.concatenate([sink_v_b, v], axis=2)
    if mask is not None and mask.dtype == mx.bool_:
        sink_mask = mx.ones(
            (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
            dtype=mx.bool_,
        )
        mask = mx.concatenate([sink_mask, mask], axis=-1)
    return k, v, mask


def _scaled_sink_kv(layer: nn.Module, scale_vec: mx.array, dtype) -> tuple[mx.array, mx.array]:
    """Cast sink K/V to ``dtype`` and pre-divide sink K by ``scale_vec``."""
    sink_k = layer.sink_key_embeddings.astype(dtype) / scale_vec
    sink_v = layer.sink_value_embeddings.astype(dtype)
    return sink_k, sink_v


class LocalSelfAttention(nn.Module):
    """Multi-headed local self-attention.

    Constructor mirrors the locked subset; any flag outside that subset
    raises at construction time.
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
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        model_dimension: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()
        if num_kv_heads not in (None, num_heads):
            raise NotImplementedError("mlx_pure: GQA (num_kv_heads != num_heads) not supported")
        if max_future_horizon != 0:
            raise NotImplementedError("mlx_pure: max_future_horizon must be 0 (causal)")
        if use_bias:
            raise NotImplementedError("mlx_pure: attention bias not supported")
        if not per_dim_scale:
            raise NotImplementedError("mlx_pure: per_dim_scale=True only")
        if attention_logits_soft_cap is not None:
            raise NotImplementedError("mlx_pure: attention_logits_soft_cap=None only")
        if use_sink_scalars:
            raise NotImplementedError("mlx_pure: use_sink_scalars=False only")
        if use_kv_cache_ringbuffer:
            raise NotImplementedError("mlx_pure: ringbuffer KV cache not supported")
        if use_rope:
            raise NotImplementedError("mlx_pure: RoPE not supported (NoPE only)")
        if max_past_horizon < 0:
            raise NotImplementedError("mlx_pure: max_past_horizon must be >= 0")

        self.in_features = in_features
        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.max_past_horizon = max_past_horizon
        self.num_sink_embeddings = num_sink_embeddings
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self.model_dimension = model_dimension if model_dimension is not None else in_features

        qkv_dim = num_heads * units_per_head  # num_kv_heads == num_heads (locked)

        # Projections stored as [in, out] (sl-native layout). K and V kernels are
        # concatenated along the last axis; we split at runtime.
        self.q_proj = mx.zeros((in_features, qkv_dim), dtype=param_dtype)
        self.kv_proj = mx.zeros((in_features, 2 * qkv_dim), dtype=param_dtype)
        self.attention_dropout_prob = attention_dropout_prob
        self.attn_dropout = nn.Dropout(attention_dropout_prob)
        self.per_dim_scale = mx.zeros((units_per_head,), dtype=param_dtype)

        if num_sink_embeddings > 0:
            self.sink_key_embeddings = mx.zeros(
                (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
            )
            self.sink_value_embeddings = mx.zeros(
                (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
            )
        else:
            self.sink_key_embeddings = None
            self.sink_value_embeddings = None

        self.output_projection = EinsumDense(
            equation="...nh,dnh->...d",
            output_shape=(self.model_dimension,),
            bias_axes="",
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
        )

    def _project_qkv(self, x: mx.array):
        """Project ``x`` ([B, T, in_features]) to Q, K, V tensors
        shaped [B, num_heads, T, units_per_head]."""
        B, T, _ = x.shape
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        q = _apply_proj(self, x, "q", dtype)
        kv = _apply_proj(self, x, "kv", dtype)
        kv_dim = self.num_heads * self.units_per_head
        k, val = mx.split(kv, [kv_dim], axis=-1)
        shape = (B, T, self.num_heads, self.units_per_head)
        q_t = mx.transpose(q.reshape(shape), (0, 2, 1, 3))
        k_t = mx.transpose(k.reshape(shape), (0, 2, 1, 3))
        v_t = mx.transpose(val.reshape(shape), (0, 2, 1, 3))
        return q_t, k_t, v_t

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine") -> "LocalSelfAttention":
        """In-place int4 / int8 quantization of ``q_proj`` and
        ``kv_proj``. The ``output_projection`` EinsumDense child is
        handled separately by ``nn.quantize``'s leaf walk.

        Note: ``mlx.nn.quantize`` only visits leaf modules, so it does
        NOT reach this method on its own (``LocalSelfAttention`` has
        children). ``magenta_rt.mlx_pure.quantize.quantize_in_place``
        invokes us in a pre-pass.
        """
        _quantize_qkv_in_place(self, group_size, bits)
        return self

    def _attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array],
        *,
        sinks_in_kv: bool = False,
    ) -> mx.array:
        """Match sl's `_compute_attention` SDPA path.

        Args:
          queries: [B, num_heads, T_q, units_per_head]
          keys:    [B, num_heads, T_kv, units_per_head]
          values:  [B, num_heads, T_kv, units_per_head]
          mask:    [B, 1, T_q, T_kv] boolean (True = attend) or None.
          sinks_in_kv: when True, ``keys`` / ``values`` / ``mask``
            already include the layer's sink rows (e.g., from a primed
            ``LocalKVCache``); skip the sink prepend.
        """
        q = queries
        k = keys
        v = values

        scale_vec = _query_scale(self.per_dim_scale, self.units_per_head, q.dtype)
        q = q * scale_vec

        if self.sink_key_embeddings is not None and not sinks_in_kv:
            # JAX computes sink logits with *unscaled* queries; pre-divide
            # the sink keys by the same scale so that scaled_q @ (sink_k/scale)
            # equals unscaled_q @ sink_k.
            sink_k, sink_v = _scaled_sink_kv(self, scale_vec, q.dtype)
            k, v, mask = _prepend_sinks(
                k, v, mask, sink_k, sink_v, self.num_heads, self.units_per_head,
            )

        if self.attention_dropout_prob > 0.0 and self.training:
            # Manual attention with dropout because MLX fast.scaled_dot_product_attention
            # doesn't support dropout.
            logits = q @ mx.transpose(k, (0, 1, 3, 2))
            if mask is not None:
                logits = mx.where(mask, logits, -1e9)
            # Compute softmax in fp32 for numerical stability
            probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(q.dtype)
            probs = self.attn_dropout(probs)
            ctx = probs @ v
        else:
            ctx = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        return mx.transpose(ctx, (0, 2, 1, 3))  # [B, T, n_heads, h]

    def _prime_cache_sinks(self, cache: LocalKVCache, dtype) -> None:
        """Prime ``cache``'s reserved sink slots with this layer's sink
        embeddings, pre-divided by the per-dim query scale (so SDPA's
        scaled queries dot the un-scaled JAX-side sink logits — same
        trick ``_attend`` uses for the non-cache path). No-op when the
        cache has no sink slots or they are already primed.
        """
        if cache.num_sinks == 0 or cache._sinks_primed:
            return
        if self.sink_key_embeddings is None:
            raise RuntimeError(
                "cache reserves sink slots but layer has no sink embeddings"
            )
        scale_vec = _query_scale(self.per_dim_scale, self.units_per_head, dtype)
        sink_k_pre, sink_v_cast = _scaled_sink_kv(self, scale_vec, dtype)
        cache.prime_sinks(sink_k_pre, sink_v_cast)

    def init_cache(self, cache: LocalKVCache, *, batch: int, dtype) -> None:
        """Eagerly prepare ``cache`` for streaming: prime its sink slots
        and zero-allocate its rolling-window buffers — the setup the
        lazy path otherwise does on the first streaming ``__call__``.
        Lets a neutral streaming state be built without a warmup step.
        """
        self._prime_cache_sinks(cache, dtype)
        cache.init_cache(
            batch=batch, n_kv_heads=self.num_heads,
            k_head_dim=self.units_per_head, v_head_dim=self.units_per_head,
            dtype=dtype,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[LocalKVCache] = None,
    ) -> mx.array:
        """Forward pass.

        - ``cache=None``: full-sequence forward. Builds the local-causal
          mask internally (banded with width ``max_past_horizon+1``).
          Sink embeddings (if configured) are prepended inside ``_attend``.
        - ``cache`` provided: streaming step. The cache holds reserved
          sink slots at the front and a sliding window of past tokens.
          Mask is supplied by ``cache.make_mask`` and threaded through.
        """
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)

        if cache is None:
            valid_mask = _banded_causal_mask(T, T, self.max_past_horizon) if T > 1 else None
            return self.output_projection(self._attend(q, k, v, valid_mask))

        # Streaming step. Sinks live directly in the cache: prime once
        # with the layer's sink K already pre-divided by the per-dim
        # scale, then we never have to drop+re-prepend sinks per step.
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        self._prime_cache_sinks(cache, dtype)

        full_k, full_v = cache.update_and_fetch(k, v)

        # Streaming mask comes from the cache (sinks-aware, [T, num_sinks + window]).
        if mask is None:
            mask = cache.make_mask(T)
        if mask is not None and mask.ndim == 2:
            mask = mask[None, None, :, :]
            mask = mx.broadcast_to(mask, (B, 1, mask.shape[2], mask.shape[3]))

        # Sinks already baked into full_k/full_v/mask by the cache; tell
        # _attend not to prepend them again.
        sinks_in_kv = cache.num_sinks > 0 and self.sink_key_embeddings is not None
        return self.output_projection(
            self._attend(q, full_k, full_v, mask, sinks_in_kv=sinks_in_kv)
        )


class StreamingCrossAttention(nn.Module):
    """Streaming cross-attention.

    The source comes in step-by-step (synchronous with the decoder) and
    its K/V are appended to a rolling local-KV buffer. Each decoder
    query attends to up to ``max_past_horizon`` past source frames plus
    the current source frame.

    Locked subset (mirrors ``streaming_cross_attention_*`` settings in
    ``model.py``): ``QueryAndKeyValueProjection`` (Q from decoder,
    combined KV from source), no bias, no RoPE, no soft cap, fully
    causal (max_future_horizon=0), no sink embeddings, no ringbuffer,
    no query delay buffer.
    """

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
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        model_dimension: Optional[int] = None,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()
        if num_kv_heads not in (None, num_heads):
            raise NotImplementedError("mlx_pure: GQA not supported")
        if use_bias:
            raise NotImplementedError("mlx_pure: cross-attn bias not supported")
        if not per_dim_scale:
            raise NotImplementedError("mlx_pure: per_dim_scale=True only")
        if max_past_horizon < 1:
            raise ValueError(f"max_past_horizon must be >= 1, got {max_past_horizon}")

        self.num_heads = num_heads
        self.units_per_head = units_per_head
        self.max_past_horizon = max_past_horizon
        self.num_sink_embeddings = num_sink_embeddings
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self.model_dimension = model_dimension if model_dimension is not None else in_features

        qkv_dim = num_heads * units_per_head

        # Q from decoder input; combined K+V from source.
        self.q_proj = mx.zeros((in_features, qkv_dim), dtype=param_dtype)
        self.kv_proj = mx.zeros((source_features, 2 * qkv_dim), dtype=param_dtype)
        self.per_dim_scale = mx.zeros((units_per_head,), dtype=param_dtype)

        if num_sink_embeddings > 0:
            self.sink_key_embeddings = mx.zeros(
                (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
            )
            self.sink_value_embeddings = mx.zeros(
                (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
            )
        else:
            self.sink_key_embeddings = None
            self.sink_value_embeddings = None

        self.output_projection = EinsumDense(
            equation="...nh,dnh->...d",
            output_shape=(self.model_dimension,),
            bias_axes="",
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        self.attention_dropout_prob = attention_dropout_prob
        self.attn_dropout = nn.Dropout(attention_dropout_prob)

    def _project_q(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        q = _apply_proj(self, x, "q", dtype)
        return mx.transpose(q.reshape(B, T, self.num_heads, self.units_per_head), (0, 2, 1, 3))

    def _project_kv(self, source: mx.array) -> tuple[mx.array, mx.array]:
        B, T, _ = source.shape
        dtype = self.compute_dtype if self.compute_dtype is not None else source.dtype
        kv = _apply_proj(self, source, "kv", dtype)
        kv_dim = self.num_heads * self.units_per_head
        k, v = mx.split(kv, [kv_dim], axis=-1)
        shape = (B, T, self.num_heads, self.units_per_head)
        k_t = mx.transpose(k.reshape(shape), (0, 2, 1, 3))
        v_t = mx.transpose(v.reshape(shape), (0, 2, 1, 3))
        return k_t, v_t

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine") -> "StreamingCrossAttention":
        """In-place int4 / int8 quantization of ``q_proj`` and
        ``kv_proj``. See ``LocalSelfAttention.to_quantized`` for the
        ``nn.quantize``-only-walks-leaves caveat.
        """
        _quantize_qkv_in_place(self, group_size, bits)
        return self

    def init_cache(self, cache: LocalKVCache, *, batch: int, dtype) -> None:
        """Eagerly zero-allocate ``cache``'s rolling source-K/V window —
        the setup the lazy path otherwise does on the first streaming
        ``__call__``. The cross cache reserves no sink slots (sinks are
        prepended fresh per step inside ``_attend_and_project``), so
        there is nothing to prime.
        """
        cache.init_cache(
            batch=batch, n_kv_heads=self.num_heads,
            k_head_dim=self.units_per_head, v_head_dim=self.units_per_head,
            dtype=dtype,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        source: mx.array,
        cache: Optional[LocalKVCache] = None,
    ) -> mx.array:
        """Streaming cross-attention forward.

        - ``cache=None``: full-sequence mode. ``source`` must have the
          same time dimension as ``x`` (synchronous streaming) and the
          banded local-causal mask is built internally.
        - ``cache`` provided: streaming step. ``x`` and ``source`` must
          have matching time dim ``S`` (typically 1). The source
          K/V projections are appended to the cache and attended to.
        """
        q = self._project_q(x)  # [B, n_heads, T_q, h]
        k_full, v_full = self._project_kv(source)  # [B, n_heads, T_kv, h]

        if cache is None:
            B, T_q = x.shape[0], q.shape[2]
            T_kv = k_full.shape[2]
            if T_q != T_kv:
                raise ValueError(
                    f"non-streaming cross-attn requires T_q == T_kv, got {T_q} vs {T_kv}"
                )
            mask = mx.broadcast_to(
                _banded_causal_mask(T_q, T_kv, self.max_past_horizon),
                (B, 1, T_q, T_kv),
            )
            return self._attend_and_project(q, k_full, v_full, mask)

        # Streaming step: append source K/V to the cache, then attend.
        full_k_t, full_v_t = cache.update_and_fetch(k_full, v_full)
        # Cache returns the full fixed-shape [num_sinks + max_past] slice;
        # mask hides unfilled slots so they don't contribute to softmax.
        B, T_q = q.shape[0], q.shape[2]
        cache_mask = cache.make_mask(T_q)  # [T_q, num_sinks + max_past]
        mask = mx.broadcast_to(
            cache_mask[None, None, :, :],
            (B, 1, T_q, cache_mask.shape[1]),
        )
        return self._attend_and_project(q, full_k_t, full_v_t, mask)

    def _attend_and_project(self, q, k, v, mask) -> mx.array:
        scale_vec = _query_scale(self.per_dim_scale, self.units_per_head, q.dtype)
        q = q * scale_vec

        if self.sink_key_embeddings is not None:
            # Sinks always attended to. Pre-divide sink keys by query scale so
            # that scaled_q @ (sink_k / scale) == unscaled_q @ sink_k (matches sl).
            sink_k, sink_v = _scaled_sink_kv(self, scale_vec, q.dtype)
            k, v, mask = _prepend_sinks(
                k, v, mask, sink_k, sink_v, self.num_heads, self.units_per_head,
            )

        if self.attention_dropout_prob > 0.0 and self.training:
            # Manual attention with dropout because MLX fast.scaled_dot_product_attention
            # doesn't support dropout.
            logits = q @ mx.transpose(k, (0, 1, 3, 2))
            if mask is not None:
                logits = mx.where(mask, logits, -1e9)
            # Compute softmax in fp32 for numerical stability
            probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(q.dtype)
            probs = self.attn_dropout(probs)
            ctx = probs @ v
        else:
            ctx = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        ctx = mx.transpose(ctx, (0, 2, 1, 3))  # [B, T, n_heads, h]
        return self.output_projection(ctx)
