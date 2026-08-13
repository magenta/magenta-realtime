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

"""Transformer block, stack, multi-channel embedding, encoder.

Mirrors the structure of ``magenta_rt/mlx/transformer.py`` for the
shipping configs only. Each sub-block is a primer-hybrid residual
wrapper::

    y = x + post_norm( body( pre_norm(x) ) )

where ``body`` is one of: self-attention + output-projection,
cross-attention + output-projection, or FFN(linear + gelu + linear).
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from typing import Sequence as _Seq

from .attention import LocalSelfAttention, StreamingCrossAttention
from .cache import LocalKVCache
from .layers import Dense, LayerNorm, RMSNorm


LEVEL_AXIS = -2
"""Axis indexing the RVQ-level / channel dimension of multi-channel tokens."""


class FFN(nn.Module):
    """Two-Dense FFN with primer_hybrid norm wrapping (pre + post RMS).

    Locked features: ``ffn_gated=False``, ``ffn_use_bias=True``,
    ``ffn_activation=nn.gelu_approx``.
    """

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        activation: Callable[[mx.array], mx.array] = nn.gelu_approx,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(model_dim, eps=eps)
        self.post_norm = RMSNorm(model_dim, eps=eps)
        self.ffn_layer1 = Dense(
            model_dim, hidden_dim, bias=True, activation=activation,
            compute_dtype=compute_dtype, param_dtype=param_dtype,
        )
        self.act_dropout = nn.Dropout(dropout_prob)
        self.ffn_layer2 = Dense(
            hidden_dim, model_dim, bias=True,
            compute_dtype=compute_dtype, param_dtype=param_dtype,
        )
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.pre_norm(x)
        h = self.ffn_layer1(h)
        h = self.act_dropout(h)
        h = self.ffn_layer2(h)
        h = self.post_norm(h)
        return x + self.dropout(h)


class SelfAttentionBlock(nn.Module):
    """Self-attention + output-projection wrapped in primer-hybrid norms.

    Mirrors ``SLSelfAttention.make()`` for the locked feature set.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(model_dim, eps=eps)
        self.post_norm = RMSNorm(model_dim, eps=eps)
        self.attention = LocalSelfAttention(
            in_features=model_dim,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            per_dim_scale=True,
            num_sink_embeddings=num_sinks,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            model_dimension=model_dim,
            attention_dropout_prob=attention_dropout_prob,
        )
        # Residual dropout on the self-attention sublayer output.
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[LocalKVCache] = None,
    ) -> mx.array:
        h = self.pre_norm(x)
        h = self.attention(h, mask=mask, cache=cache)
        h = self.post_norm(h)
        return x + self.dropout(h)


class CrossAttentionBlock(nn.Module):
    """Streaming cross-attention block (primer-hybrid norm wrapping)."""

    def __init__(
        self,
        *,
        model_dim: int,
        source_features: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(model_dim, eps=eps)
        self.post_norm = RMSNorm(model_dim, eps=eps)
        self.attention = StreamingCrossAttention(
            in_features=model_dim,
            source_features=source_features,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            num_sink_embeddings=num_sinks,
            per_dim_scale=True,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            model_dimension=model_dim,
            attention_dropout_prob=attention_dropout_prob,
        )
        # Residual dropout on the cross-attention sublayer output.
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(
        self,
        x: mx.array,
        *,
        source: mx.array,
        cache: Optional[LocalKVCache] = None,
    ) -> mx.array:
        h = self.pre_norm(x)
        h = self.attention(h, source=source, cache=cache)
        h = self.post_norm(h)
        return x + self.dropout(h)


class TransformerBlock(nn.Module):
    """One transformer layer: self-attention [+ cross-attention] + FFN.

    Each sub-block is a primer-hybrid residual wrapper.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        units_per_head: int,
        ffn_dim: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        use_cross_attention: bool = False,
        cross_attn_source_features: Optional[int] = None,
        cross_attn_max_past_horizon: Optional[int] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        self_attn_dropout_prob: Optional[float] = None,
    ):
        super().__init__()
        # Self-attn attention dropout rate: defaults to dropout_prob.
        self_attn_p = (
            dropout_prob if self_attn_dropout_prob is None else self_attn_dropout_prob
        )
        self.self_attn = SelfAttentionBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            num_sinks=num_sinks,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            eps=eps,
            dropout_prob=dropout_prob,
            attention_dropout_prob=self_attn_p,
        )
        if use_cross_attention:
            if cross_attn_source_features is None or cross_attn_max_past_horizon is None:
                raise ValueError(
                    "cross_attn_source_features and cross_attn_max_past_horizon "
                    "are required when use_cross_attention=True"
                )
            self.cross_attn = CrossAttentionBlock(
                model_dim=model_dim,
                source_features=cross_attn_source_features,
                num_heads=num_heads,
                units_per_head=units_per_head,
                max_past_horizon=cross_attn_max_past_horizon,
                num_sinks=num_sinks,
                compute_dtype=compute_dtype,
                param_dtype=param_dtype,
                eps=eps,
                dropout_prob=dropout_prob,
                attention_dropout_prob=dropout_prob,
            )
        else:
            self.cross_attn = None
        self.ffn = FFN(
            model_dim=model_dim,
            hidden_dim=ffn_dim,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            eps=eps,
            dropout_prob=dropout_prob,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        self_mask: Optional[mx.array] = None,
        self_cache: Optional[LocalKVCache] = None,
        source: Optional[mx.array] = None,
        cross_cache: Optional[LocalKVCache] = None,
    ) -> mx.array:
        x = self.self_attn(x, mask=self_mask, cache=self_cache)
        if self.cross_attn is not None:
            if source is None:
                raise ValueError("source required when cross_attn is enabled")
            x = self.cross_attn(x, source=source, cache=cross_cache)
        x = self.ffn(x)
        return x


class Transformer(nn.Module):
    """Stack of TransformerBlocks (locked-feature subset).

    Each block has its own per-layer self-cache and cross-cache; the
    `make_caches` method returns lists for the user to thread through
    the streaming loop.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        units_per_head: int,
        ffn_dim: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        use_cross_attention: bool = False,
        cross_attn_source_features: Optional[int] = None,
        cross_attn_max_past_horizon: Optional[int] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        self_attn_dropout_prob: Optional[float] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_past_horizon = max_past_horizon
        self.num_sinks = num_sinks
        self.cross_attn_max_past_horizon = cross_attn_max_past_horizon
        self.layers = [
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                units_per_head=units_per_head,
                ffn_dim=ffn_dim,
                max_past_horizon=max_past_horizon,
                num_sinks=num_sinks,
                use_cross_attention=use_cross_attention,
                cross_attn_source_features=cross_attn_source_features,
                cross_attn_max_past_horizon=cross_attn_max_past_horizon,
                compute_dtype=compute_dtype,
                param_dtype=param_dtype,
                eps=eps,
                dropout_prob=dropout_prob,
                self_attn_dropout_prob=self_attn_dropout_prob,
            )
            for _ in range(num_layers)
        ]

    def make_self_caches(self) -> list[LocalKVCache]:
        return [
            LocalKVCache(window_size=self.max_past_horizon + 1, num_sinks=self.num_sinks)
            for _ in range(self.num_layers)
        ]

    def make_cross_caches(self) -> list[LocalKVCache]:
        if self.cross_attn_max_past_horizon is None:
            raise RuntimeError("cross-attention not enabled on this Transformer")
        return [
            LocalKVCache(window_size=self.cross_attn_max_past_horizon + 1, num_sinks=0)
            for _ in range(self.num_layers)
        ]

    def init_cache(
        self,
        self_caches: list[LocalKVCache],
        cross_caches: Optional[list[LocalKVCache]] = None,
        *,
        batch: int,
        dtype,
    ) -> None:
        """Eagerly allocate every per-layer self/cross attention KV cache.

        ``self_caches`` / ``cross_caches`` are the lists produced by
        :meth:`make_self_caches` / :meth:`make_cross_caches` (as held in
        a ``SamplerState``). Each is zero-allocated and — for the
        self-attention caches — sink-primed *in place*, exactly the
        streaming setup the lazy path otherwise performs on the first
        ``__call__``. This lets a content-neutral streaming state be
        prepared without running a warmup step.
        """
        for i, blk in enumerate(self.layers):
            blk.self_attn.attention.init_cache(
                self_caches[i], batch=batch, dtype=dtype,
            )
            if blk.cross_attn is not None:
                if cross_caches is None or i >= len(cross_caches):
                    raise ValueError(
                        "cross_caches must be provided when the transformer "
                        "has cross-attention layers"
                    )
                blk.cross_attn.attention.init_cache(
                    cross_caches[i], batch=batch, dtype=dtype,
                )

    def __call__(
        self,
        x: mx.array,
        *,
        self_caches: Optional[list[LocalKVCache]] = None,
        source: Optional[mx.array] = None,
        cross_caches: Optional[list[LocalKVCache]] = None,
        self_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self_caches is None:
            self_caches = [None] * self.num_layers
        if cross_caches is None:
            cross_caches = [None] * self.num_layers
        for i, blk in enumerate(self.layers):
            x = blk(
                x,
                self_mask=self_mask,
                self_cache=self_caches[i],
                source=source,
                cross_cache=cross_caches[i],
            )
        return x


class MultiChannelEmbedding(nn.Module):
    """Looks up per-channel embeddings for a [B, T, num_channels] integer
    sequence and (optionally) reduces over the channel axis.

    Mirrors ``magenta_rt.mlx.transformer.MultiChannelEmbedding``: a single
    flat embedding table of size
    ``num_reserved_embeddings + sum(num_embeddings_per_channel)`` (rounded
    up to a multiple of 128 by default), with per-channel offsets applied
    at lookup time. Optional reduction (``reduction_fn``, e.g. ``mean``)
    aggregates over the channel axis (``LEVEL_AXIS = -2`` of the embedded
    tensor).
    """

    def __init__(
        self,
        *,
        dimension: int,
        num_embeddings_per_channel: _Seq[int],
        num_channels: int,
        num_reserved_embeddings: int = 0,
        reduction_fn: Optional[Callable[..., mx.array]] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
        round_num_embeddings_to_multiple_of_128: bool = True,
    ):
        super().__init__()
        if len(num_embeddings_per_channel) != num_channels:
            raise ValueError(
                f"num_embeddings_per_channel length {len(num_embeddings_per_channel)}"
                f" != num_channels {num_channels}"
            )
        self.dimension = dimension
        self.num_channels = num_channels
        self.num_reserved_embeddings = num_reserved_embeddings
        self.reduction_fn = reduction_fn
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype

        total = num_reserved_embeddings + sum(num_embeddings_per_channel)
        if round_num_embeddings_to_multiple_of_128:
            total = (total + 127) // 128 * 128
        self.embedding = mx.zeros((total, dimension), dtype=param_dtype)
        self._offsets = mx.cumsum(
            mx.array([0] + list(num_embeddings_per_channel)[:-1], mx.int32)
        )

    def __call__(self, ids: mx.array) -> mx.array:
        """``ids`` shape: [B, T, num_channels] int32.

        Returns shape ``[B, T, dimension]`` (after reduction) or
        ``[B, T, num_channels, dimension]`` if no reduction is configured.
        """
        if ids.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected channel dim {self.num_channels}, got {ids.shape[-1]}"
            )
        embedding = self.embedding
        if self.compute_dtype is not None:
            embedding = embedding.astype(self.compute_dtype)
        offsets = self._offsets
        if self.num_reserved_embeddings:
            offsets = mx.where(
                ids < self.num_reserved_embeddings,
                mx.array(0, dtype=offsets.dtype),
                offsets[None, None, :],
            )
        embedded = mx.take(embedding, ids + offsets, axis=0)
        if self.reduction_fn is not None:
            embedded = self.reduction_fn(embedded, axis=LEVEL_AXIS)
        return embedded


class Encoder(nn.Module):
    """Encoder: ``MultiChannelEmbedding`` (or scalar Embedding + Scale) +
    optional body + final LayerNorm.

    For the shipping configs the body is the identity, so this is just the
    embedding pipeline. Mirrors the structure of
    ``magenta_rt.mlx.depthformer.Encoder.__init__`` so the parameter tree
    lines up.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        embedding_dimension: int,
        body: Optional[nn.Module] = None,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.embedding = embedding
        self.body = body  # None means identity
        self.encoder_ln = LayerNorm(embedding_dimension, eps=1e-6, affine=True, bias=True)

    def __call__(self, tokens: mx.array) -> mx.array:
        h = self.embedding(tokens)
        if self.body is not None:
            h = self.body(h)
        return self.encoder_ln(h)
