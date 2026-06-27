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

"""Transformer block, stack, multichannel embedding, encoder.

Each transformer block owns its KV caches as child modules (via the
:class:`LocalSelfAttention` / :class:`StreamingCrossAttention` it
composes). The streaming pipeline is armed by
``self.set_attributes(streaming=True)`` plus a recursive
``init_cache(batch=batch, dtype=dtype)`` walk.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from flax import nnx

from .attention import LocalSelfAttention, StreamingCrossAttention
from .cache import LocalKVCache
from .conv import remove_cache as _remove_cache


# Matches ``flax.linen.linear.default_embed_init`` (sl's choice for
# ``Embedding`` and friends).
_default_embed_init = nnx.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0,
)


def _gelu_approx(x: jnp.ndarray) -> jnp.ndarray:
    return nnx.gelu(x, approximate=True)


class FFN(nnx.Module):
    """Two-Linear FFN with pre/post RMSNorm wrapping."""

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = _gelu_approx,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.activation = activation
        self.pre_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.post_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.ffn_layer1 = nnx.Linear(
            model_dim, hidden_dim, use_bias=True,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        self.ffn_layer2 = nnx.Linear(
            hidden_dim, model_dim, use_bias=True,
            param_dtype=param_dtype, dtype=dtype, rngs=rngs,
        )
        # Feed-forward network (FFN) uses two T5.1.1-style dropout layers:
        # act_dropout after the activation, and dropout on the sublayer output.
        self.act_dropout = nnx.Dropout(dropout_prob, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.pre_norm(x)
        h = self.ffn_layer1(h)
        h = self.activation(h)
        h = self.act_dropout(h)
        h = self.ffn_layer2(h)
        h = self.post_norm(h)
        h = self.dropout(h)
        return (x + h).astype(x.dtype)


class SelfAttentionBlock(nnx.Module):
    """Self-attention residual with pre/post RMSNorm wrapping."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.pre_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.post_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.attention = LocalSelfAttention(
            in_features=model_dim,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            per_dim_scale=True,
            num_sink_embeddings=num_sinks,
            param_dtype=param_dtype,
            dtype=dtype,
            model_dimension=model_dim,
            attention_dropout_prob=attention_dropout_prob,
            rngs=rngs,
        )
        # Residual dropout on the attention sublayer output.
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        h = self.pre_norm(x)
        h = self.attention(h, mask=mask)
        h = self.post_norm(h)
        h = self.dropout(h)
        return (x + h).astype(x.dtype)


class CrossAttentionBlock(nnx.Module):
    """Streaming cross-attention residual with pre/post RMSNorm wrapping."""

    def __init__(
        self,
        *,
        model_dim: int,
        source_features: int,
        num_heads: int,
        units_per_head: int,
        max_past_horizon: int,
        num_sinks: int = 0,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.pre_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.post_norm = nnx.RMSNorm(num_features=model_dim, epsilon=eps, param_dtype=param_dtype, rngs=rngs)
        self.attention = StreamingCrossAttention(
            in_features=model_dim,
            source_features=source_features,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            num_sink_embeddings=num_sinks,
            per_dim_scale=True,
            param_dtype=param_dtype,
            dtype=dtype,
            model_dimension=model_dim,
            attention_dropout_prob=attention_dropout_prob,
            rngs=rngs,
        )
        # Residual dropout on the cross-attention output.
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, source: jnp.ndarray) -> jnp.ndarray:
        h = self.pre_norm(x)
        h = self.attention(h, source=source)
        h = self.post_norm(h)
        h = self.dropout(h)
        return (x + h).astype(x.dtype)


class TransformerBlock(nnx.Module):
    """One transformer layer: self-attention [+ cross-attention] + FFN."""

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
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.self_attn = SelfAttentionBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_past_horizon=max_past_horizon,
            num_sinks=num_sinks,
            param_dtype=param_dtype,
            dtype=dtype,
            eps=eps,
            dropout_prob=dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            rngs=rngs,
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
                param_dtype=param_dtype,
                dtype=dtype,
                eps=eps,
                dropout_prob=dropout_prob,
                attention_dropout_prob=attention_dropout_prob,
                rngs=rngs,
            )
        else:
            self.cross_attn = None
        self.ffn = FFN(
            model_dim=model_dim, hidden_dim=ffn_dim,
            param_dtype=param_dtype, dtype=dtype, eps=eps,
            dropout_prob=dropout_prob, rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        self_mask: Optional[jnp.ndarray] = None,
        source: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        x = self.self_attn(x, mask=self_mask)
        if self.cross_attn is not None:
            if source is None:
                raise ValueError("source required when cross_attn is enabled")
            x = self.cross_attn(x, source=source)
        x = self.ffn(x)
        return x


class Transformer(nnx.Module):
    """Stack of TransformerBlocks."""

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
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        eps: float = 1e-6,
        dropout_prob: float = 0.0,
        remat: bool = False,
        attention_dropout_prob: Optional[float] = None,
        rngs: nnx.Rngs = None,
    ):
        self.num_layers = num_layers
        self.max_past_horizon = max_past_horizon
        self.num_sinks = num_sinks
        self.cross_attn_max_past_horizon = cross_attn_max_past_horizon
        self.param_dtype = param_dtype
        self.dtype = dtype
        # Gradient checkpointing: when True, the full-sequence __call__ wraps each
        # scanned layer in nnx.remat so only ONE layer's activations are live
        # during backprop (the rest are recomputed) — cutting training activation
        # memory from O(num_layers) to ~O(1) layer. Numerically identical (just
        # recompute); off by default so inference / the streaming step are
        # untouched. Set True only for memory-bound SFT (e.g. mrt2_base on 16 GB).
        self.remat = remat
        # dropout_prob>0 adds per-layer nnx.Dropout in the blocks. The per-layer
        # nnx.split_rngs + nnx.vmap build gives each stacked layer its own dropout
        # RngStream, and the nnx.scan forward threads it as live RngState:
        # verified that masks differ per layer and per step, and that eval() fully
        # disables them. Default 0.0 = no dropout (inference path always uses 0).
        attn_dropout = attention_dropout_prob if attention_dropout_prob is not None else dropout_prob

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make_block(rng):
            return TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                units_per_head=units_per_head,
                ffn_dim=ffn_dim,
                max_past_horizon=max_past_horizon,
                num_sinks=num_sinks,
                use_cross_attention=use_cross_attention,
                cross_attn_source_features=cross_attn_source_features,
                cross_attn_max_past_horizon=cross_attn_max_past_horizon,
                param_dtype=param_dtype,
                dtype=dtype,
                eps=eps,
                dropout_prob=dropout_prob,
                attention_dropout_prob=attn_dropout,
                rngs=rng,
            )

        self.layers = make_block(rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        source: Optional[jnp.ndarray] = None,
        self_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # x flows through the stack at its own dtype; ``nnx.Linear``
        # promotes via ``dtype`` (if set) or matches kernel
        # (param_dtype) otherwise.
        def forward(x, layer):
            return layer(x, self_mask=self_mask, source=source)

        # Gradient checkpointing per scanned layer (training memory; see __init__).
        body = nnx.remat(forward) if self.remat else forward
        scanned = nnx.scan(
            body, length=self.num_layers,
            in_axes=(nnx.Carry, 0), out_axes=nnx.Carry,
        )
        return scanned(x, self.layers)

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        """Allocate attention KV caches on every self / cross attention
        layer in the stack. Use together with
        ``model.set_attributes(streaming=True)`` for streaming inference.

        Layers are always built via ``nnx.split_rngs`` + ``nnx.vmap``
        (see ``__init__``), so the cache allocation has to be vmapped
        too — each layer in the stack gets its own KV slot.
        """
        @nnx.vmap(in_axes=0, out_axes=0)
        def init_layers_cache(layers):
            for _path, m in nnx.iter_modules(layers):
                if isinstance(m, (LocalSelfAttention, StreamingCrossAttention)):
                    m.init_cache(batch=batch, dtype=dtype)
            return layers
        self.layers = init_layers_cache(self.layers)

    def remove_cache(self) -> None:
        """Clear every nnx.Cache slot in the subtree (attention KV +
        any conv left-context that may have been stashed). Called
        before starting a fresh streaming session.
        """
        _remove_cache(self)

    def soft_reset_caches(self) -> None:
        """In-place reset of the rolling-window position on every
        attention cache in the stack, without changing slot types or
        shapes. Used by the depthformer to restart depth
        autoregression at the start of every temporal frame in a
        ``nnx.jit`` / ``nnx.scan``-friendly way.
        """
        for _path, m in nnx.iter_modules(self):
            if isinstance(m, LocalKVCache):
                m.soft_reset()


class MultiChannelEmbedding(nnx.Module):
    """Per-channel embedding table for ``[B, T, num_channels]`` int ids,
    optionally reduced over the channel axis.
    """

    def __init__(
        self,
        *,
        dimension: int,
        num_embeddings_per_channel: Sequence[int],
        num_channels: int,
        num_reserved_embeddings: int = 0,
        reduction_fn: Optional[Callable[..., jnp.ndarray]] = None,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        round_num_embeddings_to_multiple_of_128: bool = True,
        rngs: nnx.Rngs = None,
    ):
        if len(num_embeddings_per_channel) != num_channels:
            raise ValueError(
                f"num_embeddings_per_channel length "
                f"{len(num_embeddings_per_channel)} != num_channels {num_channels}"
            )
        self.dimension = dimension
        self.num_channels = num_channels
        self.num_reserved_embeddings = num_reserved_embeddings
        self.reduction_fn = reduction_fn
        self.dtype = dtype
        self.param_dtype = param_dtype

        total = num_reserved_embeddings + sum(num_embeddings_per_channel)
        if round_num_embeddings_to_multiple_of_128:
            total = (total + 127) // 128 * 128
        self.embedding = nnx.Param(
            _default_embed_init(rngs.params(), (total, dimension), param_dtype)
        )
        # Offsets stored as a tuple of Python ints — a *static* attribute
        # (not a graph data leaf), so the module stays compatible with
        # nnx.cached_partial; materialized as a jnp array at call time.
        self._offsets = tuple(
            np.cumsum(
                np.array([0] + list(num_embeddings_per_channel)[:-1], dtype=np.int32)
            ).tolist()
        )

    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        if ids.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected channel dim {self.num_channels}, got {ids.shape[-1]}"
            )
        embedding = self.embedding[...]
        if self.dtype is not None:
            embedding = embedding.astype(self.dtype)
        offsets = jnp.asarray(self._offsets, dtype=jnp.int32)
        if self.num_reserved_embeddings:
            offsets = jnp.where(
                ids < self.num_reserved_embeddings,
                jnp.array(0, dtype=offsets.dtype),
                offsets[None, None, :],
            )
        embedded = jnp.take(embedding, ids + offsets, axis=0)
        if self.reduction_fn is not None:
            embedded = self.reduction_fn(embedded, axis=-2)
        return embedded


class Encoder(nnx.Module):
    """``MultiChannelEmbedding`` + optional body + final ``LayerNorm``."""

    def __init__(
        self,
        *,
        embedding: nnx.Module,
        embedding_dimension: int,
        body: Optional[nnx.Module] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.embedding = embedding
        self.body = body
        self.encoder_ln = nnx.LayerNorm(
            num_features=embedding_dimension, epsilon=1e-6,
            use_bias=True, use_scale=True,
            param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        h = self.embedding(tokens)
        if self.body is not None:
            h = self.body(h)
        return self.encoder_ln(h)

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        """Forward to ``self.body.init_cache`` if the body has one."""
        if self.body is not None and hasattr(self.body, "init_cache"):
            self.body.init_cache(batch=batch, dtype=dtype)

    def remove_cache(self) -> None:
        _remove_cache(self)


class MulanEmbedder(nnx.Module):
    """Pretrained-MusicCoCa ("mulan") dequantizing embedder.

    Mirrors the Linen ``mulan_embedder`` (offset -> ``mulan_dequantizer``
    Embedding[rvq_levels*per_rvq_vocab, embedding_size] -> sum over the
    rvq-level axis -> ``depth_input_adapter`` Dense[embedding_size ->
    model_dims]). Input ``[B, T, rvq_truncation_level]`` -> ``[B, T, out_dim]``.
    """

    def __init__(
        self,
        *,
        rvq_levels: int,
        rvq_truncation_level: int,
        per_rvq_vocab_size: int,
        embedding_size: int,
        out_dim: int,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.dtype = dtype
        # Per-level vocab offsets as a tuple of Python ints — a *static*
        # attribute (not a graph data leaf), so the module stays
        # compatible with nnx.cached_partial (see MultiChannelEmbedding).
        self._offsets = tuple(
            int(i) * per_rvq_vocab_size for i in range(rvq_truncation_level)
        )
        self.mulan_dequantizer = nnx.Embed(
            num_embeddings=rvq_levels * per_rvq_vocab_size,
            features=embedding_size,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.depth_input_adapter = nnx.Linear(
            embedding_size, out_dim, use_bias=False,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        emb = self.mulan_dequantizer(ids + jnp.asarray(self._offsets, dtype=jnp.int32))
        if self.dtype is not None:
            emb = emb.astype(self.dtype)
        summed = jnp.sum(emb, axis=-2)
        return self.depth_input_adapter(summed)


class BranchedEncoderEmbedding(nnx.Module):
    """Two-branch encoder embedding (mulan + regular) combined by mean,
    mirroring the Linen ``branch_config`` Parallel with ``MEAN``."""

    def __init__(
        self,
        *,
        mulan_embedder: MulanEmbedder,
        regular_embedder: "MultiChannelEmbedding",
        mulan_channels: int,
        num_channels: int,
    ):
        self.mulan_embedder = mulan_embedder
        self.regular_embedder = regular_embedder
        self.mulan_channels = mulan_channels
        self.num_channels = num_channels

    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        mulan = self.mulan_embedder(ids[..., : self.mulan_channels])
        regular = self.regular_embedder(ids[..., self.mulan_channels :])
        return (mulan + regular) * 0.5
