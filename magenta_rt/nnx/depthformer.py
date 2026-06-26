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

"""Depthformer encoder/decoder.

* Per-layer attention KV caches live on the attention modules
  (via :class:`~magenta_rt.nnx.cache.LocalKVCache`), populated by
  ``model.init_cache(batch, dtype)``.
* The depth transformer's autoregressive sampling resets its caches
  at the start of every temporal step.
* :class:`DecodeState` variables store ``rng_state``,
  ``previous_frame``, and ``step_counter`` directly on the
  decoder module for clean encapsulation.
* CFG batch convention: ``[full, partial_1, …]`` per group.
"""

from __future__ import annotations

from typing import Optional

import jax

import dataclasses
import math

from einops import rearrange
from jax import numpy as jnp
from flax import nnx
from flax.nnx.module import first_from

from .conv import remove_cache as _remove_cache
from .sample_utils import sample_categorical_with_temperature
from .transformer import (
    BranchedEncoderEmbedding,
    Encoder,
    MulanEmbedder,
    MultiChannelEmbedding,
    Transformer,
)


def _mean_in_f32(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.mean(x.astype(jnp.float32), axis=axis).astype(x.dtype)


class ConditioningDropout(nnx.Module):
    """Whole-source conditioning dropout.

    Constructed unconditionally: a rate of 0 (no rng stream), eval/deterministic
    mode, makes ``__call__`` a pure no-op, so callers never branch on whether
    dropout is enabled.
    """

    def __init__(self, p: float, *, rngs: Optional[nnx.Rngs] = None):
        self.p = p
        self.deterministic = True
        # Only carry an rng stream when dropout is actually active.
        if p > 0 and rngs is not None:
            self.dropout_rng = rngs.dropout.fork()
        else:
            self.dropout_rng = None

    def __call__(self, x: jnp.ndarray, *, deterministic: Optional[bool] = None) -> jnp.ndarray:
        det = first_from(
            deterministic,
            self.deterministic,
            error_msg="ConditioningDropout needs `deterministic` as an argument "
                      "or attribute.",
        )
        if det or self.dropout_rng is None:
            return x
        keep_prob = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = self.dropout_rng.bernoulli(p=keep_prob, shape=mask_shape)
        return jnp.where(mask, x, 0)


class DecodeState(nnx.Variable):
    """Internal mutable variable slot for autoregressive decode context."""
    pass


class ScaledEmbedding(nnx.Module):
    """Token embedding × sqrt(dim) (matches sl decoder embedder layout
    of ``Serial([Embedding, Scale(sqrt(dim))])``).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: Optional[jnp.dtype] = None,
        rngs: nnx.Rngs = None,
    ):
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size, features=dim,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.scale = float(math.sqrt(dim))

    def __call__(self, ids):
        return self.embedding(ids) * self.scale


class DepthformerDecoder(nnx.Module):
    """Temporal + depth transformer with autoregressive sampling.

    Parameter tree matches the Linen safetensors checkpoint layout
    so the weight bridge can populate it directly:

    * ``embedder`` — :class:`MultiChannelEmbedding` or compatible.
    * ``temporal`` — :class:`Transformer` with cross-attention.
    * ``depth_input_adapter`` — optional ``Dense`` (None when
      ``model_dim == depth_dim``).
    * ``depth`` — :class:`Transformer` (no cross-attention).
    * ``final_ln`` — ``LayerNorm`` over depth-transformer output.
    * ``to_logits`` — ``Dense`` (no activation, no bias) projecting to
      the full vocab.
    """

    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        num_reserved_tokens: int,
        vocab_size: int,
        sos_id: int = 0,
        num_active_codebooks: Optional[int] = None,
        model_dim: int,
        depth_dim: int,
        temporal: Transformer,
        depth: Transformer,
        depth_input_adapter: Optional[nnx.Linear] = None,
        embedder: nnx.Module = None,
        final_ln: Optional[nnx.LayerNorm] = None,
        to_logits: Optional[nnx.Linear] = None,
        soft_cap_logits: Optional[float] = None,
        temporal_input_dropout_prob: float = 0.0,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_reserved_tokens = num_reserved_tokens
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.num_active_codebooks = num_active_codebooks or num_codebooks
        self.model_dim = model_dim
        self.depth_dim = depth_dim
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.soft_cap_logits = soft_cap_logits
        self.temporal_input_dropout_prob = temporal_input_dropout_prob
        self.deterministic = False
        if rngs is not None and temporal_input_dropout_prob > 0:
            self.dropout_rng = rngs.dropout.fork()
        else:
            self.dropout_rng = None

        self.step_counter = DecodeState(jnp.array(0, dtype=jnp.int32))
        self.previous_frame = DecodeState(nnx.data(None))
        self.rng_state = DecodeState(nnx.data(None))  # `init_streaming` will set this to nnx.Rngs

        if embedder is None:
            raise ValueError("embedder is required")
        if final_ln is None:
            final_ln = nnx.LayerNorm(
                num_features=depth_dim, epsilon=1e-6,
                use_bias=True, use_scale=True,
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
            )
        if to_logits is None:
            # use_bias=True: the checkpoint's depth-body ``to_logits``
            # carries a bias (sl/jax ``Dense`` defaults to a bias and
            # the trained weights include it). Building this Linear
            # without a bias makes the loader silently drop that
            # parameter (``to_logits.bias is None`` short-circuits the
            # bias copy), shifting every logit by the missing bias.
            to_logits = nnx.Linear(
                depth_dim, vocab_size, use_bias=True,
                param_dtype=param_dtype, dtype=dtype, rngs=rngs,
            )

        self.embedder = embedder
        self.temporal = temporal
        self.depth_input_adapter = depth_input_adapter
        self.depth = depth
        self.final_ln = final_ln
        self.to_logits = to_logits

    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        """Allocate temporal-transformer KV caches AND depth-transformer
        KV caches up front. The depth caches get a per-frame
        ``soft_reset`` inside ``step`` (just zeros ``cache_index``);
        pre-allocating them here means ``step`` never has to allocate
        new slots, which is required for ``nnx.jit`` / ``nnx.scan``.
        """
        self.temporal.init_cache(batch=batch, dtype=dtype)
        self.depth.init_cache(batch=batch, dtype=dtype)

    def remove_cache(self) -> None:
        _remove_cache(self)

    def _embed_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return self.embedder(tokens)

    def _temporal_input(self, embedded: jnp.ndarray) -> jnp.ndarray:
        return _mean_in_f32(embedded, axis=-2)

    def _adapt_depth(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.depth_input_adapter is None:
            return x
        return self.depth_input_adapter(x)

    def _logits(self, depth_out: jnp.ndarray) -> jnp.ndarray:
        return self.to_logits(self.final_ln(depth_out))

    def __call__(
        self,
        tokens: jnp.ndarray,
        *,
        encoded_source: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Full-sequence forward.

        Args:
            tokens: ``[B, T, num_codebooks]`` int32. SOS prepended internally.
            encoded_source: ``[B, T, source_dim]`` for cross-attention.

        Returns:
            ``[B, T, num_codebooks, vocab_size]`` logits.
        """
        B, T, Q = tokens.shape
        sos = jnp.full((B, 1, Q), self.sos_id, dtype=tokens.dtype)
        padded = jnp.concatenate([sos, tokens], axis=1)
        embedded = self._embed_tokens(padded)  # [B, T+1, Q, D]
        temporal_inputs = self._temporal_input(embedded)[:, :-1]  # [B, T, D]

        if self.dropout_rng is not None and not self.deterministic:
            drop_example = self.dropout_rng.uniform(
                shape=(B,) + (1,) * (temporal_inputs.ndim - 1),
                dtype=temporal_inputs.dtype,
            )
            temporal_inputs = jnp.where(
                drop_example >= self.temporal_input_dropout_prob,
                temporal_inputs,
                0.0,
            )

        temporal_outputs = self.temporal(temporal_inputs, source=encoded_source)

        depth_inputs = jnp.concatenate(
            [
                temporal_outputs[..., None, :],  # [B, T, 1, D]
                embedded[:, 1:, :-1],            # [B, T, Q-1, D]
            ],
            axis=-2,
        )  # [B, T, Q, D]
        depth_inputs = self._adapt_depth(depth_inputs)
        bt_inputs = rearrange(depth_inputs, "b t q d -> (b t) q d")
        depth_out = self.depth(bt_inputs)
        depth_out = rearrange(depth_out, "(b t) q d -> b t q d", b=B)
        logits = self._logits(depth_out)
        if self.soft_cap_logits is not None:
            cap = self.soft_cap_logits
            logits = jnp.tanh(logits / cap) * cap
        return logits

    def init_streaming(
        self,
        batch_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        sos_frame = jnp.full(
            (batch_size, 1, self.num_codebooks), self.sos_id, dtype=jnp.int32,
        )
        self.step_counter.set_value(jnp.zeros((batch_size,), dtype=jnp.int32))
        self.previous_frame.set_value(sos_frame)
        self.rng_state.set_value(rngs.fork())

    def step(
        self,
        *,
        encoded_source: jnp.ndarray,
        temperature: float | jnp.ndarray = 1.0,
        top_k: Optional[int | jnp.ndarray] = None,
        top_p: Optional[float | jnp.ndarray] = None,
        cfg_scales: Optional[list[float | jnp.ndarray]] = None,
        cfg_arity: int = 0,
        forced_tokens: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """One streaming temporal step.

        Returns ``sampled_tokens [B, 1, num_codebooks]``.
        Mutates internal KV caches and decode variables in place.
        """
        rngs = self.rng_state.get_value()
        previous_frame = self.previous_frame[...]  # [B, 1, Q]

        embedded_frame = self._embed_tokens(previous_frame)
        temporal_inputs = self._temporal_input(embedded_frame)

        # Temporal body assumed in streaming mode (caller called
        # init_cache + set_attributes(streaming=True)).
        temporal_outputs = self.temporal(temporal_inputs, source=encoded_source)

        if forced_tokens is not None and forced_tokens.shape[1] > 0:
            self.previous_frame.set_value(forced_tokens.astype(jnp.int32))
            self.step_counter.set_value(self.step_counter[...] + 1)
            return forced_tokens.astype(jnp.int32)

        # Reset depth caches per frame: each frame starts depth
        # autoregression from scratch. ``soft_reset_caches`` just
        # zeros ``cache_index[...]`` in place — safe inside
        # ``nnx.jit`` / ``nnx.scan`` because no slot types or shapes
        # change. Caches are guaranteed allocated by ``init_cache``,
        # which the streaming contract requires before ``step``.
        self.depth.soft_reset_caches()

        active = self.num_active_codebooks
        min_vs = (
            self.num_reserved_tokens
            + jnp.arange(active, dtype=jnp.int32) * self.codebook_size
        )

        step_keys = rngs.split(active)

        codebook_size = self.codebook_size
        soft_cap = self.soft_cap_logits
        logits_fn = self._logits
        embed_fn = self.embedder
        adapt_fn = self._adapt_depth

        # ``self.depth`` is threaded through the scan carry (NOT via
        # ``in_axes=None``) so that its KV-cache mutations round-trip
        # back to the outer module — required for the inner scan to
        # compose with the outer streaming scan in
        # :mod:`magenta_rt.nnx.generate`.
        @nnx.scan(
            in_axes=(nnx.Carry, 0, 0),
            out_axes=(nnx.Carry, 0),
        )
        def depth_body(carry, min_v, step_rng):
            depth_input, depth_xfmr = carry
            depth_out = depth_xfmr(depth_input)
            logits = logits_fn(depth_out)
            if soft_cap is not None:
                logits = jnp.tanh(logits / soft_cap) * soft_cap
            sample_q = sample_categorical_with_temperature(
                logits.astype(jnp.float32),
                rng_key=step_rng(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scales=cfg_scales,
                cfg_arity=cfg_arity,
                valid_range=(min_v, min_v + codebook_size),
            )  # [B, 1]
            next_depth_input = embed_fn(sample_q[..., None]).squeeze(-2)
            next_depth_input = adapt_fn(next_depth_input)
            return (next_depth_input, depth_xfmr), sample_q

        init_depth_input = self._adapt_depth(temporal_outputs)
        # Match the scan body's output dtype (embedder's dtype)
        # so the carry dtype is invariant across iterations.
        if self.dtype is not None:
            init_depth_input = init_depth_input.astype(self.dtype)
        _, sampled_arr = depth_body(
            (init_depth_input, self.depth), min_vs, step_keys,
        )
        # sampled_arr: [active, B, 1] -> [B, 1, active]
        sampled_active = jnp.moveaxis(sampled_arr, 0, -1)

        # Pad inactive codebooks with the first valid token of each.
        if active < self.num_codebooks:
            b_, t_ = sampled_active.shape[:2]
            dummy_vals = (
                self.num_reserved_tokens
                + jnp.arange(active, self.num_codebooks, dtype=jnp.int32)
                * self.codebook_size
            )
            pad = jnp.broadcast_to(
                dummy_vals, (b_, t_, self.num_codebooks - active),
            ).astype(sampled_active.dtype)
            sampled_tokens = jnp.concatenate([sampled_active, pad], axis=-1)
        else:
            sampled_tokens = sampled_active

        self.rng_state.set_value(rngs)
        self.previous_frame.set_value(sampled_tokens)
        self.step_counter.set_value(self.step_counter[...] + 1)
        return sampled_tokens


class EncoderDecoder(nnx.Module):
    """Top-level encoder + decoder orchestrator."""

    def __init__(
        self,
        *,
        encoder: nnx.Module,
        decoder: DepthformerDecoder,
        whole_source_dropout_rate: float = 0.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.whole_source_dropout_rate = whole_source_dropout_rate
        # Constructed unconditionally — ConditioningDropout is a no-op at rate 0.
        self.conditioning_dropout = ConditioningDropout(
            p=whole_source_dropout_rate, rngs=rngs,
        )

    def encode(self, source: jnp.ndarray, *, deterministic: Optional[bool] = None) -> jnp.ndarray:
        source = self.conditioning_dropout(source, deterministic=deterministic)
        return self.encoder(source)

    @classmethod
    def from_config(
        cls,
        config,
        *,
        num_active_codebooks: Optional[int] = None,
        depth_num_layers: Optional[int] = None,
        depth_model_dims: Optional[int] = None,
        depth_hidden_dims: Optional[int] = None,
        dropout_prob: float = 0.0,
        remat: bool = False,
        rngs: nnx.Rngs = None,
    ) -> EncoderDecoder:
        """Centralized neural network graph compiler mapping a configuration
        spec directly into the fully-wired module hierarchy.

        ``dropout_prob`` adds per-layer transformer dropout. It is an EXPLICIT
        argument defaulting to 0.0 rather than read from ``config.dropout_prob``
        so the inference path (``MagentaRT2Sampler.from_preset``) stays dropout-free;
        the SFT trainer passes ``config.dropout_prob`` and is responsible for
        ``model.train()`` / ``model.eval()`` discipline.
        """
        whole_source_dropout_rate = getattr(config, "whole_source_dropout_rate", 0.0)
        temporal_input_dropout_prob = getattr(config, "temporal_input_dropout_prob", 0.0)

        encoder_spec = config.encoder_size
        decoder_temporal_spec = config.decoder_temporal_size
        depth_overrides = {}
        if depth_num_layers is not None:
            depth_overrides["num_layers"] = depth_num_layers
        if depth_model_dims is not None:
            depth_overrides["model_dims"] = depth_model_dims
        if depth_hidden_dims is not None:
            depth_overrides["hidden_dims"] = depth_hidden_dims
        # Fall back to the model class's own decoder_depth_size when not
        # explicitly overridden, matching JAX behaviour.
        depth_overrides["dropout_prob"] = dropout_prob
        decoder_depth_spec = dataclasses.replace(
            config.decoder_depth_size, **depth_overrides
        )

        # Encoder embedding: branched (pretrained-MusicCoCa) or plain.
        if config.use_pretrained_musiccoca_embedder:
            musiccoca_cfg = config.input_configs[0]
            mulan = MulanEmbedder(
                rvq_levels=musiccoca_cfg.rvq_levels,
                rvq_truncation_level=musiccoca_cfg.rvq_truncation_level,
                per_rvq_vocab_size=musiccoca_cfg.per_rvq_vocab_size,
                embedding_size=musiccoca_cfg.embedding_size,
                out_dim=encoder_spec.model_dims,
                dtype=config.dtype, param_dtype=config.param_dtype, rngs=rngs,
            )
            num_embeddings_per_channel = []
            for cfg in config.input_configs[1:]:
                num_embeddings_per_channel += [cfg.per_rvq_vocab_size] * cfg.rvq_truncation_level
            num_regular_channels = (
                config.input_num_channels - musiccoca_cfg.rvq_truncation_level
            )
            regular = MultiChannelEmbedding(
                num_embeddings_per_channel=num_embeddings_per_channel,
                dimension=encoder_spec.model_dims,
                num_channels=num_regular_channels,
                reduction_fn=_mean_in_f32,
                param_dtype=config.param_dtype, dtype=config.dtype, rngs=rngs,
            )
            encoder_embedding = BranchedEncoderEmbedding(
                mulan_embedder=mulan, regular_embedder=regular,
                mulan_channels=musiccoca_cfg.rvq_truncation_level,
                num_channels=config.input_num_channels,
            )
        else:
            num_embeddings_per_channel = []
            for cfg in config.input_configs:
                num_embeddings_per_channel += [cfg.per_rvq_vocab_size] * cfg.rvq_truncation_level
            encoder_embedding = MultiChannelEmbedding(
                num_embeddings_per_channel=num_embeddings_per_channel,
                dimension=encoder_spec.model_dims,
                num_channels=config.input_num_channels,
                reduction_fn=_mean_in_f32,
                param_dtype=config.param_dtype,
                dtype=config.dtype,
                rngs=rngs,
            )
        encoder = Encoder(
            embedding=encoder_embedding,
            embedding_dimension=encoder_spec.model_dims,
            body=None,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

        # Decoder embedder.
        target_cfg = config.target_tokens_config
        embedder = ScaledEmbedding(
            target_cfg.vocab_size, decoder_temporal_spec.model_dims,
            param_dtype=config.param_dtype, dtype=config.dtype,
            rngs=rngs,
        )

        # Temporal & depth transformers.
        temporal = Transformer(
            num_layers=decoder_temporal_spec.num_layers,
            model_dim=decoder_temporal_spec.model_dims,
            num_heads=decoder_temporal_spec.num_heads,
            units_per_head=decoder_temporal_spec.dim_per_head,
            ffn_dim=decoder_temporal_spec.hidden_dims,
            max_past_horizon=config.decoder_temporal_self_attention_max_past_horizon,
            num_sinks=config.num_attention_sink_embeddings,
            use_cross_attention=True,
            cross_attn_source_features=encoder_spec.model_dims,
            cross_attn_max_past_horizon=config.decoder_temporal_cross_attention_max_past_horizon,
            param_dtype=config.param_dtype,
            dtype=config.dtype,
            dropout_prob=dropout_prob,
            attention_dropout_prob=config.temporal_self_attention_dropout_prob,
            remat=remat,
            rngs=rngs,
        )
        depth = Transformer(
            num_layers=decoder_depth_spec.num_layers,
            model_dim=decoder_depth_spec.model_dims,
            num_heads=decoder_depth_spec.num_heads,
            units_per_head=decoder_depth_spec.dim_per_head,
            ffn_dim=decoder_depth_spec.hidden_dims,
            max_past_horizon=target_cfg.rvq_truncation_level,
            num_sinks=0,
            use_cross_attention=False,
            param_dtype=config.param_dtype,
            dtype=config.dtype,
            dropout_prob=dropout_prob,
            remat=remat,
            rngs=rngs,
        )

        depth_input_adapter = None
        if decoder_temporal_spec.model_dims != decoder_depth_spec.model_dims:
            depth_input_adapter = nnx.Linear(
                decoder_temporal_spec.model_dims,
                decoder_depth_spec.model_dims,
                use_bias=False,
                param_dtype=config.param_dtype,
                dtype=config.dtype,
                rngs=rngs,
            )

        decoder = DepthformerDecoder(
            num_codebooks=target_cfg.rvq_truncation_level,
            codebook_size=target_cfg.codebook_size,
            num_reserved_tokens=target_cfg.num_extra_tokens,
            vocab_size=target_cfg.vocab_size,
            sos_id=0,
            num_active_codebooks=num_active_codebooks,
            model_dim=decoder_temporal_spec.model_dims,
            depth_dim=decoder_depth_spec.model_dims,
            temporal=temporal,
            depth=depth,
            depth_input_adapter=depth_input_adapter,
            embedder=embedder,
            soft_cap_logits=30.0,
            temporal_input_dropout_prob=temporal_input_dropout_prob,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

        return cls(
            encoder=encoder,
            decoder=decoder,
            whole_source_dropout_rate=whole_source_dropout_rate,
            rngs=rngs,
        )


    def init_cache(self, *, batch: int, dtype=jnp.float32) -> None:
        self.encoder.init_cache(batch=batch, dtype=dtype)
        self.decoder.init_cache(batch=batch, dtype=dtype)

    def remove_cache(self) -> None:
        _remove_cache(self)

    def init_streaming(self, batch_size: int, *, rngs: nnx.Rngs) -> None:
        self.decoder.init_streaming(batch_size, rngs=rngs)

    def step(
        self,
        *,
        source_tokens: jnp.ndarray,
        **sampling_kwargs,
    ) -> jnp.ndarray:
        """One streaming step."""
        source_frame = self.encoder(source_tokens)
        return self.decoder.step(
            encoded_source=source_frame, **sampling_kwargs,
        )
