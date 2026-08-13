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

Mirrors the structure of ``magenta_rt.mlx.depthformer`` but uses
idiomatic MLX patterns: caches threaded explicitly, no ``Sequence``
wrapper, and a single ``step`` method instead of the sl
``step_with_emits`` / ``get_initial_state`` split. State is a small
``SamplerState`` ``NamedTuple`` carried by the caller.

* ``num_active_codebooks`` may be ``< num_codebooks`` (depth latency
  optimization). Remaining levels are filled with the first valid
  token of each codebook.
* ``soft_cap_logits`` (optional float) applies a tanh cap to depth-body
  logits before sampling (matches sl's
  ``MultivariateDecoder.Config.soft_cap_logits``).
* CFG batch convention: ``[full, partial_1, ..., partial_n]`` per
  group, with batch size ``orig_B * (cfg_arity + 1)``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import LocalKVCache
from .layers import Dense, LayerNorm
from .sample_utils import sample_categorical_with_temperature
from .transformer import Transformer
def _mean_in_f32(x: mx.array, axis: int) -> mx.array:
    return mx.mean(x.astype(mx.float32), axis=axis).astype(x.dtype)


class ConditioningDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self._p_1 = 1 - p
    def __call__(self, x: mx.array) -> mx.array:
        if self.p == 0.0 or not self.training:
            return x
        B = x.shape[0]
        mask_shape = (B,) + (1,) * (x.ndim - 1)
        mask = mx.random.bernoulli(self._p_1, mask_shape)
        return mx.where(mask, x, 0)


class TemporalCaches(NamedTuple):
    """Per-temporal-layer self- and cross-attn caches."""

    self_caches: list[LocalKVCache]
    cross_caches: list[LocalKVCache]


class SamplerState(NamedTuple):
    """Streaming-decode state for one frame.

    ``rng`` is shape ``[batch_size]`` of ``mx.random.key`` arrays; one
    fresh sub-key is split off per step. ``previous_frame`` is the most
    recent decoded frame's tokens, shape ``[B, 1, num_codebooks]``,
    used as the depthformer's input on the next step. ``temporal``
    holds the layer-wise temporal-transformer caches. ``step`` counts
    completed temporal frames. The SpectroStream codec carries its own
    streaming state on the codec module itself (see
    :meth:`mlx_pure.spectrostream.SpectroStream.enable_streaming`); it is
    not part of the depthformer state pytree.
    """

    rng: mx.array
    previous_frame: mx.array
    temporal: TemporalCaches
    step: mx.array


class DepthformerDecoder(nn.Module):
    """``MultivariateDecoder`` — temporal + depth transformer with
    autoregressive sampling.

    Constructor mirrors the parameter tree of
    ``magenta_rt.mlx.depthformer.MultivariateDecoder`` so
    :func:`mlx_pure.load_weights.mirror_params` can populate weights
    deterministically:

    * ``embedder`` — token embedding (Embedding + sqrt(d) scale wrapped
      in a small helper).
    * ``temporal_body`` — :class:`Transformer` (with cross-attention).
    * ``depth_input_adapter`` — optional Dense (None when
      ``model_dim == depth_dim``).
    * ``depth_transformer`` — :class:`Transformer` (no cross-attention).
    * ``final_ln`` — :class:`LayerNorm` over the depth-transformer
      output.
    * ``to_logits`` — :class:`Dense` (no activation, no bias) projecting
      to the full vocab.

    The depth body operates on a length-``num_codebooks`` sequence per
    temporal frame. For each frame, the temporal output is concatenated
    with the previous codebooks' embeddings and fed to the depth
    transformer. Sampling at depth step ``q`` masks all logits outside
    the codebook-``q`` valid range
    ``[num_reserved + q*codebook_size, num_reserved + (q+1)*codebook_size)``.
    """

    def __init__(
        self,
        *,
        # Vocab / token sizes (matches sl).
        num_codebooks: int,
        codebook_size: int,
        num_reserved_tokens: int,
        vocab_size: int,  # = num_codebooks * codebook_size + num_reserved_tokens
        sos_id: int = 0,
        num_active_codebooks: Optional[int] = None,
        # Dimensions.
        model_dim: int,
        depth_dim: int,
        # Temporal transformer.
        temporal: Transformer,
        # Depth transformer.
        depth: Transformer,
        # Optional depth-input adapter (Dense) when model_dim != depth_dim.
        depth_input_adapter: Optional[Dense] = None,
        # Token embedder (used for both temporal input and depth input).
        embedder: nn.Module = None,
        # Final LayerNorm + to_logits Dense.
        final_ln: Optional[nn.LayerNorm] = None,
        to_logits: Optional[Dense] = None,
        # Optional tanh soft-cap on logits before sampling (matches
        # sl's MultivariateDecoder.Config.soft_cap_logits).
        soft_cap_logits: Optional[float] = None,
        temporal_input_dropout_prob: float = 0.0,
        # Dtypes.
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_reserved_tokens = num_reserved_tokens
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.num_active_codebooks = num_active_codebooks or num_codebooks
        self.model_dim = model_dim
        self.depth_dim = depth_dim
        self.temporal_input_dropout_prob = temporal_input_dropout_prob
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype

        if embedder is None:
            raise ValueError("embedder is required")
        if final_ln is None:
            final_ln = LayerNorm(depth_dim, eps=1e-6, affine=True, bias=True)
        if to_logits is None:
            to_logits = Dense(
                depth_dim, vocab_size, bias=False,
                compute_dtype=compute_dtype, param_dtype=param_dtype,
            )

        self.embedder = embedder
        self.temporal = temporal
        self.depth_input_adapter = depth_input_adapter
        self.depth = depth
        self.final_ln = final_ln
        self.to_logits = to_logits
        self.soft_cap_logits = soft_cap_logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_tokens(self, tokens: mx.array) -> mx.array:
        """tokens: [B, T, num_codebooks] int32 → [B, T, num_codebooks, model_dim]."""
        return self.embedder(tokens)

    def _temporal_input(self, embedded: mx.array) -> mx.array:
        """Mean over codebooks: [B, T, num_codebooks, D] → [B, T, D]."""
        return _mean_in_f32(embedded, axis=-2)

    def _adapt_depth(self, x: mx.array) -> mx.array:
        if self.depth_input_adapter is None:
            return x
        return self.depth_input_adapter(x)

    def _logits(self, depth_out: mx.array) -> mx.array:
        return self.to_logits(self.final_ln(depth_out))

    # ------------------------------------------------------------------
    # Non-streaming forward (training-style, for parity)
    # ------------------------------------------------------------------

    def __call__(
        self,
        tokens: mx.array,
        *,
        encoded_source: Optional[mx.array] = None,
    ) -> mx.array:
        """Full-sequence forward.

        Args:
            tokens: ``[B, T, num_codebooks]`` int32 — the target codebook
                tokens. SOS is prepended internally.
            encoded_source: ``[B, T, source_dim]`` for cross-attention.
                Required when temporal transformer has cross-attn.

        Returns:
            ``[B, T, num_codebooks, vocab_size]`` logits.
        """
        B, T, Q = tokens.shape
        # Prepend SOS frame: shape [B, T+1, Q].
        sos = mx.full((B, 1, Q), self.sos_id, dtype=tokens.dtype)
        padded = mx.concatenate([sos, tokens], axis=1)
        embedded = self._embed_tokens(padded)  # [B, T+1, Q, D]
        temporal_inputs = self._temporal_input(embedded)[:, :-1]  # [B, T, D]

        if self.temporal_input_dropout_prob > 0 and self.training:
            drop_example = mx.random.uniform(shape=(B,) + (1,) * (temporal_inputs.ndim - 1))
            temporal_inputs = mx.where(
                drop_example >= self.temporal_input_dropout_prob,
                temporal_inputs,
                0.0,
            )

        temporal_outputs = self.temporal(
            temporal_inputs,
            source=encoded_source,
        )  # [B, T, D]

        # Depth inputs: [temporal_out_t, embed(token_t,0), embed(token_t,1), ...]
        # along the depth (codebook) axis.
        # embedded[:, 1:] has shape [B, T, Q, D]; take all but last codebook
        # for use as depth-input rows 1..Q-1; row 0 is the temporal output.
        depth_inputs = mx.concatenate(
            [
                temporal_outputs[..., None, :],     # [B, T, 1, D]
                embedded[:, 1:, :-1],                # [B, T, Q-1, D]
            ],
            axis=-2,
        )  # [B, T, Q, D]
        depth_inputs = self._adapt_depth(depth_inputs)
        # Flatten batch×time for the depth transformer.
        bt_inputs = depth_inputs.reshape(B * T, Q, -1)
        depth_out = self.depth(bt_inputs)  # [B*T, Q, depth_dim]
        depth_out = depth_out.reshape(B, T, Q, -1)
        logits = self._logits(depth_out)
        if self.soft_cap_logits is not None:
            cap = self.soft_cap_logits
            logits = mx.tanh(logits / cap) * cap
        return logits

    # ------------------------------------------------------------------
    # Streaming step
    # ------------------------------------------------------------------

    def make_initial_state(
        self,
        batch_size: int,
        *,
        encoded_source: Optional[mx.array] = None,
        seed: int = 42,
    ) -> SamplerState:
        sos_frame = mx.full((batch_size, 1, self.num_codebooks), self.sos_id, dtype=mx.int32)
        rng = mx.stack([mx.random.key(seed + i) for i in range(batch_size)])
        return SamplerState(
            rng=rng,
            previous_frame=sos_frame,
            temporal=TemporalCaches(
                self_caches=self.temporal.make_self_caches(),
                cross_caches=self.temporal.make_cross_caches() if self.temporal.layers[0].cross_attn is not None else [],
            ),
            step=mx.zeros((batch_size,), dtype=mx.int32),
        )

    def init_cache(
        self, state: SamplerState, *, batch: int, dtype: Optional[mx.Dtype] = None,
    ) -> None:
        """Eagerly allocate the temporal transformer's per-layer KV
        caches held in ``state`` (self + cross), priming sink slots —
        so a content-neutral streaming state is ready *without* running
        a warmup step.

        The depth transformer's caches are created fresh inside
        :meth:`step` every frame, so they need no pre-allocation here.
        ``dtype`` defaults to the decoder's ``compute_dtype`` — the
        dtype the K/V projections (hence the lazily-allocated caches)
        would otherwise produce.
        """
        if dtype is None:
            dtype = self.compute_dtype if self.compute_dtype is not None else mx.float32
        self.temporal.init_cache(
            state.temporal.self_caches,
            state.temporal.cross_caches,
            batch=batch,
            dtype=dtype,
        )

    def step(
        self,
        state: SamplerState,
        *,
        encoded_source: mx.array,
        temperature: float | mx.array = 1.0,
        top_k: Optional[int | mx.array] = None,
        top_p: Optional[float | mx.array] = None,
        cfg_scales: Optional[list[float | mx.array]] = None,
        cfg_arity: int = 0,
        forced_tokens: Optional[mx.array] = None,
    ) -> tuple[mx.array, SamplerState]:
        """One streaming temporal step.

        Args:
            state: current streaming state (temporal + depth caches, RNG).
            encoded_source: ``[B, 1, source_dim]`` — the source frame for
                this step (synchronous with the decoder).
            temperature / top_k / top_p / cfg_scales / cfg_arity:
                sampling parameters; identical convention to
                :func:`sample_utils.sample_categorical_with_temperature`.
            forced_tokens: optional ``[B, 1, num_codebooks]`` int32 — when
                provided and non-empty, skips the depth sampling loop and
                returns these tokens directly (after still updating the
                temporal cache from ``previous_frame``). MLX-specific
                convenience matching ``magenta_rt.mlx.depthformer``'s
                ``forced_tokens`` argument.

        Returns:
            ``(sampled_tokens [B, 1, num_codebooks], new_state)``.
        """
        rng = state.rng
        previous_frame = state.previous_frame  # [B, 1, Q]

        # Embed the previous frame, mean over codebooks → temporal input.
        embedded_frame = self._embed_tokens(previous_frame)  # [B, 1, Q, D]
        temporal_inputs = self._temporal_input(embedded_frame)  # [B, 1, D]

        temporal_outputs = self.temporal(
            temporal_inputs,
            source=encoded_source,
            self_caches=state.temporal.self_caches,
            cross_caches=state.temporal.cross_caches,
        )  # [B, 1, D]

        if forced_tokens is not None and forced_tokens.shape[1] > 0:
            new_state = SamplerState(
                rng=rng,
                previous_frame=forced_tokens.astype(mx.int32),
                temporal=state.temporal,
                step=state.step + 1,
            )
            return forced_tokens.astype(mx.int32), new_state

        # Run depth steps autoregressively per codebook.
        B = temporal_outputs.shape[0]
        depth_caches = self.depth.make_self_caches()
        depth_input = self._adapt_depth(temporal_outputs)  # [B, 1, depth_dim]
        sampled = []
        active = self.num_active_codebooks
        for q in range(active):
            depth_out = self.depth(depth_input, self_caches=depth_caches)  # [B, 1, depth_dim]
            logits = self._logits(depth_out)  # [B, 1, V]
            if self.soft_cap_logits is not None:
                cap = self.soft_cap_logits
                logits = mx.tanh(logits / cap) * cap
            min_v = self.num_reserved_tokens + q * self.codebook_size
            max_v = min_v + self.codebook_size
            sample_q = sample_categorical_with_temperature(
                logits.astype(mx.float32),
                rng_key=rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scales=cfg_scales,
                cfg_arity=cfg_arity,
                valid_range=(min_v, max_v),
            )  # [B, 1]
            sampled.append(sample_q)
            # Advance RNG: split off a new sub-key per batch element.
            rng = mx.stack([mx.random.split(_rng)[1] for _rng in rng])
            # Embed sampled token to feed into next depth step.
            depth_input = self.embedder(sample_q[..., None])  # [B, 1, 1, D]
            # _embed_tokens on a 1-channel input returns [B,T,1,D] with
            # mean reducing — but Embedding alone doesn't reduce.
            # Use embedder directly without the mean reduction:
            depth_input = depth_input.squeeze(-2)  # [B, 1, D]
            depth_input = self._adapt_depth(depth_input)

        # Pad inactive codebooks with the first valid token of each.
        if active < self.num_codebooks:
            for q in range(active, self.num_codebooks):
                dummy_val = self.num_reserved_tokens + q * self.codebook_size
                dummy = mx.full(sampled[0].shape, dummy_val, dtype=sampled[0].dtype)
                sampled.append(dummy)

        # Stack along codebook axis → [B, 1, Q].
        sampled_tokens = mx.stack(sampled, axis=-1)

        new_state = SamplerState(
            rng=rng,
            previous_frame=sampled_tokens,
            temporal=state.temporal,  # mutated in-place inside transformer call
            step=state.step + 1,
        )
        return sampled_tokens, new_state


class EncoderDecoder(nn.Module):
    """Top-level encoder + decoder orchestrator.

    The encoder is run once per source frame (streaming, synchronous
    with the decoder). The decoder is the depthformer.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,  # mlx_pure.transformer.Encoder
        decoder: DepthformerDecoder,
        whole_source_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if whole_source_dropout_rate > 0:
            self.conditioning_dropout = ConditioningDropout(whole_source_dropout_rate)
        else:
            self.conditioning_dropout = None

    def encode(self, source_tokens: mx.array) -> mx.array:
        """Encode a [B, T, source_channels] sequence to [B, T, source_dim]."""
        if self.conditioning_dropout is not None:
            source_tokens = self.conditioning_dropout(source_tokens)
        return self.encoder(source_tokens)

    def make_initial_state(
        self,
        batch_size: int,
        *,
        seed: int = 42,
    ) -> SamplerState:
        return self.decoder.make_initial_state(batch_size, seed=seed)

    def init_cache(
        self, state: SamplerState, *, batch: int, dtype: Optional[mx.Dtype] = None,
    ) -> None:
        """Eagerly allocate (and sink-prime) the temporal KV caches held
        in ``state`` — forwards to :meth:`DepthformerDecoder.init_cache`.
        """
        self.decoder.init_cache(state, batch=batch, dtype=dtype)

    def step(
        self,
        state: SamplerState,
        *,
        source_frame: mx.array,
        **sampling_kwargs,
    ) -> tuple[mx.array, SamplerState]:
        """One streaming step. ``source_frame`` is one *encoded* frame.

        The encoder is the caller's responsibility — typically run via
        ``encode`` either once at the beginning or per step (we follow
        sl's "synchronous streaming" convention; per-step is the default).
        """
        return self.decoder.step(state, encoded_source=source_frame, **sampling_kwargs)
