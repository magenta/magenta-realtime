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

"""Magenta-RT pure specification configuration structures and model registry.

Houses stateless configuration that defines exact checkpoint shapes
independent of runtime module instances, plus the ``MagentaRT2ModelBase`` /
``MagentaRT2ModelSmall`` model classes whose :meth:`build_decoder` factory
constructs a fully-wired :class:`mlx_pure.depthformer.EncoderDecoder`.

The framework-agnostic ``ModelSpec`` / ``TokensConfig`` dataclasses and the
canonical presets are imported from :mod:`magenta_rt.config` (the single
source of truth shared with the JAX / sl-MLX implementations); this module
only adds the pure-MLX *building* logic on top.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Sequence
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from magenta_rt.config import (  # noqa: F401
    CFG_CONDITIONING_DRUMS,
    CFG_CONDITIONING_MUSICCOCA_NOTES,
    DRUM_PIANOROLL,
    L_SHALLOW_TPU_OPTIMIZED,
    L_SHALLOW_TPU_OPTIMIZED_6,
    L_TPU_OPTIMIZED,
    M_SHALLOW_TPU_OPTIMIZED,
    MUSICCOCA,
    ModelSpec,
    NUM_RESERVED_TOKENS,
    PIANOROLL_WITH_ONSETS,
    S,
    SPECTROSTREAM,
    TOKEN_DROPOUT_PROB,
    TokensConfig,
    XXL_SHALLOW,
)

from . import depthformer
from . import transformer as mrt_pure_t
from .layers import Dense
from .spectrostream import SpectroStream, ResidualVectorQuantizer


# -----------------------------------------------------------------------------
# Encoder-embedding building blocks.
# -----------------------------------------------------------------------------


class ScaledEmbedding(nn.Module):
    """Token embedding × sqrt(dim) (matches sl decoder embedder layout
    of ``Serial([Embedding, Scale(sqrt(dim))])``).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        param_dtype: mx.Dtype = mx.float32,
        compute_dtype: mx.Dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = float(math.sqrt(dim))

    def __call__(self, ids):
        return self.embedding(ids) * self.scale


def _scaled_embedding(
    vocab_size: int, dim: int, *,
    param_dtype: mx.Dtype, compute_dtype: mx.Dtype,
) -> nn.Module:
    """Backwards-compatible factory; returns a :class:`ScaledEmbedding`."""
    return ScaledEmbedding(
        vocab_size, dim,
        param_dtype=param_dtype, compute_dtype=compute_dtype,
    )


def _mean_in_f32_reduction(x: mx.array, axis: int) -> mx.array:
    return mx.mean(x.astype(mx.float32), axis=axis).astype(x.dtype)


class MusicCoCaEmbedder(nn.Module):
    """Pretrained-MusicCoCa ("mulan") dequantizing embedder.

    Mirrors the sl ``mulan_embedder`` Serial in
    ``magenta_rt.mlx.model.MagentaRT2ModelBase.depthformer_config``:

        offset = arange(rvq_truncation_level) * per_rvq_vocab_size
        ids -> ids + offset
            -> Embedding(rvq_levels * per_rvq_vocab_size, embedding_size)
            -> sum over the rvq-level axis
            -> Dense(embedding_size -> model_dims, bias=False)

    Input ``ids`` shape ``[B, T, rvq_truncation_level]`` (int32);
    output ``[B, T, out_dim]``.
    """

    def __init__(
        self,
        *,
        rvq_levels: int,
        rvq_truncation_level: int,
        per_rvq_vocab_size: int,
        embedding_size: int,
        out_dim: int,
        compute_dtype: Optional[mx.Dtype] = None,
        param_dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.rvq_truncation_level = rvq_truncation_level
        self.compute_dtype = compute_dtype
        # Constant per-level offset into the flat dequantizer table.
        self._offset = (
            mx.arange(rvq_truncation_level, dtype=mx.int32) * per_rvq_vocab_size
        )
        self.mulan_dequantizer = nn.Embedding(
            rvq_levels * per_rvq_vocab_size, embedding_size
        )
        self.mulan_dequantizer.weight = self.mulan_dequantizer.weight.astype(
            param_dtype
        )
        self.depth_input_adapter = Dense(
            embedding_size, out_dim, bias=False,
            compute_dtype=compute_dtype, param_dtype=param_dtype,
        )

    def __call__(self, ids: mx.array) -> mx.array:
        emb = self.mulan_dequantizer(ids + self._offset)
        if self.compute_dtype is not None:
            emb = emb.astype(self.compute_dtype)
        summed = mx.sum(emb, axis=-2)
        return self.depth_input_adapter(summed)


class BranchedEncoderEmbedding(nn.Module):
    """Two-branch encoder embedding combined by mean.

    Mirrors the sl ``branch_config([mulan_channels, regular_channels],
    mulan_embedder, regular_embedder)`` Parallel with
    ``CombinationMode.MEAN``: the first ``mulan_channels`` channels go
    through :class:`MusicCoCaEmbedder`, the rest through a
    :class:`~magenta_rt.mlx_pure.transformer.MultiChannelEmbedding`, and the
    two ``[B, T, model_dims]`` outputs are averaged.
    """

    def __init__(
        self,
        *,
        mulan_embedder: MusicCoCaEmbedder,
        regular_embedder: mrt_pure_t.MultiChannelEmbedding,
        mulan_channels: int,
        num_channels: int,
    ):
        super().__init__()
        self.mulan_embedder = mulan_embedder
        self.regular_embedder = regular_embedder
        self.mulan_channels = mulan_channels
        # Exposed so callers (e.g. generate.py) can read the total input width
        # the same way they read a plain MultiChannelEmbedding.num_channels.
        self.num_channels = num_channels

    def __call__(self, ids: mx.array) -> mx.array:
        mulan = self.mulan_embedder(ids[..., : self.mulan_channels])
        regular = self.regular_embedder(ids[..., self.mulan_channels :])
        return (mulan + regular) * 0.5


# -----------------------------------------------------------------------------
# Model classes — mirror magenta_rt.{jax,mlx}.model.MagentaRT2Model*.
# -----------------------------------------------------------------------------


class MagentaRT2ModelBase(metaclass=abc.ABCMeta):
    """Pure-MLX base spec, parallel to
    ``magenta_rt.mlx.model.MagentaRT2ModelBase``.

    :meth:`build_decoder` constructs a fully-wired
    :class:`mlx_pure.depthformer.EncoderDecoder` against the canonical
    ``mrt2`` feature set (12-token pretrained MusicCoCa + pianoroll-onsets +
    drums + two CFG-conditioning channels).
    """

    encoder_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED
    decoder_temporal_size: ModelSpec = XXL_SHALLOW
    decoder_depth_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED_6

    self_attention_use_separate_qkv: bool = True
    cross_attention_use_separate_kv: bool = True
    temporal_transformer_self_attention_use_kv_cache_ringbuffer: bool = False
    temporal_transformer_cross_attention_use_kv_cache_ringbuffer: bool = False

    param_dtype: mx.Dtype = mx.float32
    compute_dtype: mx.Dtype = mx.bfloat16

    num_attention_sink_embeddings: int = 1
    use_attention_sink_scalars: bool = False
    use_rope: bool = False  # NoPE

    use_pretrained_musiccoca_embedder: bool = True

    # 20s * 25Hz / 20 layers = 25 frames per layer.
    encoder_max_past_horizon: int = 25
    decoder_temporal_self_attention_max_past_horizon: int = 25
    decoder_temporal_cross_attention_max_past_horizon: int = 25

    sampling_eval_seconds: int = 60
    top_k: Optional[int] = 40
    top_p: Optional[float] = None
    cf_guidance_scale: float | tuple[float, ...] = (4.0, 2.0, 4.0)

    # Residual dropout for SFT (wired into the temporal+depth transformers).
    # MUST default to 0.0: mlx_pure inference never sets eval mode, so any
    # nonzero default would drop activations at inference time and break parity.
    # SFT opts in via SFTConfig.dropout_prob (build_model sets this on the spec).
    # ``temporal_self_attention_dropout_prob`` optionally overrides the rate
    # on the temporal self-attention residual only.
    dropout_prob: float = 0.0
    temporal_self_attention_dropout_prob: Optional[float] = None
    whole_source_dropout_rate: float = 0.0
    temporal_input_dropout_prob: float = 0.0

    spectrostream: TokensConfig = SPECTROSTREAM

    @property
    def target_tokens_config(self) -> TokensConfig:
        return dataclasses.replace(
            self.spectrostream, key="ss_target_tokens",
            frame_rate=SPECTROSTREAM.frame_rate,
        )

    @property
    def input_configs(self) -> Sequence[TokensConfig]:
        return (
            MUSICCOCA,
            PIANOROLL_WITH_ONSETS,
            DRUM_PIANOROLL,
            CFG_CONDITIONING_MUSICCOCA_NOTES,
            CFG_CONDITIONING_DRUMS,
        )

    @property
    def input_num_channels(self) -> int:
        return sum(cfg.rvq_truncation_level for cfg in self.input_configs)

    # -------------------------------------------------------------------
    # Encoder-embedding factory.
    # -------------------------------------------------------------------

    def _build_encoder_embedding(self, encoder_spec: ModelSpec) -> nn.Module:
        """Branched (pretrained-MusicCoCa) or plain multi-channel embedding."""
        if self.use_pretrained_musiccoca_embedder:
            musiccoca_cfg = self.input_configs[0]
            assert musiccoca_cfg.key == "mulan_tokens_25hz", (
                f"Expected first input config to be MusicCoCa, got {musiccoca_cfg.key}"
            )
            mulan = MusicCoCaEmbedder(
                rvq_levels=musiccoca_cfg.rvq_levels,
                rvq_truncation_level=musiccoca_cfg.rvq_truncation_level,
                per_rvq_vocab_size=musiccoca_cfg.per_rvq_vocab_size,
                embedding_size=musiccoca_cfg.embedding_size,
                out_dim=encoder_spec.model_dims,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            num_embeddings_per_channel = []
            for cfg in self.input_configs[1:]:
                num_embeddings_per_channel += [
                    cfg.per_rvq_vocab_size
                ] * cfg.rvq_truncation_level
            num_regular_channels = (
                self.input_num_channels - musiccoca_cfg.rvq_truncation_level
            )
            regular = mrt_pure_t.MultiChannelEmbedding(
                num_embeddings_per_channel=num_embeddings_per_channel,
                dimension=encoder_spec.model_dims,
                num_channels=num_regular_channels,
                reduction_fn=_mean_in_f32_reduction,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
            )
            return BranchedEncoderEmbedding(
                mulan_embedder=mulan,
                regular_embedder=regular,
                mulan_channels=musiccoca_cfg.rvq_truncation_level,
                num_channels=self.input_num_channels,
            )

        # Plain single-table embedder (e.g. TinyTestPreset).
        num_embeddings_per_channel = []
        for cfg in self.input_configs:
            num_embeddings_per_channel += [
                cfg.per_rvq_vocab_size
            ] * cfg.rvq_truncation_level
        return mrt_pure_t.MultiChannelEmbedding(
            num_embeddings_per_channel=num_embeddings_per_channel,
            dimension=encoder_spec.model_dims,
            num_channels=self.input_num_channels,
            reduction_fn=_mean_in_f32_reduction,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
        )

    # -------------------------------------------------------------------
    # Builders.
    # -------------------------------------------------------------------

    def build_decoder(
        self,
        *,
        num_active_codebooks: Optional[int] = None,
    ) -> depthformer.EncoderDecoder:
        encoder_spec = self.encoder_size
        decoder_temporal_spec = self.decoder_temporal_size
        decoder_depth_spec = self.decoder_depth_size

        encoder = mrt_pure_t.Encoder(
            embedding=self._build_encoder_embedding(encoder_spec),
            embedding_dimension=encoder_spec.model_dims,
            body=None,  # Identity body for shipping configs.
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
        )

        # ---------- Decoder embedder ----------
        target_cfg = self.target_tokens_config
        embedder = _scaled_embedding(
            target_cfg.vocab_size, decoder_temporal_spec.model_dims,
            param_dtype=self.param_dtype, compute_dtype=self.compute_dtype,
        )

        # ---------- Temporal & depth transformers ----------
        temporal = mrt_pure_t.Transformer(
            num_layers=decoder_temporal_spec.num_layers,
            model_dim=decoder_temporal_spec.model_dims,
            num_heads=decoder_temporal_spec.num_heads,
            units_per_head=decoder_temporal_spec.dim_per_head,
            ffn_dim=decoder_temporal_spec.hidden_dims,
            max_past_horizon=self.decoder_temporal_self_attention_max_past_horizon,
            num_sinks=self.num_attention_sink_embeddings,
            use_cross_attention=True,
            cross_attn_source_features=encoder_spec.model_dims,
            cross_attn_max_past_horizon=self.decoder_temporal_cross_attention_max_past_horizon,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            # Residual dropout. 0.0 by default → no-op (inference never sets
            # eval mode, so it must be off unless SFT opts in via ModelSpec.dropout_prob).
            dropout_prob=self.dropout_prob,
            self_attn_dropout_prob=self.temporal_self_attention_dropout_prob,
        )
        depth = mrt_pure_t.Transformer(
            num_layers=decoder_depth_spec.num_layers,
            model_dim=decoder_depth_spec.model_dims,
            num_heads=decoder_depth_spec.num_heads,
            units_per_head=decoder_depth_spec.dim_per_head,
            ffn_dim=decoder_depth_spec.hidden_dims,
            max_past_horizon=target_cfg.rvq_truncation_level,
            num_sinks=0,
            use_cross_attention=False,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            dropout_prob=self.dropout_prob,
        )

        # Optional depth-input adapter when widths differ.
        depth_input_adapter = None
        if decoder_temporal_spec.model_dims != decoder_depth_spec.model_dims:
            depth_input_adapter = Dense(
                decoder_temporal_spec.model_dims,
                decoder_depth_spec.model_dims,
                bias=False,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )

        decoder = depthformer.DepthformerDecoder(
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
            temporal_input_dropout_prob=self.temporal_input_dropout_prob,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

        return depthformer.EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            whole_source_dropout_rate=self.whole_source_dropout_rate,
        )

    def build_spectrostream(self) -> SpectroStream:
        target_cfg = self.target_tokens_config
        quantizer = ResidualVectorQuantizer(
            num_quantizers=64,
            num_embeddings=1024, embedding_dim=256,
            use_unique_codes=False,
            truncation_level=target_cfg.rvq_truncation_level,
        )
        return SpectroStream(
            stft_frame_length=960, stft_frame_step=480, stft_fft_length=960,
            ratios=((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1)),
            mults=(2, 1, 2, 1, 1, 2, 1),
            is_resnet=True, activation_fn=nn.elu,
            num_bins=480, num_channels=4,
            channel_splits=2, channel_recombo_block=-2,
            num_features=256,
            causal=True,
            encoder_base_conv_depth=32, encoder_base_conv_size=7,
            decoder_base_conv_depth=64, decoder_base_conv_size=7,
            keep_dc=True,
            decoder_lookahead=1,
            quantizer=quantizer,
        )


class MagentaRT2ModelSmall(MagentaRT2ModelBase):
    encoder_size: ModelSpec = S
    decoder_temporal_size: ModelSpec = L_TPU_OPTIMIZED
    decoder_depth_size: ModelSpec = M_SHALLOW_TPU_OPTIMIZED

    # 20s * 25Hz / 12 layers = 41 frames per layer.
    encoder_max_past_horizon: int = 41
    decoder_temporal_self_attention_max_past_horizon: int = 41
    decoder_temporal_cross_attention_max_past_horizon: int = 41


class TinyTestPreset(MagentaRT2ModelBase):
    """Tiny untrained model for fast smoke tests (random init, no checkpoint).

    Uses a single-channel plain embedder (no pretrained MusicCoCa branch).
    """

    encoder_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64, ffn_use_gated_activation=False
    )
    decoder_temporal_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64, ffn_use_gated_activation=False
    )
    decoder_depth_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64, ffn_use_gated_activation=False
    )

    num_attention_sink_embeddings: int = 1
    use_pretrained_musiccoca_embedder: bool = False

    encoder_max_past_horizon: int = 3
    decoder_temporal_self_attention_max_past_horizon: int = 3
    decoder_temporal_cross_attention_max_past_horizon: int = 3

    @property
    def target_tokens_config(self) -> TokensConfig:
        return TokensConfig(
            key="ss_target_tokens",
            codebook_size=8,
            rvq_levels=3,
            rvq_truncation_level=3,
            num_extra_tokens=4,
            frame_rate=25.0,
        )

    @property
    def input_configs(self) -> Sequence[TokensConfig]:
        return (
            TokensConfig(
                key="tiny_input",
                codebook_size=24,
                rvq_levels=1,
                rvq_truncation_level=1,
                num_extra_tokens=4,
                frame_rate=25.0,
            ),
        )

    def build_spectrostream(self) -> SpectroStream:
        target_cfg = self.target_tokens_config
        quantizer = ResidualVectorQuantizer(
            num_quantizers=target_cfg.rvq_truncation_level,
            num_embeddings=4, embedding_dim=16,
            use_unique_codes=False,
        )
        return SpectroStream(
            stft_frame_length=64, stft_frame_step=32, stft_fft_length=64,
            ratios=((1, 2), (2, 1)), mults=(2, 1),
            is_resnet=True, activation_fn=nn.elu,
            num_bins=32, num_channels=2, num_features=16,
            causal=True,
            encoder_base_conv_depth=8, encoder_base_conv_size=3,
            decoder_base_conv_depth=8, decoder_base_conv_size=3,
            quantizer=quantizer,
        )


MODEL_REGISTRY: dict[str, type[MagentaRT2ModelBase]] = {
    "mrt2_base": MagentaRT2ModelBase,
    "mrt2_small": MagentaRT2ModelSmall,
    "tiny": TinyTestPreset,
}


def get_model_class(name: str) -> type[MagentaRT2ModelBase]:
    if name not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available: {avail}")
    return MODEL_REGISTRY[name]
