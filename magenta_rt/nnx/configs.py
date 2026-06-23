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

"""Magenta-RT NNX model specs and registry.

Holds the stateless ``MagentaRT2ModelBase`` / ``MagentaRT2ModelSmall`` specs
(mirroring ``magenta_rt.jax.model``); the neural-network graph compilation is
centralized in ``EncoderDecoder.from_config`` / ``MagentaRT2Sampler.from_preset``.

The framework-agnostic ``ModelSpec`` / ``TokensConfig`` dataclasses and the
canonical presets come from the shared :mod:`magenta_rt.config`.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Optional

import jax.numpy as jnp

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


class MagentaRT2ModelBase(metaclass=abc.ABCMeta):
    """NNX base spec, parallel to ``magenta_rt.jax.model.MagentaRT2ModelBase``."""

    encoder_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED
    decoder_temporal_size: ModelSpec = XXL_SHALLOW
    decoder_depth_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED_6

    self_attention_use_separate_qkv: bool = True
    cross_attention_use_separate_kv: bool = True
    temporal_transformer_self_attention_use_kv_cache_ringbuffer: bool = False
    temporal_transformer_cross_attention_use_kv_cache_ringbuffer: bool = False

    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.bfloat16

    num_attention_sink_embeddings: int = 1
    use_attention_sink_scalars: bool = False
    use_rope: bool = False  # NoPE

    use_pretrained_musiccoca_embedder: bool = True

    encoder_max_past_horizon: int = 25
    decoder_temporal_self_attention_max_past_horizon: int = 25
    decoder_temporal_cross_attention_max_past_horizon: int = 25

    sampling_eval_seconds: int = 60
    top_k: Optional[int] = 40
    top_p: Optional[float] = None
    cf_guidance_scale: float | tuple[float, ...] = (4.0, 2.0, 4.0)

    # NOTE(dropout): these spec fields are NOT auto-applied. nnx transformer
    # dropout is implemented (magenta_rt.nnx.transformer) and applied only when
    # EncoderDecoder.from_config(dropout_prob=...) is passed a nonzero value
    # explicitly (the SFT trainer wires config.dropout_prob; the inference path
    # leaves it 0). The rng-under-scan threading and the train/eval +
    # cached_partial graph behavior are verified (see transformer.py and the
    # trainer's step-binding site). Dropout placement and configuration overrides
    # are reconciled against the ground truth.
    dropout_prob: float = 0.1
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


class MagentaRT2ModelSmall(MagentaRT2ModelBase):
    encoder_size: ModelSpec = S
    decoder_temporal_size: ModelSpec = L_TPU_OPTIMIZED
    decoder_depth_size: ModelSpec = M_SHALLOW_TPU_OPTIMIZED

    encoder_max_past_horizon: int = 41
    decoder_temporal_self_attention_max_past_horizon: int = 41
    decoder_temporal_cross_attention_max_past_horizon: int = 41


class TinyTestPreset(MagentaRT2ModelBase):
    """Tiny untrained model for fast smoke tests (single-channel, no MusicCoCa)."""

    encoder_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64,
        ffn_use_gated_activation=False,
    )
    decoder_temporal_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64,
        ffn_use_gated_activation=False,
    )
    decoder_depth_size: ModelSpec = ModelSpec(
        num_layers=1, model_dims=32, num_heads=4, dim_per_head=8, hidden_dims=64,
        ffn_use_gated_activation=False,
    )

    use_pretrained_musiccoca_embedder: bool = False
    encoder_max_past_horizon: int = 3
    decoder_temporal_self_attention_max_past_horizon: int = 3
    decoder_temporal_cross_attention_max_past_horizon: int = 3

    @property
    def target_tokens_config(self) -> TokensConfig:
        return TokensConfig(
            key="ss_target_tokens", codebook_size=8, rvq_levels=3,
            rvq_truncation_level=3, num_extra_tokens=4, frame_rate=25.0,
        )

    @property
    def input_configs(self) -> Sequence[TokensConfig]:
        return (
            TokensConfig(
                key="tiny_input", codebook_size=24, rvq_levels=1,
                rvq_truncation_level=1, num_extra_tokens=4, frame_rate=25.0,
            ),
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
