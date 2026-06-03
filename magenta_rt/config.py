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

"""Shared model configuration: ModelSpec, TokensConfig, and token constants.

These definitions are framework-agnostic (no JAX / MLX imports) and are shared
between the JAX and MLX model implementations.
"""

import dataclasses


NUM_RESERVED_TOKENS = 6


# ---------------------------------------------------------------------------
# ModelSpec — Transformer architecture sizes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Specification of the Transformer spec used by Depthformer."""

  num_layers: int = 12
  model_dims: int = 1024
  hidden_dims: int = 1024 * 4
  num_heads: int = 16
  dim_per_head: int = 64
  dropout_prob: float = 0.0
  use_mqa: bool = False
  ffn_use_gated_activation: bool = True
  use_repeat_layers: bool = False


S = ModelSpec(
    num_layers=6,
    model_dims=256,
    hidden_dims=1024,
    num_heads=8,
    dim_per_head=32,
    ffn_use_gated_activation=False,
)

M_SHALLOW_TPU_OPTIMIZED = ModelSpec(
    num_layers=2,
    model_dims=768,
    hidden_dims=3072,
    num_heads=6,
    dim_per_head=128,
    ffn_use_gated_activation=False,
)

L_TPU_OPTIMIZED = ModelSpec(
    num_layers=12,
    model_dims=1024,
    hidden_dims=4096,
    num_heads=8,
    dim_per_head=128,
    ffn_use_gated_activation=False,
)

L_SHALLOW_TPU_OPTIMIZED = ModelSpec(
    num_layers=4,
    model_dims=1024,
    hidden_dims=4096,
    num_heads=8,
    dim_per_head=128,
    ffn_use_gated_activation=False,
)

L_SHALLOW_TPU_OPTIMIZED_6 = ModelSpec(
    num_layers=6,
    model_dims=1024,
    hidden_dims=4096,
    num_heads=8,
    dim_per_head=128,
    ffn_use_gated_activation=False,
)

XXL_SHALLOW = ModelSpec(
    num_layers=20,
    model_dims=3072,
    hidden_dims=8192,
    num_heads=24,
    dim_per_head=128,
    ffn_use_gated_activation=False,
)


# ---------------------------------------------------------------------------
# TokensConfig — input/output token descriptors
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class TokensConfig:
  """Describes an SS, MLM or SS2Vec tokens."""

  key: str = 'embeddings'
  codebook_size: int = -1
  rvq_levels: int = -1
  rvq_truncation_level: int = -1
  quantizer: str | None = None
  num_extra_tokens: int = NUM_RESERVED_TOKENS  # Doesn't include dropout token.
  frame_rate: float = -1
  dropout_prob: float | None = None
  embedding_size: int = -1

  @property
  def dimension(self) -> int:
    return self.rvq_truncation_level

  @property
  def vocab_size(self) -> int | None:
    total_size = (
        self.codebook_size * self.rvq_truncation_level + self.num_extra_tokens
    )
    if self.dropout_prob is not None:
      total_size += 1  # 1 more token for the dropout token.
    return total_size

  @property
  def per_rvq_vocab_size(self) -> int | None:
    # This method is used for the target vocabulary which uses a
    # multi-categorical distributions (one softmax per rvq level).
    total_size = self.codebook_size + self.num_extra_tokens
    if self.dropout_prob is not None:
      total_size += 1  # 1 more token for the dropout token.
    return total_size


# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

TOKEN_DROPOUT_PROB = 0.15

PIANOROLL_WITH_ONSETS = TokensConfig(
    key='pianoroll_with_onsets_tokens',
    codebook_size=4,
    rvq_levels=128,
    rvq_truncation_level=128,
    frame_rate=25,
    dropout_prob=TOKEN_DROPOUT_PROB,
)

DRUM_PIANOROLL = TokensConfig(
    key='drum_pianoroll_tokens',
    codebook_size=2,
    rvq_levels=1,
    rvq_truncation_level=1,
    frame_rate=25,
    dropout_prob=TOKEN_DROPOUT_PROB,
)

CFG_CONDITIONING_MUSICCOCA_NOTES = TokensConfig(
    key='cfg_conditioning_tokens',
    codebook_size=41,
    rvq_levels=2,
    rvq_truncation_level=2,
    frame_rate=25,
    dropout_prob=None,
)

CFG_CONDITIONING_DRUMS = TokensConfig(
    key='cfg_conditioning_drums_tokens',
    codebook_size=9,
    rvq_levels=1,
    rvq_truncation_level=1,
    frame_rate=25,
    dropout_prob=None,
)

SPECTROSTREAM = TokensConfig(
    key='spectrostream_tokens',
    codebook_size=1024,
    rvq_levels=64,
    rvq_truncation_level=12,
    frame_rate=25,
)

MUSICCOCA = TokensConfig(
    key='mulan_tokens_25hz',
    dropout_prob=TOKEN_DROPOUT_PROB,
    codebook_size=1024,
    rvq_levels=12,
    rvq_truncation_level=12,
    frame_rate=SPECTROSTREAM.frame_rate,
    embedding_size=768,
)
