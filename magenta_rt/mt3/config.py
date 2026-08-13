# Copyright 2025 The MT3 Authors.
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

"""MT3 configuration."""

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .spectrograms import SpectrogramConfig
from .vocabularies import VocabularyConfig, build_codec, num_embeddings, vocabulary_from_codec


@dataclass(frozen=True)
class MT3Config:
    """Configuration for the MT3 encoder-decoder transcription model.

    The network is a T5.1.1 Transformer with fixed sinusoidal positional
    embeddings and a continuous-input projection in place of encoder token
    embeddings.

    Args:
        vocab_size: Size of the output vocabulary (padded to a multiple of 128).
        emb_dim: Dimensionality of the model (hidden size).
        num_heads: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        head_dim: Dimensionality per attention head.
        mlp_dim: Dimensionality of the feed-forward inner layer.
        mlp_activations: Activations for the feed-forward block; two entries
            means a gated MLP (e.g. ('gelu', 'linear') for gated-gelu).
        dropout_rate: Dropout probability.
        logits_via_embedding: Whether to share decoder embedding and output
            logits weights.
        dtype: Data type for computations — a string (e.g. "float32") or a
            backend dtype; each backend model converts at construction.
        max_positions: Maximum sequence length supported by the fixed
            positional embeddings.
        inputs_length: Number of spectrogram frames per model input segment.
        targets_length: Maximum number of target tokens per segment.
        use_ties: Whether targets use the "tie" representation for notes that
            span segment boundaries.
        onsets_only: Whether targets contain only note onsets.
        vocab_config: Event vocabulary configuration.
        spectrogram_config: Input spectrogram configuration.
    """

    vocab_size: int = 1536
    emb_dim: int = 512
    num_heads: int = 6
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    head_dim: int = 64
    mlp_dim: int = 1024
    mlp_activations: Sequence[str] = ("gelu", "linear")
    dropout_rate: float = 0.1
    logits_via_embedding: bool = False
    dtype: Any = "float32"
    max_positions: int = 2048
    inputs_length: int = 256
    targets_length: int = 1024
    use_ties: bool = True
    onsets_only: bool = False
    vocab_config: VocabularyConfig = field(default_factory=lambda: VocabularyConfig(num_velocity_bins=1))
    spectrogram_config: SpectrogramConfig = field(default_factory=SpectrogramConfig)

    def replace(self, **kwargs) -> "MT3Config":
        """Create a new config with updated values."""
        return replace(self, **kwargs)

    @classmethod
    def from_pretrained(cls, model_type: str = "mt3") -> "MT3Config":
        """Get configuration for a pretrained MT3 checkpoint.

        Args:
            model_type: One of:
                - 'mt3': multi-task multitrack model (ISMIR 2022 paper baseline).
                - 'ismir2021': piano-only model with velocities (ISMIR 2021).
                - 'ismir2022_small': T5.1.1-small multitrack model trained with
                  mixture augmentation.
                - 'ismir2022_base': T5.1.1-base multitrack model trained with
                  mixture augmentation.

        Returns:
            MT3Config for the specified model type.
        """
        if model_type == "ismir2021":
            task = dict(
                inputs_length=512,
                use_ties=False,
                vocab_config=VocabularyConfig(num_velocity_bins=127),
            )
        elif model_type in ("mt3", "ismir2022_small", "ismir2022_base"):
            task = dict(
                inputs_length=256,
                use_ties=True,
                vocab_config=VocabularyConfig(num_velocity_bins=1),
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from: "
                "'mt3', 'ismir2021', 'ismir2022_small', 'ismir2022_base'"
            )

        network = {}
        if model_type == "ismir2022_base":
            # T5.1.1 Base.
            network = dict(
                emb_dim=768,
                num_heads=12,
                num_encoder_layers=12,
                num_decoder_layers=12,
                mlp_dim=2048,
            )

        vocab_size = num_embeddings(vocabulary_from_codec(build_codec(task["vocab_config"])))
        return cls(vocab_size=vocab_size, **task, **network)
