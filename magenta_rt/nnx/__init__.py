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

"""Pure flax.nnx implementation of Magenta-RT inference modules.

Inference-only, streaming ``step`` API, bf16 mixed precision. Loads
Linen-style safetensors checkpoints via the weight bridge.

Public symbols are re-exported here for convenience. For the full
surface, import from the submodule directly
(``from magenta_rt.nnx.attention import LocalSelfAttention``).
"""

from .attention import LocalSelfAttention, StreamingCrossAttention
from .cache import LocalKVCache, OverlapAddCache
from .conv import (
    AveragePooling2D, Conv2D, Conv2DTranspose, ParallelChannels, Upsample2D,
    remove_cache,
)
from .depthformer import DepthformerDecoder, EncoderDecoder, DecodeState
from .sample_utils import sample_categorical_with_temperature
from .signal import (
    STFT, InverseSTFT, frame, hann_window, inverse_stft_window_fn,
    overlap_and_add,
)
from .spectrostream import (
    Conv2DResidualUnit, ResidualVectorQuantizer, SpectroStream,
    SpectroStreamDecoder, SpectroStreamEncoder,
)
from .model import MagentaRT2Sampler
from .transformer import (
    Encoder, MultiChannelEmbedding, Transformer, TransformerBlock,
)
from .load_weights import load_from_jax_safetensors

__all__ = [
    # Cache
    "LocalKVCache", "OverlapAddCache",
    # Attention
    "LocalSelfAttention", "StreamingCrossAttention",
    # Transformer
    "TransformerBlock", "Transformer", "MultiChannelEmbedding", "Encoder",
    # Depthformer
    "DepthformerDecoder", "EncoderDecoder", "DecodeState",
    # SpectroStream
    "ResidualVectorQuantizer", "SpectroStreamEncoder", "SpectroStreamDecoder",
    "SpectroStream", "Conv2DResidualUnit",
    # DSP
    "STFT", "InverseSTFT", "hann_window", "inverse_stft_window_fn",
    "frame", "overlap_and_add",
    # Conv
    "Conv2D", "Conv2DTranspose", "AveragePooling2D", "Upsample2D",
    "ParallelChannels", "remove_cache",
    # Sampling
    "sample_categorical_with_temperature",
    # Model
    "MagentaRT2Sampler",
    # Weight loading
    "load_from_jax_safetensors",
]
