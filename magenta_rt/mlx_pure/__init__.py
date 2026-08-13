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

"""Pure-MLX implementation of Magenta-RT inference modules.

No runtime dependence on `sequence_layers.mlx`. Modeled on `mlx-lm`
conventions: plain `nn.Module` subclasses, mlx-lm-style cache objects
threaded through `__call__`, no `Sequence`/`MaskedSequence` wrapper.

See ``README.md`` for the public API surface, parity claims, and
remaining-work checklist.

The most common entry points are re-exported here for convenience; for
the full set, import directly from the submodule (e.g., ``from
magenta_rt.mlx_pure.attention import LocalSelfAttention``).
"""

# Cache primitives
from .cache import KVCache, LocalKVCache, OverlapAddCache

# Leaf layers (Dense + EinsumDense only; mlx.nn ships the rest)
from .layers import Dense, EinsumDense

# Attention
from .attention import LocalSelfAttention, StreamingCrossAttention

# Transformer
from .transformer import (
    TransformerBlock, Transformer, MultiChannelEmbedding, Encoder,
)

# Depthformer
from .depthformer import (
    DepthformerDecoder, EncoderDecoder, SamplerState, TemporalCaches,
)

# SpectroStream codec
from .spectrostream import (
    ResidualVectorQuantizer, SpectroStreamEncoder, SpectroStreamDecoder,
    SpectroStream, Conv2DResidualUnit,
)

# DSP
from .signal import (
    STFT, InverseSTFT, hann_window, inverse_stft_window_fn,
    frame, overlap_and_add,
)

# Conv
from .conv import (
    Conv2D, Conv2DTranspose, AveragePooling2D, Upsample2D, ParallelChannels,
)

# Sampling
from .sample_utils import sample_categorical_with_temperature

# Model orchestrator
from .model import MagentaRT2Sampler

# Specifications registry
from . import configs

# Quantization
from .quantize import quantize_in_place, gptq_calibrate_and_quantize

# Weight loading
from .load_weights import (
    load_from_safetensors,
    load_sft_depthformer_from_safetensors,
    load_via_bridge,
)

__all__ = [
    # Cache
    "KVCache", "LocalKVCache", "OverlapAddCache",
    # Layers
    "Dense", "EinsumDense",
    # Attention
    "LocalSelfAttention", "StreamingCrossAttention",
    # Transformer
    "TransformerBlock", "Transformer", "MultiChannelEmbedding", "Encoder",
    # Depthformer
    "DepthformerDecoder", "EncoderDecoder", "SamplerState", "TemporalCaches",
    # SpectroStream
    "ResidualVectorQuantizer", "SpectroStreamEncoder", "SpectroStreamDecoder",
    "SpectroStream", "Conv2DResidualUnit",
    # DSP
    "STFT", "InverseSTFT", "hann_window", "inverse_stft_window_fn",
    "frame", "overlap_and_add",
    # Conv
    "Conv2D", "Conv2DTranspose", "AveragePooling2D", "Upsample2D",
    "ParallelChannels",
    # Sampling
    "sample_categorical_with_temperature",
    # Model
    "MagentaRT2Sampler",
    # Quantization, weight loading, export
    "quantize_in_place", "gptq_calibrate_and_quantize",
    "load_from_safetensors", "load_sft_depthformer_from_safetensors",
    "load_via_bridge",
    # Submodules (importable as mlx_pure.configs)
    "configs",
]
