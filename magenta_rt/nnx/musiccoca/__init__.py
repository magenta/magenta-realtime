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

"""Pure flax.nnx MusicCoCa (reverse engineered from the TFLite exports).

Convert the TFLite weights once, then use the nnx model:

    python -m magenta_rt.nnx.musiccoca.convert

    from magenta_rt.nnx.musiccoca import MusicCoCa
    style_model = MusicCoCa()
    embedding = style_model.embed_text('staccato funk')
    tokens = style_model.tokenize(embedding)
"""

from .frontend import LogMelFrontend
from .mapper import Mapper, MapperLayer
from .model import (
    MusicCoCa,
    MusicCoCaModule,
    encode_text,
    from_safetensors,
    load_safetensors,
)
from .modules import (
    AttentionPooler,
    AudioEncoder,
    Einsum,
    LayerNorm,
    TextEncoder,
    TransformerLayer,
)
from .quantizer import EmbeddingQuantizer

__all__ = [
    "MusicCoCa",
    "MusicCoCaModule",
    "LogMelFrontend",
    "AudioEncoder",
    "TextEncoder",
    "TransformerLayer",
    "AttentionPooler",
    "LayerNorm",
    "Einsum",
    "EmbeddingQuantizer",
    "Mapper",
    "MapperLayer",
    "encode_text",
    "from_safetensors",
    "load_safetensors",
]
