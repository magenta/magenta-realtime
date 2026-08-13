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

"""MusicCoCa for the mlx backend.

This re-exports :mod:`magenta_rt.mlx_pure.musiccoca`. Both backends run
on the same MLX runtime; ``sequence_layers.mlx`` exists for *streaming*
sequence models (the depthformer / codec stack), and MusicCoCa is a
stateless embedder, so an sl-config reimplementation would duplicate the
pure-MLX modules without changing a single computed value. If sl ever
grows a reason to host MusicCoCa natively (e.g. shared sharding or
export tooling), this module is the seam where that would land.

    from magenta_rt.mlx import musiccoca
    style_model = musiccoca.MusicCoCa()
    tokens = style_model.tokenize(style_model.embed_text('staccato funk'))
"""

from ..mlx_pure.musiccoca import (  # noqa: F401
    AttentionPooler,
    AudioEncoder,
    Einsum,
    EmbeddingQuantizer,
    LayerNorm,
    LogMelFrontend,
    Mapper,
    MapperLayer,
    MusicCoCa,
    MusicCoCaModule,
    TextEncoder,
    TransformerLayer,
    encode_text,
    from_safetensors,
    load_safetensors,
)

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
