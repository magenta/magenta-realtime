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

"""SpectroStream codec subpackage.

Re-exports the codec module surface as the public
``magenta_rt.mlx_pure.spectrostream`` API. The weight-loading helpers
live in :mod:`magenta_rt.mlx_pure.spectrostream.load_weights` and are
re-exported here as well so callers can write
``from magenta_rt.mlx_pure.spectrostream import load_spectrostream_weights``.
"""

from .model import (
    Conv2DResidualUnit,
    ResidualVectorQuantizer,
    SpectroStream,
    SpectroStreamDecoder,
    SpectroStreamEncoder,
    SpectroStreamInverseSTFT,
    SpectroStreamSTFT,
)
from .load_weights import (
    load_quantizer_weights,
    load_spectrostream_weights,
    load_spectrostream_decoder_weights,
    load_spectrostream_encoder_weights,
)

__all__ = [
    "Conv2DResidualUnit",
    "ResidualVectorQuantizer",
    "SpectroStream",
    "SpectroStreamDecoder",
    "SpectroStreamEncoder",
    "SpectroStreamInverseSTFT",
    "SpectroStreamSTFT",
    "load_quantizer_weights",
    "load_spectrostream_weights",
    "load_spectrostream_decoder_weights",
    "load_spectrostream_encoder_weights",
]
