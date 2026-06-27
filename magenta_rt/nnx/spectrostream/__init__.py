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

The codec module surface (RVQ, encoder, decoder, top-level
``SpectroStream``) lives in :mod:`.model` and is re-exported here.
Parameter loading logic lives internally within :mod:`.load_weights`.
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

__all__ = [
    "Conv2DResidualUnit",
    "ResidualVectorQuantizer",
    "SpectroStream",
    "SpectroStreamDecoder",
    "SpectroStreamEncoder",
    "SpectroStreamInverseSTFT",
    "SpectroStreamSTFT",
]
