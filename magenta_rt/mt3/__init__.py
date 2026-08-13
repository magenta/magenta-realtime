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

"""Framework-neutral MT3 core.

The event vocabulary, run-length decoding, ``NoteSequence`` container,
checkpoint download/conversion, model/spectrogram configuration, and the
numpy spectrogram helpers shared by the backend MT3 implementations
(``magenta_rt.nnx.mt3``, ``magenta_rt.mlx_pure.mt3``).
"""

from .config import MT3Config
from .note_sequences import Note, NoteSequence
from .spectrograms import SpectrogramConfig
from .vocabularies import VocabularyConfig

__all__ = [
    "MT3Config",
    "Note",
    "NoteSequence",
    "SpectrogramConfig",
    "VocabularyConfig",
]
