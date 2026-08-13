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

"""MT3: Multi-Task Multitrack Music Transcription (pure MLX).

Port of ``magenta_rt.nnx.mt3`` for inference; shares the
framework-neutral core (event vocabulary, NoteSequence decoding,
checkpoint download, configs) via :mod:`magenta_rt.mt3` and loads the
same converted safetensors.

Example:
    from magenta_rt.mlx_pure.mt3 import load_model, transcribe

    model = load_model("mt3")  # or "ismir2021" for piano-only
    ns = transcribe(model, samples_16khz)
    ns.write_midi("transcription.mid")
"""

from magenta_rt.mt3 import (  # noqa: F401
    event_codec,
    note_sequences,
    run_length_encoding,
    vocabularies,
)
from magenta_rt.mt3.config import MT3Config
from magenta_rt.mt3.note_sequences import Note, NoteSequence
from magenta_rt.mt3.vocabularies import VocabularyConfig

from .inference import greedy_decode, transcribe
from .model import MT3
from .pretrained import load_model
from .spectrograms import SpectrogramConfig

__all__ = [
    "MT3",
    "MT3Config",
    "Note",
    "NoteSequence",
    "SpectrogramConfig",
    "VocabularyConfig",
    "greedy_decode",
    "load_model",
    "transcribe",
]
