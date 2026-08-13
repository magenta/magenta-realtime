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

"""Model vocabulary.

Ported from https://github.com/magenta/mt3 (vocabularies.py), with the
seqio/TensorFlow vocabulary replaced by a NumPy implementation.
"""

import dataclasses
import math

import numpy as np

from . import event_codec

DECODED_EOS_ID = -1
DECODED_INVALID_ID = -2

# MIDI constants (from note_seq)
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127
MAX_MIDI_VELOCITY = 127

# defaults for vocabulary config
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127

# T5 reserves 100 extra ids for sentinel tokens.
DEFAULT_EXTRA_IDS = 100


@dataclasses.dataclass(frozen=True)
class VocabularyConfig:
    """Vocabulary configuration parameters."""

    steps_per_second: int = DEFAULT_STEPS_PER_SECOND
    max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
    num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS


def num_velocity_bins_from_codec(codec: event_codec.Codec) -> int:
    """Get number of velocity bins from event codec."""
    lo, hi = codec.event_type_range("velocity")
    return hi - lo


def velocity_to_bin(velocity: int, num_velocity_bins: int) -> int:
    if velocity == 0:
        return 0
    else:
        return math.ceil(num_velocity_bins * velocity / MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin: int, num_velocity_bins: int) -> int:
    if velocity_bin == 0:
        return 0
    else:
        return int(MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


def build_codec(vocab_config: VocabularyConfig) -> event_codec.Codec:
    """Build event codec."""
    event_ranges = [
        event_codec.EventRange("pitch", MIN_MIDI_PITCH, MAX_MIDI_PITCH),
        # velocity bin 0 is used for note-off
        event_codec.EventRange("velocity", 0, vocab_config.num_velocity_bins),
        # used to indicate that a pitch is present at the beginning of a segment
        # (only has an "off" event as when using ties all pitch events until the
        # "tie" event belong to the tie section)
        event_codec.EventRange("tie", 0, 0),
        event_codec.EventRange("program", MIN_MIDI_PROGRAM, MAX_MIDI_PROGRAM),
        event_codec.EventRange("drum", MIN_MIDI_PITCH, MAX_MIDI_PITCH),
    ]

    return event_codec.Codec(
        max_shift_steps=(vocab_config.steps_per_second * vocab_config.max_shift_seconds),
        steps_per_second=vocab_config.steps_per_second,
        event_ranges=event_ranges,
    )


class GenericTokenVocabulary:
    """Vocabulary with pass-through encoding of tokens.

    The special tokens are 0=PAD, 1=EOS, and 2=UNK; event codec indices are
    shifted up by `num_special_tokens` to make room for them.
    """

    def __init__(self, regular_ids: int, extra_ids: int = DEFAULT_EXTRA_IDS):
        self.num_special_tokens = 3
        self.num_regular_tokens = regular_ids
        self.extra_ids = extra_ids

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def unk_id(self) -> int:
        return 2

    @property
    def base_vocab_size(self) -> int:
        return self.num_special_tokens + self.num_regular_tokens

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + self.extra_ids

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Encode event codec indices as vocabulary token ids."""
        token_ids = np.asarray(token_ids)
        if np.any(token_ids < 0) or np.any(token_ids >= self.num_regular_tokens):
            raise ValueError(
                f"token ids do not fall within valid range of [0, {self.num_regular_tokens})"
            )
        return token_ids + self.num_special_tokens

    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Decode vocabulary token ids to event codec indices.

        PAD, UNK, and extra ids are replaced with DECODED_INVALID_ID. If EOS is
        present, it and all subsequent tokens are replaced with DECODED_EOS_ID.

        Args:
            ids: 1D array of token ids.

        Returns:
            1D array of event codec indices.
        """
        ids = np.asarray(ids, dtype=np.int32)
        eos_and_after = np.cumsum(ids == self.eos_id, axis=-1).astype(bool)
        valid = (ids >= self.num_special_tokens) & (ids < self.base_vocab_size)
        return np.where(
            eos_and_after,
            DECODED_EOS_ID,
            np.where(valid, ids - self.num_special_tokens, DECODED_INVALID_ID),
        )


def vocabulary_from_codec(codec: event_codec.Codec) -> GenericTokenVocabulary:
    return GenericTokenVocabulary(codec.num_classes, extra_ids=DEFAULT_EXTRA_IDS)


def num_embeddings(vocabulary: GenericTokenVocabulary) -> int:
    """Vocabulary size as a multiple of 128 for TPU efficiency."""
    return 128 * math.ceil(vocabulary.vocab_size / 128)


def trim_eos(tokens: np.ndarray) -> np.ndarray:
    """If decoded EOS is present, remove it and everything after."""
    tokens = np.asarray(tokens, np.int32)
    if DECODED_EOS_ID in tokens:
        tokens = tokens[: np.argmax(tokens == DECODED_EOS_ID)]
    return tokens
