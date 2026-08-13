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

"""Tests for the piano-roll conditioning rasterizer."""

from __future__ import annotations

import dataclasses

import numpy as np

from magenta_rt import config as _cfg
from magenta_rt.sft.pianoroll import (
    PITCH_ON,
    PITCH_ONSET,
    notes_to_pianoroll,
    transcription_to_channels,
)


@dataclasses.dataclass
class _Note:
    pitch: int
    start_time: float
    end_time: float
    is_drum: bool = False


@dataclasses.dataclass
class _Transcription:
    notes: list


def test_onset_and_sustain_frames():
    # 25 Hz frames are 40 ms; a note from 0.2s to 0.4s spans frames 5..9.
    notes = [_Note(pitch=60, start_time=0.2, end_time=0.4)]
    roll = notes_to_pianoroll(notes, num_frames=25)
    assert roll[5, 60] == PITCH_ONSET
    np.testing.assert_array_equal(roll[6:10, 60], PITCH_ON)
    assert roll[4, 60] == 0 and roll[10, 60] == 0
    assert roll[:, :60].sum() == 0 and roll[:, 61:].sum() == 0


def test_short_note_keeps_onset_frame():
    notes = [_Note(pitch=72, start_time=1.0, end_time=1.001)]
    roll = notes_to_pianoroll(notes, num_frames=50)
    assert roll[25, 72] == PITCH_ONSET
    assert (roll[:, 72] != 0).sum() == 1


def test_rearticulation_marks_second_onset():
    notes = [
        _Note(pitch=60, start_time=0.0, end_time=1.0),
        _Note(pitch=60, start_time=0.4, end_time=0.8),
    ]
    roll = notes_to_pianoroll(notes, num_frames=30)
    assert roll[0, 60] == PITCH_ONSET
    assert roll[10, 60] == PITCH_ONSET  # second attack survives the overlap
    np.testing.assert_array_equal(roll[1:10, 60], PITCH_ON)


def test_drum_notes_never_touch_the_pitched_roll():
    # Drums are skipped entirely (no drum channel is synthesized).
    notes = [_Note(pitch=38, start_time=0.5, end_time=0.6, is_drum=True)]
    roll = notes_to_pianoroll(notes, num_frames=25)
    assert roll.sum() == 0


def test_out_of_range_clipped_and_skipped():
    notes = [
        _Note(pitch=200, start_time=0.0, end_time=1.0),   # bad pitch
        _Note(pitch=60, start_time=99.0, end_time=100.0),  # past the window
        _Note(pitch=64, start_time=0.9, end_time=10.0),    # clipped at end
    ]
    roll = notes_to_pianoroll(notes, num_frames=25)
    assert (roll[:, 60] != 0).sum() == 0
    assert roll[22, 64] == PITCH_ONSET
    np.testing.assert_array_equal(roll[23:, 64], PITCH_ON)


def test_transcription_to_channels_schema():
    t = _Transcription(notes=[
        _Note(pitch=60, start_time=0.0, end_time=0.5),
        _Note(pitch=36, start_time=0.0, end_time=0.1, is_drum=True),
    ])
    channels = transcription_to_channels(t, num_frames=50)
    roll = channels[_cfg.PIANOROLL_WITH_ONSETS.key]
    assert roll.shape == (50, 128) and roll.dtype == np.int8
    assert roll.max() < _cfg.PIANOROLL_WITH_ONSETS.codebook_size
    # The drum channel is intentionally NOT synthesized (intent directive,
    # not an onset raster); training conditions it on the dropout token.
    assert _cfg.DRUM_PIANOROLL.key not in channels
