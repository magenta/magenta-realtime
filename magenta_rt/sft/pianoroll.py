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

"""Rasterize note transcriptions into the mrt2 note piano-roll channel.

Maps a transcription (anything with a ``.notes`` list whose items carry
``pitch`` / ``start_time`` / ``end_time`` / ``is_drum`` — e.g. the
``magenta_rt.nnx.mt3`` ``NoteSequence``) onto the 25 Hz conditioning frames
used by the mrt2 source channel:

* ``pianoroll_with_onsets_tokens`` ``[frames, 128]`` int32 — per pitch:
  ``0`` off, ``1`` on (continuation), ``2`` onset (first frame of a note).
  (The value ``3`` — "on, onset or continuation" — is an inference-time
  wildcard, not something the training data contains.)

**The drum channel (``drum_pianoroll_tokens`` ``[frames, 1]``) is deliberately
NOT synthesized here.** Unlike the per-pitch note roll, it is a per-frame
*intent directive*, not an onset raster: ``-1`` = "let the model decide",
``0`` = "don't play drums", ``1`` = "please play drums" (see the comment at
``magenta_rt/mlx/export.py``). Rasterizing MT3 drum onsets into it would mark
``0`` ("suppress drums") on every non-onset frame — i.e. tell the model to
*not* play drums across ~95% of frames of drum-heavy music — which is inverted
conditioning; MT3 also badly under-detects these drums. We have no pipeline
that can label genuine drum presence/intent, so we omit the channel entirely.
At train time ``prepare_source_tokens`` then falls back to the learned
unconditional (dropout) token for it — the same ``-1`` -> "let the model
decide" the generation systems feed by default.
"""

from __future__ import annotations

import numpy as np

FRAME_RATE = 25

PITCH_OFF = 0
PITCH_ON = 1
PITCH_ONSET = 2


def notes_to_pianoroll(
    notes, num_frames: int, frame_rate: float = FRAME_RATE
) -> np.ndarray:
    """Rasterizes pitched notes into the ``pianoroll_with_onsets`` channel.

    Args:
      notes: Iterable of note objects (``pitch``, ``start_time``,
        ``end_time``, ``is_drum``). Times in seconds; pitches outside
        0..127 are skipped. A pitched note occupies the frames its
        ``[start_time, end_time)`` span overlaps (at least its onset
        frame); the onset frame is marked ``2`` and later frames ``1``.
        Drum notes (``is_drum``) are skipped — the drum channel is not
        synthesized here (see the module docstring).
      num_frames: Output length; notes past the end are clipped/skipped.
      frame_rate: Conditioning frame rate (25 Hz for mrt2).

    Returns:
      ``pianoroll [num_frames, 128] int8``. Values are a tiny categorical
      (0=off, 1=on/sustain, 2=onset), so ``int8`` stores it 4× smaller than the
      old ``int32`` with no loss. It is **signed** on purpose: this is a source
      conditioning channel, and ``prepare_source_tokens`` fills a ``-1``
      dropout/mask sentinel before offsetting — an unsigned dtype would wrap
      that to 255 and corrupt the conditioning. The training prepare path casts
      back up to int32 for the embedding lookup, so storage dtype is decoupled
      from compute.
    """
    pianoroll = np.zeros((num_frames, 128), dtype=np.int8)
    # Sort by start time so a later onset overwrites an earlier note's
    # sustain on the shared frame (re-articulation stays visible).
    for note in sorted(notes, key=lambda n: n.start_time):
        if note.is_drum:
            continue
        if not 0 <= note.pitch < 128:
            continue
        onset = int(note.start_time * frame_rate)
        if onset >= num_frames or onset < 0:
            continue
        # Frames overlapped by [start, end), but at least the onset frame.
        end = int(np.ceil(note.end_time * frame_rate))
        end = max(end, onset + 1)
        end = min(end, num_frames)
        pianoroll[onset + 1 : end, note.pitch] = np.maximum(
            pianoroll[onset + 1 : end, note.pitch], PITCH_ON
        )
        pianoroll[onset, note.pitch] = PITCH_ONSET
    return pianoroll


def transcription_to_channels(
    transcription, num_frames: int, frame_rate: float = FRAME_RATE
) -> dict[str, np.ndarray]:
    """Builds the mrt2 conditioning-channel dict from a transcription.

    Only the note piano-roll is produced; the drum channel is intentionally
    omitted (see the module docstring) and conditioned on the unconditional
    token at train time.
    """
    from magenta_rt import config as _cfg

    pianoroll = notes_to_pianoroll(transcription.notes, num_frames, frame_rate)
    return {_cfg.PIANOROLL_WITH_ONSETS.key: pianoroll}
