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

"""Framework-neutral MT3 decoding glue.

Splitting audio into model frames and stitching per-segment token
predictions into one ``NoteSequence`` — shared by the backend
``transcribe`` implementations.

Ported from https://github.com/magenta/mt3 (metrics_utils.py and the
audio-framing preprocessing).
"""

import functools
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

from . import note_sequences
from . import run_length_encoding
from .spectrograms import SpectrogramConfig, split_audio

S = TypeVar("S")
T = TypeVar("T")


def audio_to_frames(
    samples: np.ndarray, spectrogram_config: SpectrogramConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert audio samples to non-overlapping frames and frame times.

    Args:
        samples: Mono audio samples at ``spectrogram_config.sample_rate``.

    Returns:
        frames: [num_frames, hop_width] array of audio samples.
        times: [num_frames] array of frame times in seconds.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio of shape [num_samples], got {samples.shape}")
    frames = split_audio(samples, spectrogram_config)
    times = np.arange(len(frames)) / spectrogram_config.frames_per_second
    return frames, times



def decode_and_combine_predictions(
    predictions: Sequence[Mapping[str, Any]],
    init_state_fn: Callable[[], S],
    begin_segment_fn: Callable[[S], None],
    decode_tokens_fn: Callable[[S, Sequence[int], float, Optional[float]], Tuple[int, int]],
    flush_state_fn: Callable[[S], T],
) -> Tuple[T, int, int]:
    """Decode and combine a sequence of predictions to a full result.

    Args:
        predictions: List of predictions, each of which is a dictionary
            containing estimated tokens ('est_tokens') and start time
            ('start_time') fields.
        init_state_fn: Function that takes no arguments and returns an initial
            decoding state.
        begin_segment_fn: Function that updates the decoding state at the
            beginning of a segment.
        decode_tokens_fn: Function that takes a decoding state, estimated
            tokens (for a single segment), start time, and max time, and
            processes the tokens, updating the decoding state in place. Also
            returns the number of invalid and dropped events for the segment.
        flush_state_fn: Function that flushes the final decoding state into the
            result.

    Returns:
        result: The full combined decoding.
        total_invalid_events: Total number of invalid event tokens across all
            predictions.
        total_dropped_events: Total number of dropped event tokens across all
            predictions.
    """
    sorted_predictions = sorted(predictions, key=lambda pred: pred["start_time"])

    state = init_state_fn()
    total_invalid_events = 0
    total_dropped_events = 0

    for pred_idx, pred in enumerate(sorted_predictions):
        begin_segment_fn(state)

        # Depending on the audio token hop length, each symbolic token could be
        # associated with multiple audio frames. Since we split up the audio
        # frames into segments for prediction, this could lead to overlap. To
        # prevent overlap issues, ensure that the current segment does not make
        # any predictions for the time period covered by the subsequent segment.
        max_decode_time = None
        if pred_idx < len(sorted_predictions) - 1:
            max_decode_time = sorted_predictions[pred_idx + 1]["start_time"]

        invalid_events, dropped_events = decode_tokens_fn(
            state, pred["est_tokens"], pred["start_time"], max_decode_time
        )

        total_invalid_events += invalid_events
        total_dropped_events += dropped_events

    return flush_state_fn(state), total_invalid_events, total_dropped_events


def event_predictions_to_ns(
    predictions: Sequence[Mapping[str, Any]],
    codec,
    encoding_spec: note_sequences.NoteEncodingSpecType,
) -> Mapping[str, Any]:
    """Convert a sequence of predictions to a combined NoteSequence."""
    ns, total_invalid_events, total_dropped_events = decode_and_combine_predictions(
        predictions=predictions,
        init_state_fn=encoding_spec.init_decoding_state_fn,
        begin_segment_fn=encoding_spec.begin_decoding_segment_fn,
        decode_tokens_fn=functools.partial(
            run_length_encoding.decode_events,
            codec=codec,
            decode_event_fn=encoding_spec.decode_event_fn,
        ),
        flush_state_fn=encoding_spec.flush_decoding_state_fn,
    )

    return {
        "est_ns": ns,
        "est_invalid_events": total_invalid_events,
        "est_dropped_events": total_dropped_events,
    }
