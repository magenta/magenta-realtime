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

"""MT3 inference in pure MLX: audio in, NoteSequence (or MIDI) out.

Same pipeline as ``magenta_rt.nnx.mt3.inference`` (segment, spectrogram,
greedy decode, stitch), with an eager decode loop — MLX needs no
compiled ``while_loop``, and the loop exits as soon as every sequence in
the batch hits EOS.
"""

from typing import Optional

import mlx.core as mx
import numpy as np

from magenta_rt.mt3 import note_sequences, vocabularies
from magenta_rt.mt3.decoding import (  # noqa: F401 (re-exports)
    audio_to_frames,
    decode_and_combine_predictions,
    event_predictions_to_ns,
)

from .model import MT3
from .spectrograms import compute_spectrogram


def greedy_decode(
    model: MT3, encoded: mx.array, max_decode_length: Optional[int] = None
) -> np.ndarray:
    """Greedily decode token ids for a batch of encoded segments.

    Returns ``[batch, max_decode_length]`` int32; positions after EOS are
    padded with 0.
    """
    if max_decode_length is None:
        max_decode_length = model.config.targets_length
    batch_size = encoded.shape[0]
    eos_id = 1  # vocabulary EOS

    model.init_cache(batch_size, max_decode_length)
    cur_tokens = mx.zeros((batch_size, 1), dtype=mx.int32)  # BOS = 0
    out_tokens = np.zeros((batch_size, max_decode_length), np.int32)
    done = np.zeros((batch_size,), bool)

    for i in range(max_decode_length):
        logits = model.decode(encoded, cur_tokens, decode=True)
        next_tokens = np.asarray(
            mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        )
        next_tokens[done] = 0
        out_tokens[:, i] = next_tokens
        done |= next_tokens == eos_id
        if done.all():
            break
        cur_tokens = mx.array(next_tokens[:, None])

    return out_tokens


def transcribe(
    model: MT3,
    samples: np.ndarray,
    batch_size: int = 8,
) -> note_sequences.NoteSequence:
    """Transcribe an audio waveform to a NoteSequence.

    Args:
        model: Pretrained MT3 model (see ``load_model``).
        samples: Mono audio samples at
            ``model.config.spectrogram_config.sample_rate`` (16 kHz).
        batch_size: Number of audio segments to decode in parallel.

    Returns:
        The transcribed NoteSequence; use ``.write_midi(path)`` to export it.
    """
    cfg = model.config
    spectrogram_config = cfg.spectrogram_config
    codec = vocabularies.build_codec(cfg.vocab_config)
    vocabulary = vocabularies.vocabulary_from_codec(codec)
    if cfg.onsets_only:
        encoding_spec = note_sequences.NoteOnsetEncodingSpec
    elif cfg.use_ties:
        encoding_spec = note_sequences.NoteEncodingWithTiesSpec
    else:
        encoding_spec = note_sequences.NoteEncodingSpec

    # Split audio into segments of inputs_length frames, padding the end.
    frames, _ = audio_to_frames(samples, spectrogram_config)
    num_segments = -(-len(frames) // cfg.inputs_length)  # ceil
    pad_frames = num_segments * cfg.inputs_length - len(frames)
    frames = np.pad(frames, [(0, pad_frames), (0, 0)])
    segments = frames.reshape(num_segments, cfg.inputs_length * spectrogram_config.hop_width)
    start_times = np.arange(num_segments) * cfg.inputs_length / spectrogram_config.frames_per_second
    # Round down to the nearest symbolic token step.
    start_times -= start_times % (1 / codec.steps_per_second)

    predictions = []
    for batch_start in range(0, num_segments, batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        spectrograms = compute_spectrogram(mx.array(batch), spectrogram_config)
        encoded = model.encode(spectrograms)
        tokens = greedy_decode(model, encoded)
        for i in range(len(batch)):
            est_tokens = vocabularies.trim_eos(vocabulary.decode(tokens[i]))
            predictions.append(
                {"est_tokens": est_tokens, "start_time": start_times[batch_start + i]}
            )

    result = event_predictions_to_ns(predictions, codec=codec, encoding_spec=encoding_spec)
    return result["est_ns"]
