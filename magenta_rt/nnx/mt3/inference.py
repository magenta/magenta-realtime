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

"""MT3 inference: audio in, NoteSequence (or MIDI) out.

Replicates the original MT3 inference pipeline (seqio task preprocessing +
t5x decoding): audio is split into non-overlapping frames, grouped into
segments of ``config.inputs_length`` frames, converted to log mel
spectrograms, transcribed with greedy autoregressive decoding, and the
per-segment token predictions are stitched into a single NoteSequence.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from magenta_rt.mt3 import note_sequences, vocabularies
from magenta_rt.mt3.decoding import (  # noqa: F401 (re-exports)
    audio_to_frames,
    decode_and_combine_predictions,
    event_predictions_to_ns,
)

from .model import MT3
from .spectrograms import compute_spectrogram


@functools.partial(jax.jit, static_argnums=(0, 3))
def _greedy_decode_loop(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    encoded: jnp.ndarray,
    max_decode_length: int,
) -> jnp.ndarray:
    """Greedy autoregressive decoding with a KV cache.

    Args:
        graphdef: Static graph definition of an MT3 model with an initialized
            decoding cache.
        state: Corresponding model state (params + cache).
        encoded: Encoder output of shape [batch, length, emb_dim].
        max_decode_length: Maximum number of tokens to decode.

    Returns:
        Decoded token ids of shape [batch, max_decode_length]; positions after
        EOS are padded with 0.
    """
    batch_size = encoded.shape[0]
    eos_id = 1  # vocabulary EOS

    def cond_fn(carry):
        _, _, _, i, done = carry
        return (i < max_decode_length) & ~jnp.all(done)

    def body_fn(carry):
        state, cur_tokens, out_tokens, i, done = carry
        model = nnx.merge(graphdef, state)
        logits = model.decode(encoded, cur_tokens[:, None], decode=True, deterministic=True)
        next_tokens = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
        next_tokens = jnp.where(done, 0, next_tokens)
        out_tokens = jax.lax.dynamic_update_slice(out_tokens, next_tokens[:, None], (0, i))
        done = done | (next_tokens == eos_id)
        _, state = nnx.split(model)
        return state, next_tokens, out_tokens, i + 1, done

    init_carry = (
        state,
        jnp.zeros((batch_size,), jnp.int32),  # BOS = 0
        jnp.zeros((batch_size, max_decode_length), jnp.int32),
        jnp.array(0, jnp.int32),
        jnp.zeros((batch_size,), bool),
    )
    _, _, out_tokens, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return out_tokens


def greedy_decode(
    model: MT3, encoded: jnp.ndarray, max_decode_length: Optional[int] = None
) -> np.ndarray:
    """Greedily decode token ids for a batch of encoded segments."""
    if max_decode_length is None:
        max_decode_length = model.config.targets_length
    model.init_cache(encoded.shape[0], max_decode_length)
    graphdef, state = nnx.split(model)
    return np.asarray(_greedy_decode_loop(graphdef, state, encoded, max_decode_length))


def transcribe(
    model: MT3,
    samples: np.ndarray,
    batch_size: int = 8,
) -> note_sequences.NoteSequence:
    """Transcribe an audio waveform to a NoteSequence.

    Args:
        model: Pretrained MT3 model (see ``load_model``).
        samples: Mono audio samples at
            ``model.config.spectrogram_config.sample_rate`` (16 kHz). No
            loudness normalization is applied, matching the original MT3
            inference pipeline, which feeds librosa-loaded float audio in
            [-1, 1] directly to the spectrogram. Note that the ismir2022
            checkpoints were trained on peak-normalized mixtures, so
            peak-normalizing to ~1.0 is reasonable for those.
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
    frames, frame_times = audio_to_frames(samples, spectrogram_config)
    num_segments = -(-len(frames) // cfg.inputs_length)  # ceil
    pad_frames = num_segments * cfg.inputs_length - len(frames)
    frames = np.pad(frames, [(0, pad_frames), (0, 0)])
    segments = frames.reshape(num_segments, cfg.inputs_length * spectrogram_config.hop_width)
    start_times = np.arange(num_segments) * cfg.inputs_length / spectrogram_config.frames_per_second
    # Round down to the nearest symbolic token step.
    start_times -= start_times % (1 / codec.steps_per_second)

    # Transcribe each batch of segments; pad the last batch to keep a single
    # jit-compiled shape.
    predictions = []
    for batch_start in range(0, num_segments, batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        num_valid = len(batch)
        if num_valid < batch_size:
            batch = np.pad(batch, [(0, batch_size - num_valid), (0, 0)])
        spectrograms = compute_spectrogram(jnp.asarray(batch), spectrogram_config)
        encoded = model.encode(spectrograms, deterministic=True)
        tokens = greedy_decode(model, encoded)
        for i in range(num_valid):
            est_tokens = vocabularies.trim_eos(vocabulary.decode(tokens[i]))
            predictions.append(
                {"est_tokens": est_tokens, "start_time": start_times[batch_start + i]}
            )

    result = event_predictions_to_ns(predictions, codec=codec, encoding_spec=encoding_spec)
    return result["est_ns"]
