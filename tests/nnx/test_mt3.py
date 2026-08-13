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

"""Tests for the vendored MT3 (magenta_rt.nnx.mt3): network, spectrogram
frontend, and event decoding. Ported from DBraun's JAX monorepo test suite."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from flax import nnx
from jax import numpy as jnp

from magenta_rt.nnx.mt3 import MT3, MT3Config
from magenta_rt.mt3 import event_codec, note_sequences, vocabularies
from magenta_rt.nnx.mt3.inference import (
    audio_to_frames,
    event_predictions_to_ns,
    greedy_decode,
)
from magenta_rt.nnx.mt3.spectrograms import (
    SpectrogramConfig,
    compute_spectrogram,
    linear_to_mel_weight_matrix,
)

Event = event_codec.Event


@pytest.fixture(scope="module")
def tiny_model():
    config = MT3Config.from_pretrained("mt3").replace(
        num_encoder_layers=2, num_decoder_layers=2, emb_dim=64, num_heads=2, head_dim=32, mlp_dim=128
    )
    model = MT3(config, rngs=nnx.Rngs(0))
    model.eval()
    return model


def test_config_vocab_sizes():
    assert MT3Config.from_pretrained("mt3").vocab_size == 1536
    assert MT3Config.from_pretrained("ismir2022_small").vocab_size == 1536
    assert MT3Config.from_pretrained("ismir2021").vocab_size == 1664
    assert MT3Config.from_pretrained("ismir2021").inputs_length == 512
    assert MT3Config.from_pretrained("ismir2022_base").emb_dim == 768


def test_forward_shapes(tiny_model):
    cfg = tiny_model.config
    batch, enc_len, dec_len = 2, 32, 16
    x = jnp.zeros((batch, enc_len, cfg.spectrogram_config.num_mel_bins))
    decoder_input = jnp.zeros((batch, dec_len), jnp.int32)
    decoder_target = jnp.ones((batch, dec_len), jnp.int32)
    logits = tiny_model(x, decoder_input, decoder_target)
    assert logits.shape == (batch, dec_len, cfg.vocab_size)


def test_train_mode_dropout_is_stochastic(tiny_model):
    cfg = tiny_model.config
    x = jnp.ones((1, 32, cfg.spectrogram_config.num_mel_bins))
    decoder_input = jnp.zeros((1, 8), jnp.int32)
    decoder_target = jnp.ones((1, 8), jnp.int32)
    tiny_model.train()
    try:
        out1 = tiny_model(x, decoder_input, decoder_target)
        out2 = tiny_model(x, decoder_input, decoder_target)
        assert not np.allclose(out1, out2)
    finally:
        tiny_model.eval()


def test_cached_decode_matches_uncached(tiny_model):
    """Autoregressive decoding with KV cache must match full forward pass."""
    cfg = tiny_model.config
    batch, enc_len, dec_len = 2, 32, 6
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(batch, enc_len, cfg.spectrogram_config.num_mel_bins), jnp.float32)
    tokens = jnp.asarray(rng.randint(3, 100, (batch, dec_len)), jnp.int32)
    decoder_input = jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), tokens[:, :-1]], axis=1)

    encoded = tiny_model.encode(x)
    # Full (teacher-forced) pass. All-ones targets avoid padding masks.
    full_logits = tiny_model.decode(
        encoded, decoder_input, decoder_target_tokens=jnp.ones_like(tokens)
    )

    # Step-by-step pass with cache.
    tiny_model.init_cache(batch, max_decode_length=dec_len)
    step_logits = []
    for t in range(dec_len):
        step_logits.append(tiny_model.decode(encoded, decoder_input[:, t : t + 1], decode=True))
    step_logits = jnp.concatenate(step_logits, axis=1)

    np.testing.assert_allclose(step_logits, full_logits, rtol=1e-4, atol=1e-4)


def test_greedy_decode_shape(tiny_model):
    cfg = tiny_model.config
    x = jnp.zeros((2, 32, cfg.spectrogram_config.num_mel_bins))
    encoded = tiny_model.encode(x)
    tokens = greedy_decode(tiny_model, encoded, max_decode_length=12)
    assert tokens.shape == (2, 12)
    assert tokens.dtype == np.int32


def test_audio_to_frames():
    config = SpectrogramConfig()
    samples = np.zeros(10000)
    frames, times = audio_to_frames(samples, config)
    assert frames.shape == (79, 128)  # ceil(10000 / 128)
    np.testing.assert_allclose(times[1] - times[0], 128 / 16000)


def test_spectrogram_shape():
    config = SpectrogramConfig()
    samples = np.random.RandomState(0).randn(256 * 128).astype(np.float32)
    spec = compute_spectrogram(jnp.asarray(samples), config)
    assert spec.shape == (256, config.num_mel_bins)


# Computes the reference log mel spectrogram with the original DDSP/mt3
# TensorFlow ops. Run in a subprocess: importing TensorFlow into the pytest
# process deadlocks its threadpool whenever torch is loaded alongside it
# (e.g. by the t5 test modules).
_TF_REFERENCE_SCRIPT = """
import sys
import numpy as np
import tensorflow as tf

rng = np.random.RandomState(0)
samples = (rng.randn(256 * 128) * 0.1).astype(np.float32)

s = tf.signal.stft(samples, frame_length=2048, frame_step=128, pad_end=True)
mag = tf.abs(s)
mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    512, int(mag.shape[-1]), 16000, 20.0, 8000.0
)
mel = tf.tensordot(mag, mel_matrix, 1)
logmel = tf.math.log(tf.where(mel <= 0.0, 1e-5, mel)).numpy()

np.savez_compressed(sys.argv[1], logmel=logmel, mel_matrix=mel_matrix.numpy())
"""


def test_logmel_matches_tf_reference():
    """Compare against the original DDSP/mt3 TensorFlow implementation."""
    asset_path = Path(__file__).parent / "assets" / "tf_logmel_reference.npz"
    if not asset_path.exists():
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, "-c", _TF_REFERENCE_SCRIPT, str(asset_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"could not generate TF reference: {result.stderr.strip()[-500:]}")

    config = SpectrogramConfig()
    rng = np.random.RandomState(0)
    samples = (rng.randn(256 * 128) * 0.1).astype(np.float32)

    reference = np.load(asset_path)

    got = np.asarray(compute_spectrogram(jnp.asarray(samples), config))
    # TF computes the mel filterbank with float32 Eigen kernels whose log()
    # differs from NumPy by ~1 ulp, which bounds the achievable agreement.
    np.testing.assert_allclose(got, reference["logmel"], rtol=1e-3, atol=1e-3)

    ours = linear_to_mel_weight_matrix(
        config.num_mel_bins, 1025, config.sample_rate, 20.0, 8000.0
    )
    np.testing.assert_allclose(ours, reference["mel_matrix"], rtol=1e-2, atol=1e-4)


def test_codec_round_trip():
    codec = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))
    assert codec.num_classes == 1388
    for event in [
        Event("pitch", 60),
        Event("velocity", 1),
        Event("shift", 100),
        Event("tie", 0),
        Event("program", 32),
        Event("drum", 38),
    ]:
        index = codec.encode_event(event)
        assert codec.decode_event_index(index) == event


def test_vocabulary_decode():
    vocab = vocabularies.GenericTokenVocabulary(100)
    np.testing.assert_array_equal(
        vocab.decode(np.array([3, 4, 102, 0, 2, 1, 5])),
        # 3->0, 4->1, 102->99, PAD->invalid, UNK->invalid, then EOS and after
        [0, 1, 99, -2, -2, -1, -1],
    )
    np.testing.assert_array_equal(vocab.encode(np.array([0, 1, 99])), [3, 4, 102])


def test_event_predictions_to_ns():
    """Decode two segments of events (with ties) into a NoteSequence."""
    vocab_config = vocabularies.VocabularyConfig(num_velocity_bins=1)
    codec = vocabularies.build_codec(vocab_config)

    def ev(type_, value):
        return codec.encode_event(Event(type_, value))

    # Segment 1 (t=0): program 0, velocity 1, pitch 60 onset at 0.0s;
    # tie section is empty.
    seg1 = [
        ev("tie", 0),
        ev("program", 0),
        ev("velocity", 1),
        ev("pitch", 60),
        ev("shift", 100),  # 1.0s
        ev("velocity", 1),
        ev("pitch", 64),
    ]
    # Segment 2 (t=2.048s): pitch 60 and 64 still active and tied; pitch 60
    # off at 2.048 + 0.5s, pitch 64 off at end of segment flush.
    seg2 = [
        ev("program", 0),
        ev("pitch", 60),
        ev("pitch", 64),
        ev("tie", 0),
        ev("shift", 50),  # 0.5s
        ev("velocity", 0),
        ev("pitch", 60),
    ]
    predictions = [
        {"est_tokens": np.array(seg1), "start_time": 0.0},
        {"est_tokens": np.array(seg2), "start_time": 2.048},
    ]
    result = event_predictions_to_ns(
        predictions, codec=codec, encoding_spec=note_sequences.NoteEncodingWithTiesSpec
    )
    assert result["est_invalid_events"] == 0
    assert result["est_dropped_events"] == 0
    ns = result["est_ns"]
    notes = sorted(ns.notes, key=lambda n: n.start_time)
    assert len(notes) == 2
    assert notes[0].pitch == 60
    assert notes[0].start_time == 0.0
    np.testing.assert_allclose(notes[0].end_time, 2.048 + 0.5)
    assert notes[1].pitch == 64
    assert notes[1].start_time == 1.0
    assert notes[1].velocity == 127


def test_note_sequence_midi_export(tmp_path):
    pretty_midi = pytest.importorskip("pretty_midi")

    ns = note_sequences.NoteSequence()
    ns.add_note(start_time=0.0, end_time=1.0, pitch=60, velocity=100)
    ns.add_note(start_time=0.5, end_time=1.5, pitch=38, velocity=80, is_drum=True, instrument=9)
    ns.total_time = 1.5

    midi_path = tmp_path / "test.mid"
    ns.write_midi(str(midi_path))

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    assert len(pm.instruments) == 2
    pitches = sorted(note.pitch for inst in pm.instruments for note in inst.notes)
    assert pitches == [38, 60]
