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

"""Tests for the pure-MLX MT3 (magenta_rt.mlx_pure.mt3): network,
spectrogram frontend, and decode-cache correctness."""

import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from magenta_rt.mlx_pure.mt3 import MT3, MT3Config, greedy_decode
from magenta_rt.mlx_pure.mt3.spectrograms import (
    SpectrogramConfig,
    compute_spectrogram,
    linear_to_mel_weight_matrix,
)


@pytest.fixture(scope="module")
def tiny_model():
    config = MT3Config.from_pretrained("mt3").replace(
        num_encoder_layers=2, num_decoder_layers=2, emb_dim=64, num_heads=2,
        head_dim=32, mlp_dim=128,
    )
    model = MT3(config)
    # Random weights (zeros would make decode-cache parity vacuous).
    rng = np.random.RandomState(0)

    def randomize(module):
        params = module.parameters()

        def fill(tree):
            if isinstance(tree, mx.array):
                return mx.array(rng.randn(*tree.shape).astype(np.float32) * 0.02)
            if isinstance(tree, dict):
                return {k: fill(v) for k, v in tree.items()}
            if isinstance(tree, list):
                return [fill(v) for v in tree]
            return tree

        module.update(fill(params))

    randomize(model)
    model.eval()
    return model


def test_forward_shapes(tiny_model):
    cfg = tiny_model.config
    batch, enc_len, dec_len = 2, 32, 16
    x = mx.zeros((batch, enc_len, cfg.spectrogram_config.num_mel_bins))
    decoder_input = mx.zeros((batch, dec_len), dtype=mx.int32)
    decoder_target = mx.ones((batch, dec_len), dtype=mx.int32)
    logits = tiny_model(x, decoder_input, decoder_target)
    assert logits.shape == (batch, dec_len, cfg.vocab_size)


def test_cached_decode_matches_uncached(tiny_model):
    """Autoregressive decoding with KV cache must match full forward pass."""
    cfg = tiny_model.config
    batch, enc_len, dec_len = 2, 32, 6
    rng = np.random.RandomState(0)
    x = mx.array(rng.randn(batch, enc_len, cfg.spectrogram_config.num_mel_bins).astype(np.float32))
    tokens = mx.array(rng.randint(3, 100, (batch, dec_len)).astype(np.int32))
    decoder_input = mx.concatenate(
        [mx.zeros((batch, 1), dtype=mx.int32), tokens[:, :-1]], axis=1
    )

    encoded = tiny_model.encode(x)
    # Full (teacher-forced) pass. All-ones targets avoid padding masks.
    full_logits = tiny_model.decode(
        encoded, decoder_input, decoder_target_tokens=mx.ones_like(tokens)
    )

    # Step-by-step pass with cache.
    tiny_model.init_cache(batch, max_decode_length=dec_len)
    step_logits = []
    for t in range(dec_len):
        step_logits.append(tiny_model.decode(encoded, decoder_input[:, t : t + 1], decode=True))
    step_logits = mx.concatenate(step_logits, axis=1)

    np.testing.assert_allclose(
        np.asarray(step_logits), np.asarray(full_logits), rtol=1e-4, atol=1e-4
    )


def test_greedy_decode_shape(tiny_model):
    cfg = tiny_model.config
    x = mx.zeros((2, 32, cfg.spectrogram_config.num_mel_bins))
    encoded = tiny_model.encode(x)
    tokens = greedy_decode(tiny_model, encoded, max_decode_length=12)
    assert tokens.shape == (2, 12)
    assert tokens.dtype == np.int32


def test_spectrogram_shape():
    config = SpectrogramConfig()
    samples = np.random.RandomState(0).randn(256 * 128).astype(np.float32)
    spec = compute_spectrogram(mx.array(samples), config)
    assert spec.shape == (256, config.num_mel_bins)


# Same TF reference recipe as tests/nnx/test_mt3.py (shared asset path).
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
    asset_path = (
        Path(__file__).parent.parent / "nnx" / "assets" / "tf_logmel_reference.npz"
    )
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

    got = np.asarray(compute_spectrogram(mx.array(samples), config))
    np.testing.assert_allclose(got, reference["logmel"], rtol=1e-3, atol=1e-3)

    ours = linear_to_mel_weight_matrix(
        config.num_mel_bins, 1025, config.sample_rate, 20.0, 8000.0
    )
    np.testing.assert_allclose(ours, reference["mel_matrix"], rtol=1e-2, atol=1e-4)
