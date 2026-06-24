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

"""Parity tests for ``magenta_rt.nnx.signal`` vs numpy / scipy reference.

Covers:
* ``hann_window`` / ``inverse_stft_window_fn`` — COLA correctness.
* ``frame`` / ``overlap_and_add`` — numpy-reference equality (time-last,
  arbitrary leading axes) and roundtrip identity at hop=length.
* ``STFT`` / ``InverseSTFT`` — full-seq roundtrip recovery (audio is
  ``[B, C, T]``; the spec contract stays ``[B, F, freqs, C]``).
* ``InverseSTFT.step`` — concatenated streaming chunks match the
  full-sequence forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from magenta_rt.nnx.signal import (
    STFT, InverseSTFT, frame, hann_window, inverse_stft_window_fn,
    overlap_and_add,
)

from .conftest import assert_close


def test_hann_window_periodic_matches_numpy_recipe():
    w = hann_window(10, periodic=True)
    expected = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(10) / 10)
    np.testing.assert_allclose(w, expected, atol=1e-7)


def test_inverse_stft_window_fn_cola(rng_key):
    """``inverse_stft_window_fn(hop, hann)`` makes the COLA sum constant 1.

    Griffin & Lim eq. 7: when the synthesis window is constructed via
    ``inverse_stft_window_fn(hop, fw)``, the staggered overlap-add of
    ``sw * fw`` equals 1 at every sample in the steady-state region.
    """
    fl, fs = 16, 8
    fw = hann_window(fl)
    sw = inverse_stft_window_fn(fs)(fl)
    # Plenty of frames so the steady-state region is well into the
    # buffer (boundary effects only at the first / last frame_length).
    n_frames = 6
    total = fs * (n_frames - 1) + fl  # final sample emitted
    overlap_sum = np.zeros(total)
    for k in range(n_frames):
        shift = k * fs
        overlap_sum[shift: shift + fl] += fw * sw
    # Steady state: all samples after the first frame and before the last.
    np.testing.assert_allclose(overlap_sum[fl: total - fl], 1.0, atol=1e-6)


def test_frame_matches_numpy_reference(rng_key):
    """``frame`` slides along the last axis with arbitrary leading axes."""
    x = jax.random.normal(rng_key, (2, 3, 50))
    fl, fs = 9, 4
    framed = frame(x, frame_length=fl, frame_step=fs)
    nf = (50 - fl) // fs + 1
    expected = np.stack(
        [np.asarray(x)[..., i * fs:i * fs + fl] for i in range(nf)], axis=-2
    )
    assert framed.shape == (2, 3, nf, fl)
    np.testing.assert_allclose(framed, expected, atol=0)


def test_overlap_and_add_matches_numpy_reference(rng_key):
    """``overlap_and_add`` accumulates overlapping frames on the last axis."""
    F, L, fs = 5, 8, 3
    framed = jax.random.normal(rng_key, (2, 3, F, L))
    out = overlap_and_add(framed, frame_step=fs)
    T = (F - 1) * fs + L
    expected = np.zeros((2, 3, T), dtype=np.float64)
    f = np.asarray(framed)
    for i in range(F):
        expected[..., i * fs:i * fs + L] += f[..., i, :]
    assert out.shape == (2, 3, T)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_frame_overlap_add_roundtrip_at_hop_equals_length(rng_key):
    """When hop == frame_length, overlap_and_add(frame(x)) == x."""
    x = jax.random.normal(rng_key, (1, 1, 64))  # [B, C, T]
    framed = frame(x, frame_length=8, frame_step=8)
    recon = overlap_and_add(framed, frame_step=8)
    assert_close(recon, x, atol=1e-6, rtol=1e-6, name="frame_OLA_roundtrip")


def test_stft_inverse_stft_roundtrip(rng_key):
    """STFT → InverseSTFT recovers the original signal under COLA."""
    fl, fs, fft = 16, 8, 16
    fwd = STFT(frame_length=fl, frame_step=fs, fft_length=fft, time_padding="reverse_causal_valid")
    inv = InverseSTFT(
        frame_length=fl, frame_step=fs, fft_length=fft,
        window_fn=inverse_stft_window_fn(fs),
        time_padding="causal",
    )

    x = jax.random.normal(rng_key, (1, 1, 64))  # [B, C, T]
    spec = fwd(x)
    assert spec.shape[-1] == 1  # spec contract: [B, F, freqs, C]
    recon = inv(spec)  # [B, C, T]
    # Compare middle region (boundaries differ from the original under causal trim).
    cut = fl
    assert_close(recon[..., cut:cut + 32], x[..., cut:cut + 32],
                 atol=1e-4, rtol=1e-4, name="stft_istft_roundtrip")


def test_streaming_inverse_stft_matches_full(rng_key):
    """Concatenated ``InverseSTFT.step`` chunks bit-equal a single
    non-streaming forward on the joined spectrogram (within bf16/fp32
    overlap-add tolerance)."""
    fl, fs, fft = 16, 8, 16
    inv_full = InverseSTFT(
        frame_length=fl, frame_step=fs, fft_length=fft,
        window_fn=inverse_stft_window_fn(fs),
        time_padding="causal",
    )
    inv_stream = InverseSTFT(
        frame_length=fl, frame_step=fs, fft_length=fft,
        window_fn=inverse_stft_window_fn(fs),
        time_padding="causal",
    )

    # Random spectrogram (complex), in the [B, T, freqs, C] step contract.
    rng_real, rng_imag = jax.random.split(rng_key, 2)
    spec = (jax.random.normal(rng_real, (1, 6, fft // 2 + 1, 1))
            + 1j * jax.random.normal(rng_imag, (1, 6, fft // 2 + 1, 1)))

    # Full forward ([B, C, T] audio out).
    full_out = inv_full(spec)

    # Streaming forward in chunks of 1 frame; audio concatenates time-last.
    inv_stream.init_cache(batch=1, num_channels=1)
    chunks = [inv_stream.step(spec[:, t:t + 1]) for t in range(spec.shape[1])]
    stream_out = jnp.concatenate(chunks, axis=-1)

    # The streaming output emits frame_step samples per frame; total
    # length = num_frames * frame_step. Compare against the leading
    # num_frames * frame_step samples of the non-streaming output (which
    # has had ``frame_length - frame_step`` samples trimmed from the
    # right by the causal trim).
    n = stream_out.shape[-1]
    assert_close(stream_out, full_out[..., :n], atol=1e-5, rtol=1e-5,
                 name="streaming_istft_vs_full")
