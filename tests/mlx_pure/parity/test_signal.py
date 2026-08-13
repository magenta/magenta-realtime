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

"""STFT / InverseSTFT smoke + roundtrip tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from magenta_rt.mlx_pure.signal import (
    STFT,
    InverseSTFT,
    frame,
    hann_window,
    inverse_stft_window_fn,
    overlap_and_add,
)


def test_hann_window_periodic():
    w = hann_window(8, periodic=True)
    # Periodic Hann is one bin shorter than a symmetric Hann; max should be ≈ 1.
    assert abs(w.max() - 1.0) < 1e-6
    assert w[0] == 0.0


def test_inverse_window_satisfies_cola():
    """For 50% overlap with Hann analysis, the synthesis window times the
    analysis window at every hop should sum to a constant on the inner
    region (Constant Overlap-Add, COLA).
    """
    frame_length = 32
    frame_step = 16
    forward = hann_window(frame_length)
    inverse = inverse_stft_window_fn(frame_step, hann_window)(frame_length)
    prod = forward * inverse
    # The COLA sum at each output sample should be 1 for the inner region.
    sum_buf = np.zeros(frame_length * 4)
    for i in range(0, len(sum_buf) - frame_length + 1, frame_step):
        sum_buf[i : i + frame_length] += prod
    inner = sum_buf[frame_length : -frame_length]
    assert np.allclose(inner, 1.0, atol=1e-5), inner


def test_frame_overlap_roundtrip():
    """frame() then overlap_and_add() with non-overlapping frames is identity."""
    B, C, T = 1, 2, 16  # time-last
    x = mx.random.normal((B, C, T))
    framed = frame(x, frame_length=4, frame_step=4)
    assert framed.shape == (B, C, 4, 4)
    rt = overlap_and_add(framed, frame_step=4)
    assert rt.shape == (B, C, T)
    assert mx.allclose(x, rt, atol=1e-6).item()


def test_frame_matches_numpy_reference():
    """``frame`` slides along the last axis with arbitrary leading axes."""
    x = mx.random.normal((2, 3, 50))
    fl, fs = 9, 4
    framed = frame(x, frame_length=fl, frame_step=fs)
    nf = (50 - fl) // fs + 1
    expected = np.stack(
        [np.asarray(x)[..., i * fs:i * fs + fl] for i in range(nf)], axis=-2
    )
    assert framed.shape == (2, 3, nf, fl)
    np.testing.assert_allclose(np.asarray(framed), expected, atol=0)


def test_overlap_and_add_matches_numpy_reference():
    """``overlap_and_add`` accumulates overlapping frames on the last axis."""
    F, L, fs = 5, 8, 3
    framed = mx.random.normal((2, 3, F, L))
    out = overlap_and_add(framed, frame_step=fs)
    T = (F - 1) * fs + L
    expected = np.zeros((2, 3, T), dtype=np.float64)
    f = np.asarray(framed)
    for i in range(F):
        expected[..., i * fs:i * fs + L] += f[..., i, :]
    assert out.shape == (2, 3, T)
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5)


def test_stft_inverse_stft_roundtrip():
    """STFT → InverseSTFT recovers the original signal up to boundary
    effects, when using COLA-compatible window/hop and ``valid`` padding.
    """
    frame_length = 64
    frame_step = 16  # 75% overlap → COLA Hann is fine.
    fft_length = 64
    B, C, T = 1, 1, 1024  # time-last

    # Generate a smooth test signal.
    t = mx.arange(T, dtype=mx.float32)
    x = mx.sin(2 * 3.14159 * t / 50.0)[None, None, :]
    x = mx.broadcast_to(x, (B, C, T))

    stft = STFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding="valid",
    )
    # InverseSTFT now applies the user-supplied window verbatim
    # (matches sequence_layers.mlx.dsp.InverseSTFT). For COLA recovery
    # the caller passes the inverse_stft_window_fn-corrected window
    # explicitly, the same way SpectroStreamInverseSTFT does.
    from magenta_rt.mlx_pure.signal import (
        hann_window, inverse_stft_window_fn,
    )
    istft = InverseSTFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=inverse_stft_window_fn(frame_step, hann_window),
        time_padding="valid",
    )
    spec = stft(x)
    rt = istft(spec)  # [B, C, T]

    # Inner region should match closely (cropped to avoid window roll-off).
    crop = frame_length
    diff = mx.max(mx.abs(
        rt[..., crop:-crop] - x[..., crop : crop + rt.shape[-1] - 2 * crop]
    )).item()
    assert diff < 1e-3, f"max abs diff in inner region: {diff}"


def test_stft_shapes():
    """Verify STFT output shapes for the typical SpectroStream config."""
    # Magenta-RT SpectroStream uses something close to:
    frame_length = 320
    frame_step = 160
    fft_length = 320
    B, C, T = 1, 1, 16000  # time-last
    x = mx.random.normal((B, C, T)) * 0.1
    stft = STFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding="reverse_causal_valid",
    )
    spec = stft(x)
    expected_freqs = fft_length // 2 + 1
    assert spec.shape[0] == B
    assert spec.shape[2] == expected_freqs
    assert spec.shape[3] == C
    assert spec.dtype == mx.complex64


def test_inverse_stft_streaming_matches_full_seq():
    """Streaming InverseSTFT with overlap-add cache should produce the
    same output as the non-streaming full-sequence path on the inner
    region (boundary differences come from the caller's padding, not
    the streaming path itself)."""
    from magenta_rt.mlx_pure.cache import OverlapAddCache

    frame_length = 16
    frame_step = 4   # 75% overlap
    fft_length = 16
    B, F, C = 1, 12, 2

    spec = mx.random.normal((B, F, fft_length // 2 + 1, C)) + 1j * mx.random.normal((B, F, fft_length // 2 + 1, C))
    spec = spec.astype(mx.complex64)

    full = InverseSTFT(
        frame_length=frame_length, frame_step=frame_step,
        fft_length=fft_length, time_padding="valid",
    )
    full_out = full(spec)  # [B, C, T]

    # Streaming.
    streaming = InverseSTFT(
        frame_length=frame_length, frame_step=frame_step,
        fft_length=fft_length, time_padding="valid",
    )
    cache = OverlapAddCache()
    chunks = []
    for t in range(F):
        chunk = streaming.step(spec[:, t : t + 1], cache)
        chunks.append(chunk)
    stream_out = mx.concatenate(chunks, axis=-1)  # [B, C, F * frame_step]

    # The non-streaming path emits T = (F-1)*frame_step + frame_length samples;
    # the streaming path emits exactly F*frame_step samples (one frame_step
    # per call). The first F*frame_step samples of full_out should match
    # stream_out exactly, except for the *very last* (frame_length - frame_step)
    # samples which still live in the streaming cache.
    common = F * frame_step - (frame_length - frame_step)
    assert common > 0
    diff = mx.max(mx.abs(full_out[..., :common] - stream_out[..., :common])).item()
    assert diff < 1e-5, f"streaming vs full max diff: {diff}"


def test_stft_parity_vs_sl():
    """Parity vs sl's STFT for the SpectroStream-style config."""
    import sequence_layers.mlx as sl

    frame_length = 32
    frame_step = 8
    fft_length = 32
    B, T, C = 1, 256, 1

    sl_stft = sl.STFT.Config(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding="reverse_causal_valid",
        fft_padding="right",
        output_magnitude=False,
    ).make(backend="mlx")

    pure = STFT(
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        time_padding="reverse_causal_valid",
    )

    x = mx.random.normal((B, T, C)) * 0.1
    sl_seq = sl.Sequence(x, mx.ones(x.shape[:2], dtype=mx.bool_))
    sl_y = sl_stft.layer(sl_seq).values
    # sl consumes [B, T, C]; the pure STFT consumes channel-major [B, C, T].
    # Both emit the same spec contract [B, F, freqs, C].
    pure_y = pure(mx.transpose(x, (0, 2, 1)))
    diff_real = mx.max(mx.abs(sl_y.real - pure_y.real)).item()
    diff_imag = mx.max(mx.abs(sl_y.imag - pure_y.imag)).item()
    assert diff_real < 1e-4 and diff_imag < 1e-4, f"real {diff_real}, imag {diff_imag}"
