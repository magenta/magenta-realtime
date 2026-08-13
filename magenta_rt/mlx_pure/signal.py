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

"""DSP utilities ported from ``sequence_layers.mlx.dsp`` / ``signal``.

* :func:`hann_window`, :func:`hamming_window`,
  :func:`inverse_stft_window_fn` — analysis / synthesis windows.
* :func:`frame`, :func:`overlap_and_add` — framing / OLA helpers.
* :class:`STFT` — non-streaming forward STFT (stateless).
* :class:`InverseSTFT` — non-streaming :meth:`__call__` plus a
  streaming :meth:`step` driven by an
  :class:`mlx_pure.cache.OverlapAddCache`.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import numpy as np


# -----------------------------------------------------------------------------
# Windows
# -----------------------------------------------------------------------------


def hann_window(window_length: int, *, periodic: bool = True, dtype=np.float32) -> np.ndarray:
    """Periodic (or symmetric) Hann window. Numpy output for compile-time use."""
    if window_length == 1:
        return np.ones([1], dtype=dtype)
    even = 1 - window_length % 2
    n = np.asarray(window_length + int(periodic) * even - 1, dtype=dtype)
    count = np.arange(window_length, dtype=dtype)
    return np.asarray(0.5 - 0.5 * np.cos(2 * np.pi * count / n), dtype)


def hamming_window(window_length: int, *, periodic: bool = True, dtype=np.float32) -> np.ndarray:
    if window_length == 1:
        return np.ones([1], dtype=dtype)
    even = 1 - window_length % 2
    n = np.asarray(window_length + int(periodic) * even - 1, dtype=dtype)
    count = np.arange(window_length, dtype=dtype)
    return np.asarray(0.54 - 0.46 * np.cos(2 * np.pi * count / n), dtype)


def inverse_stft_window_fn(frame_step: int,
                           forward_window_fn: Callable[..., np.ndarray] = hann_window):
    """Returns a function ``inverse_window(length, dtype)`` that produces
    the COLA-correct synthesis window for ``forward_window_fn`` at a
    hop of ``frame_step``. Ports
    :func:`sequence_layers.mlx.signal.inverse_stft_window_fn` (Griffin
    & Lim equation 7).
    """

    def _fn(frame_length: int, dtype=np.float32) -> np.ndarray:
        fw = forward_window_fn(frame_length, dtype=dtype).astype(np.float32)
        denom = fw * fw
        overlaps = -(-frame_length // frame_step)  # ceiling division
        # Pad denom to overlaps * frame_step.
        pad = overlaps * frame_step - frame_length
        denom = np.pad(denom, (0, pad))
        # Sum down columns of the [overlaps, frame_step] reshape.
        denom = denom.reshape(overlaps, frame_step).sum(axis=0, keepdims=True)
        denom = np.tile(denom, (overlaps, 1)).reshape(-1)[:frame_length]
        out = np.where(denom == 0.0, 0.0, fw / denom)
        return out.astype(dtype)

    return _fn


# -----------------------------------------------------------------------------
# Framing / overlap-add
# -----------------------------------------------------------------------------


def frame(values: mx.array, frame_length: int, frame_step: int) -> mx.array:
    """Produce overlapping frames of ``values`` along the last (time) axis.

    Input shape ``[..., T]``; output shape ``[..., num_frames, frame_length]``
    with ``num_frames = max(0, (T - frame_length) // frame_step + 1)``. No
    padding is applied — the caller is responsible for any required
    boundary handling.
    """
    T = values.shape[-1]
    lead = values.shape[:-1]
    num_frames = max(0, (T - frame_length) // frame_step + 1)
    if num_frames == 0:
        return mx.zeros(lead + (0, frame_length), dtype=values.dtype)
    # Slice-and-stack: simpler than as_strided for small frame counts.
    frames = []
    for i in range(num_frames):
        start = i * frame_step
        frames.append(values[..., start : start + frame_length])
    return mx.stack(frames, axis=-2)


def overlap_and_add(framed: mx.array, frame_step: int) -> mx.array:
    """Inverse of :func:`frame`. Input ``[..., num_frames, frame_length]``
    with overlap; output ``[..., output_length]`` where
    ``output_length = (num_frames - 1) * frame_step + frame_length``.
    """
    F, L = framed.shape[-2], framed.shape[-1]
    lead = framed.shape[:-2]
    output_length = (F - 1) * frame_step + L if F > 0 else 0
    if F == 0:
        return mx.zeros(lead + (0,), dtype=framed.dtype)
    if frame_step == L:
        return framed.reshape(*lead, output_length)

    out = mx.zeros(lead + (output_length,), dtype=framed.dtype)
    frame_indices = mx.arange(F)[:, None] * frame_step + mx.arange(L)[None, :]
    return out.at[..., frame_indices].add(framed)


# -----------------------------------------------------------------------------
# STFT / InverseSTFT (non-streaming)
# -----------------------------------------------------------------------------


def _padding(time_padding: str, frame_length: int, frame_step: int) -> tuple[int, int]:
    """Compute (pad_left, pad_right) for a given padding mode.

    Mirrors the modes sl uses (see
    ``sequence_layers.mlx.convolution._explicit_padding``):
        * ``'valid'`` — no padding.
        * ``'same'`` — symmetric padding so output length matches input.
        * ``'causal'`` / ``'causal_valid'`` — pad only on the left (no
          future leakage).
        * ``'reverse_causal'`` / ``'reverse_causal_valid'`` — pad only
          on the right.
        * ``'semicausal'`` — left-heavy banded padding, output length
          ``ceil(input / stride)`` and the receptive field doesn't
          extend into the future. Used by SpectroStream's STFT.
    """
    if time_padding == "valid":
        return (0, 0)
    if time_padding == "same":
        pad = max(0, frame_length - frame_step)
        return (pad // 2, pad - pad // 2)
    if time_padding in ("causal", "causal_valid"):
        return (frame_length - 1, 0)
    if time_padding in ("reverse_causal", "reverse_causal_valid"):
        return (0, frame_length - 1)
    if time_padding == "semicausal":
        # ek = frame_length (dilation=1 for STFT framing).
        pad_left = max(frame_length - frame_step, 0)
        return (pad_left, frame_length - 1 - pad_left)
    raise NotImplementedError(f"unsupported time_padding: {time_padding}")


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class STFT:
    """Short-Time Fourier Transform (non-streaming).

    Inputs ``[B, num_channels, T]`` (time-last audio) → outputs
    ``[B, num_frames, num_freqs, num_channels]`` complex (or magnitude).
    The spec layout is the codec-model contract and is unchanged; only the
    audio side moved to ``[B, C, T]``.
    """

    def __init__(
        self,
        *,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        window_fn: Optional[Callable[..., np.ndarray]] = None,
        time_padding: str = "reverse_causal_valid",
        output_magnitude: bool = False,
    ):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.window_fn = window_fn or hann_window
        self.time_padding = time_padding
        self.output_magnitude = output_magnitude
        # Precompute analysis window.
        self._window = mx.array(self.window_fn(frame_length, dtype=np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        pad_left, pad_right = _padding(self.time_padding, self.frame_length, self.frame_step)
        if pad_left or pad_right:
            pad_widths = [(0, 0)] * x.ndim
            pad_widths[-1] = (pad_left, pad_right)
            x = mx.pad(x, pad_widths)
        framed = frame(x, self.frame_length, self.frame_step)  # [B, C, F, L]
        # Apply window along the (last) frame axis.
        framed = framed * self._window
        # Pad / truncate to fft_length on the last axis.
        if self.fft_length > self.frame_length:
            pad = [(0, 0)] * framed.ndim
            pad[-1] = (0, self.fft_length - self.frame_length)
            framed = mx.pad(framed, pad)
        elif self.fft_length < self.frame_length:
            framed = framed[..., : self.fft_length]
        spec = mx.fft.rfft(framed, axis=-1)  # [B, C, F, fft_length//2+1]
        if self.output_magnitude:
            spec = mx.abs(spec)
        # [B, C, F, freqs] -> the codec-model spec contract [B, F, freqs, C].
        return mx.transpose(spec, (0, 2, 3, 1))


class InverseSTFT:
    """Inverse STFT.

    Non-streaming: ``[B, num_frames, num_freqs, num_channels]`` complex →
    ``[B, num_channels, T_audio]`` (time-last audio).

    Streaming :meth:`step`: takes one frame
    ``[B, 1, num_freqs, num_channels]`` plus an
    :class:`mlx_pure.cache.OverlapAddCache`; emits ``frame_step`` audio
    samples ``[B, num_channels, frame_step]`` and updates the cache
    with the residual overlap region.
    """

    def __init__(
        self,
        *,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        window_fn: Optional[Callable[..., np.ndarray]] = None,
        time_padding: str = "causal",
    ):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.window_fn = window_fn or hann_window
        self.time_padding = time_padding
        # Apply the user-supplied window verbatim, matching
        # ``sequence_layers.mlx.dsp.InverseSTFT``: the COLA correction
        # is the *caller's* responsibility (e.g. SpectroStreamInverseSTFT
        # passes ``inverse_stft_window_fn(frame_step, hann_window)`` as
        # ``window_fn``). Re-applying ``inverse_stft_window_fn`` here
        # would double-correct and produce miscalibrated audio.
        self._synth_window = mx.array(
            self.window_fn(frame_length, dtype=np.float32)
        )

    def __call__(self, spec: mx.array) -> mx.array:
        # Spec arrives in the codec-model contract [B, F, freqs, C]; go
        # channel-major [B, C, F, freqs] so time stays on the last axis.
        spec = mx.transpose(spec, (0, 3, 1, 2))
        sig = mx.fft.irfft(spec, n=self.fft_length, axis=-1)
        # Trim to frame_length.
        sig = sig[..., : self.frame_length]
        # Synthesis window (broadcasts over the last axis).
        sig = sig * self._synth_window
        out = overlap_and_add(sig, self.frame_step)  # [B, C, T]
        # Trim. For overlap-add the trim convention is different from
        # the framing convention used by STFT — see
        # ``sequence_layers.mlx.dsp.OverlapAdd.layer``: causal trims
        # ``frame_length - frame_step`` from the *right*; semicausal_full
        # trims it from the *left* (and we don't drop any from the
        # right, matching sl's behaviour).
        trim = max(self.frame_length - self.frame_step, 0)
        if trim:
            if self.time_padding in ("causal", "causal_valid"):
                out = out[..., : out.shape[-1] - trim]
            elif self.time_padding == "semicausal":
                out = out[..., trim:]
        return out

    # ------------------------------------------------------------------
    # Streaming step
    # ------------------------------------------------------------------

    def step(self, spec: mx.array, cache) -> mx.array:
        """Streaming inverse STFT.

        Args:
            spec: ``[B, T, num_freqs, num_channels]`` — one or more STFT frames.
            cache: an :class:`mlx_pure.cache.OverlapAddCache`. Lazily
                initialized on first call to hold the overlap of size
                ``frame_length - frame_step``.

        Returns:
            ``[B, num_channels, T * frame_step]`` audio samples (time-last).
        """
        B, T, F, C = spec.shape
        if T == 0:
            return mx.zeros((B, C, 0), dtype=mx.float32)

        overlap = self.frame_length - self.frame_step

        # [B, T, freqs, C] -> channel-major [B, C, T, freqs].
        spec = mx.transpose(spec, (0, 3, 1, 2))
        sig = mx.fft.irfft(spec, n=self.fft_length, axis=-1)
        sig = sig[..., : self.frame_length]
        sig = sig * self._synth_window  # broadcasts over the last axis

        ola = overlap_and_add(sig, self.frame_step)  # [B, C, T*step + overlap]

        if cache.empty():
            cache.reset(ola.shape[:-1] + (overlap,), sig.dtype)

        if overlap > 0:
            ola_with_buf = ola.at[..., :overlap].add(cache.buffer)
            out = ola_with_buf[..., :T * self.frame_step]
            cache.buffer = ola_with_buf[..., T * self.frame_step:]
        else:
            out = ola[..., :T * self.frame_step]

        return out
