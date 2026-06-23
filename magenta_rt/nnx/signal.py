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

"""DSP utilities for the SpectroStream codec.

* :func:`hann_window`, :func:`hamming_window`,
  :func:`inverse_stft_window_fn` — analysis / synthesis windows.
* :func:`frame`, :func:`overlap_and_add` — framing / OLA helpers.
* :class:`STFT` — non-streaming forward STFT (stateless).
* :class:`InverseSTFT` — non-streaming + streaming ``step``. The
  streaming buffer lives on the module as an :class:`nnx.Cache` slot
  populated by :meth:`init_cache`.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
from jax import lax, numpy as jnp
import numpy as np
from flax import nnx


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
    """Returns a function ``inverse_window(length, dtype)`` that
    produces the COLA-correct synthesis window for ``forward_window_fn``
    at a hop of ``frame_step`` (Griffin & Lim eq. 7).
    """

    def _fn(frame_length: int, dtype=np.float32) -> np.ndarray:
        fw = forward_window_fn(frame_length, dtype=dtype).astype(np.float32)
        denom = fw * fw
        overlaps = -(-frame_length // frame_step)
        pad = overlaps * frame_step - frame_length
        denom = np.pad(denom, (0, pad))
        denom = denom.reshape(overlaps, frame_step).sum(axis=0, keepdims=True)
        denom = np.tile(denom, (overlaps, 1)).reshape(-1)[:frame_length]
        out = np.where(denom == 0.0, 0.0, fw / denom)
        return out.astype(dtype)

    return _fn


def frame(values: jnp.ndarray, frame_length: int, frame_step: int) -> jnp.ndarray:
    """Produce overlapping frames of ``values`` along the last (time) axis.

    Input ``[..., T]`` → ``[..., num_frames, frame_length]`` with
    ``num_frames = max(0, (T - frame_length) // frame_step + 1)``.
    """
    T = values.shape[-1]
    lead = values.shape[:-1]
    num_frames = max(0, (T - frame_length) // frame_step + 1)
    if num_frames == 0:
        return jnp.zeros(lead + (0, frame_length), dtype=values.dtype)
    # Vectorize: for each i in [0, num_frames), take a slice
    # values[..., i*frame_step : i*frame_step + frame_length]. Map over
    # the start indices and stack on the second-to-last axis.
    starts = jnp.arange(num_frames) * frame_step

    def _slice_one(start):
        return lax.dynamic_slice_in_dim(values, start, frame_length, axis=-1)

    return jax.vmap(_slice_one, in_axes=0, out_axes=-2)(starts)


def overlap_and_add(framed: jnp.ndarray, frame_step: int) -> jnp.ndarray:
    """Inverse of :func:`frame`. ``[..., F, L]`` → ``[..., T]``."""
    F, L = framed.shape[-2], framed.shape[-1]
    lead = framed.shape[:-2]
    output_length = (F - 1) * frame_step + L if F > 0 else 0
    if F == 0:
        return jnp.zeros(lead + (0,), dtype=framed.dtype)
    if frame_step == L:
        return framed.reshape(*lead, output_length)

    out = jnp.zeros(lead + (output_length,), dtype=framed.dtype)
    frame_indices = jnp.arange(F)[:, None] * frame_step + jnp.arange(L)[None, :]
    return out.at[..., frame_indices].add(framed)


def _padding(time_padding: str, frame_length: int, frame_step: int) -> tuple[int, int]:
    """``(pad_left, pad_right)`` for a given padding mode (mirrors sl)."""
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
        pad_left = max(frame_length - frame_step, 0)
        return (pad_left, frame_length - 1 - pad_left)
    raise NotImplementedError(f"unsupported time_padding: {time_padding}")


class STFT:
    """Short-Time Fourier Transform (non-streaming, stateless).

    ``[B, num_channels, T]`` (time-last audio) →
    ``[B, num_frames, num_freqs, num_channels]`` complex (or magnitude).
    The spec layout is the codec-model contract and is unchanged from the
    pre-time-last version; only the audio side moved to ``[B, C, T]``.
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
        self._window = jnp.asarray(self.window_fn(frame_length, dtype=np.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pad_left, pad_right = _padding(self.time_padding, self.frame_length, self.frame_step)
        if pad_left or pad_right:
            pad_widths = [(0, 0)] * x.ndim
            pad_widths[-1] = (pad_left, pad_right)
            x = jnp.pad(x, pad_widths)
        framed = frame(x, self.frame_length, self.frame_step)  # [B, C, F, L]
        framed = framed * self._window  # window broadcasts over the last axis
        if self.fft_length > self.frame_length:
            pad = [(0, 0)] * framed.ndim
            pad[-1] = (0, self.fft_length - self.frame_length)
            framed = jnp.pad(framed, pad)
        elif self.fft_length < self.frame_length:
            framed = framed[..., : self.fft_length]
        spec = jnp.fft.rfft(framed)  # FFT over the (default) last axis
        if self.output_magnitude:
            spec = jnp.abs(spec)
        # [B, C, F, freqs] -> the codec-model spec contract [B, F, freqs, C].
        return jnp.transpose(spec, (0, 2, 3, 1))


class InverseSTFT(nnx.Module):
    """Inverse STFT (non-streaming + streaming ``step``).

    The streaming overlap buffer lives on this module as an
    ``nnx.Cache`` slot populated by :meth:`init_cache`. When the slot
    is set and :attr:`streaming` is true, ``__call__`` routes to the
    streaming path.
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
        # Synthesis window — applied verbatim, COLA correction is the
        # caller's job (matches sl).
        self._synth_window = jnp.asarray(
            self.window_fn(frame_length, dtype=np.float32)
        )
        # Streaming state.
        self.streaming: bool = False
        self.cached_overlap: nnx.Cache | None = nnx.data(None)

    def init_cache(
        self,
        *,
        batch: int,
        num_channels: int,
        dtype=jnp.float32,
    ) -> None:
        overlap = max(self.frame_length - self.frame_step, 0)
        self.cached_overlap = nnx.Cache(
            jnp.zeros((batch, num_channels, overlap), dtype=dtype)
        )
        self.streaming = True

    def remove_cache(self) -> None:
        self.cached_overlap = nnx.data(None)
        self.streaming = False

    # --- non-streaming forward ---------------------------------------------

    def __call__(self, spec: jnp.ndarray) -> jnp.ndarray:
        # Spec arrives in the codec-model contract [B, F, freqs, C]; go
        # channel-major [B, C, F, freqs] so time stays on the last axis.
        spec = jnp.transpose(spec, (0, 3, 1, 2))
        sig = jnp.fft.irfft(spec, n=self.fft_length)  # last axis
        sig = sig[..., : self.frame_length]
        sig = sig * self._synth_window  # window broadcasts over the last axis
        out = overlap_and_add(sig, self.frame_step)  # [B, C, T]
        trim = max(self.frame_length - self.frame_step, 0)
        if trim:
            if self.time_padding in ("causal", "causal_valid"):
                out = out[..., : out.shape[-1] - trim]
            elif self.time_padding == "semicausal":
                out = out[..., trim:]
        return out

    # --- streaming step ----------------------------------------------------

    def step(self, spec: jnp.ndarray) -> jnp.ndarray:
        """Streaming inverse STFT.

        ``spec``: ``[B, T, num_freqs, num_channels]`` (one or more
        frames). Returns ``[B, num_channels, T * frame_step]`` audio.

        :meth:`init_cache` must be called first to allocate the overlap
        buffer.
        """
        if self.cached_overlap is None:
            raise RuntimeError("init_cache() must be called before step().")
        B, T, _F, C = spec.shape
        if T == 0:
            return jnp.zeros((B, C, 0), dtype=jnp.float32)

        overlap = self.frame_length - self.frame_step

        # [B, T, freqs, C] -> channel-major [B, C, T, freqs].
        spec = jnp.transpose(spec, (0, 3, 1, 2))
        sig = jnp.fft.irfft(spec, n=self.fft_length)  # last axis
        sig = sig[..., : self.frame_length]
        sig = sig * self._synth_window  # window broadcasts over the last axis

        ola = overlap_and_add(sig, self.frame_step)  # [B, C, T*step + overlap]

        if overlap > 0:
            ola_with_buf = ola.at[..., :overlap].add(self.cached_overlap[...])
            out = ola_with_buf[..., :T * self.frame_step]
            self.cached_overlap[...] = ola_with_buf[..., T * self.frame_step:]
        else:
            out = ola[..., :T * self.frame_step]

        return out
