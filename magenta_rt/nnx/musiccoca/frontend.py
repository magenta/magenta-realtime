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

"""Log-mel frontend matching MusicCoCa's ``audio_preprocessor.tflite``.

Pipeline (16 kHz mono, 10-second clips):

1. pre-emphasis ``x[n] - 0.97 * x[n-1]`` (zero initial condition),
2. framing: 400-sample (25 ms) windows, 160-sample (10 ms) hop → 998 frames,
3. periodic Hann window, zero-pad to 2048, real FFT → 1025 bins,
4. power spectrum × mel matrix (128 bands, DC bin zeroed) + 1e-3, log,
5. keep the first 992 frames → ``[B, 992, 128]``.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

SAMPLE_RATE = 16_000
CLIP_SAMPLES = 160_000
FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_LENGTH = 2048
NUM_MEL_BINS = 128
NUM_FRAMES = 992  # after truncation from 998
PREEMPHASIS = 0.97
MEL_FLOOR = 1e-3


class LogMelFrontend(nnx.Module):
    """Waveform ``[B, 160000]`` → log-mel features ``[B, 992, 128]``."""

    def __init__(self):
        self.mel_matrix = nnx.Param(
            jnp.zeros((FFT_LENGTH // 2 + 1, NUM_MEL_BINS), jnp.float32)
        )
        self.window = nnx.Param(jnp.zeros((FRAME_LENGTH,), jnp.float32))

    def __call__(
        self, waveform: jnp.ndarray, *, num_frames: int | None = NUM_FRAMES
    ) -> jnp.ndarray:
        """Waveform ``[B, S]`` → log-mel ``[B, num_frames, 128]``.

        ``num_frames`` truncates the time axis (default 992, matching one 10 s
        clip → a 62×16 patch grid). Pass ``num_frames=None`` to keep ALL frames
        of an arbitrary-length signal — the basis for windowing a long
        spectrogram in mel space (see :meth:`AudioEncoder.encode_windows`):
        framing here is local (400-sample frame, 160-sample hop, no centering),
        so the mel of a long signal restricted to a window equals the mel of
        that window standalone, except for the single-sample pre-emphasis
        initial condition at the window's first sample (immaterial after the
        encoder's global pool + RVQ).
        """
        x = waveform.astype(jnp.float32)
        shifted = jnp.pad(x, ((0, 0), (1, 0)))[:, :-1]
        x = x - PREEMPHASIS * shifted

        n = (x.shape[-1] - FRAME_LENGTH) // FRAME_HOP + 1
        starts = jnp.arange(n) * FRAME_HOP
        idx = starts[:, None] + jnp.arange(FRAME_LENGTH)[None, :]
        frames = x[:, idx] * self.window[...]

        spectrum = jnp.fft.rfft(frames, n=FFT_LENGTH, axis=-1)
        power = jnp.square(jnp.abs(spectrum))
        mel = power @ self.mel_matrix[...] + MEL_FLOOR
        logmel = jnp.log(mel)
        return logmel if num_frames is None else logmel[:, :num_frames]
