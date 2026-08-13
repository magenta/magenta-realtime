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

MLX port of :mod:`magenta_rt.nnx.musiccoca.frontend` (see there for the
recovered pipeline description).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

SAMPLE_RATE = 16_000
CLIP_SAMPLES = 160_000
FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_LENGTH = 2048
NUM_MEL_BINS = 128
NUM_FRAMES = 992  # after truncation from 998
PREEMPHASIS = 0.97
MEL_FLOOR = 1e-3


class LogMelFrontend(nn.Module):
    """Waveform ``[B, 160000]`` → log-mel features ``[B, 992, 128]``."""

    def __init__(self):
        super().__init__()
        self.mel_matrix = mx.zeros(
            (FFT_LENGTH // 2 + 1, NUM_MEL_BINS), dtype=mx.float32
        )
        self.window = mx.zeros((FRAME_LENGTH,), dtype=mx.float32)

    def __call__(self, waveform: mx.array) -> mx.array:
        x = waveform.astype(mx.float32)
        shifted = mx.pad(x, [(0, 0), (1, 0)])[:, :-1]
        x = x - PREEMPHASIS * shifted

        num_frames = (x.shape[-1] - FRAME_LENGTH) // FRAME_HOP + 1
        starts = mx.arange(num_frames) * FRAME_HOP
        idx = starts[:, None] + mx.arange(FRAME_LENGTH)[None, :]
        frames = mx.take(x, idx, axis=-1) * self.window

        spectrum = mx.fft.rfft(frames, n=FFT_LENGTH, axis=-1)
        power = mx.square(mx.abs(spectrum))
        mel = power @ self.mel_matrix + MEL_FLOOR
        return mx.log(mel)[:, :NUM_FRAMES]
