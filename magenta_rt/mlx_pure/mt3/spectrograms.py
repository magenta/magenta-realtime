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

"""Audio spectrogram functions (MLX).

The framework-neutral pieces (``SpectrogramConfig``, mel filterbank,
framing helpers) live in :mod:`magenta_rt.mt3.spectrograms` and are
re-exported here; this module adds the MLX ``compute_spectrogram``. The
STFT intentionally matches ``tf.signal.stft`` (periodic Hann window, no
centering, zero padding at the end) so that pretrained MT3 checkpoints
see identical inputs.
"""

import mlx.core as mx
import numpy as np

from magenta_rt.mt3.spectrograms import (  # noqa: F401 (re-exports)
    FFT_SIZE,
    MEL_HI_HZ,
    MEL_LO_HZ,
    SpectrogramConfig,
    hann_window,
    input_depth,
    linear_to_mel_weight_matrix,
    mel_filterbank,
    split_audio,
)


def _frame(samples: mx.array, frame_length: int, frame_step: int) -> mx.array:
    """Frame a signal like ``tf.signal.frame`` with ``pad_end=True``."""
    num_samples = samples.shape[-1]
    num_frames = -(-num_samples // frame_step)  # ceil
    pad = (num_frames - 1) * frame_step + frame_length - num_samples
    samples = mx.pad(samples, [(0, 0)] * (samples.ndim - 1) + [(0, pad)])
    indices = np.arange(frame_length)[np.newaxis, :] + frame_step * np.arange(num_frames)[:, np.newaxis]
    return mx.take(samples, mx.array(indices), axis=-1)


def compute_spectrogram(
    samples: mx.array, spectrogram_config: SpectrogramConfig
) -> mx.array:
    """Compute a log mel spectrogram.

    Args:
        samples: Audio samples of shape [..., num_samples] at
            ``spectrogram_config.sample_rate``.

    Returns:
        Log mel spectrogram of shape [..., ceil(num_samples / hop_width),
        num_mel_bins].
    """
    frames = _frame(samples, FFT_SIZE, spectrogram_config.hop_width)
    magnitude = mx.abs(mx.fft.rfft(frames * mx.array(hann_window()), n=FFT_SIZE))
    mel = magnitude @ mx.array(mel_filterbank(spectrogram_config))

    # safe_log: avoid taking the log of a non-positive number.
    return mx.log(mx.where(mel <= 0.0, mx.array(1e-5), mel))
