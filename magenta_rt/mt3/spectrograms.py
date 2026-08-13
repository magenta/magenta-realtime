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

"""MT3 spectrogram configuration and numpy helpers (framework-neutral).

Ported from https://github.com/magenta/mt3 (spectrograms.py and the
``compute_logmel`` path of DDSP's spectral_ops.py). The mel filterbank
matches ``tf.signal.linear_to_mel_weight_matrix`` (HTK mel scale, triangles
computed in the mel domain, no normalization) rather than librosa
conventions, so that pretrained MT3 checkpoints see identical inputs. The
per-backend ``compute_spectrogram`` (jax in ``magenta_rt.nnx.mt3``, MLX in
``magenta_rt.mlx_pure.mt3``) consumes these helpers.
"""

import dataclasses

import numpy as np

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
MEL_HI_HZ = 8000.0


@dataclasses.dataclass(frozen=True)
class SpectrogramConfig:
    """Spectrogram configuration parameters."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_width


def split_audio(samples: np.ndarray, spectrogram_config: SpectrogramConfig) -> np.ndarray:
    """Split audio into non-overlapping frames of size hop_width, padding the end."""
    samples = np.asarray(samples)
    frame_size = spectrogram_config.hop_width
    num_frames = -(-samples.shape[-1] // frame_size)  # ceil
    pad = num_frames * frame_size - samples.shape[-1]
    samples = np.pad(samples, [(0, 0)] * (samples.ndim - 1) + [(0, pad)])
    return samples.reshape(samples.shape[:-1] + (num_frames, frame_size))


def input_depth(spectrogram_config: SpectrogramConfig) -> int:
    return spectrogram_config.num_mel_bins


def hann_window(length: int = FFT_SIZE) -> np.ndarray:
    """Periodic Hann window, as in ``tf.signal.hann_window``."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)).astype(
        np.float32
    )


def _hertz_to_mel(frequencies_hertz: np.ndarray) -> np.ndarray:
    """HTK mel scale, as in ``tf.signal``.

    Computed in float64; differs from TF's float32 Eigen kernels by ~1 ulp,
    which bounds the log mel spectrogram difference at ~1e-3.
    """
    return 1127.0 * np.log1p(np.asarray(frequencies_hertz, np.float64) / 700.0)


def linear_to_mel_weight_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    sample_rate: float = 8000.0,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> np.ndarray:
    """NumPy port of ``tf.signal.linear_to_mel_weight_matrix``.

    Returns:
        Weight matrix of shape [num_spectrogram_bins, num_mel_bins].
    """
    # Ignore the lowest (DC) spectrogram bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

    # Each mel band's lower edge, center, and upper edge, in the mel domain.
    band_edges_mel = np.linspace(
        _hertz_to_mel(lower_edge_hertz), _hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )
    lower_edge_mel = band_edges_mel[np.newaxis, :-2]
    center_mel = band_edges_mel[np.newaxis, 1:-1]
    upper_edge_mel = band_edges_mel[np.newaxis, 2:]

    # Triangular filters computed in the mel domain.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed DC bin.
    return np.pad(weights, [(bands_to_zero, 0), (0, 0)]).astype(np.float32)


def mel_filterbank(spectrogram_config: SpectrogramConfig) -> np.ndarray:
    """The MT3 mel filterbank for a spectrogram config."""
    return linear_to_mel_weight_matrix(
        num_mel_bins=spectrogram_config.num_mel_bins,
        num_spectrogram_bins=FFT_SIZE // 2 + 1,
        sample_rate=spectrogram_config.sample_rate,
        lower_edge_hertz=MEL_LO_HZ,
        upper_edge_hertz=MEL_HI_HZ,
    )
