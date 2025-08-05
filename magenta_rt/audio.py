# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Audio processing utils."""

from typing import BinaryIO, Optional

import librosa
import numpy as np
import resampy
import soundfile as sf


class Waveform:
  """Simple audio waveform container wrapping f32 numpy array of [nsamp, nch]."""

  def __init__(self, samples: np.ndarray, sample_rate: int):
    self._samples: Optional[np.ndarray] = None
    self.samples = samples
    self._sample_rate = sample_rate

  def __len__(self):
    return self.num_samples

  @property
  def sample_rate(self) -> int:
    return self._sample_rate

  @sample_rate.setter
  def sample_rate(self, value: int):
    del value
    raise AttributeError("Sample rate should only be set on construction")

  @property
  def samples(self) -> np.ndarray:
    assert self._samples is not None
    return self._samples

  @samples.setter
  def samples(self, value: np.ndarray):
    if value.ndim == 1:
      value = value[:, np.newaxis]
    if not (value.ndim == 2 and value.shape[1] > 0):
      raise ValueError(f"Invalid shape for audio: {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
      raise TypeError(f"Samples should be np.floating. Got {value.dtype}.")
    self._samples = np.array(value, dtype=np.float32)

  @property
  def _librosa_samples(self) -> np.ndarray:
    """Returns samples in a librosa-compatible form.

    Librosa expects: np.ndarray [shape=(n,) or (2, n)]).
    """
    samples = self.samples
    if self.samples.shape[1] == 1:
      # Mono audio must be 1D.
      return self.samples.squeeze(1)
    else:
      return samples.T

  @property
  def num_samples(self) -> int:
    return self.samples.shape[0]

  @property
  def num_channels(self) -> int:
    return self.samples.shape[1]

  @property
  def seconds(self) -> float:
    return len(self) / self.sample_rate

  @property
  def peak_amplitude(self) -> float:
    return np.abs(self._samples).max()

  @property
  def peak_rms(self) -> float:
    return self.compute_peak_rms()

  def compute_rms(
      self,
      # NOTE: Librosa defaults
      hop_length_seconds: float = 512 / 22050,
      frame_length_seconds: float = 2048 / 22050,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Computes RMS amplitude."""
    rms = librosa.feature.rms(
        y=self._librosa_samples,
        hop_length=round(hop_length_seconds * self.sample_rate),
        frame_length=round(frame_length_seconds * self.sample_rate),
    )[0]
    t = np.arange(rms.shape[0]) * hop_length_seconds
    return t, rms

  def compute_peak_rms(self, *args, **kwargs) -> float:
    """Computes peak RMS amplitude over frames."""
    _, rms = self.compute_rms(*args, **kwargs)
    peak_rms = np.max(rms)
    assert peak_rms >= 0
    return peak_rms

  def apply_gain(self, gain: float, in_place: bool = False) -> "Waveform":
    """Applies linear gain to samples."""
    if in_place:
      self._samples *= gain
      return self
    else:
      return type(self)(self._samples * gain, self.sample_rate)

  def peak_normalize(
      self, max_value: float = 1.0, in_place: bool = False
  ) -> "Waveform":
    """Normalizes audio to a particular amplitude value."""

    peak = self.peak_amplitude
    if peak == 0:
      gain = 1.0
    else:
      gain = max_value / peak

    result = self.apply_gain(gain, in_place=in_place)
    return result

  def resample(self, sample_rate: int) -> "Waveform":
    if self.sample_rate == sample_rate:
      return self
    return Waveform(
        samples=resampy.resample(
            self.samples.swapaxes(0, 1), self.sample_rate, sample_rate
        ).swapaxes(0, 1),
        sample_rate=sample_rate,
    )

  def as_stereo(self) -> "Waveform":
    """Converts a multichannel waveform to stereo."""
    if self.num_channels == 1:
      return Waveform(
          np.concatenate([self.samples, self.samples], axis=1), self.sample_rate
      )
    elif self.num_channels == 2:
      return self
    else:
      raise ValueError("Unsupported number of channels for stereo conversion.")

  def as_mono(self, strategy: str = "average") -> "Waveform":
    """Converts a multichannel waveform to mono."""
    if self.num_channels == 1:
      return self
    if strategy == "average":
      return Waveform(
          self.samples.mean(axis=1, keepdims=True), self.sample_rate
      )
    elif self.num_channels == 2 and strategy == "left":
      return Waveform(self.samples[:, 0:1], self.sample_rate)
    elif self.num_channels == 2 and strategy == "right":
      return Waveform(self.samples[:, 1:2], self.sample_rate)
    else:
      raise ValueError(f"Unsupported strategy: {strategy}")

  def write(self, file: str | BinaryIO, **kwargs):
    sf.write(file, self.samples, self.sample_rate, **kwargs)

  @classmethod
  def from_file(cls, file: str | BinaryIO) -> "Waveform":
    return Waveform(*sf.read(file))

  def __getitem__(self, key: slice) -> "Waveform":
    return Waveform(self._samples[key], self.sample_rate)


def concatenate(
    waveforms: list[Waveform],
    crossfade_time: float = 0.0,
    style: str = "eqpower",
):
  """Concatenates a list of waveforms into a single waveform."""
  if not waveforms:
    raise ValueError("No waveforms to concatenate.")

  # Make sure all waveforms have common sample rate and number of channels.
  all_sample_rates = set(w.sample_rate for w in waveforms)
  if len(all_sample_rates) != 1:
    raise ValueError("All waveforms must have the same sample rate.")
  sample_rate = all_sample_rates.pop()
  all_num_channels = set(w.num_channels for w in waveforms)
  if len(all_num_channels) != 1:
    raise ValueError("All waveforms must have the same number of channels.")
  num_channels = all_num_channels.pop()

  # Compute crossfades and number of output samples
  crossfade_length_samples = round(crossfade_time * sample_rate)
  if any(w.num_samples < (crossfade_length_samples * 2) for w in waveforms):
    raise ValueError(
        "All waveforms must be longer than twice the crossfade time."
    )
  num_output_samples = (
      sum(w.num_samples - crossfade_length_samples for w in waveforms)
      + crossfade_length_samples
  )
  if style == "linear":
    # Linear crossfade
    fade_in = np.linspace(
        0, 1, crossfade_length_samples, endpoint=False, dtype=np.float32
    )
  elif style == "eqpower":
    # Equal power crossfade
    fade_in = np.sin(
        np.linspace(
            0,
            np.pi / 2,
            crossfade_length_samples,
            endpoint=False,
            dtype=np.float32,
        )
    )
  else:
    raise ValueError(f"Unsupported crossfade style: {style}")
  fade_out = np.flip(fade_in)

  # Concatenate waveforms
  samples = np.zeros((num_output_samples, num_channels), dtype=np.float32)
  sample_idx = 0
  for w in waveforms:
    # Apply fade
    w_samples = w.samples.copy()
    if crossfade_length_samples > 0:
      w_samples[:crossfade_length_samples] *= fade_in[:, np.newaxis]
      w_samples[-crossfade_length_samples:] *= fade_out[:, np.newaxis]
    samples[sample_idx : sample_idx + w.num_samples] += w_samples
    sample_idx += w.num_samples - crossfade_length_samples
  return Waveform(samples=samples, sample_rate=sample_rate)


def amp_to_db(amp: float, amp_ref: float = 1.0) -> float:
  if amp < 0:
    raise ValueError(f"Amplitude must be non-negative. Got {amp}.")
  if amp_ref <= 0:
    raise ValueError(f"Reference amplitude must be positive. Got {amp_ref}.")
  if amp == 0:
    return float("-inf")
  else:
    return float(20 * np.log10(amp / amp_ref))


def db_to_amp(db: float, amp_ref: float = 1.0) -> float:
  if amp_ref <= 0:
    raise ValueError(f"Reference amplitude must be positive. Got {amp_ref}.")
  return float(np.power(10, db / 20.0) * amp_ref)
