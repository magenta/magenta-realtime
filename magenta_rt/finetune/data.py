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

"""Data featurization utils."""

import random
from typing import Iterator, Optional

import numpy as np
import tensorflow as tf

from .. import audio as audio_lib
from .. import musiccoca
from .. import spectrostream


def _key_to_unit_interval(key):
  """Map 64-bit hexadecimal string to unit interval."""
  # Use reversed digits of key since it may have already been filtered
  # lexicographically.
  return int(key[::-1], 16) / (2**64)


def _generate_clip_slices(
    key,
    num_samples,
    clip_length_seconds,
    sample_rate,
    max_clips,
):
  """Randomly generate clip slices for given number of samples.

  Args:
    key: an optional hexadecimal key to use as a hash for random sampling
    num_samples: length of the audio to clip, in samples
    clip_length_seconds: desired clip length, in seconds
    sample_rate: number of samples per second
    max_clips: maximum number of clips to generate

  Returns:
    a list of slice objects (specified in samples)
  """
  clip_length_samples = int(sample_rate * clip_length_seconds)
  if key is not None:
    # Start at a random offset so we don't oversample beginnings.
    offset = int(_key_to_unit_interval(key) * clip_length_samples)
  else:
    offset = 0
  clip_slices = []
  for start_sample in range(
      offset, num_samples - clip_length_samples + 1, clip_length_samples
  ):
    clip_slices.append(slice(start_sample, start_sample + clip_length_samples))
  if max_clips is not None and len(clip_slices) > max_clips:
    if key is not None:
      random.seed(int(key[::-1], 16))
    clip_slices = random.sample(clip_slices, max_clips)
  return clip_slices


def _is_loud_enough(
    audio: audio_lib.Waveform,
    clip_length_seconds: float = 5.0,
    clip_stride_seconds: float = 1.0,
    quiet_thresh_db: float = -25.0,
) -> bool:
  """Checks whether all clips within the audio are above the quiet threshold.

  Args:
    audio: audio clip.
    clip_length_seconds: Length of sub clips to extract (training length).
    clip_stride_seconds: Stride for sub clip extraction.
    quiet_thresh_db: Residual peak RMS must exceed this.

  Returns:
    True if audio is "loud enough", otherwise False.
  """

  clip_length = round(clip_length_seconds * audio.sample_rate)
  assert len(audio) >= clip_length
  clip_peaks = []
  for i in range(0, len(audio), round(clip_stride_seconds * audio.sample_rate)):
    clip_slice = slice(i, i + clip_length)
    audio_clip = audio[clip_slice]
    clip_peaks.append(audio_lib.amp_to_db(audio_clip.peak_rms))

  return np.min(clip_peaks) >= quiet_thresh_db


def _array_to_feature(a: np.ndarray) -> tf.train.Feature:
  """Convert array to feature containing a serialized tensor."""
  assert a.dtype in [np.float32, np.int32, np.int64]
  if a.dtype == np.float32 and a.ndim == 1:
    return tf.train.Feature(float_list=tf.train.FloatList(value=a.tolist()))
  else:
    if a.dtype != np.float32:
      assert (
          a.min() >= np.iinfo(np.int32).min
          and a.max() <= np.iinfo(np.int32).max
      )
      a = a.astype(np.int32)
    a_bytes = tf.io.serialize_tensor(a).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[a_bytes]))


class Featurizer:
  """Base class for data featurization."""

  def __init__(
      self,
      peak_norm_dbfs: float | None = -1.0,
      filter_quiet: bool = False,
      filter_low_sample_rate: bool = False,
      filter_mono: bool = False,
      audio_len_seconds: float = 30,
      max_clips_per_example: int | None = None,
      min_clip_seconds: float | None = None,
      include_style_embeddings: bool = False,
  ):
    self._peak_norm = audio_lib.db_to_amp(peak_norm_dbfs)
    self._filter_quiet = filter_quiet
    self._filter_low_sample_rate = filter_low_sample_rate
    self._filter_mono = filter_mono
    self._audio_len_seconds = audio_len_seconds
    self._max_clips_per_example = max_clips_per_example
    self._min_clip_seconds = min_clip_seconds
    self._include_style_embeddings = include_style_embeddings

    self.setup()

  def setup(self):
    self._codec = spectrostream.SpectroStreamJAX(lazy=False)
    self._style_model = musiccoca.MusicCoCa(lazy=False)

  def _featurize(
      self,
      audio: audio_lib.Waveform,
  ) -> Optional[dict[str, np.ndarray]]:
    """Extracts features from a single audio clip."""
    if (
        len(audio) != self._audio_len_seconds * audio.sample_rate
        and self._min_clip_seconds is None
    ):
      raise NotImplementedError(
          'This pipeline is configured for inputs that are precisely '
          '{self._audio_len_seconds}s in length. Got {audio.seconds}.'
      )

    if self._filter_low_sample_rate and audio.sample_rate < 44100:
      return None

    if self._filter_mono and audio.num_channels < 2:
      return None

    # Resample
    style_audio = audio.resample(self._style_model.config.sample_rate)

    codec_audio = audio.resample(self._codec.sample_rate).as_stereo()

    # Filter
    if (
        self._filter_quiet
        and (self._min_clip_seconds is None or style_audio.seconds >= 5.0)
        and not _is_loud_enough(
            style_audio,
            clip_length_seconds=5.0,
            clip_stride_seconds=1.0,
            quiet_thresh_db=-25.0,
        )
    ):
      return None

    # Normalize
    if self._peak_norm is not None:
      style_audio.peak_normalize(self._peak_norm, in_place=True)
      codec_audio.peak_normalize(self._peak_norm, in_place=True)

    # Extract features
    acoustic_tokens = self._codec.encode(codec_audio)
    style_embedding = self._style_model.embed(
        style_audio,
        pool_across_time=False,
        pad_end=True,
    )
    style_tokens = self._style_model.tokenize(style_embedding)

    features = {
        'acoustic_tokens': acoustic_tokens,
        'style_tokens': style_tokens,
    }
    if self._include_style_embeddings:
      features['style_embeddings'] = style_embedding
    return features

  def process(self, inputs: audio_lib.Waveform) -> Iterator[tf.train.Example]:
    """Yields tf.Examples from an input waveform."""

    if (
        self._min_clip_seconds is not None
        and self._min_clip_seconds <= inputs.seconds <= self._audio_len_seconds
    ):
      # Allow examples that are shorter than audio_len_seconds
      # if they are at least min_clip_seconds
      clip_slices = [slice(0, len(inputs))]
    else:
      clip_slices = _generate_clip_slices(
          key=None,
          num_samples=len(inputs),
          clip_length_seconds=self._audio_len_seconds,
          sample_rate=inputs.sample_rate,
          max_clips=self._max_clips_per_example,
      )

    for clip_slice in clip_slices:
      clip = inputs[clip_slice]
      features = self._featurize(clip)
      if features is None:
        continue
      else:
        out_features = {}
        for k, v in features.items():
          assert k not in out_features
          out_features[k] = _array_to_feature(v)

        # Create output example
        # TODO(ilariamanco): Migrate to different proto
        out_example = tf.train.Example()
        for k, f in out_features.items():
          out_example.features.feature[k].CopyFrom(f)
        yield out_example
