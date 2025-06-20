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

"""SpectroStream model for audio tokenization.

Paper coming soon.

Example:

```python
from magenta_rt import audio, spectrostream

audio = audio.Waveform(np.random.rand(16000, 2).astype(np.float32), 16000)
codec = spectrostream.SpectroStream()
tokens = codec.encode(audio)
rt = codec.decode(tokens)
```
"""

import abc
import dataclasses
import functools
from typing import Any, List

import jax
import numpy as np
import tensorflow as tf
import tf2jax
from typing_extensions import TypeAlias

from . import asset
from . import audio
from . import utils

AcousticEmbedding: TypeAlias = np.ndarray
AcousticTokens: TypeAlias = np.ndarray
BatchAudioSamples: TypeAlias = np.ndarray
BatchAcousticEmbedding: TypeAlias = np.ndarray
BatchAcousticTokens: TypeAlias = np.ndarray


@dataclasses.dataclass
class SpectroStreamConfiguration:
  """Configuration parameters for SpectroStream."""

  sample_rate: int = 48000
  num_channels: int = 2
  frame_rate: float = 25.0
  embedding_dim: int = 256
  rvq_depth: int = 64
  rvq_codebook_size: int = 1024


class SpectroStreamBase(abc.ABC):
  """SpectroStream abstract base class."""

  def __init__(self, config: SpectroStreamConfiguration):
    self._config = config

  @property
  def config(self):
    return self._config

  @property
  def sample_rate(self) -> int:
    return self.config.sample_rate

  @property
  def num_channels(self) -> int:
    return self.config.num_channels

  @property
  def frame_rate(self) -> float:
    return self.config.frame_rate

  @property
  @abc.abstractmethod
  def _rvq_codebooks(self) -> np.ndarray:
    ...

  @functools.cached_property
  def rvq_codebooks(self) -> np.ndarray:
    """Returns the RVQ codebooks."""
    rvq_codebooks = self._rvq_codebooks
    if rvq_codebooks.shape != (
        self.config.rvq_depth,
        self.config.rvq_codebook_size,
        self.config.embedding_dim,
    ):
      raise ValueError(
          'rvq_codebooks shape must be equal to (rvq_depth, rvq_codebook_size,'
          ' style_embedding_dim).'
      )
    return rvq_codebooks

  @abc.abstractmethod
  def _embed_batch(self, samples: BatchAudioSamples) -> BatchAcousticEmbedding:
    ...

  @abc.abstractmethod
  def _reconstruct_batch(
      self, embeddings: BatchAcousticEmbedding
  ) -> BatchAudioSamples:
    ...

  def encode(
      self, waveforms: audio.Waveform | List[audio.Waveform]
  ) -> AcousticTokens | BatchAcousticTokens:
    """Encodes a waveform or a batch of waveforms into acoustic tokens."""
    # Check shape
    batch = isinstance(waveforms, list)
    waveform_batch = waveforms if batch else [waveforms]
    batch_size = len(waveform_batch)
    if any(w.num_channels != self.config.num_channels for w in waveform_batch):
      raise ValueError('All waveforms must have the same number of channels.')
    if len(set(w.sample_rate for w in waveform_batch)) != 1:
      raise ValueError('All waveforms must have the same sample rate.')
    if len(set(w.num_samples for w in waveform_batch)) != 1:
      raise ValueError('All waveforms must have the same number of samples.')

    # Resample and stack into [B, T, C] float32
    # T = num samples, C = num channels
    waveform_batch = [
        w.resample(self.config.sample_rate) for w in waveform_batch
    ]
    samples = np.stack([w.samples for w in waveform_batch])
    length_seconds = samples.shape[1] / self.config.sample_rate

    # Embed to [B, S, D] float32
    # S = num frames (T // frame_rate), D = embedding dim
    embeddings = self._embed_batch(samples)
    expected_num_frames = int(np.ceil(length_seconds * self.config.frame_rate))
    expected_shape = (
        batch_size,
        expected_num_frames,
        self.config.embedding_dim,
    )
    if embeddings.shape != expected_shape:
      raise AssertionError(
          f'Expected shape is {expected_shape}, but got {embeddings.shape}.'
      )

    # Tokenize to [B, S, K] int32
    # K = rvq depth
    result, _ = utils.rvq_quantization(
        embeddings.reshape(-1, self.config.embedding_dim), self.rvq_codebooks
    )
    result = result.reshape(batch_size, expected_num_frames, -1)
    return result if batch else result[0]

  def decode(
      self, tokens: AcousticTokens | BatchAcousticTokens
  ) -> audio.Waveform | List[audio.Waveform]:
    """Decodes acoustic tokens into waveforms."""
    # Check shape is [B, S, K] int32
    # K = rvq depth
    batch = tokens.ndim == 3
    token_batch = tokens if batch else tokens[np.newaxis]
    if token_batch.ndim != 3:
      raise ValueError('tokens must be a 3D array, got {tokens.shape}')
    if token_batch.shape[2] > self.config.rvq_depth:
      raise ValueError(
          f'token depth ({token_batch.shape[2]}) is greater than the quantizer'
          f' depth ({self.config.rvq_depth})'
      )

    # Dequantize to [B, S, D] float32
    # D = embedding dim
    batch_size, num_frames, tokens_depth = token_batch.shape
    embeddings = utils.rvq_dequantization(
        tokens.reshape(-1, tokens_depth), self.rvq_codebooks
    )
    embeddings = embeddings.reshape(
        batch_size, num_frames, self.config.embedding_dim
    )

    # Reconstruct to [B, T, C] float32
    # T = num samples, C = num channels
    samples = self._reconstruct_batch(embeddings)
    expected_num_samples = int(
        np.ceil((num_frames / self.config.frame_rate) * self.config.sample_rate)
    )
    expected_shape = (
        batch_size,
        expected_num_samples,
        self.config.num_channels,
    )
    if samples.shape != expected_shape:
      raise AssertionError(
          f'Expected shape is {expected_shape}, but got {samples.shape}.'
      )

    waveforms = [audio.Waveform(s, self.config.sample_rate) for s in samples]
    return waveforms if batch else waveforms[0]


class SpectroStreamSavedModel(SpectroStreamBase):
  """A SpectroStream model that tokenizes audio."""

  def __init__(
      self, max_rvq_depth: int = 64, skip_cache: bool = False, lazy: bool = True
  ):
    if max_rvq_depth < 0 or max_rvq_depth > 64:
      raise ValueError('max_rvq_depth must be in the range [0, 64].')
    super().__init__(
        SpectroStreamConfiguration(
            sample_rate=48000,
            frame_rate=25.0,
            embedding_dim=256,
            rvq_depth=max_rvq_depth,
            rvq_codebook_size=1024,
        )
    )
    self._skip_cache = skip_cache
    if not lazy:
      self._encoder  # pylint: disable=pointless-statement
      self._decoder  # pylint: disable=pointless-statement
      self._rvq_codebooks  # pylint: disable=pointless-statement
      silence = audio.Waveform(
          samples=np.zeros(
              (self.sample_rate, self.num_channels), dtype=np.float32
          ),
          sample_rate=self.sample_rate,
      )
      self.decode(self.encode(silence))  # warm start

  @functools.cached_property
  def _encoder(self) -> Any:
    return utils.load_model_cached(
        'tf',
        asset.fetch(
            'savedmodels/ssv2_48k_stereo/encoder',
            skip_cache=self._skip_cache,
            is_dir=True,
        ),
    )

  @functools.cached_property
  def _decoder(self) -> Any:
    return utils.load_model_cached(
        'tf',
        asset.fetch(
            'savedmodels/ssv2_48k_stereo/decoder',
            skip_cache=self._skip_cache,
            is_dir=True,
        ),
    )

  @functools.cached_property
  def _rvq_codebooks(self) -> np.ndarray:
    quantizer = utils.load_model_cached(
        'tf',
        asset.fetch(
            'savedmodels/ssv2_48k_stereo/quantizer',
            skip_cache=self._skip_cache,
            is_dir=True,
        ),
    )
    result = np.zeros(
        (
            self.config.rvq_depth,
            self.config.rvq_codebook_size,
            self.config.embedding_dim,
        ),
        dtype=np.float32,
    )
    for i in range(self.config.rvq_depth):
      var = quantizer._quantizers[i].embeddings  # pylint: disable=protected-access
      result[i] = var.numpy().T
    return result

  def _embed_batch(self, samples: BatchAudioSamples) -> BatchAcousticEmbedding:
    return self._encoder(samples).cpu().numpy()

  def _reconstruct_batch(
      self, embeddings: BatchAcousticEmbedding
  ) -> BatchAudioSamples:
    return self._decoder(embeddings).cpu().numpy()


class MockSpectroStream(SpectroStreamBase):
  """A mock SpectroStream model that returns random embeddings and tokens."""

  def __init__(
      self,
      config: SpectroStreamConfiguration = SpectroStreamConfiguration(),
      *args,
      **kwargs,
  ):
    super().__init__(config, *args, **kwargs)

  @property
  def _rvq_codebooks(self) -> np.ndarray:
    np.random.seed(0)
    return np.random.randn(
        self.config.rvq_depth,
        self.config.rvq_codebook_size,
        self.config.embedding_dim,
    ).astype(np.float32)

  def _embed_batch(self, samples: BatchAudioSamples) -> BatchAcousticEmbedding:
    num_seconds = samples.shape[1] / self.config.sample_rate
    num_frames = int(np.ceil(num_seconds * self.config.frame_rate))
    return np.random.randn(
        samples.shape[0], num_frames, self.config.embedding_dim
    ).astype(np.float32)

  def _reconstruct_batch(
      self, embeddings: BatchAcousticEmbedding
  ) -> BatchAudioSamples:
    num_seconds = embeddings.shape[1] / self.config.frame_rate
    num_samples = int(np.ceil(num_seconds * self.config.sample_rate))
    return np.random.randn(
        embeddings.shape[0], num_samples, self.config.num_channels
    ).astype(np.float32)


class SpectroStreamJAX(SpectroStreamSavedModel):
  """A tf2jax wrapped SpectroStream model."""

  @functools.cached_property
  def _encoder(self) -> Any:
    encoder = super()._encoder
    encode_fn, encoder_params = tf2jax.convert(
        encoder.__call__,
        tf.zeros((1, 48000, 2), tf.float32),
    )
    return functools.partial(jax.jit(encode_fn), encoder_params)

  @functools.cached_property
  def _decoder(self) -> Any:
    decoder = super()._decoder
    decode_fn, decoder_params = tf2jax.convert(
        decoder.__call__,
        tf.zeros((1, 50, self.config.embedding_dim), tf.float32),
    )
    return functools.partial(jax.jit(decode_fn), decoder_params)

  def _embed_batch(self, samples: BatchAudioSamples) -> BatchAcousticEmbedding:
    with tf2jax.override_config('strict_shape_check', False):
      embeddings = np.asarray(self._encoder(samples)[0])
    return embeddings

  def _reconstruct_batch(
      self, embeddings: BatchAcousticEmbedding
  ) -> BatchAudioSamples:
    with tf2jax.override_config('strict_shape_check', False):
      samples = np.asarray(self._decoder(embeddings)[0])
    return samples


SpectroStream = SpectroStreamSavedModel  # Alias to indicate default codepath.
