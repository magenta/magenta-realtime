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

"""MusicCoCa model for embedding music *style* (described by text or audio).

Builds on [Yu+ 22](https://arxiv.org/abs/2205.01917) and
[Huang+ 22](https://arxiv.org/abs/2208.12415).

Example:

```python
from magenta_rt import musiccoca

style_model = musiccoca.MusicCoCa()
prompt1 = style_model.embed('Foo')
prompt2 = style_model.embed('Bar')
tokens = style_model.tokenize(np.mean([prompt1, prompt2], axis=0))
```
"""

import abc
import dataclasses
import functools
import hashlib
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf
from typing_extensions import TypeAlias

from . import asset
from . import audio
from . import utils
import sentencepiece as sentencepiece_processor

BatchText: TypeAlias = List[str]
BatchAudio: TypeAlias = List[audio.Waveform]
TextOrAudio: TypeAlias = str | audio.Waveform
BatchTextOrAudio: TypeAlias = List[TextOrAudio]
StyleEmbedding: TypeAlias = np.ndarray
StyleTokens: TypeAlias = np.ndarray
BatchStyleEmbedding: TypeAlias = np.ndarray
BatchStyleTokens: TypeAlias = np.ndarray


# NOTE: This is the correct ordering to achieve equivalence to reference
# implementation. Variable names were not properly sorted in SavedModel.
MUSICCOCA_RVQ_VAR_ORDER = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]


@dataclasses.dataclass
class MusicCoCaConfiguration:
  """Configuration parameters for MusicCoCa."""

  sample_rate: int = 16000
  clip_length: float = 10.0
  embedding_dim: int = 768
  rvq_depth: int = 12
  rvq_codebook_size: int = 1024

  def __post_init__(self):
    if not (self.clip_length * self.sample_rate).is_integer():
      raise ValueError('Clip length must yield an integer number of samples.')

  @property
  def clip_length_samples(self) -> int:
    return round(self.clip_length * self.sample_rate)


class MusicCoCaBase(abc.ABC):
  """MusicCoCa abstract base class."""

  def __init__(self, config: MusicCoCaConfiguration):
    self._config = config

  @property
  def config(self):
    return self._config

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
  def _embed_batch_text(
      self,
      batch_text: BatchText,
  ) -> BatchStyleEmbedding:
    """Override to embed a batch of text strings.

    Args:
      batch_text: A list of text strings of length B.

    Returns:
      A batch of style embeddings of shape (B, self.config.embedding_dim).
    """
    ...

  @abc.abstractmethod
  def _embed_batch_clips(
      self,
      batch_clips: np.ndarray,
  ) -> BatchStyleEmbedding:
    """Override to embed a batch of audio clips.

    Args:
      batch_clips: A batch of audio clips of shape (B, clip_length_samples).

    Returns:
      A batch of style embeddings of shape (B, self.config.embedding_dim).
    """
    ...

  def embed_batch_text(self, batch_text: BatchText) -> BatchStyleEmbedding:
    """Embeds text into a common embedding space.

    Args:
      batch_text: A list of text strings of length B.

    Returns:
      A batch of style embeddings of shape (B, self.config.embedding_dim).
    """
    # Handle empty list.
    if not batch_text:
      return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
    # Precaution for users who aren't checking types.
    if isinstance(batch_text, str):
      raise TypeError('Called embed_batch_text with a single text string.')
    return self._embed_batch_text(batch_text)

  def embed_batch_audio(
      self,
      batch_audio: BatchAudio,
      hop_length: Optional[float] = None,
      pool_across_time: bool = True,
      pad_end: bool = True,
      mono_strategy: str = 'average',
  ) -> BatchStyleEmbedding:
    """Embeds a batch of audio into a common embedding space.

    Args:
      batch_audio: A list of B audio segments, all of the same length.
      hop_length: The hop length in seconds.
      pool_across_time: Whether to average embeddings across time.
      pad_end: Whether to pad incomplete clips.
      mono_strategy: The strategy to use for converting to mono.

    Returns:
      A batch of style embeddings of shape (B, embedding_dim) if
      pool_across_time is True, otherwise (B, num_clips, embedding_dim).
    """
    # Handle empty list.
    if not batch_audio:
      if pool_across_time:
        return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
      else:
        return np.zeros((0, 0, self.config.embedding_dim), dtype=np.float32)

    # Check that all audio clips are the same length.
    if len(set(len(a) for a in batch_audio)) != 1:
      raise NotImplementedError(
          'Batch embedding of variable-length audio is not currently supported.'
      )

    # Convert to mono and resample.
    batch_audio = [
        a.as_mono(strategy=mono_strategy).resample(self.config.sample_rate)
        for a in batch_audio
    ]

    # Split audio into frames.
    clip_length_samples = self.config.clip_length_samples
    hop_length_samples = (
        self.config.clip_length_samples
        if hop_length is None
        else round(hop_length * self.config.sample_rate)
    )
    audio_length_samples = len(batch_audio[0])
    all_clips = []
    for i in range(0, audio_length_samples, hop_length_samples):
      clips = np.array(
          [a.samples[i : i + clip_length_samples, 0] for a in batch_audio]
      )
      clip_length = clips.shape[-1]
      if clip_length < clip_length_samples:
        if pad_end:
          clips = np.pad(
              clips,
              ((0, 0), (0, clip_length_samples - clip_length)),
              mode='constant',
          )
        else:
          break
      all_clips.append(clips)
    num_audio = len(batch_audio)
    num_clips = len(all_clips)

    if num_clips == 0:
      embeddings = np.zeros(
          (num_audio, 0, self.config.embedding_dim), dtype=np.float32
      )
    else:
      # Aggregate into batch of clip_length_samples.
      # all_clips is (num_clips, num_audio, clip_length_samples)
      # Change to    (num_audio, num_clips, clip_length_samples)
      batch_clips = np.array(all_clips).swapaxes(0, 1)
      assert batch_clips.shape == (num_audio, num_clips, clip_length_samples)

      # Embed audio.
      batch_embeddings = self._embed_batch_clips(
          batch_clips.reshape((num_audio * num_clips, clip_length_samples))
      )
      expected_shape = (num_audio * num_clips, self.config.embedding_dim)
      if batch_embeddings.shape != (expected_shape):
        raise AssertionError(
            f'Audio embedding shape must be {expected_shape}, got'
            f' {batch_embeddings.shape}.'
        )

      # Reshape
      embeddings = batch_embeddings.reshape(
          (num_audio, num_clips, self.config.embedding_dim)
      )
    assert embeddings.shape == (num_audio, num_clips, self.config.embedding_dim)

    # Pool across clips uniformly spaced by hop length.
    if pool_across_time:
      embeddings = np.mean(embeddings, axis=1)

    return embeddings

  def embed(
      self,
      text_or_audio: TextOrAudio | BatchTextOrAudio,
      pool_across_time: bool = True,
      **audio_kwargs,
  ) -> StyleEmbedding | BatchStyleEmbedding:
    """Embeds text or audio into a common embedding space."""
    # Check if input is a singleton or batch.
    if isinstance(text_or_audio, list):
      batch = True
      batch_text_or_audio = text_or_audio
    else:
      batch = False
      batch_text_or_audio = [text_or_audio]

    # Partition text and audio into separate lists.
    batch_indices = []
    batch_text = []
    batch_audio = []
    for x in batch_text_or_audio:
      if isinstance(x, str):
        batch_indices.append((True, len(batch_text)))
        batch_text.append(x)
      else:
        assert isinstance(x, audio.Waveform)
        batch_indices.append((False, len(batch_audio)))
        batch_audio.append(x)

    # Check input compatibility.
    if batch_text and batch_audio and not pool_across_time:
      raise ValueError(
          'Must pool across time when embedding both text and audio.'
      )

    # Embed text.
    embeddings_text = self.embed_batch_text(batch_text)
    assert embeddings_text.shape == (
        len(batch_text),
        self.config.embedding_dim,
    )

    # Embed audio.
    embeddings_audio = self.embed_batch_audio(
        batch_audio, pool_across_time=pool_across_time, **audio_kwargs
    )
    if pool_across_time:
      assert embeddings_audio.shape == (
          len(batch_audio),
          self.config.embedding_dim,
      )
    else:
      assert (
          embeddings_audio.ndim == 3
          and embeddings_audio.shape[0] == len(batch_audio)
          and embeddings_audio.shape[2] == self.config.embedding_dim
      )

    # Combine text and audio embeddings.
    embeddings = [
        embeddings_text[i] if is_text else embeddings_audio[i]
        for is_text, i in batch_indices
    ]
    assert len(set(e.shape for e in embeddings)) <= 1

    if batch:
      return np.array(embeddings)
    else:
      return embeddings[0]

  def tokenize(
      self, embeddings: StyleEmbedding | BatchStyleEmbedding
  ) -> StyleTokens | BatchStyleTokens:
    """Tokenizes a batch of embeddings using RVQ quantization."""
    if embeddings.shape[-1] != self.config.embedding_dim:
      raise ValueError(
          f'Embedding dimension must be {self.config.embedding_dim}, got'
          f' {embeddings.shape[-1]}.'
      )
    tokens = utils.rvq_quantization(
        embeddings.reshape((-1, self.config.embedding_dim)), self.rvq_codebooks
    )[0]
    return tokens.reshape(embeddings.shape[:-1] + (self.config.rvq_depth,))

  def __call__(self, *args, **kwargs):
    return self.embed(*args, **kwargs)


class MusicCoCaV212F(MusicCoCaBase):
  """A model that embeds audio and text into a common embedding space."""

  def __init__(self, lazy: bool = True):
    super().__init__(
        MusicCoCaConfiguration(
            sample_rate=16000,
            clip_length=10.0,
            embedding_dim=768,
            rvq_depth=12,
            rvq_codebook_size=1024,
        )
    )
    if not lazy:
      self._vocab  # pylint: disable=pointless-statement
      self._encoder  # pylint: disable=pointless-statement
      self._rvq_codebooks  # pylint: disable=pointless-statement
      self.tokenize(self.embed('foo'))  # warm start

  @property
  def _encoder_path(self) -> str:
    return 'savedmodels/musiccoca_mv212f_cpu_novocab'

  @property
  def _vocab_path(self) -> str:
    return 'vocabularies/musiccoca_mv212f_vocab.model'

  @property
  def _rvq_codebooks_path(self) -> str:
    return 'savedmodels/musiccoca_mv212_quant'

  @functools.cached_property
  def _encoder(self) -> Any:
    with tf.device('/cpu:0'):
      return utils.load_model_cached(
          'tf',
          asset.fetch(self._encoder_path, is_dir=True),
      )

  @functools.cached_property
  def _vocab(self) -> Any:
    sp = sentencepiece_processor.SentencePieceProcessor()
    sp.Load(asset.fetch(self._vocab_path))
    return sp

  @functools.cached_property
  def _rvq_codebooks(self) -> np.ndarray:
    path = asset.fetch(self._rvq_codebooks_path, is_dir=True)
    var_path = f'{path}/variables/variables'
    result = np.zeros(
        (
            self.config.rvq_depth,
            self.config.rvq_codebook_size,
            self.config.embedding_dim,
        ),
        dtype=np.float32,
    )
    for k, v_name in enumerate(MUSICCOCA_RVQ_VAR_ORDER):
      var = tf.train.load_variable(
          var_path, f'variables/{v_name}/.ATTRIBUTES/VARIABLE_VALUE'
      )
      result[k] = var.T
    return result

  def _embed_batch_text(
      self,
      batch_text: BatchText,
  ) -> BatchStyleEmbedding:
    # Load MusicCoCa encoder.
    emb_text = lambda x, y: self._encoder.signatures['embed_text'](
        inputs_0=x, inputs_0_1=y
    )['contrastive_txt_embed']

    # Embed text.
    embeddings = []
    max_text_length = 128
    target_sos_id = 1
    for s in batch_text:
      # text => lowercase => ids and paddings
      labels = self._vocab.EncodeAsIds(s.lower())
      num_tokens = len(labels)

      labels = labels[: max_text_length - 1]
      num_tokens = min(num_tokens, max_text_length - 1)

      ids = [target_sos_id] + labels
      num_tokens += 1

      # pad ids to the length of max_text_length with pad value 0
      ids = ids + [0] * (max_text_length - len(ids))
      ids = np.array(ids, dtype=np.int32)
      ids = tf.reshape(ids, (1, -1))
      paddings = 1.0 - tf.sequence_mask(
          num_tokens, maxlen=max_text_length, dtype=tf.float32
      )
      paddings = tf.reshape(paddings, (1, -1))
      # ids and paddings => embeddings
      with tf.device('/cpu:0'):
        embeddings.append(emb_text(ids, paddings).numpy()[0])
    return np.array(embeddings)

  def _embed_batch_clips(
      self,
      batch_clips: np.ndarray,
  ) -> BatchStyleEmbedding:
    # Load MusicCoCa encoder.
    emb_audio = lambda x: self._encoder.signatures['embed_music'](inputs_0=x)[
        'contrastive_music_embed'
    ]

    # Embed audio.
    embeddings = []
    for c in batch_clips:
      # TODO(kehanghan): support bs>1 in SavedModel?
      with tf.device('/cpu:0'):
        embeddings.append(emb_audio(tf.constant([c])).numpy()[0])
    return np.array(embeddings)


class MockMusicCoCa(MusicCoCaBase):
  """A mock MusicCoCa model that returns random embeddings and tokens."""

  def __init__(
      self,
      config: MusicCoCaConfiguration = MusicCoCaConfiguration(),
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

  def _embed_batch_text(
      self,
      batch_text: BatchText,
  ) -> BatchStyleEmbedding:
    result = []
    for s in batch_text:
      seed = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 2**32
      np.random.seed(seed)
      result.append(
          np.random.randn(self.config.embedding_dim).astype(np.float32)
      )
    return np.array(result)

  def _embed_batch_clips(
      self,
      batch_clips: np.ndarray,
  ) -> BatchStyleEmbedding:
    result = []
    for c in batch_clips:
      seed = int(hashlib.sha256(c.tobytes()).hexdigest(), 16) % 2**32
      np.random.seed(seed)
      result.append(
          np.random.randn(self.config.embedding_dim).astype(np.float32)
      )
    return np.array(result)


MusicCoCa = MusicCoCaV212F  # Alias to indicate default codepath.
