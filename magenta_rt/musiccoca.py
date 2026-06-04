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
import pathlib
from typing import Any, List, Optional

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import sentencepiece
from typing_extensions import TypeAlias

from . import audio
from . import paths

BatchText: TypeAlias = List[str]
BatchAudio: TypeAlias = List[audio.Waveform]
TextOrAudio: TypeAlias = str | audio.Waveform
BatchTextOrAudio: TypeAlias = List[TextOrAudio]
StyleEmbedding: TypeAlias = np.ndarray
StyleTokens: TypeAlias = np.ndarray
BatchStyleEmbedding: TypeAlias = np.ndarray
BatchStyleTokens: TypeAlias = np.ndarray


def _make_interpreter(model_path: str):
  """Create a TFLite interpreter."""
  interp = Interpreter(model_path=model_path)
  interp.allocate_tensors()
  return interp


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

  @abc.abstractmethod
  def _embed_batch_text(
      self,
      batch_text: BatchText,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> BatchStyleEmbedding:
    """Override to embed a batch of text strings.

    Args:
      batch_text: A list of text strings of length B.
      use_mapper: If True, maps text embeddings to audio-space via mapper.
      seed: Random seed for mapper noise (only used when use_mapper=True).

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

  @abc.abstractmethod
  def tokenize(
      self, embeddings: StyleEmbedding | BatchStyleEmbedding
  ) -> StyleTokens | BatchStyleTokens:
    """Tokenizes a batch of embeddings using RVQ quantization."""
    ...

  def embed_batch_text(
      self,
      batch_text: BatchText,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> BatchStyleEmbedding:
    """Embeds text into a common embedding space.

    Args:
      batch_text: A list of text strings of length B.
      use_mapper: If True, maps text embeddings to audio-space via mapper.
      seed: Random seed for mapper noise (only used when use_mapper=True).

    Returns:
      A batch of style embeddings of shape (B, self.config.embedding_dim).
    """
    # Handle empty list.
    if not batch_text:
      return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
    # Precaution for users who aren't checking types.
    if isinstance(batch_text, str):
      raise TypeError('Called embed_batch_text with a single text string.')
    return self._embed_batch_text(batch_text, use_mapper=use_mapper, seed=seed)

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
      use_mapper: bool = False,
      seed: int = 0,
      **audio_kwargs,
  ) -> StyleEmbedding | BatchStyleEmbedding:
    """Embeds text or audio into a common embedding space.

    Args:
      text_or_audio: A text string, audio waveform, or batch of either.
      pool_across_time: Whether to average audio embeddings across time.
      use_mapper: If True, maps text embeddings to audio-space via mapper.
      seed: Random seed for mapper noise (only used when use_mapper=True).
      **audio_kwargs: Additional kwargs forwarded to embed_batch_audio.
    """
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
    embeddings_text = self.embed_batch_text(
        batch_text, use_mapper=use_mapper, seed=seed
    )
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

  def __call__(self, *args, **kwargs):
    return self.embed(*args, **kwargs)


class MusicCoCa(MusicCoCaBase):
  """A model that embeds audio and text into a common embedding space.

  Uses TFLite interpreters (converted from v1 SavedModels) for inference.
  Expects the following files in the resource directory:
    - spm.model              (SentencePiece vocabulary)
    - text_encoder.tflite    (text → 768-dim embedding)
    - audio_preprocessor.tflite  (raw audio → preprocessed features)
    - music_encoder.tflite       (preprocessed features → 768-dim embedding)
    - pretrained_vector_quantizer.tflite  (768-dim embedding → RVQ tokens)
  """

  def __init__(
      self,
      resource_dir: str | pathlib.Path | None = None,
      lazy: bool = True,
  ):
    super().__init__(
        MusicCoCaConfiguration(
            sample_rate=16000,
            clip_length=10.0,
            embedding_dim=768,
            rvq_depth=12,
            rvq_codebook_size=1024,
        )
    )
    self._resource_dir = pathlib.Path(
        resource_dir or paths.musiccoca_dir()
    )
    if not lazy:
      self._vocab  # pylint: disable=pointless-statement
      self._text_encoder  # pylint: disable=pointless-statement
      self._audio_preprocessor  # pylint: disable=pointless-statement
      self._music_encoder  # pylint: disable=pointless-statement
      self._quantizer  # pylint: disable=pointless-statement
      self.tokenize(self.embed('foo'))  # warm start

  # ---------------------------------------------------------------------------
  # Lazy-loaded TFLite interpreters
  # ---------------------------------------------------------------------------

  @functools.cached_property
  def _vocab(self) -> Any:
    spm_path = self._resource_dir / 'spm.model'
    if not spm_path.exists():
      raise FileNotFoundError(f'SentencePiece model not found at {spm_path}')
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(spm_path))
    return sp

  @functools.cached_property
  def _text_encoder(self) -> Any:
    path = self._resource_dir / 'text_encoder.tflite'
    if not path.exists():
      raise FileNotFoundError(f'Text encoder not found at {path}')
    return _make_interpreter(str(path))

  @functools.cached_property
  def _audio_preprocessor(self) -> Any:
    path = self._resource_dir / 'audio_preprocessor.tflite'
    if not path.exists():
      raise FileNotFoundError(f'Audio preprocessor not found at {path}')
    return _make_interpreter(str(path))

  @functools.cached_property
  def _music_encoder(self) -> Any:
    path = self._resource_dir / 'music_encoder.tflite'
    if not path.exists():
      raise FileNotFoundError(f'Music encoder not found at {path}')
    return _make_interpreter(str(path))

  @functools.cached_property
  def _quantizer(self) -> Any:
    path = self._resource_dir / 'pretrained_vector_quantizer.tflite'
    if not path.exists():
      raise FileNotFoundError(f'Vector quantizer not found at {path}')
    return _make_interpreter(str(path))

  @functools.cached_property
  def _mapper(self) -> Any:
    path = self._resource_dir / 'mapper.tflite'
    if not path.exists():
      raise FileNotFoundError(f'Mapper not found at {path}')
    return _make_interpreter(str(path))

  # ---------------------------------------------------------------------------
  # Text embedding
  # ---------------------------------------------------------------------------

  def _embed_batch_text(
      self,
      batch_text: BatchText,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> BatchStyleEmbedding:
    max_text_length = 128
    target_sos_id = 1
    interpreter = self._text_encoder
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Identify which input is int32 (ids) vs float32 (paddings).
    id_idx = -1
    pad_idx = -1
    for detail in input_details:
      if detail['dtype'] == np.int32:
        id_idx = detail['index']
        id_shape = detail['shape']
      elif detail['dtype'] == np.float32:
        pad_idx = detail['index']
        pad_shape = detail['shape']
    if id_idx == -1 or pad_idx == -1:
      raise ValueError('Could not find required inputs in text encoder')

    embeddings = []
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
      paddings = np.ones(max_text_length, dtype=np.float32)
      paddings[:num_tokens] = 0.0

      interpreter.set_tensor(id_idx, ids.reshape(id_shape))
      interpreter.set_tensor(pad_idx, paddings.reshape(pad_shape))
      interpreter.invoke()

      emb = interpreter.get_tensor(output_details[0]['index'])
      emb = emb.flatten().astype(np.float32)

      if use_mapper:
        mapper = self._mapper
        mapper_input_details = mapper.get_input_details()
        mapper_output_details = mapper.get_output_details()
        rng = np.random.RandomState(seed)
        noise = rng.randn(*emb.shape).astype(np.float32)
        mapper.set_tensor(
            mapper_input_details[0]['index'],
            emb.reshape(mapper_input_details[0]['shape']),
        )
        mapper.set_tensor(
            mapper_input_details[1]['index'],
            noise.reshape(mapper_input_details[1]['shape']),
        )
        mapper.invoke()
        emb = mapper.get_tensor(
            mapper_output_details[0]['index']
        ).flatten().astype(np.float32)
        emb = emb / np.linalg.norm(emb)

      embeddings.append(emb)

    return np.array(embeddings)

  # ---------------------------------------------------------------------------
  # Audio embedding
  # ---------------------------------------------------------------------------

  def _embed_batch_clips(
      self,
      batch_clips: np.ndarray,
  ) -> BatchStyleEmbedding:
    prep = self._audio_preprocessor
    enc = self._music_encoder
    prep_input_details = prep.get_input_details()
    prep_output_details = prep.get_output_details()
    enc_input_details = enc.get_input_details()
    enc_output_details = enc.get_output_details()

    prep_input_shape = prep_input_details[0]['shape']
    prep_input_size = int(np.prod(prep_input_shape))

    embeddings = []
    for clip in batch_clips:
      # Prepare preprocessor input.
      input_data = np.zeros(prep_input_shape, dtype=np.float32)
      flat_input = input_data.flatten()
      n_copy = min(len(clip), prep_input_size)
      flat_input[:n_copy] = clip[:n_copy]
      input_data = flat_input.reshape(prep_input_shape)

      # Run audio preprocessor.
      prep.set_tensor(prep_input_details[0]['index'], input_data)
      prep.invoke()
      prep_output = prep.get_tensor(prep_output_details[0]['index'])

      # Run music encoder.
      enc.set_tensor(enc_input_details[0]['index'], prep_output)
      enc.invoke()
      emb = enc.get_tensor(enc_output_details[0]['index'])

      embeddings.append(emb.flatten().astype(np.float32))

    return np.array(embeddings)

  # ---------------------------------------------------------------------------
  # Tokenization via quantizer TFLite
  # ---------------------------------------------------------------------------

  def tokenize(
      self, embeddings: StyleEmbedding | BatchStyleEmbedding
  ) -> StyleTokens | BatchStyleTokens:
    """Tokenizes embeddings using the pretrained vector quantizer TFLite."""
    if embeddings.shape[-1] != self.config.embedding_dim:
      raise ValueError(
          f'Embedding dimension must be {self.config.embedding_dim}, got'
          f' {embeddings.shape[-1]}.'
      )
    original_shape = embeddings.shape[:-1]
    flat_embeddings = embeddings.reshape((-1, self.config.embedding_dim))

    q = self._quantizer
    q_input_details = q.get_input_details()
    q_output_details = q.get_output_details()

    all_tokens = []
    for emb in flat_embeddings:
      q.set_tensor(
          q_input_details[0]['index'],
          emb.reshape(q_input_details[0]['shape']),
      )
      q.invoke()
      tokens = q.get_tensor(q_output_details[0]['index'])
      all_tokens.append(tokens.flatten()[:self.config.rvq_depth])

    result = np.array(all_tokens, dtype=np.int32)
    return result.reshape(original_shape + (self.config.rvq_depth,))


class MockMusicCoCa(MusicCoCaBase):
  """A mock MusicCoCa model that returns random embeddings and tokens."""

  def __init__(
      self,
      config: MusicCoCaConfiguration = MusicCoCaConfiguration(),
      *args,
      **kwargs,
  ):
    super().__init__(config, *args, **kwargs)

  def tokenize(
      self, embeddings: StyleEmbedding | BatchStyleEmbedding
  ) -> StyleTokens | BatchStyleTokens:
    """Mock tokenization returning deterministic pseudo-random tokens."""
    if embeddings.shape[-1] != self.config.embedding_dim:
      raise ValueError(
          f'Embedding dimension must be {self.config.embedding_dim}, got'
          f' {embeddings.shape[-1]}.'
      )
    seed = int(
        hashlib.sha256(embeddings.tobytes()).hexdigest(), 16
    ) % 2**32
    np.random.seed(seed)
    return np.random.randint(
        0,
        self.config.rvq_codebook_size,
        size=embeddings.shape[:-1] + (self.config.rvq_depth,),
        dtype=np.int32,
    )

  def _embed_batch_text(
      self,
      batch_text: BatchText,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> BatchStyleEmbedding:
    del use_mapper, seed
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
