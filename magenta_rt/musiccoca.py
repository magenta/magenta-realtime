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
prompt1 = style_model.embed_text('Foo')              # [embedding_dim]
prompt2 = style_model.embed_text('Bar')
tokens = style_model.tokenize(np.mean([prompt1, prompt2], axis=0))
# Batches: embed_text(['Foo', 'Bar']) -> [2, dim];
#          embed_audio(audio_tree)    -> [B, dim]  (AudioTree carries the batch)
```
"""

import abc
import dataclasses
import functools
import hashlib
import pathlib
from typing import Any, List, Optional, Sequence

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import sentencepiece

from audiotree import AudioTree

from . import paths


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
      batch_text: List[str],
      use_mapper: bool = False,
      seed: int = 0,
  ) -> np.ndarray:
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
  ) -> np.ndarray:
    """Override to embed a batch of audio clips.

    Args:
      batch_clips: A batch of audio clips of shape (B, clip_length_samples).

    Returns:
      A batch of style embeddings of shape (B, self.config.embedding_dim).
    """
    ...

  @abc.abstractmethod
  def tokenize(
      self, embeddings: np.ndarray
  ) -> np.ndarray:
    """Tokenizes a batch of embeddings using RVQ quantization."""
    ...

  def embed_text(
      self,
      text: str | Sequence[str],
      *,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> np.ndarray:
    """Embeds a text prompt (or a batch of them) into the style space.

    Args:
      text: A single prompt string, or a sequence of B prompts.
      use_mapper: If True, maps text embeddings to audio-space via the mapper.
      seed: Random seed for mapper noise (only used when use_mapper=True).

    Returns:
      ``[embedding_dim]`` for a single string, or ``[B, embedding_dim]`` for a
      sequence (``[0, embedding_dim]`` for an empty sequence).
    """
    single = isinstance(text, str)
    batch = [text] if single else list(text)
    if not batch:
      return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
    embeddings = self._embed_batch_text(batch, use_mapper=use_mapper, seed=seed)
    return embeddings[0] if single else embeddings

  def embed_audio(
      self,
      audio: AudioTree,
      *,
      hop_length: Optional[float] = None,
      pool_across_time: bool = True,
      pad_end: bool = True,
      mono_strategy: str = 'average',
  ) -> np.ndarray:
    """Embeds a (batched) ``AudioTree`` into the style space.

    The ``AudioTree`` carries its own batch axis (``waveform`` ``[B, C, T]``),
    so there is no list to pass; mono mixdown + resample happen once over the
    whole batch. Each example is split into ``clip_length`` windows spaced by
    ``hop_length`` (default: non-overlapping), each window embedded, then
    optionally averaged across windows.

    Args:
      audio: An ``AudioTree`` with ``waveform`` ``[B, C, T]`` (any channel
        count / sample rate; mixed to mono and resampled internally).
      hop_length: Window hop in seconds (default: the clip length, i.e. no
        overlap).
      pool_across_time: Average the per-window embeddings (else keep them).
      pad_end: Zero-pad a trailing partial window (else drop it).
      mono_strategy: How to mix down to mono (see ``AudioTree.to_mono``).

    Returns:
      ``[B, embedding_dim]`` if ``pool_across_time`` else
      ``[B, num_windows, embedding_dim]``.
    """
    audio = audio.to_mono(strategy=mono_strategy).resample(self.config.sample_rate)
    mono = np.asarray(audio.waveform)[:, 0, :]  # [B, T]
    batch, total_samples = mono.shape

    clip_len = self.config.clip_length_samples
    hop = (
        clip_len if hop_length is None
        else round(hop_length * self.config.sample_rate)
    )

    # Slice [B, T] into windows -> a list of [B, clip_len] (zero-pad/drop tail).
    windows = []
    for i in range(0, total_samples, hop):
      window = mono[:, i : i + clip_len]
      if window.shape[-1] < clip_len:
        if not pad_end:
          break
        window = np.pad(window, ((0, 0), (0, clip_len - window.shape[-1])))
      windows.append(window)
    num_windows = len(windows)

    if num_windows == 0:
      embeddings = np.zeros(
          (batch, 0, self.config.embedding_dim), dtype=np.float32
      )
    else:
      # [B, num_windows, clip_len] -> embed flat -> [B, num_windows, dim].
      batch_clips = np.stack(windows, axis=1)
      flat = self._embed_batch_clips(
          batch_clips.reshape(batch * num_windows, clip_len)
      )
      expected = (batch * num_windows, self.config.embedding_dim)
      if flat.shape != expected:
        raise AssertionError(
            f'Audio embedding shape must be {expected}, got {flat.shape}.'
        )
      embeddings = flat.reshape(batch, num_windows, self.config.embedding_dim)

    if pool_across_time:
      if num_windows == 0:
        return np.zeros((batch, self.config.embedding_dim), dtype=np.float32)
      return np.mean(embeddings, axis=1)
    return embeddings

  def __call__(self, text_or_audio: str | AudioTree, **kwargs):
    """Convenience dispatch for a single input: ``str`` -> :meth:`embed_text`,
    ``AudioTree`` -> :meth:`embed_audio`. For batches call those directly
    (``embed_text([...])`` or ``embed_audio(batched_audio_tree)``)."""
    if isinstance(text_or_audio, AudioTree):
      return self.embed_audio(text_or_audio, **kwargs)
    return self.embed_text(text_or_audio, **kwargs)


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
      self.tokenize(self.embed_text('foo'))  # warm start

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
      batch_text: List[str],
      use_mapper: bool = False,
      seed: int = 0,
  ) -> np.ndarray:
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
  ) -> np.ndarray:
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
      self, embeddings: np.ndarray
  ) -> np.ndarray:
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
      self, embeddings: np.ndarray
  ) -> np.ndarray:
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
      batch_text: List[str],
      use_mapper: bool = False,
      seed: int = 0,
  ) -> np.ndarray:
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
  ) -> np.ndarray:
    result = []
    for c in batch_clips:
      seed = int(hashlib.sha256(c.tobytes()).hexdigest(), 16) % 2**32
      np.random.seed(seed)
      result.append(
          np.random.randn(self.config.embedding_dim).astype(np.float32)
      )
    return np.array(result)
