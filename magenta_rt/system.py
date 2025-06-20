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

"""Defines the actual Magenta RT system."""

import abc
import dataclasses
import functools
from typing import Callable, Optional, Tuple
import warnings

import jax
import numpy as np
from typing_extensions import TypeAlias

from . import asset
from . import audio
from . import musiccoca
from . import spectrostream
from . import utils
from .depthformer import model


@dataclasses.dataclass
class MagentaRTConfiguration:
  """Configuration parameters for Magenta RT."""

  chunk_length: float = 2.0
  context_length: float = 10.0
  crossfade_length: float = 0.04
  codec_sample_rate: int = 48000
  codec_frame_rate: float = 25.0
  codec_rvq_codebook_size: int = 1024
  style_rvq_codebook_size: int = 1024
  encoder_codec_rvq_depth: int = 4
  encoder_style_rvq_depth: int = 6
  decoder_codec_rvq_depth: int = 16

  def __post_init__(self):
    if not (self.context_length / self.chunk_length).is_integer():
      raise ValueError(
          "Context length must be an integer multiple of chunk length."
      )
    for t in [
        self.chunk_length,
        self.context_length,
        self.crossfade_length,
        1 / self.codec_frame_rate,
    ]:
      if t <= 0:
        raise ValueError(f"All lengths must be positive: {t}")
      if not (t * self.codec_sample_rate).is_integer():
        raise ValueError(f"Length * sample_rate must be an integer: {t}")
      if not (t * self.codec_frame_rate).is_integer():
        raise ValueError(f"Length * frame_rate must be an integer: {t}")

  @property
  def context_num_chunks(self) -> int:
    return round(self.context_length / self.chunk_length)

  @property
  def frame_length_samples(self) -> int:
    return round(self.codec_sample_rate / self.codec_frame_rate)

  @property
  def chunk_length_samples(self) -> int:
    return round(self.chunk_length * self.codec_sample_rate)

  @property
  def chunk_length_frames(self) -> int:
    return round(self.chunk_length * self.codec_frame_rate)

  @property
  def context_length_frames(self) -> int:
    return round(self.context_length * self.codec_frame_rate)

  @property
  def crossfade_length_samples(self) -> int:
    return round(self.crossfade_length * self.codec_sample_rate)

  @property
  def crossfade_length_frames(self) -> int:
    return round(self.crossfade_length * self.codec_frame_rate)

  @property
  def chunk_tokens_shape(self) -> Tuple[int, ...]:
    return (self.chunk_length_frames, self.decoder_codec_rvq_depth)

  @property
  def context_tokens_shape(self) -> Tuple[int, ...]:
    return (self.context_length_frames, self.decoder_codec_rvq_depth)

  @property
  def vocab_reserved_tokens(self) -> dict[str, int]:
    return {
        "PAD": 0,
        "MASK": 1,
    }

  @property
  def vocab_pad_token(self) -> int:
    return self.vocab_reserved_tokens["PAD"]

  @property
  def vocab_mask_token(self) -> int:
    return self.vocab_reserved_tokens["MASK"]

  @property
  def vocab_codec_offset(self) -> int:
    return len(self.vocab_reserved_tokens)

  @property
  def vocab_codec_size(self) -> int:
    return self.decoder_codec_rvq_depth * self.codec_rvq_codebook_size

  @property
  def vocab_style_offset(self) -> int:
    return self.vocab_codec_offset + self.vocab_codec_size + 1024  # 1024 unused

  @property
  def vocab_style_size(self) -> int:
    return self.encoder_style_rvq_depth * self.style_rvq_codebook_size


class MagentaRTState:
  """State management for Magenta RT."""

  def __init__(
      self,
      config: MagentaRTConfiguration,
      context_tokens: Optional[np.ndarray] = None,
      chunk_index: int = 0,
  ):
    self._config = config
    if context_tokens is None:
      context_tokens = np.full(
          self._config.context_tokens_shape, -1, dtype=np.int32
      )
    self.context_tokens = context_tokens
    self._chunk_index = chunk_index

  @property
  def context_tokens(self) -> np.ndarray:
    assert hasattr(self, "_context_tokens")
    return self._context_tokens

  @property
  def chunk_index(self) -> int:
    return self._chunk_index

  @property
  def shape(self) -> tuple[int, ...]:
    assert self.context_tokens.shape == self._config.context_tokens_shape
    return self._config.context_tokens_shape

  @context_tokens.setter
  def context_tokens(self, value: np.ndarray):
    if value.dtype != np.int32:
      raise TypeError(f"Context tokens must be int32. Got {value.dtype}")
    if value.shape != self._config.context_tokens_shape:
      raise ValueError(
          f"Context tokens must be {self._config.context_tokens_shape}. Got"
          f" {value.shape}"
      )
    self._context_tokens = value

  def update(self, chunk_tokens: np.ndarray):
    """Updates the state with the tokens from the next chunk."""
    if chunk_tokens.dtype != np.int32:
      raise TypeError(f"Chunk tokens must be int32. Got {chunk_tokens.dtype}")
    if not (
        chunk_tokens.ndim == 2
        and chunk_tokens.shape[0] <= self._config.context_length_frames
        and chunk_tokens.shape[1] == self._config.decoder_codec_rvq_depth
    ):
      raise ValueError("Invalid chunk tokens shape. Got {chunk_tokens.shape}")
    if np.any(
        np.logical_or(
            chunk_tokens < 0,
            chunk_tokens >= self._config.codec_rvq_codebook_size,
        )
    ):
      raise ValueError(
          "Chunk tokens must be in the range [0,"
          f" {self._config.codec_rvq_codebook_size}). Got {chunk_tokens}"
      )
    self.context_tokens = np.concatenate(
        [self.context_tokens[chunk_tokens.shape[0] :], chunk_tokens],
        axis=0,
    )
    self._chunk_index += 1


class MagentaRTBase(abc.ABC):
  """Magenta RT abstract base class."""

  def __init__(
      self,
      config: MagentaRTConfiguration,
      codec: spectrostream.SpectroStreamBase,
      style_model: musiccoca.MusicCoCaBase,
  ):
    self._config = config
    self._codec = codec
    self._style_model = style_model

    # Check consistency of config and codec
    if any(
        d > self.codec.config.rvq_depth
        for d in [
            config.encoder_codec_rvq_depth,
            config.decoder_codec_rvq_depth,
        ]
    ):
      raise ValueError(
          "RVQ depth must be at least as large as the codec RVQ depth:"
          f" {config.encoder_codec_rvq_depth},"
          f" {config.decoder_codec_rvq_depth}"
      )
    if config.codec_sample_rate != self.codec.config.sample_rate:
      raise ValueError(
          "Codec sample rate must match the configuration sample rate."
      )
    if config.codec_frame_rate != self.codec.config.frame_rate:
      raise ValueError(
          "Codec frame rate must match the configuration frame rate."
      )
    if config.codec_rvq_codebook_size != self.codec.config.rvq_codebook_size:
      raise ValueError(
          "Codec RVQ codebook size must match the configuration RVQ codebook"
          " size."
      )
    # Check consistency of config and style model
    if config.encoder_style_rvq_depth > self.style_model.config.rvq_depth:
      raise ValueError(
          "Style RVQ depth must be at least as large as the style model RVQ"
          " depth."
      )
    if (
        config.style_rvq_codebook_size
        != self.style_model.config.rvq_codebook_size
    ):
      raise ValueError(
          "Style RVQ codebook size must match the configuration RVQ"
          " codebook size."
      )

  @property
  def config(self):
    return self._config

  @property
  def sample_rate(self) -> int:
    return self.codec.sample_rate

  @property
  def num_channels(self) -> int:
    return self.codec.num_channels

  @property
  def chunk_length(self) -> float:
    return self.config.chunk_length

  @property
  def codec(self):
    return self._codec

  @property
  def style_model(self):
    return self._style_model

  def init_state(self) -> MagentaRTState:
    return MagentaRTState(config=self.config)

  def embed_style(
      self, text_or_audio: str | audio.Waveform
  ) -> musiccoca.StyleEmbedding:
    result = self._style_model.embed(text_or_audio)
    assert not isinstance(result, list)
    return result

  @abc.abstractmethod
  def generate_chunk(
      self,
      state: Optional[MagentaRTState] = None,
      style: Optional[musiccoca.StyleEmbedding] = None,
      seed: Optional[int] = None,
      **kwargs,
  ) -> Tuple[audio.Waveform, MagentaRTState]:
    ...

  def __call__(self, *args, **kwargs):
    return self.generate_chunk(*args, **kwargs)


class MockMagentaRT(MagentaRTBase):
  """Mock stateless Magenta RT system that just serves noise."""

  def __init__(
      self,
      *args,
      config: MagentaRTConfiguration = MagentaRTConfiguration(),
      codec_config: spectrostream.SpectroStreamConfiguration = spectrostream.SpectroStreamConfiguration(),
      style_config: musiccoca.MusicCoCaConfiguration = musiccoca.MusicCoCaConfiguration(),
      **kwargs,
  ):
    super().__init__(
        *args,
        config=config,
        codec=spectrostream.MockSpectroStream(codec_config),
        style_model=musiccoca.MockMusicCoCa(style_config),
        **kwargs,
    )

  def generate_chunk(
      self,
      state: Optional[MagentaRTState] = None,
      style: Optional[musiccoca.StyleEmbedding] = None,
      seed: Optional[int] = None,
      **kwargs,
  ) -> Tuple[audio.Waveform, MagentaRTState]:
    # Init state and style (if not provided)
    if state is None:
      state = self.init_state()
    if style is None:
      style = np.zeros(
          (self.style_model.config.embedding_dim,), dtype=np.float32
      )

    # Tokenize style
    style_tokens = self.style_model.tokenize(style)
    del style_tokens

    # Generate audio and tokens (random noise)
    if seed is not None:
      np.random.seed(seed + state.chunk_index)
    result = audio.Waveform(
        samples=np.random.rand(
            self.config.chunk_length_samples, self.num_channels
        ),
        sample_rate=self.sample_rate,
    )
    tokens = np.random.randint(
        low=0,
        high=self.config.codec_rvq_codebook_size,
        size=(
            self.config.chunk_length_frames,
            self.config.decoder_codec_rvq_depth,
        ),
        dtype=np.int32,
    )

    # Update state
    state.update(tokens)

    return result, state


# _DeviceParams is (batch_size, num partitions, model_parallel_submesh)
_DeviceParams: TypeAlias = tuple[
    int, Optional[int], Optional[tuple[int, int, int, int]]
]
_DEVICE_TO_CONFIGURATION: dict[str, _DeviceParams] = {
    "gpu": (2, 1, None),
    "tpu:v2-8": (2, None, (2, 1, 1, 2)),
}


class MagentaRTT5X(MagentaRTBase):
  """Actual Magenta RT system via t5x InteractiveModel."""

  def __init__(
      self,
      *args,
      tag: str = "large",
      guidance_weight: float = 5.0,
      temperature: float = 1.1,
      topk: int = 40,
      device: Optional[str | _DeviceParams] = None,
      checkpoint_dir: Optional[str] = None,
      skip_cache: bool = False,
      lazy: bool = True,
      **kwargs,
  ):
    """Initializes the Magenta RT system based on `t5x.InteractiveModel`.

    Args:
      *args: Additional arguments for the base class.
      tag: The pre-trained checkpoint to use, one of ["base", "large"].
      guidance_weight: The default weight of classifier free guidance inference.
      temperature: The default temperature during inference.
      topk: The default topk parameter during inference.
      device: The device to use, or None for CPU.
      checkpoint_dir: If specified, overrides the default checkpoint directory.
      skip_cache: If True, skip fetch to disk and load checkpoint from GCP.
      lazy: Whether to load the LLM lazily.
      **kwargs: Additional keyword arguments for the base class.
    """
    if tag not in ["base", "large"]:
      raise ValueError(f"Unsupported tag: {tag}")
    if isinstance(device, str) and device not in _DEVICE_TO_CONFIGURATION:
      raise ValueError(f"Unsupported device: {device}")
    codec = spectrostream.SpectroStreamJAX(skip_cache=skip_cache, lazy=lazy)
    style_model = musiccoca.MusicCoCa(skip_cache=skip_cache, lazy=lazy)
    super().__init__(
        *args,
        config=MagentaRTConfiguration(
            chunk_length=2.0,
            context_length=10.0,
            crossfade_length=0.04,
            codec_sample_rate=codec.sample_rate,
            codec_frame_rate=codec.frame_rate,
            codec_rvq_codebook_size=codec.config.rvq_codebook_size,
            style_rvq_codebook_size=style_model.config.rvq_codebook_size,
            encoder_codec_rvq_depth=4,
            encoder_style_rvq_depth=6,
            decoder_codec_rvq_depth=16,
        ),
        codec=codec,
        style_model=style_model,
        **kwargs,
    )
    self._tag = tag
    self._guidance_weight = guidance_weight
    self._temperature = temperature
    self._topk = topk
    self._device = device
    self._skip_cache = skip_cache
    self._checkpoint_dir = checkpoint_dir
    if not lazy:
      self.warm_start()

  @property
  def _device_params(self) -> _DeviceParams:
    """Returns the (batch size, num partitions, model parallel submesh)."""
    if self._device is None:
      # Default batch size is 2 to support classifier free guidance (CFG).
      device_params = (2, 1, None)
    elif isinstance(self._device, str):
      device_params = _DEVICE_TO_CONFIGURATION[self._device]
    else:
      device_params = self._device
    return device_params

  @functools.cached_property
  def _llm(self) -> Callable:  # pylint: disable=g-bare-generic
    """Loads the t5x.InteractiveModel."""
    if self._checkpoint_dir is None:
      if self._tag == "base":
        path = "checkpoints/llm_base_x4286_c1860k"
      else:
        path = "checkpoints/llm_large_x3047_c1860k"
      checkpoint_dir = asset.fetch(
          path, is_dir=True, skip_cache=self._skip_cache, parallelism=16
      )
    else:
      checkpoint_dir = self._checkpoint_dir
    batch_size, num_partitions, model_parallel_submesh = self._device_params
    return model.load_pretrained_model(
        checkpoint_dir=checkpoint_dir,
        size=self._tag,
        default_guidance_weight=self._guidance_weight,
        default_temperature=self._temperature,
        default_topk=self._topk,
        batch_size=batch_size,
        num_partitions=num_partitions,
        model_parallel_submesh=model_parallel_submesh,
    )

  def warm_start(self):
    """Warm starts the system by generating a chunk."""
    self._llm  # pylint: disable=pointless-statement
    style = self.embed_style("a tree falls in the forest")
    self.generate_chunk(style=style)

  def generate_chunk(
      self,
      state: Optional[MagentaRTState] = None,
      style: Optional[musiccoca.StyleEmbedding] = None,
      seed: Optional[int] = None,
      **kwargs,
  ) -> Tuple[audio.Waveform, MagentaRTState]:
    """Generates a chunk of audio and returns updated state.

    Args:
      state: The current state of the system.
      style: The style embedding to use for the generation.
      seed: The seed to use for the generation.
      **kwargs: Additional keyword arguments for sampling params, e.g.
        temperature, topk, guidance_weight, max_decode_frames.

    Returns:
      A tuple of the generated audio and the updated state.
    """
    # Init state, style, and seed (if not provided)
    if state is None:
      state = self.init_state()
    if seed is None:
      seed = np.random.randint(0, 2**31)

    # Prepare codec tokens for LLM
    codec_tokens_lm = np.where(
        state.context_tokens >= 0,
        utils.rvq_to_llm(
            np.maximum(state.context_tokens, 0),
            self.config.codec_rvq_codebook_size,
            self.config.vocab_codec_offset,
        ),
        np.full_like(state.context_tokens, self.config.vocab_mask_token),
    )
    assert (
        codec_tokens_lm.shape == self.config.context_tokens_shape
    )  # (250, 16)
    assert (
        codec_tokens_lm.min() >= self.config.vocab_mask_token
        and codec_tokens_lm.max()
        < (self.config.vocab_codec_offset + self.config.vocab_codec_size)
    )  # check range [1, 16386)

    # Prepare style tokens for LLM
    if style is None:
      style_tokens_lm = np.full(
          (self.config.encoder_style_rvq_depth,),
          self.config.vocab_mask_token,
          dtype=np.int32,
      )
    else:
      if style.shape != (self.style_model.config.embedding_dim,):
        raise ValueError(f"Invalid style shape: {style.shape}")
      style_tokens = self.style_model.tokenize(style)
      assert style_tokens.shape == (self.style_model.config.rvq_depth,)
      style_tokens = style_tokens[: self.config.encoder_style_rvq_depth]
      style_tokens_lm = utils.rvq_to_llm(
          style_tokens,
          self.config.style_rvq_codebook_size,
          self.config.vocab_style_offset,
      )
      assert (
          style_tokens_lm.min() >= self.config.vocab_style_offset
          and style_tokens_lm.max()
          < (self.config.vocab_style_offset + self.config.vocab_style_size)
      )  # check range [17140, 23554)
    assert style_tokens_lm.shape == (
        self.config.encoder_style_rvq_depth,
    )  # (6,)

    # Prepare encoder input
    batch_size, _, _ = self._device_params
    encoder_inputs_pos = np.concatenate(
        [
            codec_tokens_lm[:, : self.config.encoder_codec_rvq_depth].reshape(
                -1
            ),
            style_tokens_lm,
        ],
        axis=0,
    )
    assert encoder_inputs_pos.shape == (1006,)
    encoder_inputs_neg = encoder_inputs_pos.copy()
    encoder_inputs_neg[-self.config.encoder_style_rvq_depth :] = (
        self.config.vocab_mask_token
    )
    assert encoder_inputs_neg.shape == (1006,)
    encoder_inputs = np.stack([encoder_inputs_pos, encoder_inputs_neg], axis=0)
    assert encoder_inputs.shape == (2, 1006)

    # Generate tokens / NLL scores.
    max_decode_frames = kwargs.get(
        "max_decode_frames", self.config.chunk_length_frames
    )
    generated_tokens, _ = self._llm(
        {
            "encoder_input_tokens": encoder_inputs,
            "decoder_input_tokens": np.zeros(
                (
                    batch_size,
                    self.config.chunk_length_frames
                    * self.config.decoder_codec_rvq_depth,
                ),
                dtype=np.int32,
            ),
        },
        {
            "max_decode_steps": np.array(
                max_decode_frames * self.config.decoder_codec_rvq_depth,
                dtype=np.int32,
            ),
            "guidance_weight": kwargs.get(
                "guidance_weight", self._guidance_weight
            ),
            "temperature": kwargs.get("temperature", self._temperature),
            "topk": kwargs.get("topk", self._topk),
        },
        jax.random.PRNGKey(seed + state.chunk_index),
    )

    # Process generated tokens
    generated_tokens = np.array(generated_tokens)
    assert generated_tokens.shape == (
        batch_size,
        self.config.chunk_length_frames * self.config.decoder_codec_rvq_depth,
    )
    generated_tokens = generated_tokens[:1]  # larger batch sizes unsupported
    generated_tokens = generated_tokens.reshape(
        self.config.chunk_length_frames, self.config.decoder_codec_rvq_depth
    )  # (50, 16)
    generated_tokens = generated_tokens[:max_decode_frames]  # (N, 16)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      generated_rvq_tokens = utils.llm_to_rvq(
          generated_tokens,
          self.config.codec_rvq_codebook_size,
          self.config.vocab_codec_offset,
          safe=False,
      )

    # Decode via SpectroStream using additional frame of samples for crossfading
    # We want to generate a 2s chunk with an additional 40ms of crossfade, which
    # is one additional codec frame. Caller is responsible for actually applying
    # the crossfade.
    xfade_frames = state.context_tokens[-self.config.crossfade_length_frames :]
    if state.chunk_index == 0:
      # NOTE: This will create 40ms of gibberish at the beginning but it's OK.
      xfade_frames = np.zeros_like(xfade_frames)
    assert xfade_frames.min() >= 0
    xfade_tokens = np.concatenate([xfade_frames, generated_rvq_tokens], axis=0)
    assert xfade_tokens.shape == (
        self.config.crossfade_length_frames + max_decode_frames,
        self.config.decoder_codec_rvq_depth,
    )  # (N+1, 16)
    waveform = self.codec.decode(xfade_tokens)
    assert isinstance(waveform, audio.Waveform)
    assert waveform.samples.shape == (
        self.config.crossfade_length_samples
        + max_decode_frames * self.config.frame_length_samples,
        self.num_channels,
    )  # ((N+1)*1920, 2)

    # Update state
    state.update(generated_rvq_tokens)

    return (waveform, state)


MagentaRT = MagentaRTT5X  # Alias to indicate default codepath.
