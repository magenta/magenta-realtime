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

"""Magenta RealTime system for streaming audio generation."""

import dataclasses
import functools
import logging
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sequence_layers.mlx as sl

from . import depthformer
from . import spectrostream
from . import model as model_configs
from .load_weights import load_weights
from .. import audio
from .. import musiccoca
from .. import paths


logger = logging.getLogger(__name__)

NUM_RESERVED_TOKENS = 6


def discretize_cfg(value: float, step: float, max_bin: int) -> int:
  """Map a CFG scale in [-1.0, 7.0] to a discrete conditioning token index.

  Used by the in-process generate path. The exported ``.mlxfn`` bins the same
  scales with the equivalent MLX ops in ``_discretize_cfg_token()``
  (``mlx/export.py``), and the C++ runtime now feeds raw float scales to that
  exported function rather than binning them itself (the old C++
  ``discretize_cfg()`` has been removed). The two implementations agree except
  at exact bin boundaries, where float32 (mlxfn) vs float64 (here) rounding can
  differ by one bin.

  Args:
    value: CFG scale; clamped to [-1.0, 7.0].
    step: Quantization step (0.2 for musiccoca/notes, 1.0 for drums).
    max_bin: Largest valid token index (40 for musiccoca/notes, 8 for drums).

  Returns:
    Token index in [0, max_bin].
  """
  clamped = max(-1.0, min(7.0, value))
  bin_index = int(round((clamped - (-1.0)) / step))
  return max(0, min(max_bin, bin_index))


def _float_samples_to_int16(samples: mx.array, gain: float = 0.5) -> mx.array:
  # Gain is applied to reduce potential clipping artifacts when converting from
  # float to int16. Similar logic is used here for Lyria RT model export.
  samples = mx.clip(gain * samples, -1, 1)
  samples = mx.round((mx.iinfo(mx.int16).max + 0.5) * samples - 0.5)
  return samples.astype(mx.int16)


def convert_from_unique_codes(
    tokens: mx.array, codebook_size: int = 1024
) -> mx.array:
  """Transforms Depthformer's unique indexing scheme to non-unique indices.

  This should invert the result of convert_to_unique_codes.

  Args:
    tokens: Array of tokens using the unique indexing scheme.
    codebook_size: Size of the codebook.

  Returns:
    Array of tokens using the non-unique indexing scheme.
  """
  if codebook_size < NUM_RESERVED_TOKENS:
    raise ValueError(
        'Codebook size must be at least common.NUM_RESERVED_TOKENS.'
    )
  return (tokens - NUM_RESERVED_TOKENS) % codebook_size


class MagentaRT2Sampler(sl.SerialCombinatorMixin, sl.Emitting):
  """Streaming system that samples from a depthformer and decodes SpectroStream."""

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Config for MagentaRT2Sampler."""

    depthformer: depthformer.EncoderDecoder.Config
    spectrostream: spectrostream.SpectroStream.Config

    # Sampler inputs:
    mask_token_id: int = NUM_RESERVED_TOKENS

    # Sampler outputs:
    int16_outputs: bool = True
    forced_spectrostream_outputs_key: str | None = 'forced_spectrostream_outputs'
    # When set, the forced_spectrostream_outputs will be transformed from the
    # raw SpectroStream non-unique values to the depthformer output format.
    transform_forced_spectrostream_outputs: bool = True

    name: str | None = None

    def make(self) -> 'MagentaRT2Sampler':
      return MagentaRT2Sampler(self)

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg

    self.depthformer = cfg.depthformer.make()
    self.spectrostream = spectrostream.SpectroStream(cfg.spectrostream)
    assert self.spectrostream.quantizer is not None
    output_codebooks = self.depthformer.decoder.config.num_codebooks
    output_channels = (
        self.spectrostream.embeddings_to_waveform_layer.get_output_shape((
            self.spectrostream.config.num_features,
        ))
    )
    output_dtype = self.spectrostream.config.compute_dtype

    self.layers = [
        self.depthformer.get_sampler_sequence_layer(),
        sl.Lambda.Config(
            functools.partial(
                convert_from_unique_codes,
                codebook_size=cfg.spectrostream.quantizer.num_embeddings,
            ),
            expected_input_spec=sl.ShapeDType((output_codebooks,), mx.int32),
            mask_required=False,
        ).make(),
        self.spectrostream.quantizer.codes_to_embeddings_layer,
        self.spectrostream.embeddings_to_waveform_layer,
        sl.Lambda.Config(
            _float_samples_to_int16,
            mask_required=False,
            expected_input_spec=sl.ShapeDType(output_channels, output_dtype),
        ).make(),
    ]


NotesArray = list[int] | np.ndarray
MagentaRT2State = sl.State


# ---------------------------------------------------------------------------
# Checkpoint registry – maps model size names to checkpoint filenames.
# ---------------------------------------------------------------------------

_CHECKPOINT_REGISTRY: dict[str, str] = {
    'mrt2_base': 'mrt2_base.safetensors',
    'mrt2_small': 'mrt2_small.safetensors',
}


class MagentaRT2System:
  """A MagentaRT2 streaming system that takes style and notes inputs and generates audio.

  Example::

      mrt = MagentaRT2System(size='mrt2_base')
      embedding = mrt.embed_style('disco funk')
      wav, state = mrt.generate(style=embedding, frames=25)
  """

  def __init__(
      self,
      size: str = 'mrt2_base',
      style_model: musiccoca.MusicCoCa | None = None,
      checkpoint: str | None = None,
      temperature: float = 1.3,
      top_k: int = 40,
      cfg_musiccoca: float = 3.0,
      cfg_notes: float = 1.0,
      cfg_drums: float = 1.0,
      bits: int | None = 8,
      quantize_group_size: int | None = None,
  ):
    """Initialise the system: build model, load weights, optionally quantize.

    Args:
      size: Model variant name (must be a key in MODEL_REGISTRY).
      style_model: MusicCoCa instance for text/audio → embedding.  If None, a
          default MusicCoCa is created.
      checkpoint: Override checkpoint filename. If None, looked up from size.
      temperature: Sampling temperature.
      top_k: Top-k sampling threshold.
      cfg_musiccoca: Classifier-free guidance scale for MusicCoCa.
      cfg_notes: Classifier-free guidance scale for notes.
      cfg_drums: Classifier-free guidance scale for drums.
      bits: Quantization bit width (4 or 8). None means no quantization.
      quantize_group_size: Group size for quantization. If None, defaults to
          32 for 4-bit and 64 for 8-bit.
    """

    self._model = model_configs.get_model_class(size)()
    self._size = size
    self._style_model = style_model or musiccoca.MusicCoCa()

    depthformer_config = self._model.depthformer_config()
    rvq_truncation = self._model.spectrostream.rvq_truncation_level
    spectrostream_config = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=rvq_truncation,
        use_unique_codes=False,
    )
    self._sample_rate = int(spectrostream_config.audio_sample_rate)
    self._sampler = MagentaRT2Sampler.Config(
        depthformer=depthformer_config,
        spectrostream=spectrostream_config,
        int16_outputs=False,
    ).make()

    # --- Load checkpoint ---
    if checkpoint is None:
      if size not in _CHECKPOINT_REGISTRY:
        raise ValueError(
            f"No default checkpoint for size '{size}'. "
            f"Available: {list(_CHECKPOINT_REGISTRY.keys())}. "
            f"Pass checkpoint= explicitly."
        )
      checkpoint = _CHECKPOINT_REGISTRY[size]

    checkpoint_path = paths.checkpoints_dir() / checkpoint
    logger.info('Loading checkpoint: %s', checkpoint_path)
    load_weights(
        self._sampler, checkpoint_path,
        num_input_channels=self._model.input_num_channels,
    )

    # --- Quantize for performance ---
    if bits and bits < 32:
      if quantize_group_size is None:
        quantize_group_size = 32 if bits == 4 else 64
      logger.info('Quantizing to %d-bit (group_size=%d).', bits, quantize_group_size)
      nn.quantize(self._sampler, group_size=quantize_group_size, bits=bits)

    # --- Sampling defaults ---
    self.temperature = temperature
    self.top_k = top_k
    self.cfg_musiccoca = cfg_musiccoca
    self.cfg_notes = cfg_notes
    self.cfg_drums = cfg_drums

    # --- Derived constants ---
    self._num_musiccoca_tokens = self._model.input_configs[0].rvq_truncation_level
    self._num_notes = self._model.input_configs[1].rvq_truncation_level
    self._drum_tokens = self._model.input_configs[2].rvq_truncation_level
    self._cfg_tokens = sum(
        cfg.rvq_truncation_level for cfg in self._model.input_configs[3:]
    )
    self._num_channels = (
        self._num_musiccoca_tokens
        + self._num_notes
        + self._drum_tokens
        + self._cfg_tokens
    )

    # --- Warm up ---
    self._warmup()

  def _warmup(self, steps: int = 5):
    """Run a few dummy streaming steps to warm up MLX kernel caches."""
    logger.info('Warming up...')
    t0 = time.time()

    dummy_style = [1] * self._num_musiccoca_tokens
    dummy_notes = [-1] * self._num_notes
    dummy_drums = [-1] * self._drum_tokens
    dummy_cfg = [-1] * self._cfg_tokens
    block, constants = self._build_conditioning(dummy_style, dummy_notes, dummy_drums, dummy_cfg)

    input_spec = sl.ChannelSpec(
        shape=(self._num_channels,), dtype=mx.int32,
    )
    state = self._sampler.get_initial_state(
        1, input_spec, constants=constants, training=False,
    )
    for _ in range(steps):
      y, state, _ = self._sampler.step_with_emits(
          x=block, state=state, constants=constants, training=False,
      )
      mx.eval(y.values)

    logger.info('Warm-up done (%.1fs).', time.time() - t0)

  def embed_style(
      self, text_or_audio: str | audio.Waveform,
      pool_across_time: bool = True,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> musiccoca.StyleEmbedding:
    """Embed text or audio into a style embedding vector."""
    result = self._style_model.embed(
        text_or_audio, pool_across_time, use_mapper, seed
    )
    assert not isinstance(result, list)
    return result

  def tokenize_style(
      self, embedding: musiccoca.StyleEmbedding,
  ) -> musiccoca.StyleTokens:
    """Tokenize a style embedding into RVQ tokens."""
    return self._style_model.tokenize(embedding)

  def _build_conditioning(
      self,
      style: musiccoca.StyleTokens | list[int],
      notes: list[int] | None = None,
      drums: list[int] | None = None,
      cfgs: list[int] | None = None,
      temperature: float | None = None,
      top_k: int | None = None,
  ) -> tuple[sl.Sequence, dict]:
    """Build the conditioning block and constants dict for streaming.

    Returns:
      (block, constants) where block is the positive conditioning sequence
      and constants contains temperature, top_k, CFG scales, and negative
      conditioning sequences.
    """

    style_tokens = list(style) if not isinstance(style, list) else style
    notes_tokens = notes if notes is not None else [-1] * self._num_notes
    drums_tokens = drums if drums is not None else [-1] * self._drum_tokens
    # Default cfgs_tokens [20, 20, 4] => cfg values [3.0, 3.0, 3.0]
    if cfgs is None:
      cfgs_tokens = [
          int((self.cfg_musiccoca + 1.0) / 0.2),
          int((self.cfg_notes + 1.0) / 0.2),
          4,
      ]
    else:
      cfgs_tokens = cfgs

    assert len(style_tokens) == self._num_musiccoca_tokens, (
        f'Expected {self._num_musiccoca_tokens} style tokens, got {len(style_tokens)}'
    )
    assert len(notes_tokens) == self._num_notes, (
        f'Expected {self._num_notes} notes, got {len(notes_tokens)}'
    )
    assert len(drums_tokens) == self._drum_tokens, (
        f'Expected {self._drum_tokens} drums, got {len(drums_tokens)}'
    )
    assert len(cfgs_tokens) == self._cfg_tokens, (
        f'Expected {self._cfg_tokens} CFG tokens, got {len(cfgs_tokens)}'
    )

    offset = NUM_RESERVED_TOKENS + 1  # +1 for dropout token

    # Positive conditioning.
    cond = np.array(style_tokens + notes_tokens + drums_tokens + cfgs_tokens, dtype=np.int32) + offset
    block = sl.Sequence(
        mx.array(cond.reshape(1, 1, -1), dtype=mx.int32),
        mx.array([[True]], dtype=mx.bool_),
    )

    temperature = self.temperature if temperature is None else temperature
    top_k = self.top_k if top_k is None else top_k
    constants = {
        'temperature': mx.array([temperature]),
        'top_k': mx.array([top_k], dtype=mx.int32),
    }
    return block, constants

  def generate(
      self,
      style: musiccoca.StyleEmbedding | None = None,
      notes: list[int] | None = None,
      drums: list[int] | None = None,
      cfg_musiccoca: float | None = None,
      cfg_notes: float | None = None,
      cfg_drums: float | None = None,
      temperature: float | None = None,
      top_k: int | None = None,
      frames: int = 25,
      state: MagentaRT2State | None = None,
  ) -> tuple[audio.Waveform, MagentaRT2State]:
    """Generate audio from style conditioning.

    Args:
      style: Style embedding (768-dim ndarray from embed_style). None means
          unconditional/masked.
      notes: Notes conditioning (128 ints). Each slot represents the state of
          the corresponding pitch (0-127). The state can be:
          -1: means the pitch is masked out.
           0: means the pitch is off.
           1: means the pitch is on, but it's not the first time.
           2: means the pitch is on for the first time (i.e., onset)
           3: means the pitch is on (model has the freedom to play it as an
              onset or continuation).
        None means masked/silent (all pitches masked out).
      drums: Drums conditioning (1 int).
        -1: means masked
         0: no drum
         1: play drum
      cfg_musiccoca: MusicCoCa classifier-free-guidance scale, a float in
        [-1.0, 7.0]. None falls back to ``self.cfg_musiccoca``. Discretized to
        a conditioning token with a 0.2 step (token 0 -> -1.0, token 1 -> -0.8,
        ..., token 40 -> 7.0).
      cfg_notes: Notes CFG scale, a float in [-1.0, 7.0]. None falls back to
        ``self.cfg_notes``. Discretized with a 0.2 step like cfg_musiccoca.
      cfg_drums: Drums CFG scale, a float in [-1.0, 7.0]. None falls back to
        ``self.cfg_drums``. Discretized with a 1.0 step (token 0 -> -1.0,
        token 1 -> 0.0, ..., token 8 -> 7.0).
      temperature: Sampling temperature. None falls back to
        ``self.temperature``.
      top_k: Top-k sampling threshold. None falls back to ``self.top_k``.
      frames: Number of frames to generate (25 frames = 1 second at 48kHz).
      state: Streaming state from a previous call. If None, a fresh state is
          created.

    Returns:
      (waveform, state) — a Waveform at 48kHz stereo, and the updated state
      for continuation.
    """
    # --- Resolve style to tokens ---
    if style is None:
      style_tokens = [-1] * self._num_musiccoca_tokens
    else:
      style_tokens = self._style_model.tokenize(style).tolist()

    # Pad or truncate to expected length.
    if len(style_tokens) < self._num_musiccoca_tokens:
      style_tokens = style_tokens + [-1] * (self._num_musiccoca_tokens - len(style_tokens))
    style_tokens = style_tokens[:self._num_musiccoca_tokens]

    # --- Resolve CFG scales and discretize to conditioning tokens ---
    cfg_musiccoca = self.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
    cfg_notes = self.cfg_notes if cfg_notes is None else cfg_notes
    cfg_drums = self.cfg_drums if cfg_drums is None else cfg_drums
    cfgs = [
        discretize_cfg(cfg_musiccoca, 0.2, 40),
        discretize_cfg(cfg_notes, 0.2, 40),
        discretize_cfg(cfg_drums, 1.0, 8),
    ]

    # --- Build conditioning ---
    block, constants = self._build_conditioning(
        style_tokens, notes, drums, cfgs, temperature, top_k
    )

    # --- Init state if needed ---
    if state is None:
      init_constants = {}
      input_spec = sl.ChannelSpec(
          shape=(self._num_channels,), dtype=mx.int32,
      )
      state = self._sampler.get_initial_state(
          1, input_spec, constants=init_constants, training=False,
      )

    # --- Streaming generation ---
    results = []
    t0 = time.time()
    for _ in range(frames):
      step_output, state, _ = self._sampler.step_with_emits(
          x=block, state=state, constants=constants, training=False,
      )
      mx.eval(step_output.values)
      results.append(step_output)

    elapsed = time.time() - t0
    ms_per_step = (elapsed / frames) * 1000
    logger.debug(
        'Generated %d frames in %.2fs (%.1f ms/step, %.1f steps/s)',
        frames, elapsed, ms_per_step, frames / elapsed,
    )

    # --- Assemble audio ---
    samples = sl.Sequence.concatenate_sequences(results).values[0]
    samples = np.array(samples)
    # samples shape: [T*1920, 2] (interleaved stereo float32)
    waveform = audio.Waveform(samples.astype(np.float32) / 32768.0, sample_rate=self._sample_rate)

    return waveform, state


class MagentaRT2SystemMlxfn:
  """A MagentaRT2 system that uses an exported .mlxfn for inference.

  Equivalent to MagentaRT2System but skips Python model construction,
  weight loading, and quantization — all of that is baked into the .mlxfn
  at export time.

  Example::

      mrt = MagentaRT2SystemMlxfn(size='mrt2_base')
      embedding = mrt.embed_style('disco funk')
      wav, state = mrt.generate(style=embedding, frames=25)
  """

  # The exported mlxfn uses NUM_RESERVED_TOKENS + 1 (dropout token) as offset.
  _TOKEN_OFFSET = NUM_RESERVED_TOKENS + 1

  def __init__(
      self,
      size: str | None = None,
      style_model: musiccoca.MusicCoCa | None = None,
      temperature: float = 1.3,
      top_k: int = 40,
      cfg_musiccoca: float = 3.0,
      cfg_notes: float = 1.0,
      cfg_drums: float = 1.0,
      warmup_steps: int = 5,
  ):
    """Initialise from an exported .mlxfn model directory.

    Args:
      size: Model size, either "mrt2_base" or "mrt2_large".
      style_model: MusicCoCa instance for text/audio → embedding. If None,
          a default MusicCoCa is created.
      temperature: Sampling temperature.
      top_k: Top-k sampling threshold.
      cfg_musiccoca: Classifier-free guidance scale for MusicCoCa.
      cfg_notes: Classifier-free guidance scale for notes.
      cfg_drums: Classifier-free guidance scale for drums.
      warmup_steps: Number of warmup inference steps.
    """
    model_name = size or paths.DEFAULT_MODEL_NAME
    model_path = paths.models_dir() / model_name

    basename = model_path.name
    mlxfn_path = str(model_path / f'{basename}.mlxfn')
    state_path = str(model_path / f'{basename}_state.safetensors')

    # --- Load exported function + state ---
    logger.info('Loading mlxfn: %s', mlxfn_path)
    self._fn = mx.import_function(mlxfn_path)

    state_dict = mx.load(state_path)
    self._initial_state = []
    for i in range(len(state_dict)):
      key = f'state_{i}'
      if key not in state_dict:
        break
      self._initial_state.append(state_dict[key])
    mx.eval(self._initial_state)
    logger.info('Loaded %d state arrays', len(self._initial_state))

    self._rvq_depth = 12

    # --- Style model ---
    self._style_model = style_model or musiccoca.MusicCoCa()

    # --- Conditioning layout ---
    self._num_musiccoca_tokens = 12
    self._num_notes = 128
    self._num_drums = 1
    self._num_cfgs = 3  # musiccoca-cfg, notes-cfg, drums-cfg

    # --- Sampling parameters ---
    self.temperature = temperature
    self.top_k = top_k
    self.cfg_musiccoca = cfg_musiccoca
    self.cfg_notes = cfg_notes
    self.cfg_drums = cfg_drums

    self._sample_rate = 48_000

    # --- Warm up ---
    self._warmup(warmup_steps)

  def _warmup(self, steps: int = 5):
    """Run a few dummy steps to warm up MLX kernel caches."""
    logger.info('Warming up (%d steps)...', steps)
    t0 = time.time()
    args = self._build_mlxfn_args(
        style_tokens=[-1] * self._num_musiccoca_tokens,
    )
    state = list(self._initial_state)
    for _ in range(steps):
      outputs = self._fn(args + state)
      mx.eval(outputs)
      state = list(outputs[1:])
    logger.info('Warm-up done (%.1fs).', time.time() - t0)

  def embed_style(
      self, text_or_audio: str | audio.Waveform,
      pool_across_time: bool = True,
      use_mapper: bool = False,
      seed: int = 0,
  ) -> musiccoca.StyleEmbedding:
    """Embed text or audio into a style embedding vector."""
    result = self._style_model.embed(
        text_or_audio, pool_across_time, use_mapper, seed
    )
    assert not isinstance(result, list)
    return result

  def tokenize_style(
      self, embedding: musiccoca.StyleEmbedding,
  ) -> musiccoca.StyleTokens:
    """Tokenize a style embedding into RVQ tokens."""
    return self._style_model.tokenize(embedding)

  def _build_mlxfn_args(
      self,
      style_tokens: list[int],
      notes: list[int] | None = None,
      drums: list[int] | None = None,
      cfg_musiccoca: float | None = None,
      cfg_notes: float | None = None,
      cfg_drums: float | None = None,
      temperature: float | None = None,
      top_k: int | None = None,
  ) -> list[mx.array]:
    """Build the flat arg list expected by the exported mlxfn.

    The exported function signature is:
        fn([cond, temperature, top_k, cfg_musiccoca, cfg_notes, cfg_drums,
            neg_musiccoca, neg_notes, forced_tokens, *state])

    Returns:
      List of mx.array arguments (without state — caller appends that).
    """
    notes_tokens = notes if notes is not None else [-1] * self._num_notes
    drums_tokens = drums if drums is not None else [-1] * self._num_drums

    cfg_musiccoca = self.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
    cfg_notes = self.cfg_notes if cfg_notes is None else cfg_notes
    cfg_drums = self.cfg_drums if cfg_drums is None else cfg_drums
    temperature = self.temperature if temperature is None else temperature
    top_k = self.top_k if top_k is None else top_k

    offset = self._TOKEN_OFFSET

    # Positive conditioning (12 + 128 + 1 = 141 tokens)
    cond = np.array(
        style_tokens + notes_tokens + drums_tokens,
        dtype=np.int32,
    ) + offset
    cond_array = mx.array(cond.reshape(1, 1, -1), dtype=mx.int32)

    # Negative conditioning (masked style)
    masked_style = [-1] * len(style_tokens)
    neg_mc = np.array(
        masked_style + notes_tokens + drums_tokens,
        dtype=np.int32,
    ) + offset
    neg_mc_array = mx.array(neg_mc.reshape(1, 1, -1), dtype=mx.int32)

    # Negative conditioning (masked notes)
    masked_notes = [-1] * len(notes_tokens)
    neg_n = np.array(
        style_tokens + masked_notes + drums_tokens,
        dtype=np.int32,
    ) + offset
    neg_n_array = mx.array(neg_n.reshape(1, 1, -1), dtype=mx.int32)

    return [
        cond_array,
        mx.array([temperature]),
        mx.array([top_k], dtype=mx.int32),
        mx.array([cfg_musiccoca]),
        mx.array([cfg_notes]),
        mx.array([cfg_drums]),
        neg_mc_array,
        neg_n_array,
        mx.zeros((1, 0, self._rvq_depth), dtype=mx.int32),  # forced_tokens
    ]

  def generate(
      self,
      style: musiccoca.StyleEmbedding | None = None,
      notes: list[int] | None = None,
      drums: list[int] | None = None,
      cfg_musiccoca: float | None = None,
      cfg_notes: float | None = None,
      cfg_drums: float | None = None,
      temperature: float | None = None,
      top_k: int | None = None,
      frames: int = 25,
      state: list[mx.array] | None = None,
  ) -> tuple[audio.Waveform, list[mx.array]]:
    """Generate audio from style conditioning.

    Args:
      style: Style embedding from embed_style(). None means unconditional.
      notes: Notes conditioning (128 ints, see MagentaRT2System.generate).
      drums: Drums conditioning (1 int). -1=masked, 0=off, 1=on.
      cfg_musiccoca: Classifier-free guidance scale for MusicCoCa.
      cfg_notes: Classifier-free guidance scale for notes.
      cfg_drums: Classifier-free guidance scale for drums.
      temperature: Sampling temperature.
      top_k: Top-k sampling threshold.
      frames: Number of frames to generate (25 frames ≈ 1 second at 48kHz).
      state: Streaming state from a previous call. If None, uses initial state.

    Returns:
      (waveform, state) — Waveform at 48kHz stereo, and the updated state.
    """
    # --- Resolve style to tokens ---
    if style is None:
      style_tokens = [-1] * self._num_musiccoca_tokens
    else:
      style_tokens = self._style_model.tokenize(style).tolist()

    # Pad / truncate
    if len(style_tokens) < self._num_musiccoca_tokens:
      style_tokens += [-1] * (self._num_musiccoca_tokens - len(style_tokens))
    style_tokens = style_tokens[:self._num_musiccoca_tokens]

    # --- Build args ---
    args = self._build_mlxfn_args(
        style_tokens, notes, drums, cfg_musiccoca, cfg_notes, cfg_drums,
        temperature, top_k
    )

    # --- Init state if needed ---
    if state is None:
      state = list(self._initial_state)

    # --- Streaming generation ---
    audio_frames = []
    t0 = time.time()
    for _ in range(frames):
      outputs = self._fn(args + state)
      mx.eval(outputs)
      audio_frames.append(np.array(outputs[0]))  # (1, 2, 1920)
      state = list(outputs[1:])

    elapsed = time.time() - t0
    ms_per_step = (elapsed / frames) * 1000
    logger.debug(
        'Generated %d frames in %.2fs (%.1f ms/step, %.1f steps/s)',
        frames, elapsed, ms_per_step, frames / elapsed,
    )

    # --- Assemble audio ---
    # Each frame is (1, 2, T), concatenate along time axis
    all_audio = np.concatenate(audio_frames, axis=-1)  # (1, 2, total_samples)
    samples = all_audio[0].T  # (total_samples, 2)

    waveform = audio.Waveform(
        samples.astype(np.float32) / 32768.0,
        sample_rate=self._sample_rate,
    )
    return waveform, state
