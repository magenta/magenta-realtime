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

import jax
import jax.numpy as jnp
import numpy as np
import sequence_layers.jax as sl
import safetensors.flax as safetensors_flax
import flax.traverse_util as flaxtu

from . import depthformer
from . import model as model_configs
from . import spectrostream
from .. import audio
from .. import musiccoca
from .. import paths


logger = logging.getLogger(__name__)

NUM_RESERVED_TOKENS = 6


def discretize_cfg(value: float, step: float, max_bin: int) -> int:
  """Map a CFG scale in [-1.0, 7.0] to a discrete conditioning token index.

  Used by the in-process generate path. The exported MLX ``.mlxfn`` bins the
  same scales with the equivalent MLX ops in ``_discretize_cfg_token()``
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


def _float_samples_to_int16(samples: jax.Array, gain: float = 0.5) -> jax.Array:
  # Gain is applied to reduce potential clipping artifacts when converting from
  # float to int16. Similar logic is used here for Lyria RT model export.
  samples = jnp.clip(gain * samples, -1, 1)
  samples = jnp.round((jnp.iinfo(jnp.int16).max + 0.5) * samples - 0.5)
  return samples.astype(jnp.int16)


def convert_from_unique_codes(
    tokens: jax.Array, codebook_size: int = 1024
) -> jax.Array:
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
  """A sampler that samples tokens from a depthformer and decodes them into waveforms."""

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
      return MagentaRT2Sampler(self, name=self.name)

  cfg: Config

  def setup(self):
    self.depthformer = self.cfg.depthformer.make()
    # TODO(kehanghan): change to self.spectrostream after checkpoint field renaming.
    self.soundstream = self.cfg.spectrostream.make()
    assert self.cfg.spectrostream.quantizer is not None
    assert self.soundstream.quantizer is not None
    output_codebooks = self.depthformer.decoder.config.num_codebooks
    output_channels = (
        self.soundstream.embeddings_to_waveform_layer.get_output_shape((
            self.soundstream.config.num_features,
        ))
    )
    output_dtype = self.soundstream.config.compute_dtype

    self.layers = [
        self.depthformer.get_sampler_sequence_layer(),
        sl.Lambda.Config(
            functools.partial(
                convert_from_unique_codes,
                codebook_size=self.cfg.spectrostream.quantizer.num_embeddings,
            ),
            expected_input_spec=sl.ShapeDType((output_codebooks,), jnp.int32),
            mask_required=False,
        ).make(),
        self.soundstream.quantizer.codes_to_embeddings_layer,
        self.soundstream.embeddings_to_waveform_layer,
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


def _load_jax_weights(path) -> dict:
  """Load safetensors checkpoint as nested Flax param dict."""
  flat_weights = safetensors_flax.load_file(str(path))
  nested_dict = {tuple(k.split('/')): v for k, v in flat_weights.items()}
  return flaxtu.unflatten_dict(nested_dict)


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
  ):
    """Initialise the system: build model, load weights, JIT-compile.

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
    self._params = _load_jax_weights(checkpoint_path)

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

    # --- AOT-compiled functions ---
    self._jit_init_state = None
    self._jit_streaming_step = None
    self._compile()

  def _compile(self):
    """AOT-compile init_state and streaming_step."""

    if self._jit_streaming_step is not None:
      return

    batch_size = 1
    input_channel_spec = jax.ShapeDtypeStruct(
        [self._num_channels], jnp.int32,
    )
    rngs = {
        'params': jax.random.PRNGKey(42),
        'random': jax.random.PRNGKey(0),
    }

    @jax.jit
    def _init_state(params, constants):
      return self._sampler.apply(
          params, batch_size, input_channel_spec,
          constants=constants, training=False, rngs=rngs,
          method=self._sampler.get_initial_state,
      )

    @functools.partial(jax.jit, donate_argnums=(3,))
    def _streaming_step(params, x, constants, state):
      return self._sampler.apply(
          params, x=x, state=state, constants=constants,
          training=False, rngs=rngs,
          method=self._sampler.step_with_emits,
      )

    self._jit_init_state = _init_state

    # AOT compile streaming_step with concrete args.
    logger.info('Compiling...')
    t0 = time.time()

    # Create dummy conditioning to get concrete shapes.
    dummy_style = [1] * self._num_musiccoca_tokens
    dummy_notes = [-1] * self._num_notes
    dummy_drums = [-1] * self._drum_tokens
    dummy_cfg = [-1] * self._cfg_tokens
    block, constants = self._build_conditioning(dummy_style, dummy_notes, dummy_drums, dummy_cfg)

    init_constants = {}
    state = self._jit_init_state(self._params, init_constants)
    self._jit_streaming_step = _streaming_step.lower(
        self._params, block, constants, state
    ).compile()

    logger.info('Compilation time: %.1fs', time.time() - t0)

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
    # default cfgs_tokens [20, 20, 4] => cfg values [3.0, 3.0, 3.0]
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
    block = sl.Sequence.from_values(cond.reshape(1, 1, -1))

    temperature = self.temperature if temperature is None else temperature
    top_k = self.top_k if top_k is None else top_k
    constants = {
        'temperature': jnp.array([temperature]),
        'top_k': jnp.array([top_k], dtype=jnp.int32),
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
      state = self._jit_init_state(self._params, init_constants)

    # --- Streaming generation ---
    results = []
    t0 = time.time()
    for _ in range(frames):
      step_output, state, _ = self._jit_streaming_step(
          self._params, block, constants, state
      )
      results.append(step_output)

    # --- Assemble audio ---
    samples = sl.Sequence.concatenate_sequences(results).values[0]
    samples = jax.device_get(samples).astype(np.int16)
    elapsed = time.time() - t0
    ms_per_step = (elapsed / frames) * 1000
    logger.debug(
        'Generated %d frames in %.2fs (%.1f ms/step, %.1f steps/s)',
        frames, elapsed, ms_per_step, frames / elapsed,
    )
    # samples shape: [T*1920, 2] (interleaved stereo int16)
    waveform = audio.Waveform(samples.astype(np.float32) / 32768.0, sample_rate=self._sample_rate)

    return waveform, state
