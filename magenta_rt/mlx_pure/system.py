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

"""Magenta RealTime system for the pure-MLX backend.

Mirrors the ``magenta_rt.jax`` / ``magenta_rt.mlx`` system API::

    mrt = MagentaRT2System(size='mrt2_small')
    embedding = mrt.embed_style('disco funk')
    audio, state = mrt.generate(style=embedding, frames=25)
    audio, state = mrt.generate(style=embedding, frames=25, state=state)

Like those systems, CFG uses the *trained conditioning tokens* (the ``cfgs``
channels) with batch = N styles. The classifier-free-guidance logit-mixing
path (``cfg_arity`` / ``cfg_scales`` with stacked negative rows) and GPTQ
calibration remain available on the lower-level ``magenta_rt.mlx_pure.generate``
research script.

State is the depthformer's :class:`~magenta_rt.mlx_pure.depthformer.SamplerState`
NamedTuple, threaded functionally — multiple streams can be interleaved on one
system instance (matching the jax/mlx contract). Caveat: the SpectroStream
codec's streaming buffers live on the codec module, so interleaved streams
share codec state; for fully independent streams use separate systems.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import depthformer
from .configs import get_model_class
from .model import MagentaRT2Sampler
from audiotree import AudioTree

from .. import conditioning
from .. import paths

logger = logging.getLogger(__name__)

_CHECKPOINT_REGISTRY: dict[str, str] = {
    'mrt2_base': 'mrt2_base.safetensors',
    'mrt2_small': 'mrt2_small.safetensors',
}

MagentaRT2State = depthformer.SamplerState


def _resolve_uniform(value, default, name: str):
    """Resolve a scalar-or-uniform-array sampling argument to a Python scalar.

    The mlx_pure depthformer step takes shared scalars; per-element values
    are not supported yet (unlike the jax/mlx systems). A length-N array is
    accepted when all its elements are equal.
    """
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size and np.all(arr == arr.reshape(-1)[0]):
        return arr.reshape(-1)[0].item()
    raise ValueError(
        f'{name}: per-element values are not supported on the mlx_pure '
        'backend yet; pass a shared scalar.'
    )


class MagentaRT2System:
    """A MagentaRT2 streaming system (pure MLX) for conditioned audio.

    Example::

        mrt = MagentaRT2System(size='mrt2_small', bits=4)
        embedding = mrt.embed_style('disco funk')
        audio, state = mrt.generate(style=embedding, frames=25)
    """

    def __init__(
        self,
        size: str = 'mrt2_base',
        style_model=None,
        checkpoint: str | None = None,
        restore: bool = True,
        temperature: float = 1.3,
        top_k: int = 40,
        cfg_musiccoca: float = 3.0,
        cfg_notes: float = 1.0,
        cfg_drums: float = 1.0,
        seed: int = 0,
        bits: int | None = None,
        quantize_group_size: int | None = None,
        model: MagentaRT2Sampler | None = None,
    ):
        """Initialise the system: build, load weights, optionally quantize.

        Args:
          size: Model variant name (a ``magenta_rt.mlx_pure.configs``
              ``MODEL_REGISTRY`` key: ``mrt2_base`` / ``mrt2_small`` / ``tiny``).
          style_model: MusicCoCa instance for text/audio -> embedding. If None,
              one is created lazily on first ``embed_style`` call.
          checkpoint: Override checkpoint filename (relative to the checkpoints
              directory, or an absolute path). If None, looked up from size.
          restore: Load checkpoint weights (default). ``restore=False`` keeps
              the random initialization — useful for smoke tests.
          temperature: Sampling temperature default.
          top_k: Top-k sampling threshold default.
          cfg_musiccoca: CFG scale default for MusicCoCa.
          cfg_notes: CFG scale default for notes.
          cfg_drums: CFG scale default for drums.
          seed: Seed for the sampling rng of each fresh stream.
          bits: Nearest-rounding quantization bit width for the depthformer
              (the codec stays full precision). None means no quantization.
              For GPTQ use the ``magenta_rt.mlx_pure.generate`` script.
          quantize_group_size: Quantizer group size. If None, defaults to 32
              for 4-bit and 64 otherwise.
          model: Pre-built :class:`MagentaRT2Sampler` to wrap instead of building
              one from ``size`` (e.g. a hand-built tiny model in tests).
        """
        self._spec = get_model_class(size)()
        self._size = size
        if model is None:
            model = MagentaRT2Sampler.from_preset(size, int16_outputs=False)
        self._model = model

        if restore:
            if checkpoint is None:
                if size not in _CHECKPOINT_REGISTRY:
                    raise ValueError(
                        f"No default checkpoint for size '{size}'. "
                        f"Available: {list(_CHECKPOINT_REGISTRY.keys())}. "
                        f"Pass checkpoint= explicitly."
                    )
                checkpoint = _CHECKPOINT_REGISTRY[size]
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = paths.checkpoints_dir() / checkpoint_path
            logger.info('Loading checkpoint: %s', checkpoint_path)
            self._model.load_from_safetensors(checkpoint_path, model_name=size)

        if bits and bits < 32:
            from .quantize import quantize_in_place
            gs = quantize_group_size or (32 if bits == 4 else 64)
            logger.info('Quantizing depthformer to %d-bit (group_size=%d).', bits, gs)
            quantize_in_place(self._model.depthformer, group_size=gs, bits=bits)

        # --- Sampling defaults ---
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_notes = cfg_notes
        self.cfg_drums = cfg_drums

        # --- Derived constants ---
        self._seed = seed
        self._style_model_instance = style_model
        self._input_num_channels = (
            self._model.depthformer.encoder.embedding.num_channels
        )
        self._has_style_channels = self._input_num_channels > 1
        if self._has_style_channels:
            cfgs = self._spec.input_configs
            self._num_musiccoca_tokens = cfgs[0].rvq_truncation_level
            self._num_notes = cfgs[1].rvq_truncation_level
            self._drum_tokens = cfgs[2].rvq_truncation_level
            self._cfg_tokens = sum(c.rvq_truncation_level for c in cfgs[3:])

    # -------------------------------------------------------------------
    # Style embedding (MusicCoCa, lazy)
    # -------------------------------------------------------------------

    @property
    def _style_model(self):
        if self._style_model_instance is None:
            from .. import musiccoca
            self._style_model_instance = musiccoca.MusicCoCa()
        return self._style_model_instance

    def embed_style(
        self, text_or_audio,
        pool_across_time: bool = True,
        use_mapper: bool = False,
        seed: int = 0,
    ):
        """Embed text or audio into a style embedding vector."""
        if isinstance(text_or_audio, str):
            return self._style_model.embed_text(
                text_or_audio, use_mapper=use_mapper, seed=seed)
        embeddings = self._style_model.embed_audio(
            text_or_audio, pool_across_time=pool_across_time)
        # Single clip ([1, C, T]) -> [dim]; a batched tree -> [B, dim].
        return embeddings[0] if embeddings.shape[0] == 1 else embeddings

    def embed_styles(
        self, texts_or_audio,
        pool_across_time: bool = True,
        use_mapper: bool = False,
        seed: int = 0,
    ):
        """Embed a batch of texts/audio into a ``[N, dim]`` style embedding."""
        if isinstance(texts_or_audio, AudioTree):
            return self._style_model.embed_audio(
                texts_or_audio, pool_across_time=pool_across_time)
        return self._style_model.embed_text(
            list(texts_or_audio), use_mapper=use_mapper, seed=seed)

    def tokenize_style(self, embedding):
        """Tokenize a style embedding into RVQ tokens."""
        return self._style_model.tokenize(embedding)

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        # The mlx_pure SpectroStream doesn't carry a sample rate; the codec
        # is 48 kHz by construction (mirrors ``mlx_pure.generate``).
        return getattr(self._model.spectrostream, 'sample_rate', 48_000)

    def _build_conditioning(self, batch_style_tokens, notes, drums, cfgs):
        """Build the ``[N, 1, C]`` conditioning block as an mx array."""
        if not self._has_style_channels:
            offset = self._model.num_reserved_tokens + 1
            n = len(batch_style_tokens)
            return mx.full((n, 1, 1), offset, dtype=mx.int32)
        cond = conditioning.build_conditioning_rows(
            batch_style_tokens, notes, drums, cfgs,
            num_musiccoca=self._num_musiccoca_tokens,
            num_notes=self._num_notes,
            drum_tokens=self._drum_tokens,
            cfg_tokens=self._cfg_tokens,
            offset=self._model.num_reserved_tokens + 1,
        )
        return mx.array(cond, dtype=mx.int32)

    def generate(
        self,
        style=None,
        notes=None,
        drums=None,
        cfg_musiccoca: float | None = None,
        cfg_notes: float | None = None,
        cfg_drums: float | None = None,
        cfgs=None,
        temperature=None,
        top_k=None,
        frames: int = 25,
        state: MagentaRT2State | None = None,
    ) -> tuple[AudioTree, MagentaRT2State]:
        """Generate audio from style conditioning.

        Same contract as ``magenta_rt.mlx.system.MagentaRT2System.generate``
        (see that docstring for the full conditioning semantics), except
        ``temperature`` / ``top_k`` must be shared scalars (per-element values
        raise). ``state`` is a :class:`SamplerState`; None starts a fresh
        stream.

        Returns:
          (waveform, state) — a batched ``AudioTree`` (``waveform``
          ``[N, 2, T]`` float32 in [-1, 1], ``codes`` ``[N, frames, Q]`` RVQ
          codes) and the updated state for continuation.
        """
        # --- Resolve style to a batch of token rows ---
        if style is None:
            n_style = self._num_musiccoca_tokens if self._has_style_channels else 0
            batch_style_tokens = [[-1] * n_style]
        else:
            if not self._has_style_channels:
                raise ValueError(
                    f"Preset '{self._size}' has no style conditioning channels."
                )
            toks = self._style_model.tokenize(style)
            batch_style_tokens = conditioning.normalize_style_rows(
                toks, self._num_musiccoca_tokens
            )
        N = len(batch_style_tokens)

        # --- Resolve CFG conditioning tokens ---
        if cfgs is None:
            cfg_musiccoca = self.cfg_musiccoca if cfg_musiccoca is None else cfg_musiccoca
            cfg_notes = self.cfg_notes if cfg_notes is None else cfg_notes
            cfg_drums = self.cfg_drums if cfg_drums is None else cfg_drums
            cfgs = [
                conditioning.discretize_cfg(cfg_musiccoca, 0.2, 40),
                conditioning.discretize_cfg(cfg_notes, 0.2, 40),
                conditioning.discretize_cfg(cfg_drums, 1.0, 8),
            ]

        block = self._build_conditioning(batch_style_tokens, notes, drums, cfgs)
        temperature = _resolve_uniform(temperature, self.temperature, 'temperature')
        top_k = _resolve_uniform(top_k, self.top_k, 'top_k')

        # --- Init state if needed ---
        if state is None:
            state = self._model.make_initial_state(N, seed=self._seed)

        # --- Streaming generation ---
        audio_chunks = []
        code_chunks = []
        for _ in range(frames):
            wav_chunk, codes, state = self._model.step_with_codes(
                state, source_tokens=block,
                temperature=temperature, top_k=top_k,
            )
            mx.eval(wav_chunk, codes)
            audio_chunks.append(wav_chunk)
            code_chunks.append(codes)

        # Audio is channel-major [N, C, T] — chunks concatenate time-last;
        # codes are frame-major [N, frames, Q] — chunks concatenate on axis 1.
        waveform = np.asarray(
            mx.concatenate(audio_chunks, axis=-1), dtype=np.float32
        )
        if waveform.ndim == 2:
            # The codes_to_waveform delegator path may emit [B, T] (no
            # channel axis); normalize to channel-major [B, 1, T].
            waveform = waveform[:, None, :]
        codes = np.asarray(mx.concatenate(code_chunks, axis=1))

        tree = AudioTree(
            waveform, sample_rate=self.sample_rate, codes=codes,
        )
        return tree, state
