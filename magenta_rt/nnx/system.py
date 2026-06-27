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

"""Magenta RealTime system for the flax.nnx backend.

Mirrors the ``magenta_rt.jax`` / ``magenta_rt.mlx`` system API::

    mrt = MagentaRT2System(size='mrt2_small')
    embedding = mrt.embed_style('disco funk')
    audio_tree, state = mrt.generate(style=embedding, frames=25)
    audio_tree, state = mrt.generate(style=embedding, frames=25, state=state)

Like those systems, CFG uses the *trained conditioning tokens* (the ``cfgs``
channels) with batch = N styles. The classifier-free-guidance logit-mixing
path (``cfg_arity`` / ``cfg_scales`` with stacked negative rows) remains
available on the lower-level ``magenta_rt.nnx.generate`` research script.

State note: ``generate`` splits the model once into a constant parameter
partition (held on the system) and a *stream* partition — the KV caches, codec
overlap buffers, decode state and sampling rng — which is threaded through the
jitted step and returned inside :class:`MagentaRT2State`. Like the jax/mlx
``sl.State`` pytrees, these handles are independent values: ``state=None``
starts a fresh stream and any prior handle stays valid, so multiple streams can
be interleaved across one system instance.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from .model import MagentaRT2Sampler, get_model_class
from audiotree import AudioTree

from .. import audio
from .. import conditioning
from .. import paths

logger = logging.getLogger(__name__)

_CHECKPOINT_REGISTRY: dict[str, str] = {
    'mrt2_base': 'mrt2_base.safetensors',
    'mrt2_small': 'mrt2_small.safetensors',
}


@dataclasses.dataclass(frozen=True)
class MagentaRT2State:
    """Streaming continuation state for :meth:`MagentaRT2System.generate`.

    Carries the model's *stream* partition (KV / codec caches, decode state and
    sampling rng) split out of the nnx module as a pytree, so streams are
    independent values that can be interleaved across one system instance.
    """

    batch_size: int
    stream: Any


def _resolve_uniform(value, default, name: str):
    """Resolve a scalar-or-uniform-array sampling argument to a Python scalar.

    The nnx depthformer step takes shared scalars; per-element values are not
    supported yet (unlike the jax/mlx systems). A length-N array is accepted
    when all its elements are equal.
    """
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size and np.all(arr == arr.reshape(-1)[0]):
        return arr.reshape(-1)[0].item()
    raise ValueError(
        f'{name}: per-element values are not supported on the nnx backend '
        'yet; pass a shared scalar.'
    )


class MagentaRT2System:
    """A MagentaRT2 streaming system (nnx) for style/notes-conditioned audio.

    Example::

        mrt = MagentaRT2System(size='mrt2_small')
        embedding = mrt.embed_style('disco funk')
        audio_tree, state = mrt.generate(style=embedding, frames=25)
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
        jit: bool = True,
        model: MagentaRT2Sampler | None = None,
    ):
        """Initialise the system: build the model and load weights.

        Args:
          size: Model variant name (a ``magenta_rt.nnx.model.MODEL_REGISTRY``
              key: ``mrt2_base`` / ``mrt2_small``).
          style_model: MusicCoCa instance for text/audio -> embedding. If None,
              one is created lazily on first ``embed_style`` call (the TFLite
              models are not needed for token-conditioned generation).
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
          jit: Wrap the streaming step in ``jax.jit`` (recompiles per distinct
              ``(temperature, top_k)``). ``jit=False`` runs the step eagerly
              (no jit, no donation) for debugging.
          model: Pre-built :class:`MagentaRT2Sampler` to wrap instead of building
              one from ``size`` (e.g. a hand-built tiny model in tests).
        """
        self._spec = get_model_class(size)()
        self._size = size
        if model is None:
            model = MagentaRT2Sampler.from_preset(
                size, int16_outputs=False, rngs=nnx.Rngs(seed),
            )
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
            self._model.load_checkpoint(checkpoint_path)

        # --- Sampling defaults ---
        self.temperature = temperature
        self.top_k = top_k
        self.cfg_musiccoca = cfg_musiccoca
        self.cfg_notes = cfg_notes
        self.cfg_drums = cfg_drums

        # --- Derived constants ---
        self._seed = seed
        self._jit = jit
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

        # --- Functional streaming state ---
        # The model is split once per fresh stream into a constant graphdef +
        # params (held here, shared across streams) and a stream pytree (held in
        # each MagentaRT2State). Step callables are cached per (temperature,
        # top_k, donate); the cache is dropped whenever the graphdef changes.
        self._graphdef: Any = None
        self._params: Any = None
        self._step_cache: dict[tuple, Any] = {}

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
        return self._model.sample_rate

    def _make_step(self, temperature: float, top_k: int, *, donate: bool):
        """Per-``(temperature, top_k, donate)`` functional ``jax.jit`` step.

        Mirrors the jax backend's state threading. Rather than re-splitting the
        whole module graph in Python every call (as ``nnx.jit`` does), the model
        is split once per fresh stream into a constant ``params`` partition and a
        ``stream`` partition (KV / codec caches, decode state, sampling rng).
        This body merges them, runs one step, and splits the updated ``stream``
        back out; ``merge`` / ``split`` are traced once and become no-ops at
        runtime, leaving only array-pytree dispatch — markedly faster than the
        per-call ``nnx.jit`` split/merge.

        ``params`` is passed as a constant (never donated); ``stream`` is donated
        when ``donate`` so its cache buffers update in place. A freshly armed
        stream's first step runs with ``donate=False``, since its buffers may
        alias (which would make ``jax`` reject donating the same buffer twice).
        ``graphdef`` is captured in the closure rather than passed as a static
        arg, so there is no per-call graphdef hashing; it is invariant across
        streams, so the cache stays warm when streams are interleaved.
        ``jit=False`` returns an eager merge/step/split closure (no jit, no
        donation) for debugging.
        """
        graphdef = self._graphdef
        # Donation is meaningful only under jit; collapse the eager keys.
        key = (float(temperature), int(top_k), bool(donate) and self._jit)
        if key not in self._step_cache:
            def _step(params, stream, source_tokens):
                model = nnx.merge(graphdef, params, stream)
                tree = model.step(
                    source_tokens=source_tokens,
                    temperature=temperature, top_k=top_k,
                )
                _, _, new_stream = nnx.split(model, nnx.Param, ...)
                return tree, new_stream

            if self._jit:
                _step = jax.jit(_step, donate_argnums=1) if donate else jax.jit(_step)
            self._step_cache[key] = _step
        return self._step_cache[key]

    def _make_scan(self, temperature: float, top_k: int, *, donate: bool):
        """Per-``(temperature, top_k, donate)`` functional ``nnx.scan`` covering
        all frames in one call.

        Same functional contract as :meth:`_make_step` (constant ``params``,
        threaded ``stream``), but the merged model runs through an ``nnx.scan``
        whose carry threads the module, so its caches update in place across all
        frames with no per-frame copy — the lossless fast path for batch
        generation. Always jitted (``lax.scan`` compiles regardless of
        ``self._jit``); ``stream`` is donated when ``donate``. Consumes the whole
        stacked block at once, so it is not for interactive streaming.
        """
        graphdef = self._graphdef
        key = (float(temperature), int(top_k), bool(donate), "scan")
        if key not in self._step_cache:
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 1))
            def _scan_body(model, source_tokens):
                tree = model.step(
                    source_tokens=source_tokens,
                    temperature=temperature, top_k=top_k,
                )
                return model, tree

            def _run(params, stream, stacked):
                model = nnx.merge(graphdef, params, stream)
                model, stacked_tree = _scan_body(model, stacked)
                _, _, new_stream = nnx.split(model, nnx.Param, ...)
                return stacked_tree, new_stream

            self._step_cache[key] = (
                jax.jit(_run, donate_argnums=1) if donate else jax.jit(_run)
            )
        return self._step_cache[key]

    def _build_conditioning(self, batch_style_tokens, notes, drums, cfgs):
        """Build the ``[N, 1, C]`` conditioning block as a jnp array."""
        if not self._has_style_channels:
            # Single-channel presets (e.g. ``tiny``) have no conditioning
            # segments — emit the masked/offset token.
            offset = self._model.num_reserved_tokens + 1
            n = len(batch_style_tokens)
            return jnp.full((n, 1, 1), offset, dtype=jnp.int32)
        cond = conditioning.build_conditioning_rows(
            batch_style_tokens, notes, drums, cfgs,
            num_musiccoca=self._num_musiccoca_tokens,
            num_notes=self._num_notes,
            drum_tokens=self._drum_tokens,
            cfg_tokens=self._cfg_tokens,
            offset=self._model.num_reserved_tokens + 1,
        )
        return jnp.asarray(cond)

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
        scan: bool = False,
    ) -> tuple[AudioTree, MagentaRT2State]:
        """Generate audio from style conditioning.

        Same contract as ``magenta_rt.mlx.system.MagentaRT2System.generate``
        (see that docstring for the full conditioning semantics), with one
        backend difference: ``temperature`` / ``top_k`` must be shared scalars
        (per-element values raise).

        ``state`` is the returned :class:`MagentaRT2State`, which carries the
        stream pytree; ``state=None`` starts a fresh stream. Streams are
        independent values, so handles stay valid across other ``generate``
        calls and multiple streams can be interleaved on one system.

        ``scan=True`` runs all ``frames`` through a single ``nnx.scan`` (the
        stream threaded through the carry, caches updated in place) instead of
        the per-step Python loop. It returns identical audio and is markedly
        faster for batch generation, but consumes the whole block at once, so it
        is not for interactive frame-by-frame streaming (use the default
        per-step path, continuing via ``state``, there).

        Returns:
          (waveform, state) — a batched ``AudioTree`` (``waveform``
          ``[N, 2, T]`` float32 in [-1, 1], ``codes`` ``[N, frames, Q]`` RVQ
          codes) and the continuation state.
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

        # --- Streaming state: resolve the stream pytree to thread ---
        if state is None:
            # Fresh stream: arm the module template, then split it into a
            # constant params partition (held on the system, shared across
            # streams) and the stream partition threaded through the step.
            self._model.init_streaming(
                batch_size=N, rngs=nnx.Rngs(self._seed),
            )
            # graphdef + params are invariant across streams (the graphdef
            # encodes structure, not the batch-shaped leaves, and the params
            # never change), so the step cache stays warm across fresh streams —
            # the cached closures capture an equivalent graphdef, and jit
            # recompiles per stream-leaf shape on its own.
            self._graphdef, self._params, stream = nnx.split(
                self._model, nnx.Param, ...,
            )
            fresh = True
        else:
            if state.batch_size != N:
                raise ValueError(
                    f'state batch_size {state.batch_size} does not match the '
                    f'conditioning batch size {N}'
                )
            stream = state.stream
            fresh = False

        # --- Streaming generation (functional: thread the stream pytree) ---
        if scan and frames > 1:
            # Fast batch path: one nnx.scan over all frames, the stream threaded
            # through the carry (in-place cache updates). Same audio as the
            # per-step loop, just fused. The conditioning row is constant per
            # call, so broadcast the single block over the step axis. A freshly
            # armed stream is not donated (its buffers may alias).
            stacked = jnp.broadcast_to(block[None], (frames,) + block.shape)
            run = self._make_scan(temperature, top_k, donate=not fresh)
            stacked_tree, stream = run(self._params, stream, stacked)
            # step leaves: waveform [N, C, t], codes [N, 1, Q]; out_axes=1 adds
            # the step axis -> waveform [N, frames, C, t], codes [N, frames, 1, Q].
            waveform = np.asarray(
                rearrange(stacked_tree.waveform, "n s c t -> n c (s t)"),
                dtype=np.float32,
            )
            codes = np.asarray(
                rearrange(stacked_tree.codes, "n s o q -> n (s o) q")
            )
            tree = AudioTree(
                waveform, sample_rate=self.sample_rate, codes=codes,
            )
        else:
            trees = []
            for i in range(frames):
                # A freshly armed stream's first step runs without donation (its
                # buffers may alias); every later step donates for in-place
                # cache updates.
                step = self._make_step(
                    temperature, top_k, donate=not (fresh and i == 0),
                )
                tree, stream = step(self._params, stream, block)
                trees.append(tree)

            tree = jax.device_get(audio.concatenate(trees))

        return tree, MagentaRT2State(batch_size=N, stream=stream)
