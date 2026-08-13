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

"""Grain-based SFT data pipeline. Pure-numpy from source to batch.

Callers cast to ``jnp.asarray`` / ``mx.array`` at the consumption boundary,
so the same pipeline serves the NNX, JAX, and MLX trainers.

Example schema (raw codebook indices, no offsets) — mrt2 source channels keyed
by their ``magenta_rt.config`` names, plus the SpectroStream target. Shapes
refer to ``[T, features]`` (a leading batch axis—``1`` for raw examples,
``B`` for batched pipelines—is prepended on every leaf):
    soundstream_tokens             : [T, target_rvq] int32 in [0, codebook_size)
    mulan_tokens_25hz              : [T, 12]  int32 in [-1, musiccoca_codebook_size)
    pianoroll_with_onsets_tokens   : [T, 128] int32 in [0, 4)
    drum_pianoroll_tokens          : [T, 1]   int32 in [0, 2)
    cfg_conditioning_tokens        : [T, 2]   int32 in [0, 41)
    cfg_conditioning_drums_tokens  : [T, 1]   int32 in [0, 9)
(``mulan_tokens_25hz`` carries quantized MusicCoCa style tokens — the export
step quantizes raw MusicCoCa embeddings before writing.)

The on-disk format is an **audiotree ``TreeWriter`` export** — one directory
with ``manifest.json`` + one memmap per leaf, written as an *AudioTree
pytree* (``codes`` + conditioning ``metadata``, optionally ``waveform``) by
``magenta_rt.sft.export``. Read directly with audiotree's
``TreeDataSource``. While each individual raw record reconstructs as an
:class:`audiotree.AudioTree` with a batch-1 leading axis on every leaf,
the pipeline batches them together via ``.batch(batch_size, batch_fn=AudioTree.batch)``
so that the final collated ``AudioTree`` has the requested ``batch_size``.
The whole dataset shares a fixed per-example window length, so write
windows at least as long as the training crop.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import grain
import numpy as np

from ._grain_flags import ensure_absl_flags_parsed

from audiotree import AudioTree
from audiotree.sources import TreeDataSource


# ---------------------------------------------------------------------------
# Augmentation primitives (pure numpy, no jax / mlx)
# ---------------------------------------------------------------------------


def apply_musiccoca_sticky(
    musiccoca_tokens: np.ndarray,
    sticky_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    T = musiccoca_tokens.shape[0]
    if T <= 1:
        return musiccoca_tokens.copy()
    repeat_mask = rng.random(T - 1) < sticky_prob
    source_indices = np.arange(T)
    for i in range(1, T):
        if repeat_mask[i - 1]:
            source_indices[i] = source_indices[i - 1]
    return musiccoca_tokens[source_indices].copy()


def prepare_source_tokens(
    example: dict[str, np.ndarray],
    input_configs: Sequence,
    rng: np.random.Generator,
) -> np.ndarray:
    """Concatenate the per-channel conditioning tokens with offsets applied.

    A configured channel that is *missing* from the example falls back to its
    learned unconditional (dropout) token — the ``-1 -> dropout-token``
    mapping inference uses — provided the channel reserves one
    (``dropout_prob is not None``). This is how an export without MT3
    transcription trains against the full mrt2 source spec: the piano-roll
    streams condition on "dropped" everywhere.
    """
    parts = []
    for cfg in input_configs:
        if cfg.key in example:
            tokens = example[cfg.key].copy()
            if tokens.shape[-1] > cfg.rvq_truncation_level:
                tokens = tokens[..., : cfg.rvq_truncation_level]
            if cfg.dropout_prob is not None and cfg.dropout_prob > 0:
                if rng.random() < cfg.dropout_prob:
                    tokens = np.full_like(tokens, -1)
        else:
            if cfg.dropout_prob is None:
                raise KeyError(
                    f"example is missing conditioning channel {cfg.key!r}, "
                    "which reserves no dropout token to fall back to"
                )
            reference = next(
                (example[c.key] for c in input_configs if c.key in example),
                None,
            )
            if reference is None:
                raise KeyError(
                    "cannot infer the frame count for missing channel "
                    f"{cfg.key!r}: no configured channel is present"
                )
            shape = reference.shape[:-1] + (cfg.rvq_truncation_level,)
            tokens = np.full(shape, -1, dtype=np.int16)
        # Every conditioning channel reserves the dropout slot at index
        # ``num_extra_tokens``, so real tokens start at ``num_extra_tokens + 1``.
        # This matches the inference convention shared by all four backends —
        # ``jax``/``mlx`` ``_build_conditioning`` and
        # ``conditioning.build_conditioning_rows`` both use
        # ``offset = NUM_RESERVED_TOKENS + 1`` for the whole conditioning row,
        # *including* the no-dropout CFG channels. (Those CFG channels never emit
        # the dropout token, but the pretrained encoder embedding still reserves
        # the slot — verified against the mrt2_small checkpoint — so they must be
        # offset by ``+1`` too. Using ``+0`` here conditioned every CFG token one
        # embedding row too low relative to every generation path.)
        offset = cfg.num_extra_tokens + 1
        # int16 is the natural width for source token ids — the largest offset
        # source value is ~(per_rvq_vocab_size + num_extra) ≈ 1k, far below the
        # int16 max (32767). Keeping the loader compact (no int32 upcast) is
        # lossless and matches the storage dtypes; the per-channel cast also
        # normalizes the mixed stored dtypes (int8 pianoroll, int16 mulan,
        # int32-synthesized CFG) to one type for the concat. The embedding
        # lookups are dtype-agnostic, so the model consumes int16 directly.
        # ``test_token_dtypes`` guards the headroom so a future vocab bump that
        # would overflow int16 fails loudly rather than wrapping silently.
        parts.append((tokens + offset).astype(np.int16))
    return np.concatenate(parts, axis=-1)


def _array_namespace(arr):
    """Return the array module (numpy / jax.numpy / mlx.core) backing ``arr``.

    Lets token-prep run on host numpy (the grain pipeline) *or* on a device
    array (the trainer's GPU-encoded codes) without importing jax/mlx on the
    pure-numpy path — the backend module is imported lazily, only when a device
    array is actually passed in.
    """
    module = type(arr).__module__
    if module.startswith(("jax", "jaxlib")):
        import jax.numpy as xnp

        return xnp
    if module.startswith("mlx"):
        import mlx.core as xnp

        return xnp
    return np


def prepare_target_tokens(soundstream_tokens, target_config):
    """Map raw SpectroStream RVQ codes ``[..., D]`` to depthformer target tokens.

    Codes deeper than ``target_config.rvq_truncation_level`` are dropped first
    (the depthformer only models the truncated depth; full-depth codec output —
    e.g. a ``magenta_rt.sft.export`` dataset or on-the-fly encoding — stores all
    levels for flexibility). Then adds per-depth offsets (``depth *
    codebook_size``) so each codebook occupies a disjoint range, plus the
    reserved ``num_extra_tokens``. Backend-neutral: runs on numpy (the grain
    path) or a jax/mlx device array (the on-the-fly GPU encode path), so
    ``AudioTree.codes`` / ``waveform_to_codes`` output can be turned into
    targets on-device.
    """
    xp = _array_namespace(soundstream_tokens)
    truncation = getattr(target_config, "rvq_truncation_level", None)
    if truncation is not None and soundstream_tokens.shape[-1] > truncation:
        soundstream_tokens = soundstream_tokens[..., :truncation]
    D = soundstream_tokens.shape[-1]
    # int16 output (no int32 upcast): the largest target id is
    # ``(codebook-1) + (depth-1)*codebook + num_extra`` — e.g. 12,293 for
    # mrt2_{small,base} — well within the int16 max (32,767), and the
    # intermediate ``tokens + depth_offsets`` (≤ 12,287) does not overflow
    # either. Lossless, keeps the loader compact, and the embedding lookup
    # accepts int16. ``test_token_dtypes`` pins the headroom against a future
    # codebook/depth bump that would exceed int16.
    tokens = soundstream_tokens.astype(xp.int16)
    depth_offsets = (xp.arange(D) * target_config.codebook_size).astype(xp.int16)
    return (tokens + depth_offsets + target_config.num_extra_tokens).astype(xp.int16)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def create_audiotree_dataset(
    data_dir: str,
    *,
    batch_size: int,
    crop_length_seconds: Optional[int] = None,
    input_configs: Sequence,
    target_config,
    seed: int,
    musiccoca_sticky_prob: float = 0.995,
    num_workers: int = 0,
    per_worker_buffer_size: int = 2,
    tree_exclude_prefixes: Iterable[str] = (),
    load_into_memory: bool = False,
    cfg_fixed_scales=None,
    style_jitter_std: float = 0.0,
    style_jitter_prob: float = 1.0,
    style_tokens=None,
) -> grain.IterDataset[AudioTree]:
    """Audiotree-style SFT pipeline: yields batched :class:`audiotree.AudioTree`.

    Flows an AudioTree (waveform / codes / metadata) rather than a
    ``{source, target}`` dict: the grain stages crop, sticky-augment, and fill
    ``extras['source']`` while carrying ``codes`` (or raw ``waveform``). The
    *trainer* then calls ``sft.augment_batch`` to build ``extras['target']`` —
    and, when ``codes`` are absent, encode ``waveform`` to codes on device.
    ``target_config`` is accepted for signature parity (the target is built at
    the trainer boundary).

    CFG-conditioning channels (``input_configs`` entries whose key starts
    with ``cfg_conditioning``) are synthesized per example by
    :class:`magenta_rt.sft.transforms.PrepareCFG` when the data does not
    carry them (offline exports store only audio-derived channels). By
    default a fresh guidance scale is sampled uniformly per example; pass
    ``cfg_fixed_scales={key: scale_or_scales}`` to pin them (e.g.
    ``{"cfg_conditioning_tokens": (3.0, 1.0),
    "cfg_conditioning_drums_tokens": 1.0}`` to train at the inference
    defaults). Data that already has the channels is untouched.

    ``style_jitter_std > 0`` enables embedding-space style augmentation
    (:class:`magenta_rt.sft.transforms.StyleEmbeddingJitter`): the stored
    MusicCoCa embedding is perturbed and re-quantized per example, replacing
    the style tokens. Requires exports that carry
    ``extras['musiccoca_embedding']`` and the converted MusicCoCa
    safetensors on disk; a no-op for data without embeddings.

    ``crop_length_seconds=None`` (the default) trains on each record at its
    full stored length — no :class:`AudioTreeRandomCrop`. Fixed-length
    TreeWriter exports already share one shape, so they batch cleanly without
    a crop; pass an int (seconds) only to randomly crop/zero-pad for
    augmentation (e.g. drawing 2 s windows from longer records).
    """
    from . import transforms as _T  # local import: transforms imports this module

    source = TreeDataSource(data_dir, exclude_prefixes=list(tree_exclude_prefixes),
                            load_into_memory=load_into_memory)

    cfg_configs = tuple(
        cfg for cfg in input_configs if cfg.key.startswith("cfg_conditioning")
    )

    ds = grain.MapDataset.source(source).seed(seed).shuffle().repeat()
    # crop_length_seconds=None (default) → no crop: each record trains at its
    # full stored length. Fixed-length TreeWriter exports already share one
    # shape, so they batch cleanly without a crop. An int randomly crops (or
    # zero-pads) to that many seconds of 25 Hz frames for augmentation.
    if crop_length_seconds is not None:
        ds = ds.random_map(_T.AudioTreeRandomCrop(int(crop_length_seconds * 25)))
    # Single-style-prompt recipe (training-time): overlay one fixed MusicCoCa
    # token row on every example (e.g. for a codec-only export that carries no
    # per-clip style). Before the style-touching transforms below.
    if style_tokens is not None:
        ds = ds.map(_T.AddFixedStyle(np.asarray(style_tokens, dtype=np.int32)))
    ds = (
        ds.random_map(
            _T.StyleEmbeddingJitter(style_jitter_std, style_jitter_prob)
        )
        .random_map(_T.AudioTreeMusicCoCaSticky(musiccoca_sticky_prob))
        .random_map(_T.PrepareCFG(cfg_configs, fixed_scales=cfg_fixed_scales))
        .random_map(_T.PrepareSource(tuple(input_configs)))
        .to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
        .batch(batch_size, batch_fn=AudioTree.batch)
    )

    if num_workers > 0:
        ensure_absl_flags_parsed()
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=num_workers,
                per_worker_buffer_size=per_worker_buffer_size,
            ),
        )
    return ds
