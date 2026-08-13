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

"""AudioTree-flavored SFT transforms (audiotree-style).

Grain transforms that operate on an :class:`audiotree.AudioTree`, so one
container — audio ``waveform`` (``[B, C, T]``), neural-codec ``codes``
(frame-major ``[B, T, D]``), and the conditioning arrays in ``metadata`` —
flows from the grain source to the trainer across every backend
(NNX / JAX / MLX / mlx_pure).

The keystone is :class:`EncodeWithCodec`, mirroring
``audiotree.transforms.functional.encode_with_codec``: use ``codes`` if present,
otherwise encode ``waveform`` with a backend codec on device. The codec is
duck-typed (anything exposing ``waveform_to_codes``), so the *same* transform
serves the nnx SpectroStream (jax/nnx trainers) and the mlx_pure SpectroStream
(mlx trainer); each trainer passes its own.

:func:`augment_batch` applies a list of transforms at the trainer's consumption
boundary — *outside* jit, where a GPU codec can run — exactly the audiotree
pattern (``audiotree.transforms`` is numpy/grain; reach for jax only here).
"""

from __future__ import annotations

import dataclasses
import functools

import grain
import numpy as np

from audiotree import AudioTree
from magenta_rt.config import (
    CFG_CONDITIONING_DRUMS,
    CFG_CONDITIONING_MUSICCOCA_NOTES,
    MUSICCOCA as _MUSICCOCA,
)
from magenta_rt.conditioning import discretize_cfg

from .data import prepare_target_tokens, prepare_source_tokens, apply_musiccoca_sticky


@dataclasses.dataclass
class EncodeWithCodec(grain.transforms.Map):
  """Populate ``codes`` from ``waveform`` via a neural codec, if absent.

  Mirrors audiotree's ``encode_with_codec``: a no-op when ``codes`` are already
  present (pre-tokenized data), otherwise ``codes =
  codec.waveform_to_codes(waveform)``. ``codec`` is any object exposing
  ``waveform_to_codes(audio) -> [..., num_codebooks]`` (nnx or mlx_pure
  ``SpectroStream``; both consume channel-major ``[B, C, T]`` audio). Apply at
  the trainer boundary (outside jit) so the codec runs on device;
  ``waveform_fn`` adapts the ``[B, C, T]`` layout to whatever the codec
  expects (default: pass ``waveform`` through).
  """

  codec: object
  waveform_fn: object = None

  def map(self, audio: AudioTree) -> AudioTree:
    if audio.codes is not None:
      return audio
    waveform = (
        audio.waveform if self.waveform_fn is None else self.waveform_fn(audio.waveform)
    )
    return audio.replace(codes=self.codec.waveform_to_codes(waveform))


@dataclasses.dataclass
class PrepareTarget(grain.transforms.Map):
  """Write depthformer target tokens into ``metadata[key]`` from ``codes``.

  ``codes`` are raw SpectroStream RVQ codes (``[..., D]``);
  ``prepare_target_tokens`` maps them to the unique depthformer target scheme.
  Backend-neutral (numpy or a jax/mlx device array), so it runs on-device right
  after :class:`EncodeWithCodec`.
  """

  target_config: object
  key: str = "target"

  def map(self, audio: AudioTree) -> AudioTree:
    if audio.codes is None:
      raise ValueError(
          "PrepareTarget requires codes; run EncodeWithCodec first (or supply "
          "pre-tokenized data)."
      )
    target = prepare_target_tokens(audio.codes, self.target_config)
    return audio.replace_extras(**{self.key: target})


@dataclasses.dataclass
class PrepareSource(grain.transforms.RandomMap):
  """Write concatenated source conditioning tokens into ``metadata[key]``.

  Reads the per-channel conditioning arrays from ``metadata`` (keyed by their
  ``magenta_rt.config`` names) and applies ``prepare_source_tokens`` (per-channel
  rvq truncation, offsets, and stochastic input dropout). A ``RandomMap`` because
  dropout is stochastic; run it per-example in the grain pipeline (before
  batching) so dropout decisions are independent across the batch.
  """

  input_configs: tuple
  key: str = "source"

  def random_map(self, audio: AudioTree, rng) -> AudioTree:
    source = prepare_source_tokens(audio.extras, self.input_configs, rng)
    return audio.replace_extras(**{self.key: source})


@dataclasses.dataclass
class AddFixedStyle(grain.transforms.Map):
  """Overlay one fixed MusicCoCa style-token row onto every example.

  The single-style-prompt recipe applied at *training* time: set
  ``extras['mulan_tokens_25hz']`` to ``tokens`` (one RVQ row, e.g. from a text
  prompt) broadcast across all frames, so every example conditions on one
  constant style. This lets a **codec-only** export (no per-clip style; the
  channel would otherwise fall back to the learned dropout token) be fine-tuned
  toward a fixed style without re-exporting. Any existing MusicCoCa channel is
  overwritten. Run before :class:`PrepareSource` (and before
  :class:`StyleEmbeddingJitter` / :class:`AudioTreeMusicCoCaSticky`, which then
  see this constant channel).
  """

  tokens: np.ndarray  # [rvq] int token row
  musiccoca_key: str = _MUSICCOCA.key

  def map(self, audio: AudioTree) -> AudioTree:
    n_frames = _frame_length(audio)
    row = np.asarray(self.tokens, dtype=np.int32).reshape(-1)  # [rvq]
    B = _batch_size(audio)
    mulan = np.broadcast_to(row, (B, n_frames, row.shape[0])).copy()  # [B,T,rvq]
    return audio.replace_extras(**{self.musiccoca_key: mulan})


# ---------------------------------------------------------------------------
# Grain-pipeline transforms. These support both unbatched (batch-1) and
# batched (batch-B) AudioTrees — waveform ``[B, nch, nsamp]``, codes ``[B, T, D]``
# and metadata arrays ``[B, T, ch]`` (where ``B`` is the batch size).
#
# Metadata arrays come in two kinds: *frame-aligned* conditioning channels
# ``[B, T, ch]`` (ndim >= 3, sharing the 25 Hz frame axis with ``codes``) and
# *static* per-example arrays like a ``[B, 768]`` MusicCoCa embedding
# (ndim < 3). Time transforms (cropping) only touch the frame-aligned kind;
# static arrays pass through untouched and simply concatenate in the batch.
# ---------------------------------------------------------------------------


def _is_frame_aligned(arr) -> bool:
  """True for ``[B, T, ch]``-style extras that share the frame axis.

  ``extras`` is a free-form payload dict and audiotree supports non-array leaves
  in it (strings, for instance), so this has to answer False for anything
  without a shape rather than raise. The export's own provenance no longer
  lives here — ``filepath`` / ``offset`` are AudioTree-managed — but a caller's
  ``extras`` may still carry one.
  """
  if not hasattr(arr, "ndim"):
    return False
  return arr.ndim >= 3


def _batch_size(audio: AudioTree) -> int:
  """Get the batch size (leading axis length) of the AudioTree.

  Tries ``audio.batch_size`` property first; if that raises ValueError
  (waveform/codes/latents all None), infers from the first metadata array.
  """
  try:
    return audio.batch_size
  except ValueError:
    for value in audio.extras.values():
      if hasattr(value, "shape") and value.ndim >= 1:
        return value.shape[0]
    return 1


def _frame_length(audio: AudioTree) -> int:
  """Number of token frames (axis-1 length) for an AudioTree."""
  if audio.codes is not None:
    return audio.codes.shape[1]
  for value in audio.extras.values():
    if _is_frame_aligned(value):
      return value.shape[1]
  raise ValueError(
      "cannot determine frame length: AudioTree has no codes and no "
      "frame-aligned ([1, T, ch]) metadata"
  )


def _pad_axis1(arr, length):
  """Zero-pad ``arr`` along axis 1 up to ``length`` (no-op if already >=)."""
  short = length - arr.shape[1]
  if short <= 0:
    return arr
  pad = [(0, 0), (0, short)] + [(0, 0)] * (arr.ndim - 2)
  return np.pad(arr, pad, constant_values=0)


def _pad_last(arr, length):
  """Zero-pad ``arr`` along the last axis up to ``length`` (no-op if >=)."""
  short = length - arr.shape[-1]
  if short <= 0:
    return arr
  pad = [(0, 0)] * (arr.ndim - 1) + [(0, short)]
  return np.pad(arr, pad, constant_values=0)


@dataclasses.dataclass
class AudioTreeRandomCrop(grain.transforms.RandomMap):
  """Randomly crop (or zero-pad) a batch-1 AudioTree to ``crop_frames`` frames.

  Frame-rate arrays (``codes`` and every frame-aligned ``metadata`` array)
  are cropped along axis 1; ``waveform`` is cropped along its last (time)
  axis by ``samples_per_frame`` (derived per-example as ``nsamp //
  num_frames``) so audio stays aligned with the token frames without
  hard-coding the codec hop. Short examples are zero-padded. Static metadata
  (e.g. a ``[1, 768]`` MusicCoCa embedding) passes through untouched.
  """

  crop_frames: int

  def random_map(self, audio: AudioTree, rng) -> AudioTree:
    n_frames = _frame_length(audio)
    cf = self.crop_frames
    spf = audio.waveform.shape[-1] // n_frames if audio.waveform is not None else 0
    start = 0 if n_frames <= cf else int(rng.integers(0, n_frames - cf + 1))
    pad = n_frames <= cf

    def crop_frame(a):
      seg = a[:, start:start + cf]
      return _pad_axis1(seg, cf) if pad else seg

    def crop_samp(a):
      seg = a[..., start * spf:(start + cf) * spf]
      return _pad_last(seg, cf * spf) if pad else seg

    return audio.replace(
        waveform=None if audio.waveform is None else crop_samp(audio.waveform),
        codes=None if audio.codes is None else crop_frame(audio.codes),
        extras={
            k: crop_frame(v) if _is_frame_aligned(v) else v
            for k, v in audio.extras.items()
        },
    )


@dataclasses.dataclass
class PrepareCFG(grain.transforms.RandomMap):
  """Fill missing CFG-conditioning channels with per-example tokens.

  The mrt2 source includes two channels that carry the classifier-free
  guidance *strength* the model is conditioned on at inference time
  (``cfg_conditioning_tokens`` ``[T, 2]`` for musiccoca/notes and
  ``cfg_conditioning_drums_tokens`` ``[T, 1]`` for drums; scales in
  ``[-1, 7]`` discretized by ``magenta_rt.conditioning.discretize_cfg``).
  They are properties of the *training recipe*, not of the audio, so the
  offline export (``magenta_rt.sft.export``) does not store them — this
  transform synthesizes them in the grain pipeline instead.

  Per example, each configured channel that is **absent** from
  ``metadata`` gets one token row, constant across the example's frames
  (matching how the inference systems hold the CFG tokens constant over
  time). Channels already present in the data pass through untouched,
  so this is always safe to apply.

  Two modes per channel:

  * **sampled** (default): tokens drawn uniformly from
    ``[0, codebook_size)`` — i.e. a fresh CFG scale per example. This
    exercises the pretrained model's full guidance-token range, keeping
    its CFG responsiveness intact through fine-tuning.
  * **fixed** (``fixed_scales={key: scale_or_per_channel_scales}``):
    every example gets the token(s) for the given float scale(s) in
    ``[-1, 7]``. Use this to specialize the model at the guidance
    strengths you will actually run at inference (e.g.
    ``{"cfg_conditioning_tokens": (3.0, 1.0)}`` for the system
    defaults).

  Note on provenance: the original mrt2 training recipe for these
  channels is not public; sampling-uniform is the conservative default
  for SFT-from-pretrained because it avoids collapsing the token ↔
  guidance-strength association the checkpoint already carries.
  """

  cfg_configs: tuple = (CFG_CONDITIONING_MUSICCOCA_NOTES, CFG_CONDITIONING_DRUMS)
  fixed_scales: object = None  # Optional[Mapping[str, float | Sequence[float]]]

  def random_map(self, audio: AudioTree, rng) -> AudioTree:
    updates = {}
    num_frames = None
    B = _batch_size(audio)
    for cfg in self.cfg_configs:
      if cfg.key in audio.extras:
        continue
      if num_frames is None:
        num_frames = _frame_length(audio)
      width = cfg.rvq_truncation_level
      if self.fixed_scales and cfg.key in self.fixed_scales:
        scales = self.fixed_scales[cfg.key]
        if not isinstance(scales, (tuple, list)):
          scales = (scales,) * width
        if len(scales) != width:
          raise ValueError(
              f"{cfg.key}: expected {width} scale(s), got {len(scales)}"
          )
        # discretize_cfg bins [-1, 7] with step 8 / (codebook_size - 1).
        step = 8.0 / (cfg.codebook_size - 1)
        row = np.array(
            [discretize_cfg(s, step, cfg.codebook_size - 1) for s in scales],
            dtype=np.int32,
        )
        updates[cfg.key] = np.tile(row, (B, num_frames, 1))  # [B, T, width]
      else:
        row = rng.integers(0, cfg.codebook_size, size=(B, width), dtype=np.int32)
        updates[cfg.key] = np.tile(row[:, None, :], (1, num_frames, 1))  # [B, T, width]
    if not updates:
      return audio
    return audio.replace_extras(**updates)


@dataclasses.dataclass
class AudioTreeMusicCoCaSticky(grain.transforms.RandomMap):
  """Apply 'sticky' MusicCoCa augmentation to the style metadata channel.

  Repeats each style frame with probability ``sticky_prob`` (so the style token
  stream changes less often), matching the legacy dict-pipeline behavior. A
  no-op if the AudioTree carries no MusicCoCa channel.
  """

  sticky_prob: float
  musiccoca_key: str = _MUSICCOCA.key

  def random_map(self, audio: AudioTree, rng) -> AudioTree:
    if self.musiccoca_key not in audio.extras:
      return audio
    arr = audio.extras[self.musiccoca_key]  # [B, T, ch]
    B = _batch_size(audio)
    sticky = np.stack([
        apply_musiccoca_sticky(arr[b], self.sticky_prob, rng)
        for b in range(B)
    ], axis=0)
    return audio.replace_extras(**{self.musiccoca_key: sticky})


def rvq_tokenize(embedding: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
  """Greedy residual-VQ tokenization in numpy (MusicCoCa's quantizer).

  Same algorithm as ``magenta_rt.nnx.musiccoca.EmbeddingQuantizer`` /
  the original TFLite quantizer, in plain numpy so it can run inside
  grain worker processes: per stage, pick the codebook entry nearest the
  running residual and subtract it.

  Args:
    embedding: ``[dim]`` float embedding.
    codebooks: ``[depth, codebook_size, dim]`` codebooks (the
      ``quantizer.codebooks`` tensor from the converted MusicCoCa
      safetensors).

  Returns:
    ``[depth]`` int32 token indices.
  """
  residual = embedding.astype(np.float64)
  tokens = np.empty(codebooks.shape[0], dtype=np.int32)
  for stage, codebook in enumerate(codebooks):
    cb = codebook.astype(np.float64)
    distances = (
        np.sum(residual**2) - 2.0 * cb @ residual + np.sum(cb**2, axis=-1)
    )
    idx = int(np.argmin(distances))
    tokens[stage] = idx
    residual = residual - cb[idx]
  return tokens


@dataclasses.dataclass
class StyleEmbeddingJitter(grain.transforms.RandomMap):
  """Style augmentation in MusicCoCa *embedding space*.

  The offline export stores the raw 768-dim MusicCoCa embedding per
  window (``extras['musiccoca_embedding']``) alongside its quantized
  tokens. That makes a much richer augmentation possible than the
  token-level sticky repeat: perturb the *embedding* with Gaussian noise
  and re-quantize, yielding style tokens from the neighborhood of the
  true style rather than an arbitrary token corruption. Near-duplicate
  styles map to the same tokens (RVQ cells are coarse), so small jitter
  produces a realistic mix of identical and slightly-shifted token rows.

  Per example, with probability ``prob``: ``emb' = emb + N(0, (noise_std
  · rms(emb))²)`` (noise scaled to the embedding's per-dim RMS, so one
  ``noise_std`` works across un-normalized MusicCoCa embeddings), then
  ``extras['mulan_tokens_25hz']`` is rewritten with the re-quantized
  tokens (constant across the window's frames, like the export) and the
  stored embedding is updated to match. A no-op when the example carries
  no embedding (an export with ``save_embedding=False`` or
  ``tree_exclude_prefixes=["extras.musiccoca_embedding"]``).

  Requires the converted MusicCoCa codebooks: pass ``codebooks``
  directly (tests) or leave None to load ``quantizer.codebooks`` from
  ``codebooks_path`` (default: the ``musiccoca_nnx.safetensors`` produced
  by ``python -m magenta_rt.nnx.musiccoca.convert``, next to the TFLite
  resources). The from-disk load is a ``cached_property`` resolved on first
  use — i.e. **per worker process**, not at construction. This is
  deliberate for ``mp_prefetch`` safety: the codebooks are ~38 MB, so
  loading them in ``__init__`` / ``__post_init__`` would pickle that array
  to every grain worker (and hold it in the main process, which never runs
  the transform). Lazily, each worker loads it from disk once on first
  ``random_map``.

  Apply *before* :class:`AudioTreeMusicCoCaSticky` (jitter picks the
  window's style; sticky then time-augments it).
  """

  noise_std: float
  prob: float = 1.0
  embedding_key: str = "musiccoca_embedding"
  tokens_key: str = _MUSICCOCA.key
  codebooks: object = None        # Optional[np.ndarray] [depth, size, dim]
  codebooks_path: object = None   # Optional[str | Path]

  @functools.cached_property
  def _codebooks(self) -> np.ndarray:
    """``[depth, codebook_size, dim]`` codebooks, resolved once per process.

    A ``cached_property`` (not ``__post_init__``) so the ~38 MB array is
    loaded lazily in whichever process first runs the transform — the grain
    worker under ``mp_prefetch`` — rather than pickled to every worker.
    An explicitly-passed ``codebooks`` array takes precedence.
    """
    if self.codebooks is not None:
      return self.codebooks
    from safetensors.numpy import load_file

    from magenta_rt import paths

    path = self.codebooks_path or (
        paths.musiccoca_dir() / "musiccoca_nnx.safetensors"
    )
    return load_file(str(path))["quantizer.codebooks"]

  def random_map(self, audio: AudioTree, rng) -> AudioTree:
    if self.noise_std <= 0 or self.embedding_key not in audio.extras:
      return audio
    if rng.random() >= self.prob:
      return audio
    embedding = np.asarray(audio.extras[self.embedding_key], np.float32)  # [B, dim]
    B = embedding.shape[0]
    rms = np.sqrt(np.mean(np.square(embedding), axis=-1, keepdims=True))  # [B, 1]
    jittered = embedding + rng.normal(
        0.0, self.noise_std * rms, embedding.shape
    ).astype(np.float32)  # [B, dim]
    tokens = np.stack([
        rvq_tokenize(jittered[b], self._codebooks)
        for b in range(B)
    ], axis=0)  # [B, depth]
    num_frames = _frame_length(audio)
    return audio.replace_extras(**{
        self.embedding_key: jittered,
        self.tokens_key: np.tile(tokens[:, None, :], (1, num_frames, 1)),
    })


def augment_batch(rng, batch, transforms) -> AudioTree:
  """Apply a list of grain transforms to ``batch`` at the trainer boundary.

  The audiotree pattern: runs *outside* jit, so a transform may execute a codec
  on device (e.g. :class:`EncodeWithCodec`). ``RandomMap`` transforms each get a
  freshly split child rng; ``Map`` transforms are applied directly. ``rng`` is a
  numpy ``Generator``, or ``None`` for a Map-only list (e.g. encode +
  prepare-target) — pass one whenever the list contains a ``RandomMap``.
  Returns the transformed ``batch`` (an :class:`AudioTree`).
  """
  for transform in transforms:
    if isinstance(transform, grain.transforms.RandomMap):
      child = np.random.Generator(np.random.PCG64(int(rng.integers(2**63))))
      batch = transform.random_map(batch, child)
    elif isinstance(transform, grain.transforms.Map):
      batch = transform.map(batch)
    else:
      raise TypeError(f"Unsupported transform type: {type(transform)!r}")
  return batch


def to_source_target(batch, target_config, *, codec=None, asarray=None):
  """Trainer boundary: a batched ``AudioTree`` -> ``(source, target)`` arrays.

  The single consumption point shared by the NNX, JAX and MLX trainers. Runs the
  unified :func:`augment_batch` boundary — encode ``waveform`` -> ``codes`` when
  a ``codec`` is given and ``codes`` are absent (the on-the-fly GPU path),
  otherwise use the pre-tokenized ``codes`` — then builds the depthformer target
  and returns ``extras['source']`` / ``extras['target']`` cast via
  ``asarray`` (``jnp.asarray`` / ``mx.array``; default identity).
  """
  cast = asarray or (lambda x: x)
  transforms = []
  if codec is not None:
    if batch.waveform is not None and batch.codes is None:
      # Move audio to device so the codec runs on GPU.
      batch = batch.replace(waveform=cast(batch.waveform))
    transforms.append(EncodeWithCodec(codec))
  transforms.append(PrepareTarget(target_config))
  batch = augment_batch(None, batch, transforms)
  return cast(batch.extras["source"]), cast(batch.extras["target"])


@dataclasses.dataclass
class ExactLength(grain.transforms.Map):
  """Trim or zero-pad an AudioTree waveform to exactly ``window_samples``."""

  window_samples: int

  def map(self, audio: AudioTree) -> AudioTree:
    # Decoders may come back a few samples long or short of
    # duration * sample_rate; enforce the exact window so every record
    # shares the TreeWriter leaf shape.
    waveform = audio.waveform
    if waveform.shape[-1] > self.window_samples:
      return audio.replace(waveform=waveform[..., :self.window_samples])
    if waveform.shape[-1] < self.window_samples:
      pad = [(0, 0)] * (waveform.ndim - 1)
      pad.append((0, self.window_samples - waveform.shape[-1]))
      return audio.replace(waveform=np.pad(waveform, pad))
    return audio
