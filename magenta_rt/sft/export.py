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

"""Offline SFT dataset precompute: salient excerpts → codes + MusicCoCa.

An audiotree prerender pipeline: ``audiotree.sources.create_audio_dataset``
draws fixed-duration *salient* excerpts (multi-try loudness search via
``ExcerptConfig``) from directories of audio files, grain worker processes
read + preprocess them in parallel, and the main process encodes each batch with a
SpectroStream codec and a MusicCoCa style model and writes the results as an
*AudioTree pytree* via audiotree's ``TreeWriter`` (``manifest.json`` + one
memmap per leaf). The raw ``waveform`` is **not** saved — each record carries:

* ``codes``                            ``[1, T, D]``  SpectroStream RVQ codes
* ``extras['mulan_tokens_25hz']``    ``[1, T, 12]`` quantized style tokens,
  the per-excerpt embedding's RVQ tokens broadcast across the excerpt's frames
* ``extras['musiccoca_embedding']``  ``[1, 768]``   raw style embedding
  (static per-example metadata; the training pipeline passes it through —
  exclude it via ``create_audiotree_dataset(tree_exclude_prefixes=
  ["extras.musiccoca_embedding"])`` to leave it on disk)
* ``filepath`` / ``offset`` — provenance in AudioTree's own container (read
  back via the ``.filepath`` / ``.offset`` properties): the source audio file
  and the excerpt's offset in seconds within it

With a ``transcriber`` (e.g. :func:`mt3_transcriber`), the note piano-roll
conditioning channel is added per excerpt (see
:mod:`magenta_rt.sft.pianoroll`):

* ``extras['pianoroll_with_onsets_tokens']``  ``[1, T, 128]`` int8 in
  ``[0, 3)`` — off / on / onset per pitch

The drum channel (``drum_pianoroll_tokens``) is intentionally **not** produced:
it is a per-frame intent directive (``-1`` "let the model decide" / ``0`` "don't
play drums" / ``1`` "please play drums"), not an onset raster, and cannot be
labeled from a transcription (see :mod:`magenta_rt.sft.pianoroll`). With it
absent, training conditions drums on the learned unconditional (dropout) token.

audiotree's ``TreeDataSource`` reconstructs these records as
:class:`audiotree.AudioTree` directly, and the resulting dataset is plain
numpy at read time — the same export feeds the NNX, JAX, and MLX trainers
(``create_audiotree_dataset`` reads it natively).

Both models are duck-typed, so any backend works:

* ``codec``: exposes ``waveform_to_codes([B, C, T] @ 48 kHz) -> [B, T, D]``
  (the nnx / mlx_pure / mlx ``SpectroStream``).
* ``style_model``: a :class:`magenta_rt.musiccoca.MusicCoCaBase`
  (TFLite-backed or any of the nnx / mlx_pure / jax / mlx ports).
* ``transcriber``: a callable ``(mono_samples_16k) -> transcription`` whose
  result has ``.notes`` (pitch / start_time / end_time / is_drum) — e.g.
  :func:`mt3_transcriber` wrapping ``magenta_rt.nnx.mt3``.

Example:

    from magenta_rt.nnx import MagentaRT2Sampler
    from magenta_rt.nnx.musiccoca import MusicCoCa
    from magenta_rt.sft.export import export_tree_dataset

    mrt = MagentaRT2Sampler.from_preset("mrt2_small", rngs=nnx.Rngs(0))
    mrt.load_checkpoint("checkpoints/<name>.safetensors")
    export_tree_dataset(
        "~/Datasets/my_audio", "datasets/my_sft",
        codec=mrt.spectrostream, style_model=MusicCoCa(),
        num_samples=1024,
    )

One constraint inherited from ``TreeWriter``: every record shares a fixed
per-leaf shape, so pick ``duration`` at least as long as the training crop
(``crop_length_seconds``); ``AudioTreeRandomCrop`` still picks a random
sub-window within each record at train time.
"""

from __future__ import annotations

import dataclasses
import functools

import pathlib
from typing import Optional, Sequence

import numpy as np

from audiotree import AudioTree
from audiotree import ExcerptConfig, TreeWriter
from audiotree.sources import create_audio_dataset
from audiotree.transforms import stereo
from audiotree.sources.core import _load_excerpt
import grain

from magenta_rt.config import MUSICCOCA as _MUSICCOCA

from ._grain_flags import ensure_absl_flags_parsed
from .pianoroll import transcription_to_channels
from .transforms import ExactLength

# SpectroStream operating point.
SAMPLE_RATE = 48_000
FRAME_RATE = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE  # 1920

# Audio sample rate expected for MT3 transcription.
MT3_SAMPLE_RATE = 16_000

EMBEDDING_KEY = "musiccoca_embedding"


# ---------------------------------------------------------------------------
# Per-excerpt level normalization for ``export_tree_dataset(normalize=)`` is
# supplied entirely by ``audiotree.transforms`` — no local helpers:
#   * ``peak_norm()``                            -> peak 1.0
#   * ``volume_norm(min_db=L, max_db=L)``        -> a fixed LUFS target
#   * ``volume_change(min_db=D, max_db=D)``      -> a fixed dB gain
#     (a linear gain g is ``D = 20*log10(g)`` dB)
# These are grain transforms that pickle into ``mp_prefetch`` workers; the
# export hook below applies a ``RandomMap`` via ``.random_map`` (it inherits a
# seed derived from the pipeline's root ``.seed(seed)``) and any other callable
# via ``.map``.
# ---------------------------------------------------------------------------
def mt3_transcriber(
    model_type: str = "mt3", batch_size: int = 8, backend: str = "nnx"
):
    """Builds a ``transcriber`` callable backed by an MT3 port.

    Loads the pretrained checkpoint (downloading it on first use) and
    returns ``transcribe_fn(mono_samples_16k) -> NoteSequence``.

    Args:
      model_type: MT3 checkpoint type ('mt3' multitrack by default).
      batch_size: Audio segments decoded in parallel.
      backend: "nnx" (JAX) or "mlx_pure" (Apple Silicon); both load the
        same converted safetensors and decode identical tokens.
    """
    if backend == "nnx":
        from magenta_rt.nnx import mt3
    elif backend == "mlx_pure":
        from magenta_rt.mlx_pure import mt3
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'nnx' or 'mlx_pure'")

    model = mt3.load_model(model_type)

    def transcribe_fn(samples: np.ndarray):
        return mt3.transcribe(model, samples, batch_size=batch_size)

    return transcribe_fn


def discover_audio_files(
    sources: str | pathlib.Path | Sequence[str | pathlib.Path],
    extensions: Optional[Sequence[str]] = None,
) -> list[str]:
    """Recursively list audio files under ``sources`` (hidden files skipped).

    Returns a sorted list, so a seeded shuffle + split is reproducible —
    the basis for a leak-free **file-level** train/val split (the same audio
    file never appears in both, unlike excerpt-level seeds over a shared
    file pool). Uses audiotree's discovery so the file set matches what
    ``create_audio_dataset`` would have walked.
    """
    from audiotree.sources import find_audio_files

    if isinstance(sources, (str, pathlib.Path)):
        sources = [sources]
    # `find_audio_files` already returns a sorted, de-duplicated list, and
    # applies audiotree's own default extension set when given None.
    return find_audio_files(
        [str(s) for s in sources],
        list(extensions) if extensions is not None else None,
    )


def split_audio_files(
    files: Sequence[str], *, val_fraction: float, split_seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministic file-level split → ``(train_files, val_files)``.

    Holds out ``round(len(files) * val_fraction)`` whole files for
    validation. A fixed ``split_seed`` over the sorted file list makes the
    partition reproducible and auditable.
    """
    import random

    shuffled = list(files)
    random.Random(split_seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


@dataclasses.dataclass(frozen=True)
class _FrameLayout:
    """Per-excerpt 25 Hz frame bookkeeping for one drawn window.

    The drawn window has ``expected_frames``; ``target_frames`` are written. In
    time-varying mode the window is ``[head_frames | target | look_frames]``;
    otherwise a symmetric ``trim_frames`` crop applies downstream and
    ``target_frames == expected_frames``.
    """

    expected_frames: int
    target_frames: int
    head_frames: int
    look_frames: int
    tv_hop: float
    time_varying: bool


def _resolve_frame_layout(
    *,
    duration: float,
    trim_frames: int,
    musiccoca_time_varying: bool,
    musiccoca_hop_seconds: Optional[float],
    musiccoca_lookahead_seconds: float,
    head_trim_seconds: float,
    style_model,
    style_prompt: Optional[str],
) -> _FrameLayout:
    """Validate the duration / trim / time-varying options and derive the layout."""
    window_samples = round(duration * SAMPLE_RATE)
    if window_samples % SAMPLES_PER_FRAME:
        raise ValueError(
            f"duration={duration} is not a whole number of "
            f"{FRAME_RATE} Hz frames."
        )
    expected_frames = window_samples // SAMPLES_PER_FRAME
    if trim_frames < 0 or 2 * trim_frames >= expected_frames:
        raise ValueError(
            f"trim_frames={trim_frames} must satisfy 0 <= 2*trim_frames < "
            f"{expected_frames} (the number of {FRAME_RATE} Hz frames in a "
            f"{duration}s excerpt)."
        )

    # Time-varying MusicCoCa lays the drawn window out as
    #   [ head_trim (codec warm-up) | TARGET | look-ahead (MusicCoCa-only) ]
    # Codes + pianoroll are kept on TARGET; each target frame's LEADING 10 s
    # MusicCoCa window reaches into the look-ahead, so the look-ahead must be
    # >= MusicCoCa clip_length for the last target frame's window to be full.
    tv_hop = (
        musiccoca_hop_seconds if musiccoca_hop_seconds is not None
        else 1.0
    )
    head_frames = round(head_trim_seconds * FRAME_RATE)
    look_frames = round(musiccoca_lookahead_seconds * FRAME_RATE)
    if musiccoca_time_varying:
        if style_model is None or style_prompt is not None:
            raise ValueError(
                "musiccoca_time_varying needs a per-clip audio style_model "
                "(and no style_prompt)."
            )
        if trim_frames:
            raise ValueError(
                "musiccoca_time_varying uses head_trim_seconds / "
                "musiccoca_lookahead_seconds, not the symmetric trim_frames."
            )
        # TODO(alignment): look-ahead == MusicCoCa clip_length assumes the
        # LEADING-window convention chosen for this dataset. Revisit if the base
        # model used a causal-trailing window (then this would be a lead-IN).
        target_frames = expected_frames - head_frames - look_frames
        if target_frames <= 0:
            raise ValueError(
                f"time-varying export: duration={duration}s ({expected_frames} "
                f"frames) - head_trim ({head_frames}) - lookahead "
                f"({look_frames}) leaves no target frames; raise --duration."
            )
    else:
        target_frames = expected_frames
    return _FrameLayout(
        expected_frames=expected_frames,
        target_frames=target_frames,
        head_frames=head_frames,
        look_frames=look_frames,
        tv_hop=tv_hop,
        time_varying=musiccoca_time_varying,
    )


def _resolve_fixed_style(*, style_prompt, style_model, style_embedding, style_tokens):
    """Single-prompt mode: the one fixed ``(embedding, tokens)`` reused for every
    record, or ``(None, None)`` when ``style_prompt`` is unset.

    Callers SHOULD precompute ``style_embedding`` / ``style_tokens`` in a SEPARATE
    PROCESS and pass them in: MusicCoCa's text path loads a SentencePiece tokenizer
    (C++) whose runtime deadlocks/aborts ("mutex lock failed") once
    grain/audiotree/jax are live — and this export runs under grain. (The audio
    path never loads SentencePiece, so it coexists with grain fine.) The
    in-function ``embed_text`` fallback is kept only for the MockMusicCoCa unit
    test; production callers pass the arrays. See ``magenta_rt.sft.embed_prompt``.
    """
    if style_prompt is None:
        return None, None
    if style_embedding is not None and style_tokens is not None:
        return (
            np.asarray(style_embedding, dtype=np.float32).reshape(-1),
            np.asarray(style_tokens, dtype=np.int32).reshape(-1),
        )
    if style_model is None:
        raise ValueError(
            "style_prompt was set but no style_model and no precomputed "
            "style_embedding/style_tokens were provided."
        )
    fixed_embedding = np.asarray(
        style_model.embed_text(style_prompt), dtype=np.float32
    )  # [768]
    fixed_tokens = np.asarray(
        style_model.tokenize(fixed_embedding[None]), dtype=np.int32
    )[0]  # [12]
    return fixed_embedding, fixed_tokens


def _transcribe_channels(transcriber, audio: AudioTree, frames: int) -> dict:
    """Run ``transcriber`` on each clip's 16 kHz mono samples and stack the
    per-frame conditioning channels into ``{channel: [B, frames, ...]}``.

    TODO(perf): excerpts are drawn (random offsets, ``repeat``) from a handful of
    source songs, so the same audio is re-transcribed here many times and MT3
    dominates export time. Transcribe each whole song ONCE and index into the
    cached per-frame channels by the excerpt ``offset`` (slice
    ``[offset_frame : offset_frame + frames]``) instead of re-running MT3 per
    chunk — see "Known limits / follow-ups" in ``magenta_rt/sft/README.md``.
    """
    channel_rows: dict[str, list[np.ndarray]] = {}
    audio = audio.to_mono().resample(MT3_SAMPLE_RATE)
    for clip in audio:
        # MT3 transcribes ONE clip at a time. Drop the (B=1, C=1) axes to the
        # (T,) mono samples it expects; squeeze(axis=(0, 1)) raises if a real
        # batch or multi-channel waveform ever slips in.  (B, C, T) -> (T,)
        # TODO: take advantage of parallelism within the transcriber.
        samples = np.squeeze(
            np.asarray(clip.waveform, dtype=np.float32), axis=(0, 1)
        )
        for key, value in transcription_to_channels(
            transcriber(samples), frames, FRAME_RATE
        ).items():
            channel_rows.setdefault(key, []).append(value)
    return {key: np.stack(rows) for key, rows in channel_rows.items()}


class _BatchEncoder:
    """Encodes one grain batch (pure-numpy ``AudioTree``) into a record
    ``AudioTree``: SpectroStream codes + MusicCoCa style tokens (+ optional MT3
    piano-roll), in either broadcast (one style per clip) or time-varying mode.
    """

    def __init__(
        self,
        *,
        codec,
        style_model,
        transcriber,
        save_embedding,
        trim_frames,
        layout: _FrameLayout,
        style_prompt,
        fixed_embedding,
        fixed_tokens,
        musiccoca_window_subbatch,
        musiccoca_scan,
    ):
        self._codec = codec
        self._style_model = style_model
        self._transcriber = transcriber
        self._save_embedding = save_embedding
        self._trim_frames = trim_frames
        self._layout = layout
        self._style_prompt = style_prompt
        self._fixed_embedding = fixed_embedding
        self._fixed_tokens = fixed_tokens
        self._musiccoca_window_subbatch = musiccoca_window_subbatch
        self._musiccoca_scan = musiccoca_scan
        # Style is computed when a text prompt OR an audio style model is given;
        # otherwise the export is codec-only.
        self._compute_musiccoca = (
            style_prompt is not None or style_model is not None
        )

    def __call__(self, batch: AudioTree) -> AudioTree:
        # The codec accepts the AudioTree directly (it unwraps the waveform and
        # checks the sample rate), so there's no separate raw-array handle.
        codes = np.asarray(
            self._codec.waveform_to_codes(batch)
        ).astype(np.int32)
        frames = codes.shape[1]
        if self._layout.time_varying:
            return self._encode_time_varying(batch, codes, frames)
        return self._encode_broadcast(batch, codes, frames)

    def _encode_time_varying(self, batch: AudioTree, codes, frames: int) -> AudioTree:
        """``[head | TARGET | look-ahead]``: codes + pianoroll on TARGET, a
        per-frame ``mulan_tokens_25hz`` from leading 10 s windows reaching into
        the look-ahead region."""
        layout = self._layout
        tlo, thi = layout.head_frames, frames - layout.look_frames
        n = thi - tlo
        # Codes are the TARGET tokens (RVQ indices 0-1023, never -1) → unsigned
        # uint16 is lossless here and halves the largest stored leaf.
        codes_t = codes[:, tlo:thi, :].astype(np.uint16)

        extras = {}
        # One MusicCoCa embedding per target frame, from a LEADING [t, t+10s]
        # window. The windowing + shared single log-mel live in the style model
        # (the nnx port computes the mel once and windows it; musiccoca_scan
        # streams the windows through nnx.scan for very large spectrograms).
        win_emb = self._style_model.embed_audio_windows(
            batch,
            start_seconds=tlo / FRAME_RATE,
            hop_seconds=layout.tv_hop,
            num_windows=n,
            scan=self._musiccoca_scan,
            window_subbatch=self._musiccoca_window_subbatch,
        )  # [B, n, 768]
        # mulan is a SOURCE channel that prepare_source_tokens may later mask with
        # a -1 dropout sentinel, so store it SIGNED (int16); an unsigned dtype
        # would wrap that -1 to 65535.
        extras[_MUSICCOCA.key] = np.asarray(
            self._style_model.tokenize(win_emb), dtype=np.int16
        )  # [B, n, 12]
        # No EMBEDDING_KEY: there is no single per-clip embedding in this mode.
        # TODO: train-time StyleEmbeddingJitter re-tokenizes from
        # ``musiccoca_embedding``; with per-frame tokens it would need per-frame
        # embeddings [B, n, 768] on disk. Skipped (jitter defaults off); the
        # trainer consumes the stored per-frame tokens directly.

        # Pianoroll over the TARGET sub-window only (its frames must align to the
        # kept codes / style tokens, not the head-trim or look-ahead regions).
        if self._transcriber is not None:
            tl_s, th_s = tlo * SAMPLES_PER_FRAME, thi * SAMPLES_PER_FRAME
            clips = batch.replace(waveform=batch.waveform[:, :, tl_s:th_s])
            extras.update(_transcribe_channels(self._transcriber, clips, n))

        # Derive the record from ``batch`` rather than building a fresh tree:
        # the loader's provenance (``filepath`` / ``offset``) lives in
        # AudioTree's own container and rides along untouched, so it needs no
        # restating here. ``clear_lufs`` because the cached loudness describes
        # the waveform being dropped, and a stale value is worse than none.
        return (
            batch.replace(waveform=None, codes=codes_t)
            .replace_extras(**extras)
            .clear_lufs()
        )

    def _encode_broadcast(self, batch: AudioTree, codes, frames: int) -> AudioTree:
        """One MusicCoCa embedding per clip, broadcast across the excerpt's
        frames, with an optional symmetric ``trim_frames`` crop."""
        bsz = batch.waveform.shape[0]
        extras = {}
        if self._compute_musiccoca:
            if self._style_prompt is not None:
                # Same fixed text embedding/tokens for every record in the batch.
                embeddings = np.broadcast_to(
                    self._fixed_embedding, (bsz, 768)
                ).copy()
                style_tokens = np.broadcast_to(
                    self._fixed_tokens, (bsz, self._fixed_tokens.shape[0])
                ).copy()
            else:
                # One batched AudioTree ([B, 2, W] @ 48 kHz); embed_audio mixes
                # to mono + resamples to 16 kHz over the whole batch internally.
                embeddings = np.asarray(
                    self._style_model.embed_audio(batch), dtype=np.float32,
                )  # [B, 768]
                style_tokens = np.asarray(
                    self._style_model.tokenize(embeddings), dtype=np.int32
                )  # [B, 12]
            extras[_MUSICCOCA.key] = np.repeat(
                style_tokens[:, None, :], frames, axis=1
            )
            if self._save_embedding:
                extras[EMBEDDING_KEY] = embeddings


        if self._transcriber is not None:
            extras.update(
                _transcribe_channels(self._transcriber, batch, frames)
            )

        # Discard the outer ``trim_frames`` frames from the codes and every
        # per-frame conditioning channel (those whose axis-1 spans the excerpt's
        # frames), keeping the central window with full codec context.
        if self._trim_frames:
            lo, hi = self._trim_frames, frames - self._trim_frames
            codes = codes[:, lo:hi, :]
            extras = {
                key: (value[:, lo:hi]
                      if getattr(value, "ndim", 0) >= 3
                      and value.shape[1] == frames else value)
                for key, value in extras.items()
            }

        # Compact storage dtypes — lossless for these value ranges; the training
        # prepare_target_tokens / prepare_source_tokens paths keep these compact
        # dtypes (int16) into the model's embedding lookup, so storage is cheap.
        # ``codes`` are the TARGET (RVQ indices 0-1023, never -1) → unsigned
        # uint16 is safe and halves the largest non-pianoroll leaf. ``mulan`` is
        # a SOURCE channel that prepare_source_tokens may fill with a -1
        # dropout/mask sentinel, so it stays SIGNED int16 (an unsigned -1 would
        # wrap to 65535). The note pianoroll already arrives as int8 from
        # ``notes_to_pianoroll``. ``musiccoca_embedding`` stays fp32 (precision
        # matters for re-tokenize + StyleEmbeddingJitter).
        codes = codes.astype(np.uint16)
        if _MUSICCOCA.key in extras:
            extras[_MUSICCOCA.key] = np.asarray(
                extras[_MUSICCOCA.key], dtype=np.int16
            )
        # Derive the record from ``batch`` rather than building a fresh tree:
        # the loader's provenance (``filepath`` / ``offset``) lives in
        # AudioTree's own container and rides along untouched, so it needs no
        # restating here. ``clear_lufs`` because the cached loudness describes
        # the waveform being dropped, and a stale value is worse than none.
        return (
            batch.replace(waveform=None, codes=codes)
            .replace_extras(**extras)
            .clear_lufs()
        )


def _build_excerpt_dataset(
    grain,
    create_audio_dataset,
    stereo,
    *,
    sources,
    files,
    duration,
    seed,
    excerpt,
    extensions,
    normalize,
    num_samples,
    batch_size,
    worker_count,
    worker_buffer_size,
    profile,
):
    """The grain salient-excerpt pipeline → a batched IterDataset of pure-numpy
    ``AudioTree`` batches: decode → ``ExactLength`` → stereo → near-silence
    filter → optional normalize → cap at ``num_samples`` → batch → optional
    ``mp_prefetch``.

    The pipeline stays pure numpy (no GPU code ever runs in a worker); the model
    encode happens in the *consuming* loop. ``mp_prefetch`` overlaps the next
    batch's worker decode with the current batch's main-process encode.
    """
    window_samples = round(duration * SAMPLE_RATE)
    if files is not None:
        # Build the same salient-excerpt pipeline create_audio_dataset builds,
        # but from an explicit file list (the file-level split path).
        load_fn = functools.partial(
            _load_excerpt,
            sample_rate=SAMPLE_RATE,
            duration=duration,
            mono=False,
            pad_mode="wrap",
            excerpt=excerpt,
        )
        ds = (
            grain.MapDataset.source(files)
            .seed(seed)
            .shuffle()
            .repeat()
            .seed(seed)
            .random_map(load_fn)
        )
    else:
        ds = create_audio_dataset(
            sources=[str(s) for s in sources],
            shuffle=True,
            num_epochs=None,  # unbounded stream (was `repeat=True` pre-1.0)
            shuffle_seed=seed,
            sample_rate=SAMPLE_RATE,
            mono=False,
            duration=duration,
            pad_mode="wrap",
            extensions=list(extensions) if extensions is not None else None,
            excerpt=excerpt,
        )
    ds = ds.map(ExactLength(window_samples))
    ds = ds.map(stereo())
    # ExactLength/stereo rebuild the waveform, dropping the loudness the excerpt
    # search computed (and mono inputs always pass through stereo()), so recompute
    # it on the final waveform before the near-silence filter — otherwise
    # ``lufs`` is None and the filter raises on ``lufs[0]``.
    ds = ds.map(AudioTree.replace_lufs)
    ds = ds.filter(lambda audio: audio.lufs[0] > -60.)
    # Optional per-excerpt level normalization, applied AFTER the near-silence
    # filter (so quiet noise isn't boosted to full level) and BEFORE encoding.
    # Accepts a grain ``RandomMap`` (e.g. ``audiotree.transforms.volume_norm``
    # for a LUFS target) or a plain ``AudioTree -> AudioTree`` callable (e.g.
    # peak-normalize or a fixed gain).
    if normalize is not None:
        if isinstance(normalize, grain.transforms.RandomMap):
            ds = ds.random_map(normalize)
        else:
            ds = ds.map(normalize)
    ds = ds.to_iter_dataset(
        grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    )
    ds = ds.batch(batch_size, drop_remainder=False, batch_fn=AudioTree.batch)
    if worker_count > 0:
        ensure_absl_flags_parsed()
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=worker_count,
                per_worker_buffer_size=worker_buffer_size,
            )
        )
    # Bound the stream AFTER the workers are joined, not before. `mp_prefetch`
    # replays the whole upstream pipeline in each worker, so a limit placed
    # above it is applied once *per worker* and the export writes
    # ``worker_count * num_samples`` records — silently ignoring --num-samples
    # (the source is an unbounded ``num_epochs=None`` stream, so nothing else
    # stops it). Limiting here counts batches off the joined stream, so the
    # total is ``num_samples`` rounded up to a whole batch regardless of
    # ``worker_count``.
    ds = grain.experimental.LimitIterDataset(
        ds, -(-num_samples // batch_size)  # ceil division: whole batches
    )
    if profile:
        ds = grain.experimental.WithOptionsIterDataset(
            ds,
            grain.experimental.DatasetOptions(
                execution_tracking_mode=(
                    grain.experimental.ExecutionTrackingMode.STAGE_TIMING
                )
            ),
        )
    return ds


def export_tree_dataset(
    sources: str | pathlib.Path | Sequence[str | pathlib.Path] | None,
    out_dir: str | pathlib.Path,
    *,
    codec,
    style_model=None,
    num_samples: int,
    files: Optional[Sequence[str]] = None,
    transcriber=None,
    duration: float = 10.0,
    trim_frames: int = 0,
    batch_size: int = 4,
    seed: int = 0,
    excerpt: Optional[ExcerptConfig] = None,
    normalize=None,
    extensions: Optional[Sequence[str]] = None,
    worker_count: int = 0,
    worker_buffer_size: int = 1,
    save_embedding: bool = True,
    musiccoca_time_varying: bool = False,
    musiccoca_hop_seconds: float = 1.0,
    musiccoca_lookahead_seconds: float = 10.0,
    head_trim_seconds: float = 0.0,
    musiccoca_window_subbatch: int = 128,
    musiccoca_scan: bool = False,
    style_prompt: Optional[str] = None,
    style_embedding=None,
    style_tokens=None,
    dataset_metadata: Optional[dict] = None,
    pbar=None,
    profile: bool = False,
) -> str:
    """Encodes salient audio excerpts into a ``TreeWriter`` SFT dataset.

    Args:
      sources: Directory (or directories) of audio files, searched
        recursively (anything ``soundfile``/``librosa`` read). Pass ``None``
        when supplying ``files`` directly.
      out_dir: Output directory for ``manifest.json`` + memmaps.
      files: Explicit list of audio file paths to draw excerpts from
        (mutually exclusive with ``sources``). Use with
        :func:`discover_audio_files` + :func:`split_audio_files` for a
        leak-free file-level train/val split. The list is recorded in the
        manifest (``source_files``) for auditing.
      codec: Duck-typed SpectroStream — ``waveform_to_codes([B, C, T] @
        48 kHz) -> [B, T, D]`` (channel-major audio, frame-major codes).
      style_model: A ``MusicCoCaBase`` (any backend); used per excerpt via
        ``embed_audio`` (handles mono mixdown + 16 kHz resample) and
        ``tokenize``. Pass ``None`` (and no ``style_prompt``) for a
        **codec-only** export: no MusicCoCa runs, no ``mulan_tokens_25hz`` /
        ``musiccoca_embedding`` is written, and the training pipeline falls
        back to the learned unconditional (dropout) style token for these
        records.
      style_prompt: If set, condition the whole dataset on a single fixed *text*
        prompt (e.g. ``"electronic dance music"``) instead of each excerpt's own
        audio-derived MusicCoCa embedding. The prompt is embedded + tokenized
        once and written identically to every record. This is the single-style
        recipe: it removes the per-clip style fingerprint (the 12 RVQ tokens
        nearly uniquely identify an excerpt — an easy SFT shortcut to memorize),
        so the LoRA learns the *style* anchored to one prompt, and inference
        conditions on that same text. Also skips per-excerpt audio embedding
        (faster export).
      num_samples: Number of excerpts to draw and encode. Files are
        shuffled and repeated, with a fresh excerpt position per visit, so
        this can exceed the number of source files.
      transcriber: Optional ``(mono_samples_16k) -> transcription`` callable
        (e.g. :func:`mt3_transcriber`); adds the two piano-roll conditioning
        channels per excerpt. None skips transcription.
      duration: Excerpt length in seconds (a whole number of 25 Hz frames).
        The default (10 s = 250 frames) matches one MusicCoCa clip, so each
        record gets one style embedding.
      trim_frames: Frames to discard from *each* side of every encoded excerpt
        (codes and per-frame conditioning channels) after encoding. The codec
        is run on the full ``duration`` window, then the outer ``trim_frames``
        frames are dropped, so the kept central frames carry full codec
        receptive-field context on both sides (no chunk-boundary edge effects).
        E.g. ``duration=4.0, trim_frames=25`` encodes 4 s (100 frames @ 25 Hz)
        and keeps the central 2 s (50 frames). 0 = keep the whole excerpt.
      batch_size: Excerpts encoded per codec / style-model call.
      seed: Seed for the file shuffle and excerpt positions.
      normalize: Optional per-excerpt level normalization applied after the
        near-silence filter and before encoding. Either a grain ``RandomMap``
        (e.g. ``audiotree.transforms.volume_norm(min_db=L, max_db=L)`` for a
        fixed ``L`` LUFS target) or a plain ``AudioTree -> AudioTree`` callable
        (e.g. a peak-normalize or fixed-gain map). ``None`` leaves levels as-is.
      excerpt: ``audiotree.ExcerptConfig`` controlling the
        loudness-guided excerpt search. Defaults to a ``"loudest"`` 8-try search
        with a -60 LUFS cutoff — permissive enough to reject only near-silence,
        so excerpts cover the whole track (intros, breakdowns, quiet passages)
        rather than over-weighting the loudest sections (a higher cutoff like
        -40 biases toward drops/choruses and freezes positional variety).
      extensions: Audio file extensions to search for (audiotree's default:
        ``[".wav", ".flac"]``).
      worker_count: grain worker processes that read audio files and run the
        CPU-based preprocessing (excerpting / resampling) in parallel
        (0 = in-process). Model encodes stay in the main process.
      worker_buffer_size: Batches buffered per worker.
      save_embedding: Also store the raw 768-dim embedding per record. Ignored
        in time-varying MusicCoCa mode (there is no single per-clip embedding;
        see ``musiccoca_time_varying``).
      musiccoca_time_varying: Produce a TIME-VARYING ``mulan_tokens_25hz`` — a
        different MusicCoCa embedding per target frame, from a LEADING
        ``clip_length`` (10 s) window ``[t, t+10s]`` (the style of the audio the
        model is about to generate) — instead of one per-clip embedding broadcast
        across frames. This matches the base model's training. Requires drawing
        ``clip_length`` seconds of look-ahead after the target window: set
        ``duration = head_trim + target + musiccoca_lookahead_seconds`` (e.g.
        ``duration=31, head_trim_seconds=1, musiccoca_lookahead_seconds=10`` →
        a 20 s target). Codes + pianoroll are kept on the target window only.
      musiccoca_hop_seconds: MusicCoCa recompute hop for time-varying mode
        (default: 1.0). A coarser hop recomputes less often and upsamples
        (piecewise-constant) to 25 Hz — far cheaper and near-identical, since a
        10 s window barely changes in 40 ms.
      musiccoca_lookahead_seconds: Seconds of look-ahead audio after the target
        window, used only to fill the leading MusicCoCa windows of the last
        target frames (time-varying mode). Should equal MusicCoCa's clip_length
        (10 s). The codes for this trailing region are discarded.
      head_trim_seconds: Seconds of codes/conditioning discarded from the START
        of the drawn window (codec warm-up: the first frames at a hard audio
        boundary lack left context). Asymmetric counterpart to ``trim_frames``.
      musiccoca_window_subbatch: Max leading windows embedded per MusicCoCa call
        (bounds GPU memory in time-varying mode).
      musiccoca_scan: In time-varying mode, stream the leading windows through
        ``nnx.scan`` (one window resident at a time — bounded memory for a very
        large spectrogram, but sequential) instead of one batched encode. Only
        the nnx style model honors this; other backends ignore it.
      dataset_metadata: Extra user metadata for the manifest.
      pbar: Optional tqdm-style progress bar (updated per excerpt written).
      profile: Collect grain per-stage execution statistics
        (``ExecutionTrackingMode.STAGE_TIMING``) and print the summary when
        the export finishes — times the grain (decode/crop/batch) stages.
        The model encode runs in the consuming loop, not the grain pipeline,
        so it isn't in this summary (time it via the tqdm rate instead).

    Returns:
      ``out_dir`` as a string.
    """
    layout = _resolve_frame_layout(
        duration=duration,
        trim_frames=trim_frames,
        musiccoca_time_varying=musiccoca_time_varying,
        musiccoca_hop_seconds=musiccoca_hop_seconds,
        musiccoca_lookahead_seconds=musiccoca_lookahead_seconds,
        head_trim_seconds=head_trim_seconds,
        style_model=style_model,
        style_prompt=style_prompt,
    )

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if (sources is None) == (files is None):
        raise ValueError("pass exactly one of `sources` or `files`.")
    if excerpt is None:
        excerpt = ExcerptConfig(
            strategy="loudest", num_tries=8, lufs_cutoff=-60.0
        )
    if sources is not None and isinstance(sources, (str, pathlib.Path)):
        sources = [sources]
    if files is not None:
        files = [str(f) for f in files]
        if not files:
            raise ValueError("`files` is empty.")

    # Manifest is provenance only — the trainer reconstructs records from the
    # per-leaf memmaps + the audiotree-written ``structure`` key, never from this
    # user-metadata dict. So just pass the caller's ``dataset_metadata`` through.
    manifest = dict(dataset_metadata or {})
    fixed_embedding, fixed_tokens = _resolve_fixed_style(
        style_prompt=style_prompt,
        style_model=style_model,
        style_embedding=style_embedding,
        style_tokens=style_tokens,
    )
    encode = _BatchEncoder(
        codec=codec,
        style_model=style_model,
        transcriber=transcriber,
        save_embedding=save_embedding,
        trim_frames=trim_frames,
        layout=layout,
        style_prompt=style_prompt,
        fixed_embedding=fixed_embedding,
        fixed_tokens=fixed_tokens,
        musiccoca_window_subbatch=musiccoca_window_subbatch,
        musiccoca_scan=musiccoca_scan,
    )

    ds = _build_excerpt_dataset(
        grain,
        create_audio_dataset,
        stereo,
        sources=sources,
        files=files,
        duration=duration,
        seed=seed,
        excerpt=excerpt,
        extensions=extensions,
        normalize=normalize,
        num_samples=num_samples,
        batch_size=batch_size,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        profile=profile,
    )

    iterator: grain.IterDataset[AudioTree] = iter(ds)
    with TreeWriter(
        out_dir,
        expected_samples=num_samples,
        metadata=manifest,
        pbar=pbar,
    ) as writer:
        # The stream is bounded in whole batches (see `_build_excerpt_dataset`),
        # so the last one can overshoot `num_samples`; trim it here — before the
        # encode, so no compute is spent on records that are not written — and
        # the dataset holds exactly `num_samples` records for any `worker_count`
        # / `batch_size` combination.
        written = 0
        for batch in iterator:        # batch: pure-numpy AudioTree from grain
            remaining = num_samples - written
            if remaining <= 0:
                break
            if batch.batch_size > remaining:
                batch = batch[:remaining]
            writer.write(encode(batch))  # MLX/GPU/TPU encode, main process
            written += batch.batch_size
    if profile:
        try:  # re-exported from grain.experimental on newer grain
            from grain.experimental import get_execution_summary
        except ImportError:
            from grain._src.python.dataset.dataset import get_execution_summary
        from grain._src.python.dataset.stats import pretty_format_summary

        print(pretty_format_summary(get_execution_summary(iterator)))
    return str(out_dir)
