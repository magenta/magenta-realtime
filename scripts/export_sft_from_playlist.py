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

"""Export an SFT TreeWriter dataset from an explicit file list (fp32, 3-model).

This is a thin driver around :func:`magenta_rt.sft.export.export_tree_dataset`
that mirrors ``mrt sft export`` but draws its source audio from a *file list*
instead of ``--sources`` directories. By default it runs the full **fp32**
preprocessing pipeline — SpectroStream codec + MusicCoCa style + MT3
piano-roll — so each record stores SpectroStream RVQ ``codes``, per-excerpt
MusicCoCa style tokens + the 768-dim embedding, the two MT3 piano-roll
conditioning channels, and ``filepath``/``offset`` provenance. ``--no-musiccoca``
and ``--no-transcribe`` drop those streams (the trainer then falls back to the
learned unconditional dropout token for the missing ones).

**Everything runs at fp32**: the codec's default bf16 compute flips ~11.5% of
RVQ codes even in the coarse training codebooks vs the fp32 ground truth
(verified nnx-vs-jax ``waveform_to_codes`` parity), corrupting target codes;
MusicCoCa and MT3 are fp32 as well.

The file list comes from one of:

* ``--itunes-xml`` — an iTunes/Music *Library.xml* plist; every track's
  ``Location`` is parsed and converted to a WSL path.
* ``--filelist`` — a plain text file, one path per line (``#`` comments and
  blank lines ignored). Windows paths (``D:\\...`` / ``D:/...``) and
  ``file://`` URLs are converted to ``/mnt/<drive>/...`` automatically.

Windows → WSL path mapping: a drive ``D:`` becomes ``/mnt/d`` (the standard
WSL2 DrvFs mount), so ``D:/iTunes/Music/x.mp3`` →
``/mnt/d/iTunes/Music/x.mp3``. Files that don't exist on disk are dropped
(with a count), so a stale playlist still exports cleanly.

Example (smoke run, 22 s excerpts → central 20 s of codes, NVIDIA/NNX):

    python scripts/export_sft_from_playlist.py \\
        --itunes-xml iTunes-Electronic-Playlist.xml \\
        --out datasets/electronic_sft_smoke \\
        --num-samples 8 --workers 4
"""

from __future__ import annotations

import os
import pathlib
import time
from urllib.parse import unquote, urlparse

import click


def _location_to_wsl(location: str) -> str:
    """Convert an iTunes ``file://`` Location URL to a WSL path.

    ``file://localhost/D:/iTunes/Music/04%20Black%20Projects.mp3`` →
    ``/mnt/d/iTunes/Music/04 Black Projects.mp3``.
    """
    path = unquote(urlparse(location).path).lstrip("/")  # 'D:/iTunes/...'
    if len(path) >= 2 and path[1] == ":":
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


def _to_wsl(raw: str) -> str:
    """Normalize one file-list entry (URL / Windows / POSIX) to a WSL path."""
    raw = raw.strip()
    if raw.startswith("file://"):
        return _location_to_wsl(raw)
    raw = raw.replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":":  # 'D:/...' drive-letter path
        return f"/mnt/{raw[0].lower()}{raw[2:]}"
    return raw


def _files_from_itunes_xml(xml_path: str) -> list[str]:
    """Parse an iTunes Library plist into a list of WSL track paths."""
    import plistlib

    with open(xml_path, "rb") as f:
        plist = plistlib.load(f)
    tracks = plist["Tracks"]  # raises if this isn't an iTunes library
    return [
        _location_to_wsl(track["Location"])
        for track in tracks.values()
        if "Location" in track
    ]


def _files_from_filelist(list_path: str) -> list[str]:
    """Read a plain-text file list (one path per line; ``#`` comments)."""
    out = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(_to_wsl(line))
    return out


def _build_codec_nnx(model_name: str, checkpoint: str | None):
    """Build an **fp32** NNX SpectroStream codec (no depthformer kept).

    Loads the full ``mrt2`` sampler to get a correctly-bridged SpectroStream
    encoder (the standalone codec has no NNX builder), extracts the codec, and
    drops the depthformer so only the codec stays resident.

    The codec compute dtype is forced to **fp32** (params are already fp32).
    This matters for correctness: the default bf16 compute path flips ~11.5%
    of RVQ codes *even in the coarse training codebooks* relative to the fp32
    ground truth (verified by nnx-vs-jax ``waveform_to_codes`` parity), so
    target codes encoded in bf16 are degraded. At fp32 the coarse codebooks
    are bit-identical to jax and the first ``rvq_truncation_level`` (training)
    codes are exact.
    """
    import gc

    import jax.numpy as jnp
    import numpy as np
    from flax import nnx as flax_nnx

    from magenta_rt import paths
    from magenta_rt.nnx.model import MagentaRT2Sampler

    ckpt = pathlib.Path(checkpoint or f"{model_name}.safetensors")
    if not ckpt.is_absolute():
        ckpt = paths.checkpoints_dir() / ckpt

    sampler = MagentaRT2Sampler.from_preset(
        model_name, int16_outputs=False, rngs=flax_nnx.Rngs(0)
    )
    sampler.load_checkpoint(ckpt)
    spectrostream = sampler.spectrostream
    # fp32 compute (params already fp32); non-streaming full-sequence encode.
    spectrostream.set_attributes(dtype=jnp.float32, raise_if_not_found=False)
    spectrostream.set_attributes(streaming=False, raise_if_not_found=False)

    class Codec:
        """[B, C, T] numpy @ 48 kHz -> [B, T, D] numpy RVQ codes (on the GPU)."""

        def waveform_to_codes(self, audio):
            return np.asarray(spectrostream.waveform_to_codes(audio))

    del sampler  # depthformer now unreferenced — free it, keep only the codec
    gc.collect()
    return Codec()


def _build_musiccoca_nnx():
    """Build the fp32 NNX MusicCoCa style embedder.

    Verified to ~1e-5 against the original TFLite models (exact RVQ tokens).
    The export uses ``embed_audio`` only (no SentencePiece text path), so it
    coexists with the grain worker pool.
    """
    from magenta_rt.nnx.musiccoca import MusicCoCa

    return MusicCoCa()


@click.command()
@click.option("--itunes-xml", default=None,
              help="iTunes/Music Library.xml to pull track Locations from.")
@click.option("--filelist", default=None,
              help="Plain-text file list (one audio path per line).")
@click.option("--out", required=True,
              help="Output dataset directory (TreeWriter format).")
@click.option("--num-samples", type=int, required=True,
              help="Number of salient excerpts to draw and encode.")
@click.option("--duration", type=float, default=22.0,
              help="Excerpt length in seconds the codec encodes (default 22.0).")
@click.option("--trim-seconds", type=float, default=1.0,
              help="Seconds of codes to discard from EACH side after encoding "
                   "(default 1.0). With the 22 s default this keeps the central "
                   "20 s = 500 frames, with full codec context on both sides.")
@click.option("--musiccoca/--no-musiccoca", "use_musiccoca", default=True,
              help="Store per-excerpt MusicCoCa style tokens + the 768-dim "
                   "embedding (fp32 nnx port). Default on.")
@click.option("--transcribe/--no-transcribe", "use_transcribe", default=True,
              help="Store MT3 piano-roll conditioning channels (fp32 nnx "
                   "port). Default on (~0.5 s/excerpt on this GPU).")
@click.option("--mt3-batch-size", type=int, default=8,
              help="MT3 segments decoded in parallel per excerpt.")
@click.option("--batch-size", type=int, default=4,
              help="Excerpts per SpectroStream encode call.")
@click.option("--seed", type=int, default=0)
@click.option("--val-fraction", type=float, default=0.0,
              help="Hold out this fraction of FILES (whole files) into a "
                   "leak-free validation split written to <out>_val "
                   "(0 = no val set).")
@click.option("--val-num-samples", type=int, default=0,
              help="Number of excerpts for the val split (required when "
                   "--val-fraction > 0).")
@click.option("--split-seed", type=int, default=0,
              help="Seed for the deterministic file-level train/val split.")
@click.option("--model", default="mrt2_small")
@click.option("--checkpoint", default=None,
              help="Checkpoint filename or path (default <model>.safetensors).")
@click.option("--workers", type=int, default=4,
              help="grain worker processes that read + decode audio in "
                   "parallel (0 = in-process).")
@click.option("--worker-buffer-size", type=int, default=1)
@click.option("--loudness-cutoff", type=float, default=-60.0,
              help="Saliency LUFS cutoff (-60 rejects only near-silence).")
@click.option("--saliency-tries", type=int, default=8)
@click.option("--volume-norm-lufs", type=float, default=None,
              help="Normalize each excerpt to this LUFS before encoding "
                   "(audiotree volume_norm). Mutually exclusive with "
                   "--peak-normalize/--gain.")
@click.option("--peak-normalize", is_flag=True, default=False,
              help="Peak-normalize each excerpt to 1.0 before encoding "
                   "(audiotree peak_norm).")
@click.option("--gain", type=float, default=None,
              help="Apply this fixed linear gain to each excerpt before "
                   "encoding (e.g. 0.7), via audiotree volume_change "
                   "(converted to dB). Mutually exclusive with the others.")
@click.option("--write-filelist", default=None,
              help="Also write the resolved (existing) file list here.")
@click.option("--profile", is_flag=True,
              help="Print grain per-stage execution statistics at the end.")
def main(itunes_xml, filelist, out, num_samples, duration, trim_seconds,
         use_musiccoca, use_transcribe, mt3_batch_size,
         batch_size, seed, val_fraction, val_num_samples, split_seed, model,
         checkpoint, workers, worker_buffer_size, loudness_cutoff,
         saliency_tries, volume_norm_lufs, peak_normalize, gain,
         write_filelist, profile):
    """fp32 SFT export from a file list (SpectroStream + MusicCoCa + MT3)."""
    if (itunes_xml is None) == (filelist is None):
        raise click.UsageError(
            "pass exactly one of --itunes-xml or --filelist.")
    if val_fraction > 0 and val_num_samples <= 0:
        raise click.UsageError(
            "--val-num-samples must be > 0 when --val-fraction > 0.")
    if sum(x for x in (volume_norm_lufs is not None, peak_normalize,
                       gain is not None)) > 1:
        raise click.UsageError(
            "pass at most one of --volume-norm-lufs / --peak-normalize / --gain.")

    raw_files = (_files_from_itunes_xml(itunes_xml) if itunes_xml
                 else _files_from_filelist(filelist))
    files = [f for f in raw_files if os.path.isfile(f)]
    missing = len(raw_files) - len(files)
    print(f"[export] {len(raw_files)} listed, {len(files)} present on disk"
          f" ({missing} missing/skipped).")
    if not files:
        raise click.UsageError("no listed files exist on disk.")
    if write_filelist:
        with open(write_filelist, "w") as f:
            f.write("\n".join(files) + "\n")
        print(f"[export] wrote resolved file list -> {write_filelist}")

    from audiotree import ExcerptConfig
    from tqdm import tqdm

    from magenta_rt.sft.export import (
        FRAME_RATE,
        export_tree_dataset,
        mt3_transcriber,
        split_audio_files,
    )

    print(f"[export] building fp32 NNX codec ({model}) ...")
    codec = _build_codec_nnx(model, checkpoint)
    style_model = None
    if use_musiccoca:
        print("[export] building fp32 NNX MusicCoCa (per-clip audio embeddings) ...")
        style_model = _build_musiccoca_nnx()
    transcriber = None
    if use_transcribe:
        print("[export] building fp32 NNX MT3 transcriber (piano-roll) ...")
        transcriber = mt3_transcriber(backend="nnx", batch_size=mt3_batch_size)
    print(f"[export] pipeline: codec + "
          f"{'MusicCoCa' if use_musiccoca else 'no-MusicCoCa'} + "
          f"{'MT3' if use_transcribe else 'no-MT3'} (all fp32)")

    saliency = ExcerptConfig(
        strategy="loudest", num_tries=saliency_tries, lufs_cutoff=loudness_cutoff)
    trim_frames = round(trim_seconds * FRAME_RATE)
    kept_frames = round(duration * FRAME_RATE) - 2 * trim_frames

    # Optional per-excerpt level normalization (applied before encoding).
    normalize = None
    normalize_label = "none"
    if volume_norm_lufs is not None:
        from audiotree.transforms import volume_norm
        normalize = volume_norm(min_db=volume_norm_lufs, max_db=volume_norm_lufs)
        normalize_label = f"volume_norm_lufs={volume_norm_lufs}"
    elif peak_normalize:
        from audiotree.transforms import peak_norm as _peak_norm
        normalize = _peak_norm()
        normalize_label = "peak_normalize=1.0"
    elif gain is not None:
        import math

        from audiotree.transforms import volume_change
        if gain <= 0:
            raise click.UsageError("--gain must be a positive linear factor.")
        gain_db = 20.0 * math.log10(gain)
        normalize = volume_change(min_db=gain_db, max_db=gain_db)
        normalize_label = f"gain={gain} ({gain_db:+.2f} dB volume_change)"
    print(f"[export] level normalization: {normalize_label}")

    def run(out_dir, n_samples, split_files, split=None):
        tag = f" [{split}]" if split else ""
        print(f"[export]{tag} {n_samples} x {duration:.0f}s salient excerpts "
              f"from {len(split_files)} files -> {out_dir}")
        print(f"[export]{tag} trim {trim_frames} frames ({trim_seconds:.1f}s) "
              f"per side -> keep {kept_frames} frames "
              f"({kept_frames / FRAME_RATE:.1f}s) of codes per record.")
        start = time.time()
        with tqdm(total=n_samples, desc=f"Encoding{tag}", unit="ex") as pbar:
            export_tree_dataset(
                None, out_dir,
                codec=codec, style_model=style_model, transcriber=transcriber,
                files=split_files, num_samples=n_samples, duration=duration,
                trim_frames=trim_frames,
                batch_size=batch_size, seed=seed, excerpt=saliency,
                normalize=normalize,
                worker_count=workers, worker_buffer_size=worker_buffer_size,
                save_embedding=use_musiccoca,
                dataset_metadata={"source": "filelist", "model": model,
                                  "backend": "nnx", "normalize": normalize_label,
                                  "dtype": "fp32",
                                  "has_mt3": use_transcribe,
                                  **({"split": split} if split else {})},
                pbar=pbar, profile=profile,
            )
        elapsed = time.time() - start
        print(f"[export]{tag} wrote {n_samples} examples to {out_dir} in "
              f"{elapsed:.1f} s ({n_samples / elapsed:.2f} ex/s, "
              f"{elapsed / n_samples * 1000:.0f} ms/ex).")

    if val_fraction > 0:
        train_files, val_files = split_audio_files(
            files, val_fraction=val_fraction, split_seed=split_seed)
        print(f"[export] file-level split (seed {split_seed}): "
              f"{len(train_files)} train / {len(val_files)} val files "
              f"of {len(files)} total")
        if not val_files:
            raise click.UsageError("--val-fraction too small: 0 files held out.")
        run(out, num_samples, train_files, split="train")
        run(f"{out}_val", val_num_samples, val_files, split="val")
    else:
        run(out, num_samples, files)


if __name__ == "__main__":
    main()
