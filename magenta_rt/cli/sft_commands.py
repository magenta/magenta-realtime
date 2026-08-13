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

"""CLI commands for supervised fine-tuning (SFT): ``mrt sft``.

Three subcommands cover the offline SFT data path:

* ``mrt sft export`` — precompute an SFT TreeWriter dataset from directories of
  audio (SpectroStream codes + MusicCoCa style tokens + optional MT3 piano-roll).
* ``mrt sft generate`` — generate SFT evaluation audio from a dataset's
  conditioning, optionally through a LoRA/DoRA adapter, with a provenance sidecar.
* ``mrt sft inspect`` — print stats for an SFT TreeWriter dataset and optionally
  visualize a record's pianoroll.

Heavy imports (MLX / JAX / grain / audiotree / the model stack) are done lazily
inside each command body so ``mrt sft --help`` stays fast.
"""

import dataclasses
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from audiotree import ExcerptConfig
from audiotree.sources import TreeDataSource
import click
import numpy as np
import soundfile
from tqdm import tqdm

from magenta_rt.cli import main
from magenta_rt import paths


@main.group()
def sft():
    """Supervised fine-tuning: dataset export, eval-sample generation, inspection."""


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _build_codec_and_style(backend: str, checkpoint: str | None):
    """Returns (codec, style_model) with pretrained weights for ``backend``."""

    # todo: shouldn't need to use the full MagentaRT2Sampler just to get the codec.
    model_name = "mrt2_small"
    ckpt = Path(checkpoint or f"{model_name}.safetensors")
    if not ckpt.is_absolute():
        ckpt = paths.checkpoints_dir() / ckpt

    if backend == "mlx_pure":
        import mlx.core as mx

        from magenta_rt.mlx_pure import configs as mlx_pure_configs
        from magenta_rt.mlx_pure.musiccoca import MusicCoCa
        from magenta_rt.mlx_pure.spectrostream.load_weights import (
            load_spectrostream_from_linen,
        )

        # The encode's conv/STFT temporaries are large; without a cap MLX's
        # buffer cache balloons and pushes a 16 GB machine into memory
        # compression, slowing the whole pipeline several-fold.
        mx.set_cache_limit(1 << 30)

        # Build a STANDALONE pure SpectroStream codec and load it directly from
        # the standalone Linen safetensors (resources/spectrostream/) — no
        # depthformer, no MagentaRT2Sampler, no sequence_layers bridge. The
        # codec is self-contained in those two files (encoder + decoder +
        # quantizer), so the main checkpoint is never read here.
        spec = mlx_pure_configs.get_model_class(model_name)()
        spectrostream = spec.build_spectrostream()
        load_spectrostream_from_linen(spectrostream)
        sampler = None

        class Codec:
            """[B, C, T] @ 48 kHz (array or AudioTree) -> [B, T, D] numpy codes."""

            def waveform_to_codes(self, audio):
                # The export passes an AudioTree; the pure codec unwraps it (and
                # converts to mx internally). Forward as-is.
                return np.asarray(spectrostream.waveform_to_codes(audio))

    elif backend == "nnx":
        from flax import nnx as flax_nnx

        from magenta_rt.nnx.model import MagentaRT2Sampler
        from magenta_rt.nnx.musiccoca import MusicCoCa

        from magenta_rt.nnx.checkpoint_utils import load_into_abstract

        sampler = load_into_abstract(
            lambda: MagentaRT2Sampler.from_preset(
                model_name, int16_outputs=False, rngs=flax_nnx.Rngs(0)
            ),
            lambda m: m.load_checkpoint(ckpt),
        )
        spectrostream = sampler.spectrostream

        class Codec:
            def waveform_to_codes(self, audio):
                return np.asarray(spectrostream.waveform_to_codes(audio))

    else:
        raise ValueError(f"unknown backend {backend!r}")

    # The export only needs the codec; let the depthformer be collected.
    del sampler
    gc.collect()
    return Codec(), MusicCoCa()


@sft.command("export")
@click.option("--sources", multiple=True, required=True,
              help="Audio directories, searched recursively.")
@click.option("--out", required=True,
              help="Output dataset directory (TreeWriter format).")
@click.option("--num-samples", type=int, required=True,
              help="Number of excerpts to draw and encode.")
@click.option("--duration", type=float, default=10.0,
              help="Excerpt length in seconds (default 10 = one "
                   "MusicCoCa clip).")
@click.option("--batch-size", type=int, default=4,
              help="Excerpts per codec/style call.")
@click.option("--seed", type=int, default=0)
@click.option("--val-fraction", type=float, default=0.0,
              help="Hold out this fraction of FILES for a leak-free "
                   "file-level validation split (0 = no val set; "
                   "the same audio file never appears in both).")
@click.option("--val-num-samples", type=int, default=0,
              help="Number of excerpts for the val split (written to "
                   "<out>_val). Required when --val-fraction > 0.")
@click.option("--split-seed", type=int, default=0,
              help="Seed for the deterministic file-level split.")
@click.option("--backend", type=click.Choice(["mlx_pure", "nnx"]),
              default="mlx_pure",
              help="mlx_pure = Apple Silicon GPU; nnx = JAX.")
@click.option("--checkpoint", default=None,
              help="Checkpoint filename or path "
                   "(default mrt2_small.safetensors).")
@click.option("--style-prompt", default=None,
              help="Condition the whole dataset on one fixed TEXT "
                   "prompt (e.g. 'electronic dance music') instead of "
                   "per-excerpt audio MusicCoCa. The single-style "
                   "recipe: removes the per-clip style fingerprint so "
                   "the LoRA learns the style anchored to this prompt.")
@click.option("--transcribe", is_flag=True,
              help="Run MT3 per excerpt to add the piano-roll "
                   "conditioning channels (slow: dominates export "
                   "time). Without them, SFT conditions the "
                   "piano-roll streams on their learned "
                   "unconditional (dropout) tokens.")
@click.option("--mt3-batch-size", type=int, default=8)
@click.option("--workers", type=int, default=8,
              help="grain worker processes that read audio and run CPU-based "
                   "preprocessing in parallel.")
@click.option("--worker-buffer-size", type=int, default=1)
@click.option("--extensions", multiple=True,
              default=(".wav", ".flac", ".mp3", ".ogg",
                       ".aif", ".aiff", ".m4a"))
@click.option("--loudness-cutoff", type=float, default=-60.0,
              help="Saliency search LUFS cutoff. -60 rejects only "
                   "near-silence (broad whole-track coverage); raise "
                   "toward -40 to bias excerpts to the loudest "
                   "sections (drops/choruses).")
@click.option("--saliency-tries", type=int, default=8)
@click.option("--musiccoca-time-varying", is_flag=True,
              help="Per-frame time-varying MusicCoCa: a leading 10 s window "
                   "[t, t+10s] per target frame (matches the base model's "
                   "training) instead of one per-clip embedding broadcast. "
                   "Draw extra audio: --duration = head-trim + target + "
                   "lookahead (e.g. --duration 31 --head-trim 1 "
                   "--musiccoca-lookahead 10 -> a 20 s target).")
@click.option("--musiccoca-hop", type=float, default=1.0,
              help="MusicCoCa recompute hop in seconds for --musiccoca-time-"
                   "varying (default 1.0). Coarser is "
                   "far cheaper and near-identical.")
@click.option("--musiccoca-lookahead", type=float, default=10.0,
              help="Seconds of look-ahead audio after the target for the "
                   "leading MusicCoCa windows (should equal clip_length, 10 s).")
@click.option("--head-trim", type=float, default=0.0,
              help="Seconds of codes/conditioning discarded from the START "
                   "(codec warm-up; only used with --musiccoca-time-varying).")
@click.option("--musiccoca-window-subbatch", type=int, default=128,
              help="Max leading windows embedded per MusicCoCa call (GPU mem).")
@click.option("--musiccoca-scan", is_flag=True,
              help="Stream the time-varying MusicCoCa windows through nnx.scan "
                   "(bounded memory for very large spectrograms; sequential).")
@click.option("--profile", is_flag=True,
              help="Print grain per-stage execution statistics "
                   "at the end.")
def export(sources, out, num_samples, duration, batch_size, seed, val_fraction,
           val_num_samples, split_seed, backend, checkpoint,
           style_prompt, transcribe, mt3_batch_size, workers,
           worker_buffer_size, extensions, loudness_cutoff, saliency_tries,
           musiccoca_time_varying, musiccoca_hop, musiccoca_lookahead, head_trim,
           musiccoca_window_subbatch, musiccoca_scan, profile):
    """Precompute an SFT TreeWriter dataset from directories of audio files.

    Draws salient fixed-duration excerpts from the source directories
    (``audiotree.sources.create_audio_dataset`` + ``ExcerptConfig``; grain
    workers read + preprocess audio in parallel) and encodes each batch with the real
    SpectroStream codec + MusicCoCa style model (+ optionally MT3 piano-roll
    transcription) via :func:`magenta_rt.sft.export.export_tree_dataset`.

    Example (Apple Silicon, full conditioning stack):

        JAX_PLATFORMS=cpu mrt sft export \\
            --sources ~/Datasets/Electronic \\
            --out datasets/electronic_sft \\
            --num-samples 1024 --workers 8

    The output directory (``manifest.json`` + one memmap per leaf) is read
    natively by ``magenta_rt.sft.create_audiotree_dataset``.
    """
    sources = list(sources)
    extensions = list(extensions)

    if val_fraction > 0 and val_num_samples <= 0:
        raise click.UsageError(
            "--val-num-samples must be > 0 when --val-fraction > 0")

    # Embed the fixed style prompt in a SEPARATE PROCESS. The mlx_pure MusicCoCa
    # is native (no TFLite), but its text path loads a SentencePiece tokenizer
    # (C++) whose runtime deadlocks/aborts ("mutex lock failed" / abseil "Lock
    # blocking") once grain/JAX are also live in this process — and even doing
    # the embed in-process *before* importing grain poisons the later grain loop,
    # because the C++ runtime persists. Isolating it in a child process keeps
    # this process SentencePiece-free; we feed the arrays to export_tree_dataset
    # so it never re-embeds. (Per-clip AUDIO MusicCoCa, used when --style-prompt
    # is absent, never loads SentencePiece and coexists with grain fine.)
    style_embedding = style_tokens = None
    if style_prompt is not None:
        print(f"[export] embedding style prompt (isolated): "
              f"{style_prompt!r}")
        fd, tmp = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            subprocess.run(
                [sys.executable, "-m", "magenta_rt.sft.embed_prompt",
                 "--prompt", style_prompt, "--backend", backend,
                 "--out", tmp],
                check=True,
            )
            with np.load(tmp) as d:
                style_embedding = d["embedding"]
                style_tokens = d["tokens"]
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"[export] building {backend} codec + MusicCoCa")
    codec, style_model = _build_codec_and_style(
        backend, checkpoint
    )

    from magenta_rt.sft.export import (
        discover_audio_files,
        export_tree_dataset,
        mt3_transcriber,
        split_audio_files,
    )

    transcriber = None
    if transcribe:
        print(f"[export] loading MT3 ({backend}) ...")
        transcriber = mt3_transcriber(
            batch_size=mt3_batch_size, backend=backend
        )

    excerpt = ExcerptConfig(
        strategy="loudest",
        num_tries=saliency_tries,
        lufs_cutoff=loudness_cutoff,
    )
    base_metadata = {
        "sources": [str(s) for s in sources],
        "backend": backend,
    }

    def run(out_dir, n_samples, *, files=None, srcs=None, split=None):
        tag = f" [{split}]" if split else ""
        where = srcs if srcs is not None else f"{len(files)} files"
        print(f"[export]{tag} {n_samples} x {duration:.0f}s salient "
              f"excerpts from {where} -> {out_dir}")
        start = time.time()
        with tqdm(total=n_samples, desc=f"Encoding{tag}", unit="ex") as pbar:
            export_tree_dataset(
                srcs, out_dir,
                codec=codec, style_model=style_model, transcriber=transcriber,
                files=files, num_samples=n_samples, duration=duration,
                batch_size=batch_size, seed=seed,
                excerpt=excerpt, extensions=extensions,
                worker_count=workers,
                worker_buffer_size=worker_buffer_size,
                musiccoca_time_varying=musiccoca_time_varying,
                musiccoca_hop_seconds=musiccoca_hop,
                musiccoca_lookahead_seconds=musiccoca_lookahead,
                head_trim_seconds=head_trim,
                musiccoca_window_subbatch=musiccoca_window_subbatch,
                musiccoca_scan=musiccoca_scan,
                style_prompt=style_prompt,
                style_embedding=style_embedding,
                style_tokens=style_tokens,
                dataset_metadata={**base_metadata,
                                  **({"split": split} if split else {})},
                pbar=pbar, profile=profile,
            )
        elapsed = time.time() - start
        print(f"[export]{tag} wrote {n_samples} examples to {out_dir} in "
              f"{elapsed / 60:.1f} min ({n_samples / elapsed:.2f} ex/s)")

    if val_fraction > 0:
        all_files = discover_audio_files(sources, extensions)
        train_files, val_files = split_audio_files(
            all_files, val_fraction=val_fraction,
            split_seed=split_seed,
        )
        print(f"[export] file-level split (seed {split_seed}): "
              f"{len(train_files)} train / {len(val_files)} val files "
              f"of {len(all_files)} total")
        if not val_files:
            raise click.UsageError(
                "--val-fraction too small: 0 files held out")
        run(out, num_samples, files=train_files, split="train")
        run(f"{out}_val", val_num_samples, files=val_files,
            split="val")
    else:
        run(out, num_samples, srcs=sources)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@sft.command("generate")
@click.option("--model", default="mrt2_small")
@click.option("--checkpoint", default=None,
              help="Checkpoint filename/path (default <model>.safetensors).")
@click.option("--data-dir", required=True,
              help="TreeWriter dataset to pull conditioning from.")
@click.option("--out", required=True)
@click.option("--tag", default="gen")
@click.option("--indices", type=int, multiple=True, default=(0, 1, 2))
@click.option("--adapters", default=None,
              help="LoRA adapter safetensors (injected before load).")
@click.option("--lora-rank", type=int, default=8)
@click.option("--lora-alpha", type=float, default=16.0)
@click.option("--lora-all-linears/--no-lora-all-linears", default=True)
@click.option("--dora", is_flag=True, default=False,
              help="Adapters are DoRA (learned magnitude). Must match how "
                   "they were trained.")
@click.option("--lora-strength", type=float, default=1.0,
              help="Blend adapter toward the base at inference: 1.0=full, "
                   "0.0=base. A strongly-trained adapter often sounds best "
                   "around 0.6-0.8.")
@click.option("--condition", type=click.Choice(["pianoroll", "style"]),
              default="pianoroll")
@click.option("--cfg-musiccoca", type=float, default=3.0)
@click.option("--cfg-notes", type=float, default=1.0)
@click.option("--cfg-drums", type=float, default=1.0)
@click.option("--frames", type=int, default=250)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top-k", type=int, default=40)
@click.option("--seed", type=int, default=7)
def generate(model, checkpoint, data_dir, out, tag, indices, adapters,
             lora_rank, lora_alpha, lora_all_linears, dora, lora_strength,
             condition, cfg_musiccoca, cfg_notes, cfg_drums, frames,
             temperature, top_k, seed):
    """Generate SFT evaluation audio from a TreeWriter dataset's conditioning.

    For each chosen record it builds the source conditioning (MusicCoCa style +,
    optionally, the record's real MT3 piano-roll), drives the streaming step
    frame-by-frame at a fixed CFG, and writes ``<tag>_<i>.wav`` plus a sidecar
    ``<tag>_<i>.txt`` recording PROVENANCE — the source audio ``filepath`` and
    ``offset`` the MusicCoCa/MT3 conditioning was derived from, the model,
    adapters, CFG, and conditioning mode — so the generated clip can be compared
    against the track it was conditioned on.

    mrt2_base loads in bf16 (~15.6 GB peak on load); LoRA adapters (``--adapters``)
    are injected before loading. ``--condition pianoroll`` feeds the record's real
    note structure per frame (System.generate only supports a constant block);
    ``--condition style`` drops the piano-roll to its unconditional token.
    """
    indices = list(indices)

    import mlx.core as mx
    from magenta_rt import config as _cfg, conditioning, paths
    from magenta_rt.mlx_pure.configs import get_model_class
    from magenta_rt.mlx_pure.system import MagentaRT2System
    from magenta_rt.sft import lora_mlx
    from magenta_rt.sft.data import prepare_source_tokens

    mx.set_cache_limit(1 << 30)
    os.makedirs(out, exist_ok=True)
    spec = get_model_class(model)()
    src = TreeDataSource(data_dir)
    manifest = src.get_metadata() or {}
    rng = np.random.default_rng(0)

    # Deterministic conditioning: no input dropout (offset preserved), fixed CFG.
    eval_cfgs = tuple(
        dataclasses.replace(c, dropout_prob=0.0) if c.dropout_prob is not None else c
        for c in spec.input_configs
    )
    MC, DR = (_cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key,
              _cfg.CFG_CONDITIONING_DRUMS.key)
    PR = _cfg.PIANOROLL_WITH_ONSETS.key

    def build_source(record):
        e = {k: v[0].copy() for k, v in record.extras.items() if v[0].ndim >= 2}
        T = e[_cfg.MUSICCOCA.key].shape[0]
        if condition == "style" and PR in e:
            e[PR] = np.full_like(e[PR], -1)  # drop notes -> unconditional
            if _cfg.DRUM_PIANOROLL.key in e:
                e[_cfg.DRUM_PIANOROLL.key] = np.full_like(
                    e[_cfg.DRUM_PIANOROLL.key], -1)
        e[MC] = np.stack([
            np.full(T, conditioning.discretize_cfg(cfg_musiccoca, 0.2, 40)),
            np.full(T, conditioning.discretize_cfg(cfg_notes, 0.2, 40)),
        ], -1).astype(np.int32)
        e[DR] = np.full((T, 1),
                        conditioning.discretize_cfg(cfg_drums, 1.0, 8), np.int32)
        return prepare_source_tokens(e, eval_cfgs, rng)

    ckpt = checkpoint or f"{model}.safetensors"
    print(f"[gen] loading {model} ({ckpt}) ...")
    mrt = MagentaRT2System(size=model, checkpoint=ckpt)
    if adapters:
        targets = (lora_mlx.all_linear_targets if lora_all_linears
                   else lora_mlx.default_targets)
        lora_mlx.inject_lora(mrt._model.depthformer, rank=lora_rank,
                             alpha=lora_alpha, dora=dora,
                             targets=targets, seed=0)
        mrt._model.depthformer.load_weights(adapters, strict=False)
        if lora_strength != 1.0:
            n = lora_mlx.set_lora_strength(mrt._model.depthformer,
                                           lora_strength)
            print(f"[gen] lora_strength={lora_strength} on {n} adapters")
        mx.eval(mrt._model.depthformer.parameters())
        print(f"[gen] loaded adapters: {adapters} "
              f"({'DoRA' if dora else 'LoRA'})")

    for i in indices:
        record = src[i]
        source = build_source(record)  # [T, C]
        # Distinct sampler seed per clip (base seed + index). Without this every
        # index reuses one seed, so with a *fixed* style prompt — where the
        # conditioning is identical across records — all clips come out
        # bit-identical. Offsetting by the index makes each a different draw
        # (and still differs across records when conditioning varies).
        clip_seed = seed + i
        state = mrt._model.make_initial_state(1, seed=clip_seed)
        chunks = []
        for t in range(min(frames, source.shape[0])):
            wav, _codes, state = mrt._model.step_with_codes(
                state, source_tokens=mx.array(source[t][None, None]),
                temperature=temperature, top_k=top_k)
            mx.eval(wav)
            chunks.append(np.asarray(wav))
        audio = np.concatenate(chunks, axis=-1)[0].T  # [T, 2]
        peak = float(np.abs(audio).max())
        wav_path = os.path.join(out, f"{tag}_{i}.wav")
        soundfile.write(wav_path, audio * (0.9 / peak if peak > 0 else 1.0), 48_000)

        # --- provenance sidecar -------------------------------------------
        meta = record.extras
        filepaths = record.filepath  # List[str] (empty if no filepath metadata)
        filepath = filepaths[0] if filepaths else "(unknown)"
        offset = float(meta["offset"][0]) if "offset" in meta else float("nan")
        has_roll = PR in meta and not (meta[PR] < 0).all()
        rms = float(np.sqrt((audio**2).mean()))
        txt_path = os.path.join(out, f"{tag}_{i}.txt")
        with open(txt_path, "w") as f:
            f.write(f"generated: {os.path.basename(wav_path)}\n")
            f.write(f"model: {model}   checkpoint: {ckpt}\n")
            f.write(f"adapters: {adapters or '(none — baseline)'}"
                    f"  rank={lora_rank} alpha={lora_alpha}\n")
            f.write(f"conditioning: {condition} "
                    f"(piano-roll present in data: {has_roll})\n")
            f.write(f"cfg: musiccoca={cfg_musiccoca} notes={cfg_notes} "
                    f"drums={cfg_drums}  temp={temperature} "
                    f"top_k={top_k}  seed={clip_seed}\n")
            f.write(f"dataset: {data_dir}  record index: {i}\n")
            f.write("\n--- conditioning source (MusicCoCa style + MT3 "
                    "piano-roll derived from) ---\n")
            f.write(f"source audio: {filepath}\n")
            f.write(f"excerpt offset: {offset:.2f} s\n")
            f.write(f"\ngenerated rms: {rms:.4f}  duration: "
                    f"{audio.shape[0] / 48_000:.1f} s\n")
            if manifest.get("sources"):
                f.write(f"dataset sources: {manifest['sources']}\n")
        print(f"[gen] {wav_path}  rms {rms:.4f}  <- {os.path.basename(filepath)}"
              f" @ {offset:.0f}s")
    print("[gen] done")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

FRAME_RATE = 25  # SpectroStream frame rate in Hz

MUSICCOCA_KEY = 'mulan_tokens_25hz'
PIANOROLL_KEY = 'pianoroll_with_onsets_tokens'
DRUMS_KEY = 'drum_pianoroll_tokens'
EMBEDDING_KEY = 'musiccoca_embedding'


def print_record(record, index: int) -> None:
    print(f'\n{"=" * 60}')
    print(f'  record {index}')
    print(f'{"=" * 60}')

    filepaths = record.filepath  # List[str], empty if no filepath metadata
    if filepaths:
        print(f'  Source:    {filepaths[0]}')
    offset = record.extras.get('offset')
    if offset is not None:
        print(f'  Offset:    {float(offset[0]):.1f}s')

    codes = record.codes
    if codes is not None:
        codes = codes[0]  # [T, D]
        T = codes.shape[0]
        print('\n  SpectroStream codes:')
        print(f'    Shape:     {codes.shape}  dtype={codes.dtype}'
              f'  ({T / FRAME_RATE:.1f}s @ {FRAME_RATE}Hz)')
        print(f'    Range:     [{codes.min()}, {codes.max()}]')
        print(f'    Unique:    {len(np.unique(codes))} / 1024 codes used')

    style = record.extras.get(MUSICCOCA_KEY)
    if style is not None:
        style = style[0]
        print('\n  MusicCoCa tokens:')
        print(f'    Shape:     {style.shape}  dtype={style.dtype}')
        print(f'    Range:     [{style.min()}, {style.max()}]')
        constant = bool((style == style[0]).all())
        print(f'    Constant over frames: {constant}')

    emb = record.extras.get(EMBEDDING_KEY)
    if emb is not None:
        print(f'\n  MusicCoCa embedding: shape {emb.shape},'
              f' norm {np.linalg.norm(emb[0]):.2f}')

    roll = record.extras.get(PIANOROLL_KEY)
    if roll is None:
        print('\n  Pianoroll: (not stored — exported without a transcriber)')
    else:
        roll = roll[0]  # [T, 128]
        T = roll.shape[0]
        print('\n  Pianoroll:')
        print(f'    Shape:     {roll.shape}  dtype={roll.dtype}')
        print(f'    Values:    {sorted(np.unique(roll))}')
        note_frames = (roll > 0).sum()
        onset_frames = (roll == 2).sum()
        active_notes = np.unique(np.where(roll > 0)[1])
        print(f'    Notes:     {note_frames} active cells, {onset_frames} onsets')
        if len(active_notes) > 0:
            poly = (roll > 0).sum(axis=1)
            poly_active = poly[poly > 0]
            print(f'    Pitch:     {len(active_notes)} unique pitches, '
                  f'range [{active_notes.min()}, {active_notes.max()}]')
            print(f'    Polyphony: avg={poly_active.mean():.1f}, '
                  f'max={poly_active.max()}, '
                  f'active frames={len(poly_active)}/{T} '
                  f'({100 * len(poly_active) / T:.0f}%)')

    drums = record.extras.get(DRUMS_KEY)
    if drums is not None:
        print(f'  Drums:     {int(drums.sum())} onset frames')


def plot_pianoroll(record, title: str, output_path: str | None = None):
    """Visualize the pianoroll of one record."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    roll = record.extras.get(PIANOROLL_KEY)
    if roll is None:
        print('  ⚠️  No pianoroll channel to plot (exported without --transcribe).')
        return
    roll = roll[0]
    T = roll.shape[0]
    duration = T / FRAME_RATE

    active_pitches = np.where(roll.max(axis=0) > 0)[0]
    if len(active_pitches) == 0:
        print('  ⚠️  No active notes to plot.')
        return

    lo = max(0, active_pitches.min() - 4)
    hi = min(128, active_pitches.max() + 5)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[3, 1],
                             gridspec_kw={'hspace': 0.3})

    ax = axes[0]
    cmap = mcolors.ListedColormap(['#1a1a2e', '#4a9eff', '#ff6b6b'])
    ax.imshow(roll[:, lo:hi].T, aspect='auto', origin='lower', cmap=cmap,
              interpolation='nearest', extent=[0, duration, lo, hi])
    ax.set_ylabel('MIDI Pitch')
    ax.set_title(title, fontsize=12, fontweight='bold')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#1a1a2e', label='Off'),
        Patch(facecolor='#4a9eff', label='Note'),
        Patch(facecolor='#ff6b6b', label='Onset'),
    ], loc='upper right', fontsize=8)

    ax2 = axes[1]
    poly = (roll > 0).sum(axis=1)
    time_axis = np.arange(T) / FRAME_RATE
    ax2.fill_between(time_axis, poly, alpha=0.6, color='#4a9eff')
    ax2.plot(time_axis, poly, color='#2a7adf', linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Polyphony')
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, max(poly.max() + 1, 2))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'\n  Saved plot: {output_path}')
    else:
        plt.show()


@sft.command("inspect")
@click.argument("path")
@click.option("--records", type=int, default=3,
              help="Number of records to print in detail.")
@click.option("--plot", type=int, default=None, metavar="INDEX",
              help="Visualize the pianoroll of this record index.")
@click.option("--save-plot", type=str, default=None,
              help="Save the plot to this path instead of displaying.")
def inspect(path, records, plot, save_plot):
    """Inspect an SFT TreeWriter dataset: print stats and visualize pianoroll.

    PATH is the dataset directory (containing manifest.json). The dataset is a
    `magenta_rt.sft.export` TreeWriter export (manifest.json + one memmap per
    leaf), read with audiotree's TreeDataSource.
    """
    path = Path(path)
    if not (path / 'manifest.json').exists():
        raise click.UsageError(
            f'{path} has no manifest.json (not a TreeWriter dataset)')

    source = TreeDataSource(str(path))
    metadata = source.get_metadata()
    print(f'{"=" * 60}')
    print(f'  DATASET {path}')
    print(f'{"=" * 60}')
    print(f'  Records:   {len(source)}')
    for key, value in (metadata or {}).items():
        print(f'  {key}: {value}')

    for i in range(min(records, len(source))):
        print_record(source[i], i)

    if plot is not None:
        record = source[plot]
        plot_pianoroll(record, f'record {plot}', save_plot)
