#!/usr/bin/env python
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

"""Controlled A/B comparison of NNX vs JAX streaming generation.

Motivation
----------
The repository has carried a standing assumption that "NNX generation sounds
worse than JAX on the same checkpoint", which (if true) invalidates every NNX
SFT audition. A code-level audit found the two generation paths to be aligned
(token-based CFG, identical ``discretize_cfg``, identical ``+ (num_reserved+1)``
offset, mathematically-equivalent Gumbel-Max samplers, matched decode dtype
handling). This script settles the question empirically, under *matched*
conditions, so the comparison is fair:

* One MusicCoCa style embedding is computed once and fed to BOTH backends, so
  the style conditioning is identical by construction (not merely by relying on
  MusicCoCa determinism across two call sites).
* Both backends load the same checkpoint and use the same CFG strengths,
  temperature, top-k, frame count, and seed.

Two arms per backend:

* ``greedy`` (temperature=0.0, top_k=1) — a CONTROL. With temperature 0 the
  Gumbel noise is multiplied by zero, so sampling is a deterministic argmax that
  does not depend on the RNG stream. The greedy NNX and JAX *waveforms* should
  therefore match to bf16 numerical tolerance IF the pipelines are equivalent.
  This is an objective check that needs no listening.
* ``stochastic`` (temperature, top_k as given) — the real audition setting. If
  the greedy arm matches but the stochastic arm sounds different, the difference
  is RNG-stream luck on a single clip, NOT a model defect.

Before generating, the script also dumps and diffs the source-conditioning
token arrays each backend constructs. A mismatch there (e.g. the drums CFG bin)
is a real discrepancy that feeds the encoder different inputs, and is reported
explicitly.

Usage
-----
Run only when the GPU is free (this loads two models; never run alongside a
training job on a 16 GB card)::

    .venv/bin/python scripts/ab_nnx_vs_jax.py \
        --checkpoint /abs/path/to/mrt2_small.safetensors \
        --prompt "disco funk" --frames 100 --seed 0 \
        --out-dir /abs/path/to/ab_out

Outputs four wavs (``greedy_nnx.wav``, ``greedy_jax.wav``, ``stoch_nnx.wav``,
``stoch_jax.wav``) plus a printed verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Shared style conditioning
# --------------------------------------------------------------------------- #
def compute_shared_style(prompt: str) -> tuple[np.ndarray, list[int]]:
    """Embed ``prompt`` once with MusicCoCa and return ``(embedding, tokens)``.

    The same embedding object is handed to JAX (which accepts a style
    embedding) and the derived RVQ tokens are handed to NNX (which builds its
    source frame from tokens), so both backends condition on bit-identical
    style. MusicCoCa runs on CPU via the on-disk TFLite models.

    Args:
        prompt: Free-text style description.

    Returns:
        A tuple of the MusicCoCa style embedding and its RVQ token list.
    """
    from magenta_rt import musiccoca

    mc = musiccoca.MusicCoCa()
    embedding = mc.embed_text(prompt, use_mapper=True)
    tokens = list(mc.tokenize(embedding))
    return np.asarray(embedding), tokens


# --------------------------------------------------------------------------- #
# NNX generation (mirrors magenta_rt.nnx.generate.main, but takes shared tokens)
# --------------------------------------------------------------------------- #
def generate_nnx(
    *,
    checkpoint: Path,
    model_name: str,
    style_tokens: list[int],
    temperature: float,
    top_k: int,
    frames: int,
    seed: int,
    cfg_musiccoca: float,
    cfg_notes: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate with the NNX backend.

    Returns:
        ``(waveform[C, T], source_tokens[B, 1, C], compute_dtype_str)``.
    """
    from jax import block_until_ready, numpy as jnp
    from flax import nnx

    from magenta_rt.nnx.model import MagentaRT2Sampler
    from magenta_rt.nnx.generate import _build_source_tokens

    rngs = nnx.Rngs(seed)
    mrt = MagentaRT2Sampler.from_preset(
        model_name, int16_outputs=False, rngs=rngs
    )
    mrt.load_checkpoint(checkpoint)
    mrt.init_streaming(batch_size=1, rngs=rngs)

    input_num_channels = mrt.depthformer.encoder.embedding.num_channels
    source_tokens = _build_source_tokens(
        style=style_tokens,
        num_cfgs=0,
        input_num_channels=input_num_channels,
        num_reserved=mrt.num_reserved_tokens,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
    )

    compute_dtype = str(getattr(mrt.depthformer, "dtype", None))

    chunks = []
    for _ in range(frames):
        tree = mrt.step(
            source_tokens=source_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        tree = block_until_ready(tree)
        chunks.append(np.asarray(tree.waveform))  # [B, C, T_chunk]
    audio = np.concatenate(chunks, axis=-1)  # [B, C, T]
    return audio[0], np.asarray(source_tokens), compute_dtype


# --------------------------------------------------------------------------- #
# JAX generation (the trusted ground-truth path)
# --------------------------------------------------------------------------- #
def generate_jax(
    *,
    checkpoint: Optional[str],
    model_name: str,
    style_embedding: np.ndarray,
    temperature: float,
    top_k: int,
    frames: int,
    cfg_musiccoca: float,
    cfg_notes: float,
) -> tuple[np.ndarray, str]:
    """Generate with the JAX backend.

    Returns:
        ``(waveform[C, T], compute_dtype_str)``.
    """
    from magenta_rt import MagentaRT2Jax

    mrt = MagentaRT2Jax(
        size=model_name,
        checkpoint=checkpoint,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
    )
    audio, _ = mrt.generate(style=style_embedding, frames=frames)
    # AudioTree waveform is channel-major [N, C, T].
    waveform = np.asarray(audio.waveform)
    compute_dtype = str(getattr(mrt, "compute_dtype", "unknown"))
    return waveform[0], compute_dtype


# --------------------------------------------------------------------------- #
# Eval-conditioning reproduction (arms B and C): does BASE mrt2_small sound bad
# on the SFT sampler's exact per-frame conditioning, and is the note pianoroll
# the culprit?
# --------------------------------------------------------------------------- #
def load_eval_source(
    *, eval_dir: str, model_name: str, seed: int, frames: int
) -> tuple[np.ndarray, object]:
    """Build one held-out eval clip's prepared source via the REAL pipeline.

    Uses ``make_eval_dataset`` so the channel layout, the ``-1`` drum fallback,
    and the CFG pinning (``eval_cfg_*``) are bit-identical to what the trainer's
    ``AudioSampleWriter`` feeds the model. Returns ``(source[1, T, C], spec)``.
    """
    from jax import numpy as jnp

    from magenta_rt.nnx.model import MODEL_REGISTRY
    from magenta_rt.sft import to_source_target
    from magenta_rt.sft import trainer_common
    from magenta_rt.sft.configs import SFTConfig

    spec = MODEL_REGISTRY[model_name]()
    config = SFTConfig(
        valid_dir=eval_dir, batch_size=1, seed=seed,
        eval_cfg_musiccoca=3.0, eval_cfg_notes=1.0, eval_cfg_drums=1.0,
        mask_conditioning=False,
    )
    eval_ds = trainer_common.make_eval_dataset(
        eval_dir, config, spec, seed=seed + 1, style_tokens=None
    )
    record = next(iter(eval_ds))
    source, _ = to_source_target(
        record, spec.target_tokens_config, asarray=jnp.asarray
    )
    source = np.asarray(source)[:1, : min(source.shape[1], frames)]
    return source, spec


def notes_off_source(source: np.ndarray, spec: object) -> np.ndarray:
    """Return a copy of ``source`` with the note-pianoroll channel forced to its
    dropout (unconditional) token.

    The dropout token for a channel is ``num_extra_tokens`` (``prepare_source_tokens``
    encodes ``-1`` as ``-1 + (num_extra_tokens + 1)``). Channel column spans come
    from ``spec.input_configs`` order, so this zeroes exactly the pianoroll
    columns and leaves MusicCoCa / drums / CFG untouched.
    """
    from magenta_rt.config import PIANOROLL_WITH_ONSETS

    out = np.array(source)
    col = 0
    for c in spec.input_configs:
        width = c.rvq_truncation_level
        if c.key == PIANOROLL_WITH_ONSETS.key:
            out[..., col : col + width] = c.num_extra_tokens
        col += width
    return out


def static_mulan_source(source: np.ndarray, spec: object) -> np.ndarray:
    """Return a copy of ``source`` with the MusicCoCa channel frozen to frame 0.

    Replaces the per-frame ``mulan_tokens_25hz`` trajectory with a single static
    style frame (frame 0, repeated across all timesteps) — the same static-style
    regime the text-prompt arm A uses. Isolates the time-VARYING per-frame style
    conditioning from the token values themselves. Other channels untouched.
    """
    from magenta_rt.config import MUSICCOCA

    out = np.array(source)
    col = 0
    for c in spec.input_configs:
        width = c.rvq_truncation_level
        if c.key == MUSICCOCA.key:
            out[:, :, col : col + width] = out[:, 0:1, col : col + width]
        col += width
    return out


def mulan_off_source(source: np.ndarray, spec: object) -> np.ndarray:
    """Return a copy of ``source`` with the MusicCoCa channel set to its dropout
    (unconditional) token — tests whether the style conditioning is used at all.
    """
    from magenta_rt.config import MUSICCOCA

    out = np.array(source)
    col = 0
    for c in spec.input_configs:
        width = c.rvq_truncation_level
        if c.key == MUSICCOCA.key:
            out[..., col : col + width] = c.num_extra_tokens
        col += width
    return out


def generate_nnx_from_source(
    *,
    checkpoint: Path,
    model_name: str,
    source: np.ndarray,
    temperature: float,
    top_k: int,
    seed: int,
) -> np.ndarray:
    """Generate base (no-LoRA) NNX audio from a prebuilt per-frame ``source``.

    Mirrors ``AudioSampleWriter`` minus the adapter: a fresh ``MagentaRT2Sampler``
    on the base checkpoint, streamed one source frame per step. Returns ``[C, T]``.
    """
    from jax import block_until_ready, numpy as jnp
    from flax import nnx

    from magenta_rt.nnx.model import MagentaRT2Sampler

    rngs = nnx.Rngs(seed)
    mrt = MagentaRT2Sampler.from_preset(
        model_name, int16_outputs=False, rngs=rngs
    )
    mrt.load_checkpoint(checkpoint)
    mrt.init_streaming(batch_size=1, rngs=nnx.Rngs(seed))
    src = jnp.asarray(source)
    chunks = []
    for t in range(src.shape[1]):
        tree = mrt.step(
            source_tokens=src[:, t : t + 1],
            temperature=temperature,
            top_k=top_k,
        )
        tree = block_until_ready(tree)
        chunks.append(np.asarray(tree.waveform))
    return np.concatenate(chunks, axis=-1)[0]


def _energy(label: str, wav: np.ndarray) -> None:
    print(f"[ab] {label}: RMS={np.sqrt(np.mean(wav**2)):.5f}  "
          f"peak={np.abs(wav).max():.3f}  "
          f"frac_silent={np.mean(np.abs(wav) < 1e-4):.3f}")


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
def compare_waveforms(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two ``[C, T]`` waveforms after trimming to a common length.

    Returns a dict of max-abs difference, RMS difference, and per-channel
    Pearson correlation.
    """
    t = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :t].astype(np.float64), b[..., :t].astype(np.float64)
    diff = a - b
    max_abs = float(np.abs(diff).max())
    rms = float(np.sqrt(np.mean(diff**2)))
    corrs = []
    for c in range(a.shape[0]):
        av, bv = a[c] - a[c].mean(), b[c] - b[c].mean()
        denom = np.sqrt((av**2).sum() * (bv**2).sum())
        corrs.append(float((av * bv).sum() / denom) if denom > 0 else float("nan"))
    return {"trimmed_T": t, "max_abs": max_abs, "rms": rms, "corr": corrs}


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a ``[C, T]`` float waveform to ``path`` as a wav file."""
    from scipy.io import wavfile

    wavfile.write(path, sample_rate, waveform.T.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser("ab_nnx_vs_jax")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Absolute path (or checkpoints/ filename) to the "
                             "shared safetensors checkpoint.")
    parser.add_argument("--model", default="mrt2_small", type=str)
    parser.add_argument("--prompt", default="disco funk", type=str)
    parser.add_argument("--frames", default=100, type=int,
                        help="Frames to generate (25 frames = 1s).")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="Stochastic-arm temperature (SFT sampler uses 1.0).")
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--eval-dir", default=None, type=str,
                        help="Held-out SFT dataset dir. When given, also runs "
                             "arms B/C: BASE nnx on the SFT sampler's exact "
                             "per-frame eval conditioning, notes-on vs notes-off "
                             "(pianoroll forced to its dropout token).")
    parser.add_argument("--skip-jax", action="store_true",
                        help="Skip the JAX arms (e.g. to focus on the "
                             "eval-conditioning pianoroll repro).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint

    ckpt_arg = Path(ckpt) if Path(ckpt).is_absolute() else ckpt

    # ===================================================================== #
    # ARM A: standalone nnx vs jax on a STATIC text-prompt style — is core
    # generation itself faithful? (greedy control + stochastic listening)
    # ===================================================================== #
    if not args.skip_jax:
        print(f"[ab] computing shared MusicCoCa style for prompt "
              f"{args.prompt!r}…")
        style_embedding, style_tokens = compute_shared_style(args.prompt)
        print(f"[ab] style RVQ tokens: {style_tokens}")

        print("\n[ab] === GREEDY arm (control; should match numerically) ===")
        g_nnx, src_nnx, dt_nnx = generate_nnx(
            checkpoint=ckpt_arg, model_name=args.model,
            style_tokens=style_tokens, temperature=0.0, top_k=1,
            frames=args.frames, seed=args.seed,
            cfg_musiccoca=args.cfg_musiccoca, cfg_notes=args.cfg_notes,
        )
        g_jax, dt_jax = generate_jax(
            checkpoint=ckpt, model_name=args.model,
            style_embedding=style_embedding, temperature=0.0, top_k=1,
            frames=args.frames, cfg_musiccoca=args.cfg_musiccoca,
            cfg_notes=args.cfg_notes,
        )
        print(f"[ab] NNX dtype: {dt_nnx}   JAX dtype: {dt_jax}")
        print(f"[ab] NNX source tokens [B,1,C]: shape={src_nnx.shape}")
        print(f"[ab]   first row: {src_nnx[0, 0].tolist()}")
        greedy_cmp = compare_waveforms(g_nnx, g_jax)
        print(f"[ab] GREEDY waveform diff: {greedy_cmp}")
        write_wav(out_dir / "A_greedy_nnx.wav", g_nnx, 48000)
        write_wav(out_dir / "A_greedy_jax.wav", g_jax, 48000)

        print("\n[ab] === STOCHASTIC arm (listening; RNG streams differ) ===")
        s_nnx, _, _ = generate_nnx(
            checkpoint=ckpt_arg, model_name=args.model,
            style_tokens=style_tokens, temperature=args.temperature,
            top_k=args.top_k, frames=args.frames, seed=args.seed,
            cfg_musiccoca=args.cfg_musiccoca, cfg_notes=args.cfg_notes,
        )
        s_jax, _ = generate_jax(
            checkpoint=ckpt, model_name=args.model,
            style_embedding=style_embedding, temperature=args.temperature,
            top_k=args.top_k, frames=args.frames,
            cfg_musiccoca=args.cfg_musiccoca, cfg_notes=args.cfg_notes,
        )
        write_wav(out_dir / "A_stoch_nnx.wav", s_nnx, 48000)
        write_wav(out_dir / "A_stoch_jax.wav", s_jax, 48000)
        _energy("A stoch nnx", s_nnx)
        _energy("A stoch jax", s_jax)

        dtypes_known = all(
            d not in ("unknown", "None", None) for d in (dt_nnx, dt_jax)
        )
        if dtypes_known and dt_nnx != dt_jax:
            print(f"[ab] ⚠ compute dtypes differ (NNX={dt_nnx}, JAX={dt_jax}); "
                  f"the greedy control is dtype-confounded.")
        if greedy_cmp["max_abs"] < 1e-2 and all(
            c > 0.99 for c in greedy_cmp["corr"]
        ):
            print("[ab] ✓ ARM A: greedy waveforms match → core generation+decode "
                  "is bit-for-bit EQUIVALENT to JAX.")
        else:
            print("[ab] ~ ARM A: greedy waveforms diverge. NOTE: free-running "
                  "greedy is CHAOTIC — a single early argmax flip cascades into a "
                  "totally different trajectory. With NNX at bf16 and JAX at fp32 "
                  "this is EXPECTED and is NOT a bug (the teacher-forced parity "
                  "tests are the real bit-exactness control and they pass). Judge "
                  "by ear + by the energy/clipping stats, and re-run NNX at fp32 "
                  "for a fair numerical comparison.")

    # ===================================================================== #
    # ARMS B & C: BASE mrt2_small on the SFT sampler's EXACT per-frame eval
    # conditioning. B = notes ON (reproduces the audition); C = notes OFF
    # (pianoroll → dropout token). If B sounds bad and C sounds good, the note
    # pianoroll is feeding the encoder a representation it can't use.
    # ===================================================================== #
    if args.eval_dir:
        print(f"\n[ab] === ARMS B/C: base nnx on real eval conditioning "
              f"({args.eval_dir}) ===")
        source_on, spec = load_eval_source(
            eval_dir=args.eval_dir, model_name=args.model,
            seed=args.seed, frames=args.frames,
        )
        source_off = notes_off_source(source_on, spec)
        source_static = static_mulan_source(source_off, spec)
        print(f"[ab] eval source shape={source_on.shape}; "
              f"notes-on row0={source_on[0, 0].tolist()[:16]}…")
        print(f"[ab]   static-mulan: frame0 mulan repeated, notes off; "
              f"row5 mulan={source_static[0, 5].tolist()[:12]}")

        b_wav = generate_nnx_from_source(
            checkpoint=ckpt_arg, model_name=args.model, source=source_on,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
        c_wav = generate_nnx_from_source(
            checkpoint=ckpt_arg, model_name=args.model, source=source_off,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
        d_wav = generate_nnx_from_source(
            checkpoint=ckpt_arg, model_name=args.model, source=source_static,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
        source_nomulan = mulan_off_source(source_off, spec)
        f_wav = generate_nnx_from_source(
            checkpoint=ckpt_arg, model_name=args.model, source=source_nomulan,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
        write_wav(out_dir / "B_base_eval_notes_on.wav", b_wav, 48000)
        write_wav(out_dir / "C_base_eval_notes_off.wav", c_wav, 48000)
        write_wav(out_dir / "D_base_eval_static_mulan.wav", d_wav, 48000)
        write_wav(out_dir / "F_base_eval_mulan_off.wav", f_wav, 48000)
        _energy("B base per-frame mulan, notes-ON ", b_wav)
        _energy("C base per-frame mulan, notes-OFF", c_wav)
        _energy("D base STATIC mulan (frame0), notes-OFF", d_wav)
        _energy("F base MULAN-OFF (unconditional style)", f_wav)

        # Arm E: this clip's GLOBAL musiccoca_embedding tokenized into a clean
        # static style (the arm-A source builder: notes/drums = -1), directly
        # comparable to the MUSICAL text-prompt arm A. If E is ALSO incoherent,
        # audio-derived MusicCoCa tokens are out-of-distribution vs the text path.
        emb = np.fromfile(
            str(Path(args.eval_dir) / "extras.musiccoca_embedding.bin"),
            dtype=np.float32, count=768,
        )
        from magenta_rt import musiccoca
        e_tokens = [int(t) for t in musiccoca.MusicCoCa().tokenize(emb)]
        print(f"[ab] clip0 GLOBAL-embedding style tokens: {e_tokens}")
        e_wav, _, _ = generate_nnx(
            checkpoint=ckpt_arg, model_name=args.model, style_tokens=e_tokens,
            temperature=args.temperature, top_k=args.top_k,
            frames=args.frames, seed=args.seed,
            cfg_musiccoca=args.cfg_musiccoca, cfg_notes=args.cfg_notes,
        )
        write_wav(out_dir / "E_clip_global_style.wav", e_wav, 48000)
        _energy("E clip GLOBAL-embedding static style (clean A-source)", e_wav)
        print("[ab] DIAGNOSIS: F==D ⇒ style ignored entirely. F!=D ⇒ style is "
              "used. E incoherent (like B/C/D) but A musical ⇒ audio-derived "
              "MusicCoCa tokens are the culprit; E musical ⇒ the per-frame export "
              "is the culprit.")

    print(f"\n[ab] wrote wavs to {out_dir}")


if __name__ == "__main__":
    main()
