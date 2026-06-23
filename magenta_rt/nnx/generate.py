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

"""End-to-end streaming inference driver for the JAX/nnx pipeline.

Pipeline:

    source_tokens → encoder → depthformer (sampling) → RVQ codes →
    codes_to_waveform (decoder + InverseSTFT) → int16 audio

Two modes:

* ``--restore`` — load weights via
  :func:`magenta_rt.nnx.load_weights.load_from_jax_safetensors` from a
  Linen safetensors checkpoint.
* ``--skip-restore`` — randomize zero-init params from ``--seed`` and
  generate. Useful to demonstrate the pipeline runs end-to-end without
  needing real weights.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from jax import block_until_ready, numpy as jnp
from flax import nnx
from einops import rearrange
import numpy as np

from .model import MagentaRT2Sampler
from magenta_rt import paths


DEFAULT_MODEL = "mrt2_small"


def _musiccoca_tokens(prompt: str) -> list[int]:
    """Encode ``prompt`` to a MusicCoCa RVQ token list via the on-disk
    TFLite models (lazy import keeps nnx free of the dependency)."""
    from magenta_rt import musiccoca
    mc = musiccoca.MusicCoCa()
    return list(mc.tokenize(mc.embed_text(prompt, use_mapper=True)))


def _build_source_tokens(
    *,
    style: Optional[list[int]],
    num_cfgs: int,
    input_num_channels: int,
    num_reserved: int,
    num_notes: int = 128,
    num_drums: int = 1,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 1.0,
) -> jnp.ndarray:
    """Build the source conditioning tokens (one frame, batched over CFG
    variants), mirroring ``magenta_rt.mlx.system._build_conditioning``:
    ``[musiccoca(12), notes(128), drums(1), cfgs(3)] + (num_reserved + 1)``.

    CFG negatives mask the MusicCoCa (and notes) segment. The 1-channel
    tiny debug spec has no MusicCoCa branch (emit a masked channel).
    """
    off = num_reserved + 1
    if input_num_channels == 1 or style is None:
        return jnp.full((max(1, num_cfgs + 1), 1, input_num_channels), off, dtype=jnp.int32)

    notes = [-1] * num_notes
    drums = [-1] * num_drums
    cfgs = [int((cfg_musiccoca + 1.0) / 0.2), int((cfg_notes + 1.0) / 0.2), 4]

    def frame(style_seg: list[int]) -> list[int]:
        return style_seg + notes + drums + cfgs

    pos = frame(style)
    neg_musiccoca = frame([-1] * len(style))
    neg_notes = frame(style)  # notes already masked here
    if num_cfgs == 0:
        rows = [pos]
    elif num_cfgs == 1:
        rows = [pos, neg_musiccoca]
    elif num_cfgs == 2:
        rows = [pos, neg_musiccoca, neg_notes]
    else:
        raise ValueError(f"Unsupported num_cfgs: {num_cfgs}. Must be 0, 1, or 2.")

    cond = np.array(rows, dtype=np.int32) + off
    return jnp.asarray(cond[:, np.newaxis, :], dtype=jnp.int32)


def main(
    restore: bool = True,
    model_name: str = DEFAULT_MODEL,
    prompt: str = "disco funk",
    temperature: float = 1.3,
    top_k: int = 40,
    num_steps: int = 100,
    seed: int = 0,
    output_path: Optional[Path] = None,
    quiet: bool = False,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 1.0,
    # CFG for mrt2 is carried by the trained cfg-strength *conditioning tokens*
    # (set from cfg_musiccoca/cfg_notes), exactly as the jax / sl-mlx systems and
    # `mrt sft generate` do — a single forward. num_cfgs>0 ADDITIONALLY runs
    # classifier-free logit mixing over negative passes, which double-applies
    # guidance on top of those tokens (over-driven, wider/clipping output), so it
    # defaults off. (0 = single forward; 1/2 = experimental logit-space CFG.)
    num_cfgs: int = 0,
    checkpoint: Optional[str] = None,
    jit: bool = False,
    scan: bool = True,
):
    log = (lambda *a, **k: None) if quiet else print

    log(f"Building nnx system (model={model_name})…")
    rngs = nnx.Rngs(seed)
    mrt = MagentaRT2Sampler.from_preset(model_name, int16_outputs=False, rngs=rngs)

    if restore:
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = paths.checkpoints_dir() / checkpoint_path
        else:
            checkpoint_path = paths.checkpoints_dir() / f"{model_name}.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Pass --skip-restore for a random-init demo."
            )
        mrt.load_checkpoint(checkpoint_path)
    else:
        log(f"Using freshly-initialized random weights (seed={seed})…")

    batch_size = 1 if num_cfgs == 0 else (num_cfgs + 1)
    mrt.init_streaming(batch_size=batch_size, rngs=rngs)

    input_num_channels = mrt.depthformer.encoder.embedding.num_channels
    if input_num_channels > 1:
        log(f"Encoding MusicCoCa style for prompt {prompt!r}…")
        style = _musiccoca_tokens(prompt)
    else:
        style = None

    source_tokens = _build_source_tokens(
        style=style,
        num_cfgs=num_cfgs,
        input_num_channels=input_num_channels,
        num_reserved=mrt.num_reserved_tokens,
        cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes,
    )

    cfg_scales = []
    if num_cfgs >= 1:
        cfg_scales.append(cfg_musiccoca)
    if num_cfgs >= 2:
        cfg_scales.append(cfg_notes)
    cfg_scales_arg = cfg_scales if num_cfgs > 0 else None

    audio_chunks = []
    first_step_time: Optional[float] = None

    if jit:
        if scan and num_steps > 1:
            log(f"Streaming {num_steps} step(s) [nnx.scan]…")
            stacked_src = jnp.broadcast_to(
                source_tokens[None], (num_steps,) + source_tokens.shape,
            )

            # Stacks each step's ``AudioTree`` leaf arrays along axis 1, so a
            # single rearrange flattens the (steps, T_chunk) slices into one
            # contiguous (time-last) axis.
            @nnx.jit
            @nnx.scan(
                in_axes=(nnx.Carry, 0),
                out_axes=(nnx.Carry, 1),
            )
            def stream_body(model, src):
                audio_tree = model.step(
                    source_tokens=src,
                    temperature=temperature, top_k=top_k,
                    cfg_scales=cfg_scales_arg, cfg_arity=num_cfgs,
                )
                return model, audio_tree

            t0 = time.time()
            mrt, audio_tree = stream_body(mrt, stacked_src)
            audio_tree = block_until_ready(audio_tree)
            elapsed = time.time() - t0

            audio_chunks.append(rearrange(audio_tree.waveform, 'b s c t -> b c (s t)'))

            log(f"Streaming done. Scan compile + run {elapsed:.2f}s "
                f"for {num_steps} steps "
                f"({num_steps / max(elapsed, 1e-9):.2f} steps/s amortized).")
        else:
            @nnx.jit
            def step_fn(model, source_tokens):
                return model.step(
                    source_tokens=source_tokens,
                    temperature=temperature, top_k=top_k,
                    cfg_scales=cfg_scales_arg, cfg_arity=num_cfgs,
                )
            log(f"Streaming {num_steps} step(s) [nnx.jit]…")
            t0 = time.time()
            for i in range(num_steps):
                audio_tree = step_fn(mrt, source_tokens)
                audio_tree = block_until_ready(audio_tree)
                audio_chunks.append(audio_tree.waveform)
                if i == 0:
                    first_step_time = time.time() - t0

            elapsed = time.time() - t0
            log(f"Streaming done. First jitted step (compile + run) {first_step_time:.2f}s; "
                f"remaining {num_steps - 1} steps "
                f"{elapsed - (first_step_time or 0):.2f}s "
                f"({(num_steps - 1) / max(elapsed - (first_step_time or 0), 1e-9):.2f} steps/s).")
    else:
        log(f"Streaming {num_steps} step(s) [eager]…")
        t0 = time.time()
        for i in range(num_steps):
            audio_tree = mrt.step(
                source_tokens=source_tokens,
                temperature=temperature, top_k=top_k,
                cfg_scales=cfg_scales_arg, cfg_arity=num_cfgs,
            )
            audio_tree = block_until_ready(audio_tree)
            audio_chunks.append(audio_tree.waveform)
        elapsed = time.time() - t0
        log(f"Streaming done: {num_steps} steps in {elapsed:.2f}s "
            f"({num_steps / max(elapsed, 1e-9):.2f} steps/s).")

    audio = jnp.concatenate(audio_chunks, axis=-1)  # [B, C, T] time-last
    audio_np = np.asarray(audio)

    if output_path is not None:
        from scipy.io import wavfile
        assert audio_np.ndim == 3
        arr = audio_np[0].T  # [C, T] -> wavfile's [T, C]
        wavfile.write(output_path, mrt.sample_rate, arr)
        log(f"Saved audio to {output_path}")

    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser("magenta_rt.nnx.generate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-restore", dest="restore", action="store_false",
        help="Use random weights.",
    )
    parser.add_argument("--prompt", default="disco funk", type=str,
                        help="Text conditioning for MusicCoCa.")
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Frames to generate (25 frames = 1s).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
    parser.add_argument("--cfg-notes", type=float, default=1.0)
    parser.add_argument(
        "--num-cfgs", type=int, default=0, choices=[0, 1, 2],
        help="0 (default): CFG via the trained cfg-strength conditioning tokens "
             "only (single forward; matches the jax/sl-mlx systems). 1/2: also "
             "apply classifier-free logit mixing, which DOUBLE-applies guidance "
             "on top of those tokens (over-driven output) — experimental.")
    parser.add_argument(
        "--checkpoint", default=None, type=str,
        help="Filename in checkpoints/ (or absolute path) to load. "
             "Defaults to <model>.safetensors.",
    )
    parser.add_argument(
        "--adapters", default=None, type=str,
        help="Portable LoRA/DoRA adapter safetensors (recipe read from its "
             "metadata); merged into the base before generation.",
    )
    parser.add_argument(
        "--lora-strength", type=float, default=1.0,
        help="Blend the adapter toward the base (1.0=full, 0.0=base; a "
             "strongly-trained adapter often sounds best around 0.6-0.8).",
    )
    parser.add_argument(
        "--jit", dest="jit", action="store_true",
        help="Wrap the streaming loop in nnx.jit. With --scan (the "
             "default) the entire remaining loop runs as a single "
             "nnx.scan, which is the fast path. Without --scan it "
             "falls back to per-step nnx.jit.",
    )
    parser.add_argument(
        "--no-scan", dest="scan", action="store_false",
        help="Use a Python for-loop instead of nnx.scan (worse performance). "
             "Has no effect with --no-jit.",
    )
    parser.set_defaults(restore=True, jit=False, scan=True)
    args = parser.parse_args()

    main(
        restore=args.restore,
        model_name=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        num_steps=args.num_steps,
        seed=args.seed,
        output_path=args.output,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        num_cfgs=args.num_cfgs,
        checkpoint=args.checkpoint,
        adapters=args.adapters,
        lora_strength=args.lora_strength,
        jit=args.jit,
        scan=args.scan,
    )
