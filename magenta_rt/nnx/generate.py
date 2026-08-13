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

"""NNX inference / generation entry point for Magenta RealTime 2.

``main`` is the high-level demo: it drives :class:`MagentaRT2System`
(``embed_style`` -> ``generate``), mirroring the jax / mlx entry points, with
optional portable LoRA/DoRA adapters (``--adapters`` / ``--lora-strength``).

The module also exposes the lower-level ``_build_source_tokens`` /
``_musiccoca_tokens`` research helpers, which construct the raw, CFG-batched
conditioning block directly. They are not used by ``main`` (the high-level
system builds conditioning internally) but are imported by A/B harnesses such
as ``scripts/ab_nnx_vs_jax.py`` that drive the ``MagentaRT2Sampler`` step by
hand.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from jax import numpy as jnp

from magenta_rt import paths
from magenta_rt import MagentaRT2Nnx

logging.basicConfig(level=logging.INFO, force=True)


def _musiccoca_tokens(prompt: str) -> list[int]:
    """Encode ``prompt`` to a MusicCoCa RVQ token list via the on-disk TFLite
    models (lazy import keeps nnx free of the dependency)."""
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
    model_name: str = paths.DEFAULT_MODEL_NAME,
    # control
    prompt: str = "disco funk",
    temperature: float = 1.3,
    top_k: int = 40,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 0.1,
    # adapters
    adapters: Optional[str] = None,
    lora_strength: float = 1.0,
    # utils
    checkpoint: str | None = None,
    duration: float = 4.0,
    restore: bool = True,
    jit: bool = True,
    scan: bool = True,
):
    mrt = MagentaRT2Nnx(
        size=model_name,
        checkpoint=checkpoint,
        restore=restore,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        jit=jit,
    )

    # Apply a portable LoRA/DoRA adapter (self-describing safetensors: the
    # rank/alpha/DoRA/targets recipe is read from the file's metadata). The
    # system injects + loads then MERGES into the base depthformer, so the
    # streaming step runs a plain depthformer — identical to a fine-tuned
    # checkpoint, with `lora_strength` baked in. Done before the first
    # `generate` so the KV cache sees plain Linears.
    if adapters is not None:
        meta = mrt.apply_lora_adapters(adapters, strength=lora_strength)
        print(f"Applied LoRA adapters {adapters} (rank={meta['rank']} "
              f"alpha={meta['alpha']} dora={meta['dora']} "
              f"targets={meta['targets']} strength={lora_strength}).")

    embedding = mrt.embed_style(prompt, use_mapper=True)

    frames = int(duration * 25)

    # --- Benchmark ---
    start_time = time.time()
    audio, state = mrt.generate(style=embedding, frames=frames, scan=scan)
    elapsed = time.time() - start_time
    ms_per_step = (elapsed / frames) * 1000
    print(f"Generated {frames} frames in {elapsed:.1f}s "
          f"({frames/elapsed:.1f} steps/s, {ms_per_step:.1f} ms/step)")
    print(f"Target: 25 steps/s, 40 ms/step for real-time")

    # --- Save output ---
    out_path = paths.outputs_dir() / f"output_audio_nnx_{model_name}.wav"
    audio.write(str(out_path))
    print(f"Saved to {out_path} ({duration}s of audio)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("magenta_rt.nnx.generate")

    parser.add_argument("--model", default=paths.DEFAULT_MODEL_NAME, type=str,
                        help=f"Model variant name (default: {paths.DEFAULT_MODEL_NAME}).")
    parser.add_argument("--prompt", default="disco funk", type=str, help="Text conditioning for MusicCoCa.")
    parser.add_argument("--temperature", default=1.3, type=float)
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=0.1, type=float)
    parser.add_argument("--duration", default=4.0, type=float, help="Duration in seconds.")
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='Checkpoint filename in checkpoints/ directory.'
    )
    parser.add_argument(
        "--adapters", default=None, type=str,
        help="Portable LoRA/DoRA adapter safetensors (rank/alpha/DoRA/targets "
             "read from its metadata); merged into the base.",
    )
    parser.add_argument(
        "--lora-strength", default=1.0, type=float,
        help="Blend adapter toward base (1.0=full, 0.0=base; "
             "0.6-0.8 often best for a strong adapter).",
    )
    parser.add_argument(
        "--skip-restore", dest="restore", action="store_false",
        help="Use random weights.",
    )
    parser.add_argument(
        "--no-jit", dest="jit", action="store_false",
        help="Run generation eagerly without jax.jit.",
    )
    parser.add_argument(
        "--no-scan", dest="scan", action="store_false",
        help="Use step-by-step python loop instead of nnx.scan.",
    )
    parser.set_defaults(restore=True, jit=True, scan=True)
    args = parser.parse_args()

    main(
        model_name=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        adapters=args.adapters,
        lora_strength=args.lora_strength,
        checkpoint=args.checkpoint,
        duration=args.duration,
        restore=args.restore,
        jit=args.jit,
        scan=args.scan,
    )
