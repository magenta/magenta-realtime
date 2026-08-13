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

"""End-to-end streaming inference driver for the pure-MLX pipeline.

Mirrors :mod:`magenta_rt.mlx.generate` but routes through
:mod:`magenta_rt.mlx_pure` modules — no ``sequence_layers`` runtime
imports. The pipeline is:

    source_tokens → encoder → depthformer (sampling) → RVQ codes →
    codes_to_waveform (decoder + InverseSTFT) → int16 audio

Two modes:

* ``--restore`` (default off when no checkpoint is on disk) — load weights natively
  straight into the pure tree module properties via the standalone interface
  :func:`magenta_rt.mlx_pure.load_weights.load_from_safetensors`.

* Omit ``--restore`` — randomize zero-init params from ``--seed`` and
  generate. Used to demonstrate the pipeline runs end-to-end without
  needing real weights.

The shipping-config SpectroStream codec is built on the fly. For
demonstration runs you can pass ``--tiny`` to use a small registered
spec and a small SpectroStream.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from magenta_rt import paths
from .model import MagentaRT2Sampler
from .load_weights import init_random_params

DEFAULT_MODEL = "mrt2_small"


def _musiccoca_tokens(prompt: str) -> list[int]:
    """Encode ``prompt`` to a MusicCoCa RVQ token list via the on-disk
    TFLite models (``paths.resources``). Lazy import keeps the pure
    runtime free of the MusicCoCa dependency unless a prompt is used.
    """
    from magenta_rt import musiccoca
    mc = musiccoca.MusicCoCa()
    return list(mc.tokenize(mc.embed_text(prompt, use_mapper=True)))


def _build_source_tokens(
    *, style: Optional[list[int]], num_cfgs: int,
    input_num_channels: int, num_reserved: int,
    num_notes: int = 128, num_drums: int = 1,
    cfg_musiccoca: float = 3.0, cfg_notes: float = 1.0,
) -> mx.array:
    """Build the source-token block (one frame, batched over CFG variants).

    For the single-channel ``tiny`` debug spec there is no MusicCoCa
    branch — emit a masked single channel. Otherwise mirror sl's
    ``MagentaRT2System._build_conditioning`` layout:
    ``[musiccoca(12), notes(128), drums(1), cfgs(3)] + (num_reserved + 1)``
    with masked (``-1``) notes/drums and CFG token bins for the guidance
    scales. CFG negatives mask the MusicCoCa (and notes) segment.
    """
    off = num_reserved + 1
    if input_num_channels == 1 or style is None:
        return mx.full((1, 1, input_num_channels), off, dtype=mx.int32)

    notes = [-1] * num_notes
    drums = [-1] * num_drums
    # cfgs: [musiccoca_bin, notes_bin, drums_bin]; 0.2 step, +1.0 offset.
    cfgs = [int((cfg_musiccoca + 1.0) / 0.2), int((cfg_notes + 1.0) / 0.2), 4]

    def frame(style_seg: list[int]) -> np.ndarray:
        return np.array(style_seg + notes + drums + cfgs, dtype=np.int32) + off

    pos = frame(style)
    if num_cfgs == 0:
        return mx.array(pos.reshape(1, 1, -1), dtype=mx.int32)
    neg_musiccoca = frame([-1] * len(style))
    if num_cfgs == 1:
        cond = np.stack([pos, neg_musiccoca], axis=0)
        return mx.array(cond.reshape(2, 1, -1), dtype=mx.int32)
    # num_cfgs == 2: notes are already masked here, so the notes-negative
    # equals the positive notes segment (kept for batch/API parity).
    neg_notes = frame(style)
    cond = np.stack([pos, neg_musiccoca, neg_notes], axis=0)
    return mx.array(cond.reshape(3, 1, -1), dtype=mx.int32)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(
    restore: bool = False,
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
    # CFG is carried by the trained cfg-strength conditioning tokens (single
    # forward), as the jax / sl-mlx systems and `mrt sft generate` do. num_cfgs>0
    # ALSO does classifier-free logit mixing, double-applying guidance on top of
    # those tokens (over-driven, wider/clipping output) — defaults off.
    num_cfgs: int = 0,
    bridge: bool = False,
    checkpoint: Optional[str] = None,
    bits: Optional[int] = None,
    quantize_method: str = "naive",
    quantize_group_size: Optional[int] = None,
    gptq_cal_steps: int = 8,
):
    log = (lambda *a, **k: None) if quiet else print

    log(f"Building pure-MLX system (model={model_name})…")
    mrt = MagentaRT2Sampler.from_preset(model_name, int16_outputs=False)
    num_reserved = mrt.num_reserved_tokens

    if restore or bridge:
        log("Loading weights via standalone native loader…")
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = paths.checkpoints_dir() / checkpoint_path
        else:
            checkpoint_path = paths.checkpoints_dir() / f"{model_name}.safetensors"
        if checkpoint_path.exists():
            log(f"Loading safetensors weights from {checkpoint_path}…")
            mrt.load_from_safetensors(checkpoint_path, model_name=model_name)
        else:
            log(f"Checkpoint not found at {checkpoint_path}; "
                "falling back to random initialization for demo run.")
            init_random_params(mrt, seed=seed, only_zeros=False)

    else:
        log(f"Initializing random weights (seed={seed})…")
        init_random_params(mrt, seed=seed)

    input_num_channels = mrt.depthformer.encoder.embedding.num_channels
    # MusicCoCa style tokens from the text prompt (skip for the 1-channel
    # tiny debug spec, which has no MusicCoCa branch).
    if input_num_channels > 1:
        log(f"Encoding MusicCoCa style for prompt {prompt!r}…")
        style = _musiccoca_tokens(prompt)
    else:
        style = None

    batch_size = 1 if num_cfgs == 0 else (num_cfgs + 1)

    # Optional quantization. ``naive`` calls quantize_in_place; ``gptq``
    # runs a short calibration loop (gptq_cal_steps streaming steps on
    # the same source tokens used below) then GPTQ-quantizes Dense /
    # EinsumDense layers in place. Both touch only the depthformer side
    # — the SpectroStream codec stays full-precision.
    if bits and bits < 32:
        from .quantize import quantize_in_place, gptq_calibrate_and_quantize
        gs = quantize_group_size or (32 if bits == 4 else 64)
        log(f"Quantizing depthformer to {bits}-bit "
            f"(method={quantize_method}, group_size={gs})…")
        if quantize_method == "naive":
            quantize_in_place(mrt.depthformer, group_size=gs, bits=bits)
        elif quantize_method == "gptq":
            # Build a small calibration source using the first frame's
            # source tokens, then run a few streaming steps. We have to
            # construct source_tokens once here (the same expression as
            # the streaming loop further down).
            cal_source = _build_source_tokens(
                style=style, num_cfgs=num_cfgs,
                input_num_channels=input_num_channels,
                num_reserved=num_reserved,
                cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes,
            )
            def _calibrate(_root):
                cal_state = mrt.make_initial_state(batch_size=batch_size, seed=seed)
                cfg_scales = []
                if num_cfgs >= 1:
                    cfg_scales.append(cfg_musiccoca)
                if num_cfgs >= 2:
                    cfg_scales.append(cfg_notes)
                for _ in range(gptq_cal_steps):
                    _, cal_state = mrt.step(
                        cal_state, source_tokens=cal_source,
                        temperature=temperature, top_k=top_k,
                        cfg_scales=cfg_scales if num_cfgs > 0 else None,
                        cfg_arity=num_cfgs,
                    )
            gptq_calibrate_and_quantize(
                mrt.depthformer, _calibrate,
                group_size=gs, bits=bits, verbose=not quiet,
            )
        else:
            raise ValueError(
                f"unknown quantize_method {quantize_method!r}; "
                f"expected 'naive' or 'gptq'"
            )

    state = mrt.make_initial_state(batch_size=batch_size, seed=seed)

    # Drive the pipeline with a fixed source-token block (deterministic).
    source_tokens = _build_source_tokens(
        style=style, num_cfgs=num_cfgs,
        input_num_channels=input_num_channels,
        num_reserved=num_reserved,
        cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes,
    )

    log(f"Streaming {num_steps} step(s)…")
    audio_chunks = []
    t0 = time.time()

    cfg_scales = []
    if num_cfgs >= 1:
        cfg_scales.append(cfg_musiccoca)
    if num_cfgs >= 2:
        cfg_scales.append(cfg_notes)

    for i in range(num_steps):
        waveform, state = mrt.step(
            state,
            source_tokens=source_tokens,
            temperature=temperature,
            top_k=top_k,
            cfg_scales=cfg_scales if num_cfgs > 0 else None,
            cfg_arity=num_cfgs,
        )
        mx.eval(waveform)
        audio_chunks.append(waveform)
    elapsed = time.time() - t0
    log(f"Streaming done: {num_steps} steps in {elapsed:.2f}s "
        f"({num_steps / max(elapsed, 1e-9):.2f} steps/s)")

    audio = mx.concatenate(audio_chunks, axis=-1)  # time-last
    audio_np = np.array(audio)

    if output_path is not None:
        from scipy.io import wavfile
        sr = 48_000
        # audio shape: [B, T] or [B, C, T]; wavfile expects [T, C] or [T].
        if audio_np.ndim == 3:
            arr = audio_np[0].T  # [C, T] -> [T, C]
        else:
            arr = audio_np[0]
        wavfile.write(output_path, sr, arr)
        log(f"Saved audio to {output_path}")

    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser("magenta_rt.mlx_pure.generate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tiny", action="store_true", help="Use tiny test config.")
    parser.add_argument(
        "--restore", action="store_true",
        help="Load weights natively via the standalone interface: parse "
             "safetensors flat keys straight into pure module properties "
             "via mlx_pure.load_weights.load_from_safetensors.",
    )
    parser.add_argument(
        "--bridge", action="store_true",
        help="Hidden alias for --restore (layout mapping is the only "
             "supported load path; native is a stub).",
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
        help="0 (default): CFG via the trained conditioning tokens only (single "
             "forward; matches jax/sl-mlx). 1/2: also apply logit-space CFG, "
             "which double-applies guidance on top of those tokens.")
    parser.add_argument(
        "--checkpoint", default=None, type=str,
        help="Filename in checkpoints/ (or absolute path) to load via "
             "--restore. Defaults to <model>.safetensors.",
    )
    parser.add_argument(
        "--bits", default=None, type=int, choices=[3, 4, 5, 6, 8],
        help="Quantize Dense + EinsumDense layers to this bit width. "
             "Skipped when not set (or set to 32).",
    )
    parser.add_argument(
        "--quantize-method", default="naive", choices=["naive", "gptq"],
        help="`naive` calls quantize_in_place (nearest-rounding); `gptq` "
             "runs a short calibration loop and applies GPTQ "
             "error-compensated rounding. Only consulted when --bits is set.",
    )
    parser.add_argument(
        "--quantize-group-size", default=None, type=int,
        help="Group size for the quantizer (default 32 for bits=4, 64 otherwise).",
    )
    parser.add_argument(
        "--gptq-cal-steps", default=8, type=int,
        help="Number of streaming forward passes used to capture activations "
             "during GPTQ calibration.",
    )
    args = parser.parse_args()

    model_name = "tiny" if args.tiny else args.model

    main(
        restore=args.restore,
        model_name=model_name,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        num_steps=args.num_steps,
        seed=args.seed,
        output_path=args.output,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        num_cfgs=args.num_cfgs,
        bridge=args.bridge,
        checkpoint=args.checkpoint,
        bits=args.bits,
        quantize_method=args.quantize_method,
        quantize_group_size=args.quantize_group_size,
        gptq_cal_steps=args.gptq_cal_steps,
    )
