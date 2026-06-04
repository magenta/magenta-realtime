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

"""JAX inference / generation entry point for Magenta RealTime 2."""

import logging
import time

from magenta_rt import paths
from magenta_rt import MagentaRT2Jax

logging.basicConfig(level=logging.INFO, force=True)


def main(
    model_name: str = paths.DEFAULT_MODEL_NAME,
    # control
    prompt: str = "disco funk",
    temperature: float = 1.3,
    top_k: int = 40,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 0.1,
    # utils
    checkpoint: str | None = None,
    duration: float = 4.0,
):
    mrt = MagentaRT2Jax(
        size=model_name,
        checkpoint=checkpoint,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
    )

    embedding = mrt.embed_style(prompt, use_mapper=True)

    frames = int(duration * 25)

    # --- Benchmark ---
    start_time = time.time()
    wav, state = mrt.generate(style=embedding, frames=frames)
    elapsed = time.time() - start_time
    ms_per_step = (elapsed / frames) * 1000
    print(f"Generated {frames} frames in {elapsed:.1f}s "
          f"({frames/elapsed:.1f} steps/s, {ms_per_step:.1f} ms/step)")
    print(f"Target: 25 steps/s, 40 ms/step for real-time")

    # --- Save output ---
    out_path = paths.outputs_dir() / f"output_audio_jax_{model_name}.wav"
    wav.write(str(out_path))
    print(f"Saved to {out_path} ({duration}s of audio)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("run_model")

    parser.add_argument("--model", default=paths.DEFAULT_MODEL_NAME, type=str,
                        help=f"Model variant name (default: {paths.DEFAULT_MODEL_NAME}).")
    parser.add_argument("--prompt", default=None, type=str, help="Text conditioning for MusicCoCa.")
    parser.add_argument("--temperature", default=1.3, type=float)
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    parser.add_argument("--duration", default=4.0, type=float, help="Duration in seconds.")
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='Checkpoint filename in checkpoints/ directory.'
    )
    args = parser.parse_args()

    main(
        model_name=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        checkpoint=args.checkpoint,
        duration=args.duration,
    )
