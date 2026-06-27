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

"""Bulk generate audio from a CSV of text prompts using MLX backends.

Usage:

    python scripts/bulk_generate.py
    python scripts/bulk_generate.py --size mrt2_base --duration-sec 30 --batch-size 4

Outputs are saved to `outputs/eval_audio/{size}/`.
"""

import argparse
import time
import logging

import pandas as pd
from pathlib import Path

from magenta_rt import MagentaRT2Mlxfn, MagentaRT2Mlx

logging.basicConfig(level=logging.INFO, force=True)

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_PROMPTS_FILE = "./magenta_rt/data/example_prompt_set.csv"


def main():
    parser = argparse.ArgumentParser(description="Bulk generate audio from text prompts")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE,
                        help="CSV file with 'prompt_id' and 'prompt' columns")
    parser.add_argument("--size", default=None, help="Model size name (default: paths.DEFAULT_MODEL_NAME)")
    parser.add_argument("--duration-sec", default=60, type=int, help="Duration of each clip in seconds")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for parallel generation")
    parser.add_argument("--no-mlxfn", action="store_true",
                        help="Use the Python model instead of the compiled .mlxfn model")
    parser.add_argument("--temperature", default=1.1, type=float)
    parser.add_argument("--top-k", default=128, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    args = parser.parse_args()

    frames = args.duration_sec * 25  # 25 fps

    model_size = args.size or "mrt2_base"

    # --- Init system ---
    if args.batch_size > 1 or args.no_mlxfn:
        print("Initializing Python-based MLX system...")
        mrt = MagentaRT2Mlx(
            size=model_size,
            temperature=args.temperature,
            top_k=args.top_k,
            cfg_musiccoca=args.cfg_musiccoca,
            cfg_notes=args.cfg_notes,
        )
    else:
        print("Initializing compiled .mlxfn MLX system...")
        mrt = MagentaRT2Mlxfn(
            size=model_size,
            temperature=args.temperature,
            top_k=args.top_k,
            cfg_musiccoca=args.cfg_musiccoca,
            cfg_notes=args.cfg_notes,
        )

    # --- Load prompts ---
    prompts_df = pd.read_csv(args.prompts_file)
    print(f"Loaded {len(prompts_df)} prompts from {args.prompts_file}")

    output_dir = ROOT_DIR / "outputs" / "eval_audio" / model_size
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    for i in range(0, len(prompts_df), batch_size):
        chunk = prompts_df.iloc[i : i + batch_size]
        prompts = chunk["prompt"].tolist()
        prompt_ids = chunk.index.tolist()

        print(f"\nGenerating batch {i // batch_size + 1} ({len(prompts)} clips) for prompt_ids={prompt_ids}")
        for pid, prompt_text in zip(prompt_ids, prompts):
            print(f"  - {pid}: '{prompt_text}'")

        if args.batch_size > 1 or args.no_mlxfn:
            embedding = mrt.embed_styles(prompts, use_mapper=True)
        else:
            embedding = mrt.embed_style(prompts[0], use_mapper=True)

        start_time = time.time()
        audio_tree, _ = mrt.generate(style=embedding, frames=frames)
        elapsed = time.time() - start_time
        print(f"  Done in {elapsed:.1f}s ({frames/elapsed:.1f} steps/s)")

        for j, item in enumerate(audio_tree):
            pid = prompt_ids[j]
            out_path = output_dir / f"{pid}.wav"
            item.write(str(out_path))
            print(f"  Saved to {out_path}")

    print(f"\nAll done! Generated {len(prompts_df)} clips in {output_dir}")


if __name__ == "__main__":
    main()
