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

"""Bulk generate audio from a CSV of text prompts using an exported .mlxfn model.

Usage:

    python scripts/bulk_generate.py
    python scripts/bulk_generate.py --size mrt2_base --duration-sec 30

Outputs are saved to `outputs/eval_audio/{size}/`.
"""

import argparse
import time
import logging

import pandas as pd
from pathlib import Path
from scipy.io import wavfile

from magenta_rt import MagentaRT2Mlxfn

logging.basicConfig(level=logging.INFO, force=True)

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_PROMPTS_FILE = "./magenta_rt/data/example_prompt_set.csv"


def main():
    parser = argparse.ArgumentParser(description="Bulk generate audio from text prompts")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE,
                        help="CSV file with 'prompt_id' and 'prompt' columns")
    parser.add_argument("--size", default=None, help="Model size name (default: paths.DEFAULT_MODEL_NAME)")
    parser.add_argument("--duration-sec", default=60, type=int, help="Duration of each clip in seconds")
    parser.add_argument("--temperature", default=1.1, type=float)
    parser.add_argument("--top-k", default=128, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    args = parser.parse_args()

    frames = args.duration_sec * 25  # 25 fps

    # --- Init system ---
    mrt = MagentaRT2Mlxfn(
        size=args.size,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
    )

    # --- Load prompts ---
    prompts_df = pd.read_csv(args.prompts_file)
    print(f"Loaded {len(prompts_df)} prompts from {args.prompts_file}")

    output_dir = ROOT_DIR / "outputs" / "eval_audio" / (args.size or "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in prompts_df.iterrows():
        prompt_id = idx
        prompt_text = row["prompt"]

        print(f"\n[{idx+1}/{len(prompts_df)}] Generating {args.duration_sec}s for "
              f"prompt_id={prompt_id}: '{prompt_text}'")

        embedding = mrt.embed_style(prompt_text, use_mapper=True)

        start_time = time.time()
        wav, _ = mrt.generate(style=embedding, frames=frames)
        elapsed = time.time() - start_time
        print(f"  Done in {elapsed:.1f}s ({frames/elapsed:.1f} steps/s)")

        out_path = output_dir / f"{prompt_id}.wav"
        wavfile.write(str(out_path), wav.sample_rate, wav.samples)
        print(f"  Saved to {out_path}")

    print(f"\nAll done! Generated {len(prompts_df)} clips in {output_dir}")


if __name__ == "__main__":
    main()
