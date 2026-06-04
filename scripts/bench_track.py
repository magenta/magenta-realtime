#!/usr/bin/env python3
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

"""Tracked benchmark: export → build → run → log to JSONL.

Output name is auto-generated as: {MODEL_PREFIX}_int{bits}_rvq12_{YYMMDD}_{sha}

Usage:
    python scripts/bench_track.py                       # defaults: 8-bit, 100 steps
    python scripts/bench_track.py --bits 8 --steps 100  # explicit
    python scripts/bench_track.py --note "new depth decoder" --chip "M4 Max"

    # Benchmark an existing exported model (skips export):
    python scripts/bench_track.py --model-dir outputs/bench_track/mrt2_base_int8_rvq12_260604_7ff396a
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

LOG_FILE = "outputs/bench_track/benchmark_log.jsonl"
BENCHMARK_BIN = "./benchmark_build/benchmark_mlxfn"
BENCH_DIR = "outputs/bench_track"
MODEL_PREFIX = "mrt2_small"


def git_sha():
    """Get short git SHA of current HEAD."""
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def run_cmd(cmd, capture=False):
    """Run a command, streaming output live.

    If capture=True, also returns the full stdout text for parsing.
    """
    print(f"  $ {' '.join(cmd)}")
    if capture:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)
        proc.wait()
        return proc.returncode, "".join(lines)
    else:
        return subprocess.call(cmd), ""


def parse_output(text):
    """Extract latency stats and last-10 samples from benchmark stdout."""
    latency = {}
    for key, pat in [
        ("avg_ms",        r"Avg:\s+([\d.]+)\s+ms"),
        ("median_ms",     r"Median:\s+([\d.]+)\s+ms"),
        ("min_ms",        r"Min:\s+([\d.]+)\s+ms"),
        ("max_ms",        r"Max:\s+([\d.]+)\s+ms"),
        ("p95_ms",        r"P95:\s+([\d.]+)\s+ms"),
        ("p99_ms",        r"P99:\s+([\d.]+)\s+ms"),
        ("steps_per_sec", r"Throughput:\s+([\d.]+)\s+steps/s"),
    ]:
        m = re.search(pat, text)
        if m:
            latency[key] = float(m.group(1))

    status = (
        "WITHIN BUDGET" if "WITHIN BUDGET" in text else
        "EXCEEDS BUDGET" if "EXCEEDS BUDGET" in text else
        "UNKNOWN"
    )

    l_vals = [float(m.group(1)) for m in re.finditer(r"L\[\s*\d+\]\s*=\s*([-\d.]+)", text)]
    r_vals = [float(m.group(1)) for m in re.finditer(r"R\[\s*\d+\]\s*=\s*([-\d.]+)", text)]

    return latency, status, {"L": l_vals[-10:], "R": r_vals[-10:]}


def main():
    p = argparse.ArgumentParser(
        description="Export + build + benchmark → log results to JSONL",
    )
    p.add_argument("--model-dir", default=None,
                   help="Path to an existing exported model dir (skips export)")
    p.add_argument("--bits", type=int, default=8, help="Quantization bits (default: 8)")
    p.add_argument("--steps", type=int, default=100, help="Benchmark steps (default: 100)")
    p.add_argument("--note", default="", help="Optional note for this run")
    p.add_argument("--chip", default="M4 Pro", help="Mac chip type (default: M4 Pro)")
    args = p.parse_args()

    now = datetime.datetime.now().isoformat(timespec="seconds")
    skip_export = args.model_dir is not None

    if skip_export:
        # Use existing exported model
        model_dir = args.model_dir.rstrip("/")
        output_name = os.path.basename(model_dir)
        # Extract git SHA from name (last segment after _)
        sha = output_name.rsplit("_", 1)[-1] if "_" in output_name else "unknown"
    else:
        # Auto-generate output name: {MODEL_PREFIX}_int{bits}_rvq12_{YYMMDD}_{sha}
        sha = git_sha()
        date_str = datetime.datetime.now().strftime("%y%m%d")
        output_name = f"{MODEL_PREFIX}_int{args.bits}_rvq12_cfgs0_{date_str}_{sha}"
        model_dir = os.path.join(BENCH_DIR, output_name)

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 63}")
    print(f"  Tracked Benchmark{' (bench only)' if skip_export else ''}")
    print(f"{'═' * 63}")
    print(f"  Model:   {output_name}")
    print(f"  Bits:    {args.bits}  Steps: {args.steps}  SHA: {sha}  Chip: {args.chip}")
    if args.note:
        print(f"  Note:    {args.note}")
    print(f"{'═' * 63}\n")

    if not skip_export:
        # ── 1) Export ─────────────────────────────────────────────────────
        print("━━━ [1/3] Export ━━━")
        export_cmd = [
            "mrt", "mlx", "export",
            f"--checkpoint={MODEL_PREFIX}.safetensors",
            f"--model={MODEL_PREFIX}",
            "--num-cfgs=0",
            f"--bits={args.bits}",
            f"--output-name={output_name}",
            f"--output-dir={BENCH_DIR}",
        ]
        rc, _ = run_cmd(export_cmd)
        if rc != 0:
            sys.exit(f"❌ Export failed (exit {rc})")
        print()

    # ── 2) Build ──────────────────────────────────────────────────────────
    print("━━━ [2/3] Build ━━━")
    # Configure cmake if benchmark_build doesn't exist yet
    if not os.path.exists("benchmark_build/build.ninja") and \
       not os.path.exists("benchmark_build/Makefile"):
        print("  (configuring cmake...)")
        rc, _ = run_cmd(["cmake", "core/src/benchmark", "-B", "benchmark_build"])
        if rc != 0:
            sys.exit(f"❌ cmake configure failed (exit {rc})")
    rc, _ = run_cmd([
        "cmake", "--build", "benchmark_build",
        "--target", "benchmark_mlxfn", "-j10",
    ])
    if rc != 0:
        sys.exit(f"❌ Build failed (exit {rc})")
    print()

    # ── 3) Benchmark ──────────────────────────────────────────────────────
    print("━━━ [3/3] Benchmark ━━━")
    bench_cmd = [BENCHMARK_BIN, model_dir, str(args.steps)]
    rc, output = run_cmd(bench_cmd, capture=True)
    if rc != 0:
        sys.exit(f"❌ Benchmark failed (exit {rc})")

    # ── Parse & log ───────────────────────────────────────────────────────
    latency, status, samples = parse_output(output)
    record = {
        "timestamp": now,
        "git_sha": sha,
        "model_name": output_name,
        "bits": args.bits,
        "num_steps": args.steps,
        "chip": args.chip,
        "latency": latency,
        "last_10_samples": samples,
        "status": status,
    }
    if args.note:
        record["note"] = args.note

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n✅ Logged to {LOG_FILE}")


if __name__ == "__main__":
    main()
