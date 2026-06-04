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

"""Display benchmark history from outputs/benchmark_log.jsonl.

Usage:
    python scripts/bench_show.py              # latency table
    python scripts/bench_show.py --samples    # also show last-10 audio samples
    python scripts/bench_show.py --last 5     # only last 5 runs
"""

import json
import argparse
import os

LOG_FILE = "outputs/bench_track/benchmark_log.jsonl"


def main():
    p = argparse.ArgumentParser(description="Show benchmark history")
    p.add_argument("--samples", action="store_true", help="Show last-10 audio samples")
    p.add_argument("--last", type=int, default=None, help="Show only last N runs")
    args = p.parse_args()

    if not os.path.exists(LOG_FILE):
        print(f"No benchmark log found at {LOG_FILE}")
        return

    with open(LOG_FILE) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("No benchmark records found.")
        return

    if args.last:
        records = records[-args.last:]

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 120}")
    print(f"  Benchmark History ({len(records)} run{'s' if len(records) != 1 else ''})")
    print(f"{'═' * 120}")

    hdr = (
        f"  {'Date':<20}"
        f"{'Model':<48}"
        f"{'SHA':<9}"
        f"{'Avg':>7}"
        f"{'Med':>7}"
        f"{'P95':>7}"
        f"{'P99':>7}"
        f"{'St/s':>7}"
        f"  Status"
    )
    print(hdr)
    print(
        f"  {'─' * 19} "
        f"{'─' * 47} "
        f"{'─' * 8} "
        f"{'─' * 6} "
        f"{'─' * 6} "
        f"{'─' * 6} "
        f"{'─' * 6} "
        f"{'─' * 6} "
        f" {'─' * 16}"
    )

    for r in records:
        lat = r.get("latency", {})
        status_icon = "✅" if r.get("status") == "WITHIN BUDGET" else "❌"
        line = (
            f"  {r['timestamp']:<20}"
            f"{r['model_name']:<48}"
            f"{r['git_sha']:<9}"
            f"{lat.get('avg_ms', 0):>6.1f} "
            f"{lat.get('median_ms', 0):>6.1f} "
            f"{lat.get('p95_ms', 0):>6.1f} "
            f"{lat.get('p99_ms', 0):>6.1f} "
            f"{lat.get('steps_per_sec', 0):>6.1f} "
            f" {status_icon} {r.get('status', '')}"
        )
        print(line)

        # Optional info lines
        chip = r.get("chip", "")
        note = r.get("note", "")
        if chip or note:
            extra = f"[{chip}]" if chip else ""
            if note:
                extra += f"  {note}" if extra else note
            print(f"  {'':>20}↳ {extra}")

        if args.samples:
            s = r.get("last_10_samples", {})
            l_vals = s.get("L", [])
            r_vals = s.get("R", [])
            if l_vals:
                l_str = ", ".join(f"{v: .8f}" for v in l_vals)
                print(f"  {'':>20}L: [{l_str}]")
            if r_vals:
                r_str = ", ".join(f"{v: .8f}" for v in r_vals)
                print(f"  {'':>20}R: [{r_str}]")

    print(f"{'═' * 120}\n")


if __name__ == "__main__":
    main()
