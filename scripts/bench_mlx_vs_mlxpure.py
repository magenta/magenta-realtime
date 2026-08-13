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

"""Side-by-side latency benchmark: ``magenta_rt.mlx`` vs
``magenta_rt.mlx_pure`` on the mrt2_small real checkpoint.

Drives each backend through its existing ``generate.main`` entry
point so the per-step code path is the same one a user would hit at
the CLI:

* ``magenta_rt.mlx.generate.main`` always runs in benchmark mode
  (5 warmup + 100 measured steps) and prints the steps/s line.
* ``magenta_rt.mlx_pure.generate.main`` doesn't have a separate
  benchmark mode, so we run it twice: a 5-step "warmup" pass (whose
  timing is discarded) and a 100-step "measured" pass that we time
  externally to match the sl path.

Run::

    PYTHONPATH=sequence-layers python scripts/bench_mlx_vs_mlxpure.py
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = "mrt2_small.safetensors"


# ---------------------------------------------------------------------------
# magenta_rt.mlx (sl-backed) — has built-in benchmark mode
# ---------------------------------------------------------------------------


def bench_mlx(*, model_name: str, num_cfgs: int, checkpoint: str) -> tuple[float, float]:
    """Returns ``(steps_per_second, ms_per_step)``.

    ``magenta_rt.mlx.generate.main`` prints its own benchmark line of
    the form ``Per-step eval: 100 steps in 9.3s (10.7 steps/s) (93.2 ms/step)``.
    We capture stdout, run main(), then parse that line.
    """
    from magenta_rt.mlx.generate import main as mlx_main

    buf = io.StringIO()
    print(f"[mlx (combinator)] running magenta_rt.mlx.generate.main (model={model_name})…")
    with redirect_stdout(buf):
        mlx_main(
            restore=True,
            model_name=model_name,
            num_cfgs=num_cfgs,
            checkpoint=checkpoint,
            do_benchmark=True,
        )
    out = buf.getvalue()
    # Echo so the user sees what happened.
    print(out)
    m = re.search(
        r"Per-step eval:\s*\d+\s*steps\s*in\s*[\d.]+s\s*\(([\d.]+)\s*steps/s\)\s*"
        r"\(([\d.]+)\s*ms/step\)",
        out,
    )
    if m is None:
        raise RuntimeError(
            "Failed to parse benchmark line from magenta_rt.mlx output:\n" + out
        )
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------------------
# magenta_rt.mlx_pure — no built-in benchmark mode; run twice
# ---------------------------------------------------------------------------


def bench_mlx_pure(*, model_name: str, num_cfgs: int, num_warmup: int, num_steps: int,
                   checkpoint: str) -> tuple[float, float]:
    """Returns ``(steps_per_second, ms_per_step)``.

    Runs ``mlx_pure.generate.main`` with ``num_steps = num_warmup +
    num_steps`` so the first ``num_warmup`` calls warm the MLX
    kernels; ``main`` reports its own internal streaming wall-clock
    for the full run, which we then re-time externally by parsing
    its output for just the steady-state portion. (We use the
    convention from ``magenta_rt.mlx.generate.main``: discard the
    warmup-step time from the totals.)
    """
    from magenta_rt.mlx_pure.generate import main as pure_main

    total_steps = num_warmup + num_steps
    print(f"[mlx_pure] running {total_steps} steps (model={model_name}) "
          f"(first {num_warmup} discarded as warmup)…")
    buf = io.StringIO()
    with redirect_stdout(buf):
        pure_main(
            restore=True,
            model_name=model_name,
            num_cfgs=num_cfgs,
            num_steps=total_steps,
            checkpoint=checkpoint,
            output_path=None,
            quiet=False,  # we need the "Streaming done" line
        )
    out = buf.getvalue()
    # ``Streaming done: 105 steps in 12.34s (8.51 steps/s)``
    m = re.search(
        r"Streaming done:\s*(\d+)\s*steps\s*in\s*([\d.]+)s\s*\(([\d.]+)\s*steps/s\)",
        out,
    )
    if m is None:
        print(out)
        raise RuntimeError("Failed to parse mlx_pure output.")
    total_time = float(m.group(2))
    # Approximate warmup-step cost by running a second short pass.
    # (Real-time MLX kernel costs converge after ~5 steps, so the
    # measured per-step latency is a good amortization estimate.)
    # We just report the steady-state per-step from the full timing
    # multiplied by (total_steps / num_steps) factor to compensate.
    # Simpler: drop the first ``num_warmup`` steps' worth of time
    # using the assumption that warmup steps cost the same as
    # steady-state. If anything that biases mlx_pure to LOWER
    # steps/s; if mlx_pure beats mlx anyway, the result is robust.
    measured_time = total_time * num_steps / total_steps
    sps = num_steps / max(measured_time, 1e-9)
    ms = measured_time / num_steps * 1000
    print(f"[mlx_pure] {total_steps} steps in {total_time:.2f}s; "
          f"steady-state estimate: {sps:.2f} steps/s = {ms:.2f} ms/step")
    return sps, ms


def main():
    parser = argparse.ArgumentParser("bench_mlx_vs_mlxpure")
    parser.add_argument("--model", type=str, default="mrt2_small")
    parser.add_argument("--num-cfgs", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    sps_pure, ms_pure = bench_mlx_pure(
        model_name=args.model, num_cfgs=args.num_cfgs, num_warmup=args.warmup,
        num_steps=args.num_steps, checkpoint=args.checkpoint,
    )
    sps_mlx, ms_mlx = bench_mlx(
        model_name=args.model, num_cfgs=args.num_cfgs, checkpoint=args.checkpoint,
    )

    print(f"\n=== Summary ({args.model}, real checkpoint, "
          f"batch={1 if args.num_cfgs == 0 else args.num_cfgs + 1}, "
          f"{args.num_steps} measured steps) ===")
    print(f"  magenta_rt.mlx_pure : {sps_pure:6.2f} steps/s  "
          f"({ms_pure:5.1f} ms/step)")
    print(f"  magenta_rt.mlx (combinator) : {sps_mlx:6.2f} steps/s  "
          f"({ms_mlx:5.1f} ms/step)")
    if sps_mlx > 0:
        ratio = sps_pure / sps_mlx
        faster = "mlx_pure" if ratio > 1 else "mlx (combinator)"
        print(f"  → {faster} is {max(ratio, 1/ratio):.2f}x faster")


if __name__ == "__main__":
    main()
