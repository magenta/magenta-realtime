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

"""CLI commands for the pure-MLX backend: ``mrt mlx_pure {generate, benchmark, export}``.

Mirrors :mod:`magenta_rt.cli.mlx_commands` but routes to
:mod:`magenta_rt.mlx_pure` so users can hit the sl-free runtime
path with the same UX.
"""
from pathlib import Path
import subprocess
import sys

import click

from magenta_rt.cli import main
from magenta_rt import paths


@main.group("mlx-pure")
def mlx_pure():
    """Pure-MLX backend commands (no ``sequence_layers.mlx`` runtime dep)."""


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@mlx_pure.command()
@click.option("--model", default=None, type=str,
              help="Model variant name (e.g. 'mrt2_small', 'mrt2_base').")
@click.option("--tiny", is_flag=True, default=False,
              help="Use the tiny test config (random weights).")
@click.option("--num-steps", default=100, type=int)
@click.option("--temperature", default=1.3, type=float)
@click.option("--top-k", default=40, type=int)
@click.option("--cfg-musiccoca", default=3.0, type=float)
@click.option("--cfg-notes", default=1.0, type=float)
@click.option("--num-cfgs", default=None, type=int,
              help="Number of CFGs: 0=disabled (1× batch), 1=MusicCoCa only (2×), 2=MusicCoCa+notes (3×).")
@click.option("--seed", default=0, type=int)
@click.option("--skip-restore", is_flag=True, default=False,
              help="Use random weights (skip the safetensors bridge).")
@click.option("--checkpoint", default=None, type=str,
              help="Checkpoint filename in checkpoints/ directory.")
@click.option("--output", default=None, type=click.Path(),
              help="Output WAV path.")
@click.option("--bits", default=None, type=click.Choice(["2", "3", "4", "5", "6", "8"]),
              help="Bit quantization level for depthformer Dense / EinsumDense layers.")
@click.option("--quantize-method", default="naive",
              type=click.Choice(["naive", "gptq"]),
              help="``naive`` = nearest-rounding ``quantize_in_place``; "
                   "``gptq`` = calibration-driven GPTQ.")
@click.option("--quantize-group-size", default=None, type=int)
@click.option("--gptq-cal-steps", default=8, type=int)
def generate(model, tiny, num_steps, temperature, top_k, cfg_musiccoca, cfg_notes,
             num_cfgs, seed, skip_restore, checkpoint, output, bits,
             quantize_method, quantize_group_size, gptq_cal_steps):
    """Generate audio with the pure-MLX backend."""
    from magenta_rt.mlx_pure.generate import main as run

    bits = int(bits) if bits else None

    kwargs = dict(
        restore=not skip_restore,
        tiny=tiny,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        seed=seed,
        bits=bits,
        quantize_method=quantize_method,
        gptq_cal_steps=gptq_cal_steps,
    )
    if model is not None:
        kwargs["model_name"] = model
    if num_cfgs is not None:
        kwargs["num_cfgs"] = num_cfgs
    if checkpoint is not None:
        kwargs["checkpoint"] = checkpoint
    if quantize_group_size is not None:
        kwargs["quantize_group_size"] = quantize_group_size
    if output is not None:
        kwargs["output_path"] = Path(output)
    run(**kwargs)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@mlx_pure.command()
@click.option("--model", default="mrt2_small", type=str)
@click.option("--num-steps", default=100, type=int,
              help="Number of measured streaming steps.")
@click.option("--warmup", default=5, type=int,
              help="Number of warmup steps before timing starts.")
@click.option("--num-cfgs", default=2, type=int)
@click.option("--checkpoint", default=None, type=str)
def benchmark(model, num_steps, warmup, num_cfgs, checkpoint):
    """Side-by-side latency benchmark vs ``magenta_rt.mlx`` (sl-backed)
    on the same model + checkpoint. Mirrors
    ``scripts/bench_mlx_vs_mlxpure.py``.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "bench_mlx_vs_mlxpure.py"
    cmd = [
        sys.executable, str(script),
        "--num-cfgs", str(num_cfgs),
        "--num-steps", str(num_steps),
        "--warmup", str(warmup),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@mlx_pure.command("export")
@click.option("--output-name", required=True, type=str,
              help="Name for the exported model (creates "
                   "<output-dir>/<name>/<name>.mlxfn).")
@click.option("--model", default="mrt2_small", type=str)
@click.option("--bits", default=None, type=click.Choice(["2", "3", "4", "5", "6", "8"]))
@click.option("--quantize-method", default="naive",
              type=click.Choice(["naive", "gptq"]))
@click.option("--gptq-cal-steps", default=8, type=int)
@click.option("--quantize-group-size", default=None, type=int)
@click.option("--num-cfgs", default=2, type=int)
@click.option("--output-dir", default=paths.models_dir(), type=str)
@click.option("--skip-restore", is_flag=True, default=False)
@click.option("--checkpoint", default=None, type=str)
def export_cmd(output_name, model, bits, quantize_method, gptq_cal_steps,
               quantize_group_size, num_cfgs, output_dir, skip_restore, checkpoint):
    """Export a streaming-step .mlxfn for the pure-MLX backend.

    The exported function takes the source-token block + dynamic
    constants + flat state arrays and returns the next audio chunk +
    new state arrays — analogous to the ``mrt mlx export`` flow.
    """
    from magenta_rt.mlx_pure.export import main as run
    bits = int(bits) if bits else None
    kwargs = dict(
        restore=not skip_restore,
        model_name=model,
        bits=bits,
        quantize_method=quantize_method,
        gptq_cal_steps=gptq_cal_steps,
        num_cfgs=num_cfgs,
        output_name=output_name,
        output_dir=output_dir,
    )
    if quantize_group_size is not None:
        kwargs["quantize_group_size"] = quantize_group_size
    if checkpoint is not None:
        kwargs["checkpoint"] = checkpoint
    run(**kwargs)
