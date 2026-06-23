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

"""CLI commands for the flax.nnx backend: ``mrt nnx {generate}``."""
from pathlib import Path

import click

from magenta_rt.cli import main


@main.group()
def nnx():
    """flax.nnx backend commands."""


@nnx.command()
@click.option("--model", default=None, type=str,
              help="Model variant name (e.g. 'mrt2_small', 'mrt2_base').")
@click.option("--checkpoint", default=None, type=str,
              help="Checkpoint filename in checkpoints/ directory.")
@click.option("--skip-restore", is_flag=True, default=False,
              help="Use random weights.")
@click.option("--num-steps", default=5, type=int)
@click.option("--temperature", default=1.0, type=float)
@click.option("--top-k", default=40, type=int)
@click.option("--cfg-musiccoca", default=3.0, type=float)
@click.option("--cfg-notes", default=1.0, type=float)
@click.option("--num-cfgs", default=0, type=int,
              help="0 (default): CFG via trained conditioning tokens only "
                   "(single forward, matches jax/sl-mlx). 1/2: also do "
                   "logit-space CFG, which double-applies guidance.")
@click.option("--seed", default=0, type=int)
@click.option("--output", default=None, type=click.Path(),
              help="Output WAV path.")
@click.option("--jit/--no-jit", default=True,
              help="Wrap the streaming loop in nnx.jit.")
@click.option("--scan/--no-scan", default=True,
              help="Use nnx.scan inside --jit (worse performance without scan).")
def generate(model, checkpoint, skip_restore,
             num_steps, temperature, top_k, cfg_musiccoca, cfg_notes,
             num_cfgs, seed, output, jit, scan):
    """Run nnx inference."""
    from magenta_rt.nnx.generate import main as run

    kwargs = dict(
        restore=not skip_restore,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        num_cfgs=num_cfgs,
        seed=seed,
        jit=jit,
        scan=scan,
    )
    if model is not None:
        kwargs["model_name"] = model
    if checkpoint is not None:
        kwargs["checkpoint"] = checkpoint
    if output is not None:
        kwargs["output_path"] = Path(output)
    run(**kwargs)
