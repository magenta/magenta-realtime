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

"""CLI commands for the MT3 transcription model: ``mrt mt3 download``.

MT3 lives in a different place from the Magenta-RT assets handled by
``mrt models`` / ``mrt checkpoints`` (which pull from HuggingFace / the
``magenta-rt-public`` GCS bucket): the pretrained MT3 checkpoints are public in
``gs://mt3`` as raw t5x checkpoints. ``mrt mt3 download`` fetches one, converts
it to a single safetensors file under ``magenta_rt.paths.mt3_dir()``, and removes
the raw checkpoint. MT3 supplies the optional piano-roll conditioning channels
for SFT dataset export (``mrt sft export --transcribe``).
"""

import click
from absl import logging as absl_logging

from magenta_rt.cli import main
from magenta_rt.mt3.download import MODEL_TYPES


@main.group()
def mt3():
    """Manage MT3 transcription model checkpoints."""


@mt3.command()
@click.argument(
    "model_type",
    required=False,
    default="mt3",
    type=click.Choice(MODEL_TYPES, case_sensitive=False),
)
def download(model_type):
    """Download a pretrained MT3 checkpoint and convert it to safetensors.

    MODEL_TYPE defaults to 'mt3'; the others are 'ismir2021',
    'ismir2022_small', and 'ismir2022_base'. No-op if the converted
    safetensors already exists.
    """
    # Imported here (not at module top) so the heavy t5x/tensorstore path is
    # only loaded when the command actually runs.
    from magenta_rt.mt3.download import download_model

    # Surface download_model's absl-logged progress ("Downloading N files …").
    absl_logging.use_absl_handler()
    absl_logging.set_verbosity(absl_logging.INFO)

    click.echo(
        "📦 Downloading MT3 checkpoint "
        + click.style(model_type, fg="cyan")
        + " from gs://mt3 (this can take a few minutes) …"
    )
    path = download_model(model_type)
    click.echo(
        click.style(f"\n✓ MT3 '{model_type}' ready at {path}", fg="green", bold=True)
    )
