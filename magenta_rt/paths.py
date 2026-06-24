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

"""Centralized path resolution for Magenta RT.

All paths resolve under MAGENTA_HOME/magenta-rt-v2 (where MAGENTA_HOME defaults to ~/Documents/Magenta).
Override with the MAGENTA_HOME environment variable.
"""

import os
import pathlib
from typing import Union


# Configurable root for all downloaded assets and models.
_MAGENTA_BASE = pathlib.Path(
    os.environ.get("MAGENTA_HOME", pathlib.Path.home() / "Documents" / "Magenta")
)
_MAGENTA_HOME = _MAGENTA_BASE / "magenta-rt-v2"

# Default model directory name (under ~/Documents/Magenta/magenta-rt-v2/models/).
DEFAULT_MODEL_NAME = "mrt2_base"
DEFAULT_CHECKPOINT = "mrt2_base.safetensors"


def magenta_home() -> pathlib.Path:
    """Returns the magenta home directory (default: ~/Documents/Magenta/magenta-rt-v2)."""
    return _MAGENTA_HOME


def set_magenta_home(path: Union[pathlib.Path, str]) -> None:
    """Override the magenta home directory at runtime."""
    global _MAGENTA_HOME
    if isinstance(path, str):
        path = pathlib.Path(path)
    _MAGENTA_HOME = path


# ---------------------------------------------------------------------------
# Resource directories
# ---------------------------------------------------------------------------


def resources_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/resources — shared resource files (musiccoca, spectrostream)."""
    return _MAGENTA_HOME / "resources"


def musiccoca_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/resources/musiccoca — MusicCoCa TFLite models."""
    return resources_dir() / "musiccoca"


def spectrostream_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/resources/spectrostream — SpectroStream weights."""
    return resources_dir() / "spectrostream"


def models_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/models — exported .mlxfn model directories."""
    return _MAGENTA_HOME / "models"


def default_model_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/models/<DEFAULT_MODEL_NAME> — the default model to load."""
    return models_dir() / DEFAULT_MODEL_NAME


def outputs_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/outputs — generation and export outputs."""
    d = _MAGENTA_HOME / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir() -> pathlib.Path:
    """~/Documents/Magenta/magenta-rt-v2/checkpoints — full safetensors from Linen models."""
    d = _MAGENTA_HOME / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_checkpoint(filename: str) -> pathlib.Path:
    """Resolve a checkpoint file path.

    First check if literal filepath exists; fallback to ~/Documents/Magenta/magenta-rt-v2/checkpoints/<filename>.

    Args:
        filename: Checkpoint filename ending in `.safetensors`

    Returns:
        Path to the checkpoint file (may not exist yet).
    """
    if os.path.isfile(filename):
        return filename
    return checkpoints_dir() / filename

# ---------------------------------------------------------------------------
# Path resolution without fallbacks — everything in ~/Documents/Magenta/magenta-rt-v2)
# ---------------------------------------------------------------------------

def resolve_encoder_weights() -> pathlib.Path:
    """Returns ~/Documents/Magenta/magenta-rt-v2/resources/spectrostream/encoder.safetensors."""
    return spectrostream_dir() / "encoder.safetensors"


def resolve_decoder_weights() -> pathlib.Path:
    """Returns ~/Documents/Magenta/magenta-rt-v2/resources/spectrostream/decoder.safetensors."""
    return spectrostream_dir() / "decoder.safetensors"
