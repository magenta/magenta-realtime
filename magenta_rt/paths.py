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

# HuggingFace repo that mirrors the MAGENTA_HOME layout (resources/, models/,
# checkpoints/). Single source of truth, shared with the download CLI.
HF_REPO_ID = "google/magenta-realtime-2"


def _hf_token() -> "Union[str, None]":
    """HuggingFace token from the environment, if set."""
    return os.environ.get("HF_TOKEN")


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

    Resolution order:
      1. ``filename`` as a literal filepath, if it exists.
      2. ``<MAGENTA_HOME>/checkpoints/<filename>``, if it exists.
      3. The same file reused from the global HuggingFace cache (no network
         fetch — an already-populated cache from ``hf download`` or
         ``mrt checkpoints download --use-hf-cache`` is picked up here).

    Args:
        filename: Checkpoint filename ending in `.safetensors`, or a literal path.

    Returns:
        Path to the checkpoint file. Falls back to the MAGENTA_HOME path (which
        may not exist yet) when the asset is neither local nor cached.
    """
    if os.path.isfile(filename):
        return pathlib.Path(filename)
    # NB: build the candidate path directly (do not call checkpoints_dir(), which
    # mkdir's as a side effect) — resolution must not create dirs, so a cache hit
    # leaves MAGENTA_HOME untouched.
    local = _MAGENTA_HOME / "checkpoints" / filename
    if local.exists():
        return local
    cached = _resolve_from_cache(f"checkpoints/{filename}", is_dir=False)
    return cached if cached is not None else local

# ---------------------------------------------------------------------------
# Asset resolution — MAGENTA_HOME layout first, global HuggingFace cache fallback
# ---------------------------------------------------------------------------
#
# Reads resolve through the global HF cache so an asset already fetched (e.g. via
# `hf download google/magenta-realtime-2` or `mrt models download --use-hf-cache`)
# is reused instead of re-downloaded. The MAGENTA_HOME layout always wins, so
# local exports / GCS downloads / user overrides take precedence. Resolution is
# cache-reuse only (no network fetch) unless `allow_download=True`.


def _resolve_from_cache(
    repo_relative: str, *, is_dir: bool, allow_download: bool = False
) -> "Union[pathlib.Path, None]":
    """Resolve a repo-relative asset from the global HF cache.

    Returns the cached path, or None if it is not cached (and not downloaded).
    """
    import huggingface_hub  # lazy: keep paths.py import-light / offline-safe

    try:
        if is_dir:
            root = huggingface_hub.snapshot_download(
                repo_id=HF_REPO_ID,
                allow_patterns=f"{repo_relative}/*",
                token=_hf_token(),
                local_files_only=not allow_download,
            )
            resolved = pathlib.Path(root) / repo_relative
            return resolved if resolved.exists() else None
        return pathlib.Path(
            huggingface_hub.hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=repo_relative,
                token=_hf_token(),
                local_files_only=not allow_download,
            )
        )
    except Exception:
        # Not in the cache (local_files_only) or unavailable from the hub.
        return None


def resolve_asset(
    repo_relative: str, *, is_dir: bool = False, allow_download: bool = False
) -> pathlib.Path:
    """Resolve a model asset: MAGENTA_HOME layout first, else the HF cache.

    Args:
        repo_relative: Path mirroring the HF repo layout, e.g.
            ``"resources/musiccoca"`` or ``"checkpoints/mrt2_base.safetensors"``.
        is_dir: True to resolve a directory of files, False for a single file.
        allow_download: If True, fetch from the hub when missing from the cache;
            otherwise reuse the cache only (the default for load paths).

    Returns:
        An existing local path to the asset.

    Raises:
        FileNotFoundError: if the asset is neither under MAGENTA_HOME nor cached.
    """
    local = _MAGENTA_HOME / repo_relative
    if local.exists():
        return local
    cached = _resolve_from_cache(
        repo_relative, is_dir=is_dir, allow_download=allow_download
    )
    if cached is not None:
        return cached
    raise FileNotFoundError(
        f"Could not find '{repo_relative}'. Looked in '{local}' and the global "
        f"HuggingFace cache for repo '{HF_REPO_ID}'. Download it with "
        f"`mrt models download` (add `--use-hf-cache` to populate the HF cache), "
        f"or `hf download {HF_REPO_ID}`."
    )


def resolve_musiccoca_dir() -> pathlib.Path:
    """MusicCoCa resource directory, resolved from MAGENTA_HOME or the HF cache."""
    return resolve_asset("resources/musiccoca", is_dir=True)


def resolve_spectrostream_dir() -> pathlib.Path:
    """SpectroStream resource directory, resolved from MAGENTA_HOME or the HF cache."""
    return resolve_asset("resources/spectrostream", is_dir=True)


def resolve_model_dir(name: str) -> pathlib.Path:
    """Exported `.mlxfn` model directory, resolved from MAGENTA_HOME or the HF cache."""
    return resolve_asset(f"models/{name}", is_dir=True)


def resolve_encoder_weights() -> pathlib.Path:
    """SpectroStream encoder.safetensors, resolved from MAGENTA_HOME or the HF cache."""
    return resolve_spectrostream_dir() / "encoder.safetensors"


def resolve_decoder_weights() -> pathlib.Path:
    """SpectroStream decoder.safetensors, resolved from MAGENTA_HOME or the HF cache."""
    return resolve_spectrostream_dir() / "decoder.safetensors"
