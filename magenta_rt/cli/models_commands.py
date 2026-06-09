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

"""CLI commands for model management: mrt models {init,download}, mrt checkpoints {download}.

This module provides commands to fetch necessary models and resources
from HuggingFace (default) or Google Cloud Storage.

Commands:
  init     : Fetches shared base resources (e.g., musiccoca, spectrostream)
             required for general model operation.
  download : Fetches specific exported models. If no name is provided,
             an interactive prompt allows selection from available models.

Default Storage Location:
  All assets are downloaded to `~/Documents/Magenta/magenta-rt-v2/` by default.
  - `mrt models init` saves files into the `resources/` subdirectory.
  - `mrt models download` saves files into the `models/` subdirectory.
  - `mrt checkpoints download` saves files into the `checkpoints/` subdirectory.
  (This root path can be overridden using the --download-path option).
"""

import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import click

from magenta_rt.cli import main
from magenta_rt import paths

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE = os.environ.get('MAGENTA_RT_DOWNLOAD_SOURCE', 'hf')
_HF_TOKEN = os.environ.get('HF_TOKEN', None)

# GCS settings
_GCP_PROJECT = "brain-magenta"
_GCS_BUCKET = "magenta-rt-public"
_GCS_CHECKPOINTS_PREFIX = "magenta-rt-2"

# HuggingFace settings
_HF_REPO_NAME = 'google/magenta-realtime-2'

# Resources synced by `mrt models init`.
# Paths are the same in both GCS (under _GCS_CHECKPOINTS_PREFIX) and HF repo.
_INIT_RESOURCES = [
    ("resources/musiccoca", "resources/musiccoca"),
    ("resources/spectrostream", "resources/spectrostream"),
]

# Subdirectory containing downloadable models.
_MODELS_SUBDIR = "models"
_GCS_MODELS_PREFIX = f"{_GCS_CHECKPOINTS_PREFIX}/{_MODELS_SUBDIR}"

# Subdirectory containing raw safetensors checkpoints.
_CHECKPOINTS_SUBDIR = "checkpoints"
_GCS_CHECKPOINTS_DL_PREFIX = f"{_GCS_CHECKPOINTS_PREFIX}/{_CHECKPOINTS_SUBDIR}"

# Default local root for all downloaded assets.
_DEFAULT_DOWNLOAD_PATH = paths.magenta_home()


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _gcs_uri(suffix: str) -> str:
    """Build a full gs:// URI from a path suffix under the bucket."""
    return f"gs://{_GCS_BUCKET}/{suffix}"


def _get_storage_client():
    """Get a GCS storage client, or exit with a helpful error."""
    from google.cloud import storage  # noqa: E402

    # Disable mTLS to prevent "No module named 'OpenSSL'" errors when pyOpenSSL
    # is not installed in the environment.
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")

    try:
        return storage.Client.create_anonymous_client()
    except Exception as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to initialize GCS client: {e}",
            err=True,
        )
        sys.exit(1)


def _download_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download a GCS directory to a local directory."""
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    local_path.mkdir(parents=True, exist_ok=True)
    client = _get_storage_client()

    try:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        downloaded_any = False
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            relative_path = blob.name[len(prefix):]
            if not relative_path:
                continue

            local_file = local_path / relative_path

            downloaded_any = True
            click.echo(f"  Downloading {relative_path} …")

            # Create parent directories for the file if nested
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download to a temporary file first to prevent corrupted partial files
            temp_file = local_file.with_suffix(local_file.suffix + ".tmp")
            blob.download_to_filename(str(temp_file))
            temp_file.replace(local_file)

        if not downloaded_any:
            click.echo("  No files found to download.")

    except Exception as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to download from GCS: {e}\n"
            "Check your credentials, bucket permissions, and network connection.",
            err=True,
        )
        sys.exit(1)


def _list_gcs_dirs(gcs_prefix: str) -> list[str]:
    """List immediate subdirectory names under a GCS prefix."""
    prefix = gcs_prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    client = _get_storage_client()
    try:
        bucket = client.bucket(_GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=prefix, delimiter="/")

        # We must consume/iterate the blobs iterator to populate prefixes
        for _ in blobs:
            pass

        names = []
        for p in blobs.prefixes:
            # Strip trailing slash and get the basename
            name = p.rstrip("/").split("/")[-1]
            if name:
                names.append(name)
        return sorted(names)
    except Exception as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to list directories from GCS: {e}",
            err=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------


def _download_hf(repo_path: str, local_path: Path) -> None:
    """Download a directory from HuggingFace repo to a local directory.

    Args:
        repo_path: Path within the HF repo (e.g. "resources/musiccoca").
        local_path: Local destination that should mirror repo_path
            (e.g. ~/Documents/Magenta/magenta-rt-v2/resources/musiccoca).
    """
    import huggingface_hub  # noqa: E402

    local_path.mkdir(parents=True, exist_ok=True)

    # hf_hub_download saves to local_dir/filename.  Since filename is the
    # full repo-relative path (e.g. "resources/musiccoca/file.bin"),
    # local_dir must be the root *above* the repo_path portion of local_path
    # so that local_dir/filename == local_path/relative_file.
    repo_path_parts = Path(repo_path).parts
    local_dir = local_path
    for _ in repo_path_parts:
        local_dir = local_dir.parent

    try:
        # List all files under the repo path
        fs = huggingface_hub.HfFileSystem(token=_HF_TOKEN)
        hf_full_path = f"{_HF_REPO_NAME}/{repo_path}"
        entries = fs.find(hf_full_path, withdirs=False)

        if not entries:
            click.echo("  No files found to download.")
            return

        repo_prefix = f"{_HF_REPO_NAME}/"
        for entry in entries:
            # entry is like "google/magenta-realtime-2/resources/musiccoca/file.bin"
            rel_to_repo = entry[len(repo_prefix):]
            # rel_to_repo is like "resources/musiccoca/file.bin"
            if repo_path and not repo_path.endswith("/"):
                prefix_with_slash = repo_path + "/"
            else:
                prefix_with_slash = repo_path
            relative_path = rel_to_repo[len(prefix_with_slash):]

            click.echo(f"  Downloading {relative_path} …")

            huggingface_hub.hf_hub_download(
                repo_id=_HF_REPO_NAME,
                filename=rel_to_repo,
                local_dir=str(local_dir),
                token=_HF_TOKEN,
            )

    except Exception as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to download from HuggingFace: {e}\n"
            "Run `hf auth login` or set $HF_TOKEN if authentication is needed.",
            err=True,
        )
        sys.exit(1)


def _list_hf_dirs(repo_path: str) -> list[str]:
    """List immediate subdirectory names under a HuggingFace repo path."""
    import huggingface_hub  # noqa: E402

    try:
        fs = huggingface_hub.HfFileSystem(token=_HF_TOKEN)
        hf_full_path = f"{_HF_REPO_NAME}/{repo_path}"
        entries = fs.ls(hf_full_path, detail=True)

        names = []
        for entry in entries:
            if entry["type"] == "directory":
                name = entry["name"].rstrip("/").split("/")[-1]
                if name:
                    names.append(name)
        return sorted(names)
    except Exception as e:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Failed to list models from HuggingFace: {e}",
            err=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Unified download/list dispatchers
# ---------------------------------------------------------------------------


def _download(source: str, remote_path: str, local_path: Path) -> None:
    """Download from either GCS or HuggingFace."""
    if source == "gcs":
        _download_gcs(_gcs_uri(remote_path), local_path)
    elif source == "hf":
        _download_hf(remote_path, local_path)
    else:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Unknown source: {source!r}. Use 'hf' or 'gcs'.",
            err=True,
        )
        sys.exit(1)


def _list_models(source: str) -> list[str]:
    """List available model directories from either GCS or HuggingFace."""
    if source == "gcs":
        return _list_gcs_dirs(_GCS_MODELS_PREFIX)
    elif source == "hf":
        return _list_hf_dirs(_MODELS_SUBDIR)
    else:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Unknown source: {source!r}. Use 'hf' or 'gcs'.",
            err=True,
        )
        sys.exit(1)


def _list_checkpoint_files(source: str) -> list[str]:
    """List available checkpoint files from either GCS or HuggingFace."""
    if source == "gcs":
        prefix = _GCS_CHECKPOINTS_DL_PREFIX
        if not prefix.endswith("/"):
            prefix += "/"
        client = _get_storage_client()
        try:
            bucket = client.bucket(_GCS_BUCKET)
            blobs = bucket.list_blobs(prefix=prefix)
            names = []
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue
                relative = blob.name[len(prefix):]
                if relative:
                    names.append(relative)
            return sorted(names)
        except Exception as e:
            click.echo(
                click.style("Error: ", fg="red", bold=True)
                + f"Failed to list checkpoints from GCS: {e}",
                err=True,
            )
            sys.exit(1)
    elif source == "hf":
        import huggingface_hub  # noqa: E402
        try:
            fs = huggingface_hub.HfFileSystem(token=_HF_TOKEN)
            hf_full_path = f"{_HF_REPO_NAME}/{_CHECKPOINTS_SUBDIR}"
            entries = fs.ls(hf_full_path, detail=True)
            names = []
            for entry in entries:
                if entry["type"] == "file":
                    name = entry["name"].rstrip("/").split("/")[-1]
                    if name:
                        names.append(name)
            return sorted(names)
        except Exception as e:
            click.echo(
                click.style("Error: ", fg="red", bold=True)
                + f"Failed to list checkpoints from HuggingFace: {e}",
                err=True,
            )
            sys.exit(1)
    else:
        click.echo(
            click.style("Error: ", fg="red", bold=True)
            + f"Unknown source: {source!r}. Use 'hf' or 'gcs'.",
            err=True,
        )
        sys.exit(1)


def _interactive_select(
    models: list[str], downloaded: set[str]
) -> str | None:
    """Arrow-key interactive model selector.

    Shows a list of models with ✓ next to already-downloaded ones.
    Returns the selected model name, or None if cancelled.
    """
    from simple_term_menu import TerminalMenu

    entries = [
        f"{'✓ ' if name in downloaded else '  '}{name}" for name in models
    ]
    menu = TerminalMenu(
        entries,
        title="Select a model to download:",
        cursor_index=0,
        clear_screen=False,
    )
    idx = menu.show()
    return models[idx] if idx is not None else None


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


@main.group()
def models():
    """Manage model downloads."""


_source_option = click.option(
    "--source",
    type=click.Choice(["hf", "gcs"], case_sensitive=False),
    default=_DEFAULT_SOURCE,
    show_default=True,
    help="Download source: 'hf' (HuggingFace) or 'gcs' (Google Cloud Storage).",
)


@models.command()
@click.option(
    "--download-path",
    type=click.Path(),
    default=str(_DEFAULT_DOWNLOAD_PATH),
    show_default=True,
    help="Root directory for downloaded assets.",
)
@_source_option
def init(download_path, source):
    """Fetch all shared model resources (musiccoca, spectrostream)."""
    download_path = Path(download_path)
    source_label = "HuggingFace" if source == "hf" else "GCS"

    click.echo(
        click.style("Initializing model resources", bold=True)
        + f" from {source_label} → {download_path}"
    )

    for remote_suffix, local_subdir in _INIT_RESOURCES:
        if source == "gcs":
            remote_path = f"{_GCS_CHECKPOINTS_PREFIX}/{remote_suffix}"
        else:
            remote_path = remote_suffix
        dst = download_path / local_subdir
        click.echo(f"\n📦 Downloading {click.style(local_subdir, fg='cyan')} …")
        _download(source, remote_path, dst)

    click.echo(click.style("\n✓ Init complete.", fg="green", bold=True))


@models.command()
@click.argument("name", required=False, default=None)
@click.option(
    "--download-path",
    type=click.Path(),
    default=str(_DEFAULT_DOWNLOAD_PATH),
    show_default=True,
    help="Root directory for downloaded assets.",
)
@_source_option
def download(name, download_path, source):
    """Download an exported model by NAME.

    If NAME is omitted an interactive picker lists all available models.
    """
    download_path = Path(download_path)
    models_dir = download_path / "models"
    source_label = "HuggingFace" if source == "hf" else "GCS"

    if name is None:
        # Interactive selection
        click.echo(f"Fetching available models from {source_label} …")
        available = _list_models(source)

        if not available:
            click.echo(f"No models found on {source_label}.")
            return

        # Determine which models are already downloaded locally.
        downloaded = set()
        if models_dir.exists():
            downloaded = {
                p.name for p in models_dir.iterdir() if p.is_dir()
            }

        name = _interactive_select(available, downloaded)
        if name is None:
            click.echo("Cancelled.")
            return

    if source == "gcs":
        remote_path = f"{_GCS_MODELS_PREFIX}/{name}"
    else:
        remote_path = f"{_MODELS_SUBDIR}/{name}"
    dst = models_dir / name
    click.echo(
        f"\n📦 Downloading model {click.style(name, fg='cyan')}"
        f" from {source_label} → {dst} …"
    )
    _download(source, remote_path, dst)
    click.echo(click.style(f"\n✓ Model '{name}' downloaded.", fg="green", bold=True))


# ---------------------------------------------------------------------------
# Checkpoints commands
# ---------------------------------------------------------------------------


@main.group()
def checkpoints():
    """Manage raw model checkpoints (safetensors)."""


@checkpoints.command()
@click.argument("name", required=False, default=None)
@click.option(
    "--download-path",
    type=click.Path(),
    default=str(_DEFAULT_DOWNLOAD_PATH),
    show_default=True,
    help="Root directory for downloaded assets.",
)
@_source_option
def download(name, download_path, source):
    """Download a raw model checkpoint by NAME.

    If NAME is omitted an interactive picker lists all available checkpoints.
    """
    download_path = Path(download_path)
    ckpt_dir = download_path / "checkpoints"
    source_label = "HuggingFace" if source == "hf" else "GCS"

    if name is None:
        # Interactive selection
        click.echo(f"Fetching available checkpoints from {source_label} …")
        available = _list_checkpoint_files(source)

        if not available:
            click.echo(f"No checkpoints found on {source_label}.")
            return

        # Determine which checkpoints are already downloaded locally.
        downloaded = set()
        if ckpt_dir.exists():
            downloaded = {
                p.name for p in ckpt_dir.iterdir() if p.is_file()
            }

        name = _interactive_select(available, downloaded)
        if name is None:
            click.echo("Cancelled.")
            return

    # Accept a bare model name (e.g. "mrt2_small") as well as a full filename;
    # checkpoints are always .safetensors, mirroring `mrt models download`.
    if not name.endswith(".safetensors"):
        name = f"{name}.safetensors"

    # Download the single checkpoint file.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    local_file = ckpt_dir / name

    click.echo(
        f"\n📦 Downloading checkpoint {click.style(name, fg='cyan')}"
        f" from {source_label} → {local_file} …"
    )

    if source == "gcs":
        from google.cloud import storage  # noqa: E402
        client = _get_storage_client()
        bucket = client.bucket(_GCS_BUCKET)
        blob_path = f"{_GCS_CHECKPOINTS_DL_PREFIX}/{name}"
        blob = bucket.blob(blob_path)
        temp_file = local_file.with_suffix(local_file.suffix + ".tmp")
        blob.download_to_filename(str(temp_file))
        temp_file.replace(local_file)
    else:
        import huggingface_hub  # noqa: E402
        huggingface_hub.hf_hub_download(
            repo_id=_HF_REPO_NAME,
            filename=f"{_CHECKPOINTS_SUBDIR}/{name}",
            local_dir=str(download_path),
            token=_HF_TOKEN,
        )

    click.echo(click.style(f"\n✓ Checkpoint '{name}' downloaded.", fg="green", bold=True))
