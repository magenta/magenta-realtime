# Copyright 2025 The MT3 Authors.
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

"""Download pretrained MT3 checkpoints.

The original t5x checkpoints live in the public ``gs://mt3`` bucket. This
module downloads a checkpoint via the public GCS JSON/media APIs (no gcloud
required), converts it to a single safetensors file under
``magenta_rt.paths.mt3_dir()`` (default
``~/Documents/Magenta/magenta-rt-v2/resources/mt3``), and removes the raw
checkpoint files.

Usage: ``mrt mt3 download [model_type]`` (or ``python -m magenta_rt.mt3.download
[model_type]``).
"""

import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict

from absl import logging
import numpy as np

from magenta_rt import paths

GCS_BUCKET = "mt3"

MODEL_TYPES = ("mt3", "ismir2021", "ismir2022_small", "ismir2022_base")


def _list_objects(prefix: str):
    """List all object names under a prefix in the public GCS bucket."""
    names = []
    page_token = None
    while True:
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
            f"?prefix={urllib.parse.quote(prefix)}&maxResults=1000"
        )
        if page_token:
            url += f"&pageToken={page_token}"
        with urllib.request.urlopen(url) as response:
            listing = json.load(response)
        names.extend(item["name"] for item in listing.get("items", []))
        page_token = listing.get("nextPageToken")
        if not page_token:
            break
    return [name for name in names if not name.endswith("/")]


def _download_object(name: str, dest: Path):
    url = f"https://storage.googleapis.com/{GCS_BUCKET}/{urllib.parse.quote(name)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _load_t5x_checkpoint(checkpoint_dir: Path) -> Dict[str, np.ndarray]:
    """Load a t5x checkpoint directory into a flat dict of parameter arrays.

    Small parameters are stored inline in the msgpack ``checkpoint`` file;
    large ones are TensorStore (zarr) references next to it.
    """
    import tensorstore as ts
    from flax import serialization

    with open(checkpoint_dir / "checkpoint", "rb") as f:
        restored = serialization.msgpack_restore(f.read())
    target = restored["optimizer"]["target"]

    def _load(tree, prefix, out):
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict) and "kvstore" in value:
                # TensorStore spec; point its kvstore at the local download.
                spec = dict(value)
                kvstore = dict(spec["kvstore"])
                kvstore["driver"] = "file"
                kvstore["path"] = str(checkpoint_dir / kvstore["path"])
                spec["kvstore"] = kvstore
                out[path] = np.asarray(ts.open(spec, open=True).result().read().result())
            elif isinstance(value, dict):
                _load(value, path, out)
            else:
                out[path] = np.asarray(value)

    params = {}
    _load(target, "", params)
    return params


def download_model(model_type: str = "mt3") -> str:
    """Download an MT3 checkpoint and convert it to safetensors.

    Args:
        model_type: One of 'mt3', 'ismir2021', 'ismir2022_small',
            'ismir2022_base'.

    Returns:
        Path to the converted safetensors file.
    """
    from safetensors.numpy import save_file

    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {MODEL_TYPES}")

    cache_home = paths.mt3_dir()
    cache_home.mkdir(parents=True, exist_ok=True)
    safetensors_path = cache_home / f"mt3_{model_type}.safetensors"
    if safetensors_path.exists():
        return str(safetensors_path)

    prefix = f"checkpoints/{model_type}/"
    checkpoint_dir = cache_home / f"mt3_{model_type}_t5x_checkpoint"

    names = _list_objects(prefix)
    if not names:
        raise RuntimeError(f"No checkpoint files found at gs://{GCS_BUCKET}/{prefix}")
    logging.info("Downloading %d files from gs://%s/%s ...", len(names), GCS_BUCKET, prefix)
    for i, name in enumerate(names):
        _download_object(name, checkpoint_dir / name.removeprefix(prefix))
        if (i + 1) % 25 == 0:
            logging.info("Downloaded %d/%d files", i + 1, len(names))

    params = _load_t5x_checkpoint(checkpoint_dir)
    save_file(params, str(safetensors_path))
    shutil.rmtree(checkpoint_dir)
    logging.info("Saved %s", safetensors_path)

    return str(safetensors_path)


if __name__ == "__main__":
    download_model(sys.argv[1] if len(sys.argv) > 1 else "mt3")
