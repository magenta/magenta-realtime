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

"""Vendor hook: makes bundled third-party packages importable.

For each vendored package, when the package is **not** already installed as a
proper package (e.g. via pip), this module adds its vendored directory to
``sys.path`` so that the bare ``import`` statements used throughout the
codebase work unchanged. If the package **is** already installed, the hook is
a no-op for it, so a proper pip-installed version always wins.

Vendored here:

- ``sequence_layers`` (``sequence-layers/sequence_layers/``) — the
  SequenceLayers library the jax/mlx backends are built on.
- ``audiotree`` (``audiotree/audiotree/``) — a *minimal* copy of the
  ``AudioTree`` pytree container (https://github.com/DBraun/audiotree).
"""

import importlib.util
import sys
from pathlib import Path

# (importable module name, directory under _vendor/ that contains that package)
_VENDORED = (
    ("sequence_layers", "sequence-layers"),
    ("audiotree", "audiotree"),
)


def install() -> None:
  """Put each vendored package on ``sys.path`` unless it is already installed."""
  vendor_root = Path(__file__).resolve().parent
  for module_name, subdir in _VENDORED:
    if importlib.util.find_spec(module_name) is not None:
      # Already importable (pip-installed or already vendored) — leave it.
      continue
    vendor_dir = vendor_root / subdir
    vendor_path = str(vendor_dir)
    if vendor_dir.is_dir() and vendor_path not in sys.path:
      sys.path.insert(0, vendor_path)
