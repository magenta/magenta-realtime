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

"""Vendor hook: makes the bundled ``sequence_layers`` importable.

When ``sequence_layers`` is **not** already installed as a proper package
(e.g. via pip), this module adds the vendored submodule directory to
``sys.path`` so that bare ``import sequence_layers`` statements work
unchanged throughout the codebase.

If ``sequence_layers`` **is** already installed (i.e. importable), this hook
is a no-op, so a proper pip-installed version always wins.
"""

import importlib
import sys
from pathlib import Path


def install() -> None:
  """Add the vendored ``sequence-layers`` submodule to ``sys.path``."""
  try:
    importlib.import_module("sequence_layers")
    # Already available — nothing to do.
    return
  except ImportError:
    pass

  # The submodule lives at magenta_rt/_vendor/sequence-layers/ and contains
  # the sequence_layers/ Python package inside it.
  vendor_submodule = str(Path(__file__).resolve().parent / "sequence-layers")
  if Path(vendor_submodule).is_dir() and vendor_submodule not in sys.path:
    sys.path.insert(0, vendor_submodule)
