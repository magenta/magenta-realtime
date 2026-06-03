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

"""Magenta RealTime 2 — real-time streaming audio generation."""

# Activate vendored sequence_layers if it is not installed as a package.
from magenta_rt._vendor import _vendor_hook
_vendor_hook.install()
del _vendor_hook

__version__ = "2.0.2"
__all__ = ["MagentaRT2Jax", "MagentaRT2Mlx", "MagentaRT2Mlxfn"]


def __getattr__(name):
  if name == "MagentaRT2Jax":
    try:
      from magenta_rt.jax.system import MagentaRT2System
    except ImportError as e:
      raise ImportError(
          "MagentaRT2Jax requires JAX dependencies. "
          "Install them with: pip install magenta-rt[jax]"
      ) from e
    return MagentaRT2System
  if name == "MagentaRT2Mlx":
    try:
      from magenta_rt.mlx.system import MagentaRT2System
    except ImportError as e:
      raise ImportError(
          "MagentaRT2Mlx requires MLX dependencies. "
          "Install them with: pip install magenta-rt[mlx]"
      ) from e
    return MagentaRT2System
  if name == "MagentaRT2Mlxfn":
    try:
      from magenta_rt.mlx.system import MagentaRT2SystemMlxfn
    except ImportError as e:
      raise ImportError(
          "MagentaRTV2SystemMlxfn requires MLX dependencies. "
          "Install them with: pip install magenta-rt[mlx]"
      ) from e
    return MagentaRT2SystemMlxfn
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
