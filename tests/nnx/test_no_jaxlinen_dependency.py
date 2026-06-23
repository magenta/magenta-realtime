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

"""Verify the inference path of ``magenta_rt.nnx`` does not import
``magenta_rt.jax`` (the Linen-based sibling tree).

We can't realistically forbid ``flax.linen`` because ``flax``'s own
``__init__`` eagerly imports it as part of the public namespace —
that's a flax design choice. What we *can* lock in is that none of
the nnx runtime modules import the Linen-based ``magenta_rt.jax``
tree, which would defeat the point of having a separate nnx package.
"""

import subprocess
import sys


# Modules whose import path we want to keep linen-free.
_RUNTIME_MODULES = [
    "magenta_rt.nnx.attention",
    "magenta_rt.nnx.transformer",
    "magenta_rt.nnx.depthformer",
    "magenta_rt.nnx.spectrostream",
    "magenta_rt.nnx.spectrostream.model",
    "magenta_rt.nnx.model",
    "magenta_rt.nnx.signal",
    "magenta_rt.nnx.conv",
    "magenta_rt.nnx.cache",
    "magenta_rt.nnx.sample_utils",
    "magenta_rt.nnx.configs",
]


def test_runtime_modules_do_not_import_jax_tree():
    """None of the runtime nnx modules should pull in
    ``magenta_rt.jax`` (the Linen sibling tree) transitively.
    """
    code = f"""
import sys
import importlib

runtime_modules = {_RUNTIME_MODULES!r}
for name in runtime_modules:
    importlib.import_module(name)

forbidden_prefix = "magenta_rt.jax"
leaked = sorted(
    m for m in sys.modules
    if m == forbidden_prefix or m.startswith(forbidden_prefix + ".")
)
assert not leaked, f"Runtime nnx modules pulled in {{forbidden_prefix}}: {{leaked}}"
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_bridge_imports_are_lazy():
    """``magenta_rt.nnx.load_weights`` shouldn't pull in
    ``magenta_rt.jax`` at import time — only when the user actually
    invokes ``load_from_jax_safetensors`` (and even then, only the
    bridge helpers, not the Linen sibling tree).
    """
    code = """
import sys
import importlib

importlib.import_module("magenta_rt.nnx.load_weights")
leaked = sorted(m for m in sys.modules if m.startswith("magenta_rt.jax"))
assert not leaked, f"load_weights pulled in magenta_rt.jax at import time: {leaked}"
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
