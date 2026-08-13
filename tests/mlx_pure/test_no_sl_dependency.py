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

"""Regression test: ``mlx_pure`` runtime must not depend on ``sequence_layers.mlx``.

This test verifies that:
1. Importing every ``mlx_pure`` runtime module does NOT require
   ``sequence_layers.mlx`` to be importable.
2. A non-trivial forward pass (Transformer + LocalSelfAttention +
   FFN with sinks) runs without ``sequence_layers.mlx`` being
   imported into the test's process.

We test this by clearing any pre-existing ``sequence_layers.mlx``
imports from ``sys.modules`` and then running the imports / forward
pass in a clean state.
"""

import subprocess
import sys


def test_mlx_pure_imports_without_sequence_layers():
    code = """
import sys
import importlib

modules = [
    "magenta_rt.mlx_pure",
    "magenta_rt.mlx_pure.base",
    "magenta_rt.mlx_pure.cache",
    "magenta_rt.mlx_pure.layers",
    "magenta_rt.mlx_pure.attention",
    "magenta_rt.mlx_pure.transformer",
    "magenta_rt.mlx_pure.depthformer",
    "magenta_rt.mlx_pure.sample_utils",
    "magenta_rt.mlx_pure.configs",
    "magenta_rt.mlx_pure.model",
    "magenta_rt.mlx_pure.spectrostream",
    "magenta_rt.mlx_pure.signal",
    "magenta_rt.mlx_pure.conv",
    "magenta_rt.mlx_pure.export",
    "magenta_rt.mlx_pure.load_weights",
    "magenta_rt.mlx_pure.musiccoca",
    "magenta_rt.mlx_pure.mt3",
]
for name in modules:
    importlib.import_module(name)

sl_keys = [k for k in sys.modules if k == "sequence_layers.mlx" or k.startswith("sequence_layers.mlx.")]
assert not sl_keys, f"sequence_layers.mlx imported as a side-effect: {sl_keys}"
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_mlx_pure_forward_without_sequence_layers():
    """End-to-end forward pass via mlx_pure with sequence_layers.mlx
    removed from sys.modules. Confirms no late-import dependency."""
    code = """
import sys
import mlx.core as mx
from magenta_rt.mlx_pure.transformer import Transformer

t = Transformer(
    num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
    ffn_dim=32, max_past_horizon=3, num_sinks=1,
    use_cross_attention=True, cross_attn_source_features=16,
    cross_attn_max_past_horizon=3,
)
x = mx.random.normal((1, 5, 16))
src = mx.random.normal((1, 5, 16))
y = t(x, source=src)
assert y.shape == (1, 5, 16)

sl_keys = [k for k in sys.modules if k == "sequence_layers.mlx" or k.startswith("sequence_layers.mlx.")]
assert not sl_keys, f"sequence_layers.mlx imported by Transformer forward: {sl_keys}"
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
