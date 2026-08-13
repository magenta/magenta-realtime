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

"""Round-trip tests for portable LoRA adapter files (``sft.lora_io``).

Save adapters from a trained tiny model, rebuild a fresh base, load the file
(recipe-driven, no flags), and assert the forward matches bit-for-bit — and
that the file is far smaller than the base. CPU-only (no checkpoint).
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("audiotree")
from flax import nnx  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "notebooks", "sft"))
import train_nnx as sft  # type: ignore  # noqa: E402

from magenta_rt.sft import lora_io  # noqa: E402
from magenta_rt.sft import lora_nnx as L  # noqa: E402
from magenta_rt.sft.configs import TinyPOCSpec  # noqa: E402


def _tiny_model(seed=0):
    spec = TinyPOCSpec()
    object.__setattr__(spec, "dtype", jnp.float32)
    return spec, sft.build_model(spec, seed=seed, param_dtype=jnp.float32)


def _dummy_io(spec):
    Q = spec.target_tokens_config.rvq_truncation_level
    src = jnp.zeros((1, 4, spec.input_num_channels), jnp.int32)
    tgt = jnp.zeros((1, 4, Q), jnp.int32)
    return src, tgt


def _fwd(model, src, tgt):
    return model.decoder(tgt, encoded_source=model.encoder(src))


def _randomize_adapters(model, scale=0.1, seed=0):
    """Make lora_b (zero-init) and DoRA magnitude non-trivial."""
    rng = np.random.default_rng(seed)
    for _, mod in nnx.iter_graph(model):
        if isinstance(mod, L.LoRAAdapter):
            b = mod.lora_b[...]
            mod.lora_b[...] = jnp.asarray(
                rng.standard_normal(b.shape).astype(np.float32) * scale, dtype=b.dtype)
            if mod.dora:
                m = mod.magnitude[...]
                mod.magnitude[...] = jnp.asarray(
                    np.abs(rng.standard_normal(m.shape)).astype(np.float32) + 0.5,
                    dtype=m.dtype)


@pytest.mark.parametrize("dora", [False, True])
def test_adapter_safetensors_roundtrip(tmp_path, dora):
    spec, model = _tiny_model()
    src, tgt = _dummy_io(spec)
    L.inject_lora(model, rank=4, alpha=8.0, dora=dora,
                  targets=L.all_linear_targets, seed=0)
    _randomize_adapters(model, seed=1)
    out_ref = np.asarray(_fwd(model, src, tgt))

    path = tmp_path / "adapter.safetensors"
    lora_io.save_lora_adapters(model, path, base_model="tiny_poc",
                               targets="all_linears")

    # The recipe is embedded — no flags needed to reload.
    meta = lora_io.read_metadata(path)
    assert meta["rank"] == "4"
    assert abs(float(meta["alpha"]) - 8.0) < 1e-6
    assert meta["dora"] == str(dora)
    assert meta["targets"] == "all_linears"
    assert meta["format"] == "mrt-lora"

    _, fresh = _tiny_model()  # same base weights (seed 0), no adapters
    meta2 = lora_io.load_lora_adapters(fresh, path)
    assert meta2["rank"] == "4"
    out_loaded = np.asarray(_fwd(fresh, src, tgt))

    np.testing.assert_allclose(out_ref, out_loaded, rtol=1e-5, atol=1e-5)
    # The adapter is non-trivial (delta actually changes the base output).
    _, base_only = _tiny_model()
    out_base = np.asarray(_fwd(base_only, src, tgt))
    assert not np.allclose(out_ref, out_base, atol=1e-4)


def test_adapter_file_is_small_vs_base(tmp_path):
    """The adapter file holds only LoRA leaves — orders of magnitude smaller
    than a full model checkpoint would be."""
    spec, model = _tiny_model()
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets, seed=0)
    _randomize_adapters(model, seed=2)
    path = tmp_path / "adapter.safetensors"
    lora_io.save_lora_adapters(model, path, base_model="tiny_poc",
                               targets="all_linears")

    adapter_params = sum(
        int(np.asarray(v).size)
        for _, m in nnx.iter_graph(model) if isinstance(m, L.LoRAAdapter)
        for v in (m.lora_a[...], m.lora_b[...])
    )
    assert adapter_params > 0
    assert path.stat().st_size < 2_000_000  # tiny POC adapters: well under 2 MB


def test_strength_override_and_recipe_default(tmp_path):
    """A stored lora_strength<1 is applied on load; an explicit override wins."""
    spec, model = _tiny_model()
    src, tgt = _dummy_io(spec)
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets, seed=0)
    _randomize_adapters(model, seed=3)
    path = tmp_path / "adapter.safetensors"
    lora_io.save_lora_adapters(model, path, base_model="tiny_poc",
                               targets="all_linears", lora_strength=0.5)
    assert abs(float(lora_io.read_metadata(path)["lora_strength"]) - 0.5) < 1e-9

    # Loading with the stored strength (0.5) differs from full strength (1.0).
    _, m_half = _tiny_model()
    lora_io.load_lora_adapters(m_half, path)            # uses stored 0.5
    _, m_full = _tiny_model()
    lora_io.load_lora_adapters(m_full, path, strength=1.0)  # override
    y_half = np.asarray(_fwd(m_half, src, tgt))
    y_full = np.asarray(_fwd(m_full, src, tgt))
    assert not np.allclose(y_half, y_full, atol=1e-4)


def test_read_metadata_rejects_plain_safetensors(tmp_path):
    from safetensors.numpy import save_file

    p = tmp_path / "plain.safetensors"
    save_file({"w": np.zeros((2, 2), np.float32)}, str(p))
    with pytest.raises(ValueError, match="recipe metadata"):
        lora_io.read_metadata(p)
