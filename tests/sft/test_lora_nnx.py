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

"""Tests for the NNX LoRA / DoRA adapter (:mod:`magenta_rt.sft.lora_nnx`).

Fast unit tests (random-weight tiny model, no resources) covering the adapter
contract, parametrized over ``dora in {False, True}``:

  * zero-init identity — injection is a no-op at step 0 (``B=0``; for DoRA the
    magnitude init ``‖W‖_in`` makes ``W' = magnitude·W/‖W‖ = W``);
  * fuse/merge round-trip — a merged model is numerically identical to the
    adapted one (the key correctness property);
  * ``set_lora_strength(0)`` collapses every adapter back to the base.

These mirror ``tests/sft/test_lora_mlx.py`` on the NNX side. The NNX kernel is
``(..., in, out)`` (vs MLX's ``[out, in]``), so the DoRA per-output norm is over
the ``in`` axis (-2) — see ``lora_nnx._dora_kernel``.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax import nnx

# Make `notebooks/sft/` importable (train_nnx + its sibling `utils`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "sft"))

import train_nnx as sft  # type: ignore  # noqa: E402

from magenta_rt.sft.configs import TinyPOCSpec  # noqa: E402
from magenta_rt.sft import lora_nnx as L  # noqa: E402


def _tiny_model(seed=0):
    # Build in fp32 (params AND compute). The spec's default compute dtype is
    # bf16 (configs.MagentaRT2ModelBase.dtype), which makes the forward coarse
    # (~0.03 granularity) and swamps the DoRA fp32-normalize rounding and the
    # merge round-trip's contraction-order delta. fp32 gives the clean precision
    # regime the MLX twin tests run in. `dtype` isn't a declared dataclass field
    # so set it on the instance with object.__setattr__ (build_model does the
    # same for param_dtype).
    spec = TinyPOCSpec()
    object.__setattr__(spec, "dtype", jnp.float32)
    model = sft.build_model(spec, seed=seed, param_dtype=jnp.float32)
    model.eval()
    return spec, model


def _dummy_io(spec):
    Q = spec.target_tokens_config.rvq_truncation_level
    src = jnp.zeros((1, 4, spec.input_num_channels), jnp.int32)
    tgt = jnp.zeros((1, 4, Q), jnp.int32)
    return src, tgt


def _fwd(model, src, tgt):
    return model.decoder(tgt, encoded_source=model.encoder(src))


def _randomize_adapters(model, scale=0.1, seed=0):
    """Make every ``lora_b`` non-zero so the adapter delta is non-trivial
    (``lora_b`` is zero-init, which would make round-trip tests vacuous)."""
    rng = np.random.default_rng(seed)
    for _, mod in nnx.iter_graph(model):
        if isinstance(mod, L.LoRAAdapter):
            b = mod.lora_b[...]
            mod.lora_b[...] = jnp.asarray(
                rng.standard_normal(b.shape).astype(np.float32) * scale,
                dtype=b.dtype,
            )


def _adapter_count(model):
    return sum(1 for _, m in nnx.iter_graph(model) if isinstance(m, L.LoRAAdapter))


# ---- Adapter math ----------------------------------------------------------

@pytest.fixture
def exact_matmul():
    """Contract fp32 matmuls exactly for the duration of a test.

    The merge round-trip is an *algebraic* identity — folding the adapter into
    the kernel cannot change the output — so it is asserted at a tight 1e-5.
    Merged and unmerged forwards are different op sequences, though, and on
    Ampere-class GPUs (and TPU) JAX contracts fp32 matmuls in TF32/bf16 by
    default, whose ~1e-3 relative error swamps that bound. At default precision
    the assertion measures the accelerator's matmul mode rather than the merge,
    so pin precision here instead of loosening the tolerance — a real merge bug
    worth ~1e-4 must still fail on CPU, where 1e-5 is achievable.
    """
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize("dora", [False, True])
def test_zero_init_identity_all_linears(dora):
    spec, model = _tiny_model()
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)
    n = L.inject_lora(model, rank=4, alpha=8.0,
                      targets=L.all_linear_targets, dora=dora, seed=0)
    assert n > 0
    y1 = _fwd(model, src, tgt)
    # B=0 ⇒ LoRA delta is 0; DoRA's magnitude·W/‖W‖ = W. Identical to base at
    # step 0 (DoRA carries a tiny fp32 normalize/rescale rounding).
    # B=0 makes the delta exactly zero, but adding it still changes the op
    # sequence, so the result differs by fp32 ulps on some backends. The claim
    # is "the adapter is a no-op at init", not "the compiler emits identical
    # code" — same tolerance the MLX twin uses.
    assert jnp.allclose(y0, y1, atol=1e-4 if dora else 1e-5)


@pytest.mark.parametrize("dora", [False, True])
def test_zero_init_identity_default_targets(dora):
    spec, model = _tiny_model()
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)
    n = L.inject_lora(model, rank=4, alpha=8.0, dora=dora, seed=0)  # QKV default
    assert n > 0
    y1 = _fwd(model, src, tgt)
    # B=0 makes the delta exactly zero, but adding it still changes the op
    # sequence, so the result differs by fp32 ulps on some backends. The claim
    # is "the adapter is a no-op at init", not "the compiler emits identical
    # code" — same tolerance the MLX twin uses.
    assert jnp.allclose(y0, y1, atol=1e-4 if dora else 1e-5)


@pytest.mark.parametrize("dora", [False, True])
def test_set_lora_strength_zero_is_base(dora):
    """strength=0 collapses every adapter (LoRA and DoRA) back to the base."""
    spec, model = _tiny_model()
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)

    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=dora, seed=2)
    _randomize_adapters(model)
    # Adapter is active (forward moved off the base).
    assert float(jnp.max(jnp.abs(_fwd(model, src, tgt) - y0))) > 1e-4

    n = L.set_lora_strength(model, 0.0)
    assert n > 0
    assert jnp.allclose(_fwd(model, src, tgt), y0, atol=1e-4)


@pytest.mark.parametrize("dora", [False, True])
def test_set_lora_strength_count_matches_injected(dora):
    spec, model = _tiny_model()
    n_inject = L.inject_lora(model, rank=2, targets=L.all_linear_targets,
                             dora=dora, seed=0)
    n_set = L.set_lora_strength(model, 0.5)
    assert n_set == n_inject == _adapter_count(model)


# ---- DoRA magnitude is a trainable MRTLoRAParam ----------------------------

def test_dora_magnitude_is_lora_param():
    """The DoRA magnitude must be an MRTLoRAParam so it joins the trainable set
    and the adapter-only checkpoint (``wrt=MRTLoRAParam`` filters)."""
    spec, model = _tiny_model()
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=True, seed=0)
    mags = [
        m for _, m in nnx.iter_graph(model)
        if isinstance(m, L.LoRAAdapter)
    ]
    assert mags
    for adapter in mags:
        assert isinstance(adapter.magnitude, L.MRTLoRAParam)
        # Magnitude carries the kernel's leading dims + (out,) — norm over `in`.
        kshape = adapter.base.kernel[...].shape  # (..., in, out)
        assert adapter.magnitude[...].shape == kshape[:-2] + (kshape[-1],)

    # No magnitude leaves when dora=False.
    _, model2 = _tiny_model()
    L.inject_lora(model2, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=False, seed=0)
    for _, m in nnx.iter_graph(model2):
        if isinstance(m, L.LoRAAdapter):
            assert not hasattr(m, "magnitude")


# ---- Fuse / merge round-trip (the key correctness property) ----------------

@pytest.mark.parametrize("targets", [None, "all"])
@pytest.mark.parametrize("dora", [False, True])
def test_merge_round_trip(targets, dora, exact_matmul):
    spec, model = _tiny_model(seed=1)
    src, tgt = _dummy_io(spec)
    tfn = L.all_linear_targets if targets == "all" else None
    n = L.inject_lora(model, rank=4, alpha=8.0, targets=tfn, dora=dora, seed=1)
    assert n > 0
    _randomize_adapters(model, seed=3)

    y_wrapped = _fwd(model, src, tgt)
    n_merged = L.merge_lora_into_base(model)
    y_fused = _fwd(model, src, tgt)

    assert n_merged == n
    # Merged model must equal the adapted one. Plain LoRA folds the same matmul;
    # DoRA's effective-kernel forward vs matmul-with-folded-kernel differ only by
    # fp32 contraction order (tiny).
    tol = 1e-4 if dora else 1e-5
    assert float(jnp.max(jnp.abs(y_wrapped - y_fused))) <= tol
    # No adapter modules remain — plain inference module again.
    assert _adapter_count(model) == 0


@pytest.mark.parametrize("dora", [False, True])
def test_merge_round_trip_bf16_compute(dora):
    """Round-trip under the spec's default bf16 *compute* dtype (fp32 params).

    The DoRA forward builds W' in the param dtype then promotes (x, W', bias) to
    the compute dtype exactly as nnx.Linear does — so the wrapped forward tracks
    the merged-kernel forward even when compute != param dtype. Tolerance is the
    bf16 unit-in-last-place granularity (~0.03 near these magnitudes)."""
    spec = TinyPOCSpec()  # default dtype=bf16 compute, fp32 params
    model = sft.build_model(spec, seed=7)
    model.eval()
    src, tgt = _dummy_io(spec)
    n = L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                      dora=dora, seed=7)
    assert n > 0
    _randomize_adapters(model, seed=8)

    y_wrapped = _fwd(model, src, tgt)
    L.merge_lora_into_base(model)
    y_fused = _fwd(model, src, tgt)
    # bf16 ULP-scale: the wrapped and merged forwards take the same math path,
    # so any diff is just bf16 rounding of the (re)assembled kernel.
    assert float(jnp.max(jnp.abs(
        y_wrapped.astype(jnp.float32) - y_fused.astype(jnp.float32)))) <= 5e-2


@pytest.mark.parametrize("dora", [False, True])
def test_merge_honors_lora_strength(dora, exact_matmul):
    """Merging after set_lora_strength bakes the blend in: a strength-dialed
    wrapped forward equals the merged forward at that same strength."""
    spec, model = _tiny_model(seed=4)
    src, tgt = _dummy_io(spec)
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=dora, seed=4)
    _randomize_adapters(model, seed=5)
    L.set_lora_strength(model, 0.6)

    y_wrapped = _fwd(model, src, tgt)
    L.merge_lora_into_base(model)
    y_fused = _fwd(model, src, tgt)
    tol = 1e-4 if dora else 1e-5
    assert float(jnp.max(jnp.abs(y_wrapped - y_fused))) <= tol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
