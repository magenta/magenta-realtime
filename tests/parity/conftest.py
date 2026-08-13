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

"""Checkpoint-free cross-framework parity against the JAX/Linen ground truth.

``magenta_rt.jax`` (Flax Linen) is the reference implementation; ``magenta_rt.nnx``
(Flax NNX) and ``magenta_rt.mlx_pure`` (pure MLX) are ports that must match it
numerically. The real numerical guarantee otherwise lives only in the
``*/parity/test_jax_logit_parity.py`` tests, which need the multi-GB
``mrt2_small`` checkpoint and therefore **skip in CI**.

These tests close that gap with a checkpoint-free counterpart at the ``tiny``
preset: a tiny depthformer is randomly initialized in nnx (seeded), its weights
are exported to the Linen safetensors interchange format, and that one file is
loaded into jax (ground truth), nnx, and mlx_pure. One fp32 streaming step runs
through each and the encoder output, temporal-decoder output, and per-codebook
pre-soft-cap depth logits are compared to jax.

The random weights originate in nnx only because it has the cleanest rng-seeded
init + Linen exporter; jax merely loads and runs them, so it remains the
arbiter of correctness. Everything is deterministic (``nnx.Rngs(0)`` + greedy
``temperature=0``), so the comparison is reproducible run to run.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# CFG arity-2 (musiccoca + notes), greedy. The JAX no-CFG path is broken
# (``interleave_sequences`` arity collapse), so both pipelines run with CFG
# even though the tiny model has a single input channel.
CFG_A = 3.0
CFG_B = 1.0

# fp32 cross-framework round-off through the tiny stack is ~1e-6; 1e-3 keeps a
# ~1000x margin while still catching any genuine op/transpose/missing-bias
# divergence (which lands orders of magnitude above this).
ATOL = 1e-3


def find_sown(tree, key):
    """DFS for a ``sow``-ed key in a Flax intermediates tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == key:
                return v
            hit = find_sown(v, key)
            if hit is not None:
                return hit
    return None


def cond_blocks(n_in):
    """``(pos, neg_a, neg_b)`` conditioning rows for a tiny ``n_in``-channel
    input. Small valid ids (< codebook_size) keep encoder-embedding lookups in
    range; the two negatives just need to differ from ``pos`` for CFG."""
    base = np.arange(n_in, dtype=np.int32) % 20
    return base.copy(), (base + 3) % 20, (base + 7) % 20


@pytest.fixture(scope="session")
def tiny_linen_ckpt(tmp_path_factory):
    """A random-weight tiny depthformer in the Linen safetensors interchange
    format, produced by exporting a seeded nnx ``EncoderDecoder``. Shared by the
    jax / nnx / mlx_pure runners so all three load *identical* weights."""
    pytest.importorskip("jax")
    from flax import nnx
    from magenta_rt.nnx import model as nnx_configs
    from magenta_rt.nnx import depthformer as nnx_df
    from magenta_rt.sft.checkpoint import export_nnx_to_linen_safetensors

    spec = nnx_configs.get_model_class("tiny")()
    enc_dec = nnx_df.EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    out = tmp_path_factory.mktemp("parity") / "tiny_linen.safetensors"
    export_nnx_to_linen_safetensors(enc_dec, str(out))
    return out


def run_jax_fp32(ckpt):
    """One fp32 greedy streaming step of the JAX/Linen tiny depthformer.

    Returns the ground-truth encoder output, temporal-decoder output and the
    list of per-codebook pre-soft-cap depth logits (batch element 0), all as
    fp32 numpy, captured via the sown intermediates.
    """
    import jax
    import jax.numpy as jnp
    from jax import random
    import sequence_layers.jax as sl
    from magenta_rt.jax import model as jm
    from magenta_rt.jax.system import _load_jax_weights

    spec = jm.get_model_class("tiny")()
    spec.compute_dtype = jnp.float32  # params are already fp32
    n_in = sum(c.rvq_truncation_level for c in spec.input_configs)
    enc_dec = spec.depthformer_config().make()

    # The export nests depthformer params under ``params/depthformer/...``; the
    # EncoderDecoder is itself the ``apply`` target, so strip that level.
    params = {"params": _load_jax_weights(ckpt)["params"]["depthformer"]}

    pos, neg_a, neg_b = cond_blocks(n_in)
    block = sl.Sequence.from_values(jnp.asarray(pos.reshape(1, 1, -1)))
    constants = {
        "temperature": jnp.array([0.0]),
        "top_k": jnp.array([1], dtype=jnp.int32),
        "classifier_free_guidance_scale_musiccoca": jnp.array([CFG_A]),
        "classifier_free_guidance_scale_notes": jnp.array([CFG_B]),
        "classifier_free_guidance_negative_musiccoca":
            sl.Sequence.from_values(jnp.asarray(neg_a.reshape(1, 1, -1))),
        "classifier_free_guidance_negative_notes":
            sl.Sequence.from_values(jnp.asarray(neg_b.reshape(1, 1, -1))),
    }
    rngs = {"params": random.PRNGKey(0), "random": random.PRNGKey(0)}
    input_spec = jax.ShapeDtypeStruct([n_in], jnp.int32)

    def _init(mod):
        return mod.sampler.get_initial_state(
            1, input_spec, training=False, constants=constants,
        )

    def _step(mod, state):
        return mod.sampler.step(block, state, training=False, constants=constants)

    state = enc_dec.apply(params, rngs=rngs, method=_init)
    _, mutated = enc_dec.apply(
        params, state, rngs=rngs, method=_step, mutable=["intermediates"],
    )
    inter = mutated.get("intermediates", {})
    depth_logits = find_sown(inter, "depth_logits")
    assert depth_logits is not None, "JAX did not sow 'depth_logits'"
    return {
        "encoded_source": np.asarray(find_sown(inter, "encoded_source")[0], np.float32),
        "temporal_outputs": np.asarray(find_sown(inter, "temporal_outputs")[0], np.float32),
        "depth_logits": [np.asarray(a, np.float32) for a in depth_logits],
    }


@pytest.fixture(scope="session")
def jax_ground_truth(tiny_linen_ckpt):
    """fp32 JAX/Linen reference signals, computed once and shared by both the
    nnx and mlx_pure parity tests."""
    pytest.importorskip("sequence_layers.jax")
    gt = run_jax_fp32(tiny_linen_ckpt)
    # Sanity: the comparison must have teeth — the ground truth must be finite
    # and non-trivial, not all-zeros / NaN.
    enc = gt["encoded_source"]
    assert np.isfinite(enc).all(), "jax encoded_source has non-finite values"
    assert float(enc.std()) > 1e-3, "jax encoded_source is ~constant (no signal)"
    assert len(gt["depth_logits"]) >= 3, "jax produced < 3 codebooks of logits"
    return gt


def assert_matches_jax(jax_signals, other, *, label):
    """Assert another implementation's signals match the jax ground truth to
    ``ATOL`` on every signal, and that the right number of codebooks came back.
    Returns the worst per-signal ``max|diff|`` for optional logging."""
    jl, ol = jax_signals["depth_logits"], other["depth_logits"]
    assert len(ol) == len(jl), (
        f"{label}: codebook count {len(ol)} != jax {len(jl)}"
    )
    worst = 0.0

    def _one(name, a, b):
        nonlocal worst
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        assert a.shape == b.shape, f"{label} {name}: shape {a.shape} vs {b.shape}"
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        assert d < ATOL, (
            f"{label} {name}: max|diff|={d:.3e} >= {ATOL} "
            f"(|val|max={np.abs(a).max():.3f})"
        )

    _one("encoded_source", jax_signals["encoded_source"], other["encoded_source"])
    _one("temporal_outputs", jax_signals["temporal_outputs"], other["temporal_outputs"])
    for q, (a, b) in enumerate(zip(jl, ol)):
        _one(f"depth_logits[{q}]", a, b)
    return worst
