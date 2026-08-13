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

"""Direct JAX (Linen) ↔ mlx_pure parity on the real checkpoint, at fp32.

The sl-backed ``magenta_rt.mlx`` runtime is itself a port of the JAX
reference; the other parity tests in this directory pin ``mlx_pure``
against ``sl``. That leaves one blind spot: a bug *common to both* the
sl port and ``mlx_pure`` is invisible to an sl↔pure comparison. This
test closes it by going straight to the JAX/Linen source of truth.

It runs **both pipelines at fp32 compute** so the comparison is a clean
implementation-vs-implementation check with no bf16 quantization noise:

* JAX: the model spec's ``compute_dtype`` is overridden to fp32 (the
  shipped checkpoint params are already fp32).
* mlx_pure: the sl checkpoint loader's unconditional
  ``convert_to_bf16`` pass is monkeypatched to a no-op for the duration
  of the load, so the bridged depthformer keeps fp32 params; the pure
  spec is built with ``compute_dtype = param_dtype = float32``.

At fp32 the two frameworks agree to ~1e-5 (XLA vs Metal round-off), the
argmax never flips, and the autoregressive depth loop stays in
lock-step — so **all 12 codebooks** are directly comparable in a single
streaming step.

Why fp32 and not the bf16 production config
-------------------------------------------
At bf16 the pipelines diverge — but it was diagnosed to be numerical,
not structural: ``temporal_inputs`` is bit-exact, ``encoded_source``
agrees to ~1.5e-2, per-block QKV projections sit flat at ~1 bf16 ULP
without compounding, and pure's *own* bf16-vs-fp32 compute gap through
the temporal stack (``max|diff| ≈ 0.69``) is essentially the entire
JAX↔pure bf16 gap (``≈ 0.80``). bf16 rounding through a 24-block
residual stack accumulates real drift, and the discrete argmax in the
depth loop turns that into token desync past codebook ~4. fp32 removes
that noise floor, leaving a test that fails only on a genuine
implementation divergence.

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``; auto-skips
when JAX is unavailable or the checkpoint file is absent.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_NAME = "mrt2_small.safetensors"

# CFG / sampling settings — shared by both pipelines. Greedy (temp=0)
# keeps the depth loop deterministic. CFG is required: the JAX no-CFG
# path is broken (``_sample_categorical_with_temperature`` calls
# ``interleave_sequences`` with too few args when arity collapses to 1).
_CFG_MUSICCOCA = 3.0
_CFG_NOTES = 1.0
_MUSICCOCA = [679, 132, 480, 389, 160, 1010]  # "disco funk" mv212 tokens
_TOKEN_OFFSET = 7  # NUM_RESERVED_TOKENS + 1


@pytest.fixture
def smallm4air_checkpoint():
    from magenta_rt import paths as _paths
    p = pathlib.Path(_paths.resolve_checkpoint(_CHECKPOINT_NAME))
    if not p.exists():
        pytest.skip(f"checkpoint not found: {p}")
    return p


def _find_sown(tree: dict, key: str):
    """Depth-first search for a ``sow``-ed key in a Flax intermediates tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == key:
                return v
            hit = _find_sown(v, key)
            if hit is not None:
                return hit
    return None


def _cond_blocks(n_in: int):
    """``(pos, neg_musiccoca, neg_notes)`` conditioning rows, offset-applied."""
    notes = [-1] * (n_in - len(_MUSICCOCA))
    pos = np.array(_MUSICCOCA + notes, dtype=np.int32) + _TOKEN_OFFSET
    neg_musiccoca = np.array([-1] * len(_MUSICCOCA) + notes, dtype=np.int32) + _TOKEN_OFFSET
    neg_notes = np.array(_MUSICCOCA + [-1] * len(notes), dtype=np.int32) + _TOKEN_OFFSET
    return pos, neg_musiccoca, neg_notes


def _run_jax_fp32(checkpoint_path: Path) -> dict:
    """One JAX streaming step at fp32 compute. Returns sown intermediates
    as fp32 numpy: ``encoded_source`` ``[3,1,De]``, ``temporal_outputs``
    ``[3,1,Dt]``, ``depth_logits`` (list of ``[3,1,V]`` per codebook)."""
    import jax
    import jax.numpy as jnp
    from jax import random
    import sequence_layers.jax as sl
    from magenta_rt.jax import model as jm, system as jsys, spectrostream as jss
    from magenta_rt.jax.system import _load_jax_weights as load_jax_weights

    spec = jm.get_model_class("mrt2_small")()
    spec.compute_dtype = jnp.float32  # override bf16 -> fp32 (params are fp32)
    n_in = sum(c.rvq_truncation_level for c in spec.input_configs)

    mrt = jsys.MagentaRT2Sampler.Config(
        depthformer=spec.depthformer_config(),
        spectrostream=jss.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=spec.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        ),
    ).make()
    params = load_jax_weights(checkpoint_path)

    pos, neg_musiccoca, neg_notes = _cond_blocks(n_in)
    block = sl.Sequence.from_values(jnp.array(pos.reshape(1, 1, -1), dtype=jnp.int32))
    constants = {
        "temperature": jnp.array([0.0]),
        "top_k": jnp.array([1], dtype=jnp.int32),
        "classifier_free_guidance_scale_musiccoca": jnp.array([_CFG_MUSICCOCA]),
        "classifier_free_guidance_scale_notes": jnp.array([_CFG_NOTES]),
        "classifier_free_guidance_negative_musiccoca":
            sl.Sequence.from_values(jnp.array(neg_musiccoca.reshape(1, 1, -1), dtype=jnp.int32)),
        "classifier_free_guidance_negative_notes":
            sl.Sequence.from_values(jnp.array(neg_notes.reshape(1, 1, -1), dtype=jnp.int32)),
    }
    rngs = {"params": random.PRNGKey(42), "random": random.PRNGKey(0)}
    input_spec = jax.ShapeDtypeStruct([n_in], jnp.int32)

    state = mrt.apply(
        params, 1, input_spec, constants=constants, training=False,
        rngs=rngs, method=mrt.get_initial_state,
    )
    _, mutated = mrt.apply(
        params, x=block, state=state, constants=constants, training=False,
        rngs=rngs, method=mrt.step_with_emits, mutable=["intermediates"],
    )
    inter = mutated.get("intermediates", {})
    depth_logits = _find_sown(inter, "depth_logits")
    assert depth_logits is not None, "JAX did not sow 'depth_logits'"
    return {
        "encoded_source": np.asarray(_find_sown(inter, "encoded_source")[0], np.float32),
        "temporal_outputs": np.asarray(_find_sown(inter, "temporal_outputs")[0], np.float32),
        "depth_logits": [np.asarray(a, np.float32) for a in depth_logits],
    }


def _run_pure_fp32(checkpoint_path: Path) -> dict:
    """One mlx_pure streaming step at fp32 compute. Bridges the same
    checkpoint in with the sl loader's bf16 conversion suppressed, so
    the depthformer keeps fp32 params. Returns the matching
    intermediates as fp32 numpy, captured through the real code path."""
    import mlx.core as mx
    import sequence_layers.mlx as sl  # noqa: F401  (import side effects)
    from magenta_rt.mlx import model as cm, spectrostream as css, system as csys
    from magenta_rt.mlx import load_weights as lw_module
    from magenta_rt.mlx_pure import configs as pure_configs
    from magenta_rt.mlx_pure.load_weights import load_depthformer_weights

    cspec = cm.get_model_class("mrt2_small")()
    rvq = cspec.spectrostream.rvq_truncation_level
    combinator = csys.MagentaRT2Sampler.Config(
        depthformer=cspec.depthformer_config(),
        spectrostream=css.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq, use_unique_codes=False,
        ),
        int16_outputs=False,
    ).make()

    # Suppress the loader's unconditional fp32->bf16 depthformer cast so
    # the bridged params stay fp32. Restored in `finally`.
    _orig_convert = lw_module.convert_to_bf16
    lw_module.convert_to_bf16 = lambda _module: None
    try:
        lw_module.load_weights(
            combinator, checkpoint_path,
            num_input_channels=cspec.input_num_channels,
        )
    finally:
        lw_module.convert_to_bf16 = _orig_convert

    pspec = pure_configs.get_model_class("mrt2_small")()
    pspec_compute, pspec_param = pspec.compute_dtype, pspec.param_dtype
    pspec.compute_dtype = mx.float32
    pspec.param_dtype = mx.float32
    try:
        enc_dec = pspec.build_decoder()
    finally:  # restore class-level attrs (specs are shared classes)
        pspec.compute_dtype, pspec.param_dtype = pspec_compute, pspec_param
    load_depthformer_weights(enc_dec, combinator.depthformer)

    n_in = cspec.input_num_channels
    pos, neg_musiccoca, neg_notes = _cond_blocks(n_in)
    src = mx.array(np.stack([pos, neg_musiccoca, neg_notes], axis=0).reshape(3, 1, -1),
                   dtype=mx.int32)
    decoder = enc_dec.decoder

    # ---- encoder + temporal stack (deterministic) ----
    encoded = enc_dec.encode(src)
    state = enc_dec.make_initial_state(batch_size=3, seed=0)
    embedded = decoder._embed_tokens(state.previous_frame)
    temporal_inputs = decoder._temporal_input(embedded)
    temporal_out = decoder.temporal(
        temporal_inputs, source=encoded,
        self_caches=state.temporal.self_caches,
        cross_caches=state.temporal.cross_caches,
    )

    # ---- all 12 codebooks via the real step (spy on _logits) ----
    captured: list = []
    orig_logits = decoder._logits

    def _spy(d):
        out = orig_logits(d)
        captured.append(np.asarray(out.astype(mx.float32)))
        return out

    decoder._logits = _spy
    try:
        state2 = enc_dec.make_initial_state(batch_size=3, seed=0)
        enc_dec.step(
            state2, source_frame=encoded,
            temperature=0.0, top_k=1,
            cfg_scales=[_CFG_MUSICCOCA, _CFG_NOTES], cfg_arity=2,
        )
    finally:
        decoder._logits = orig_logits

    return {
        "encoded_source": np.asarray(encoded.astype(mx.float32)),
        "temporal_outputs": np.asarray(temporal_out.astype(mx.float32)),
        "depth_logits": captured,
    }


@pytest.mark.checkpoint
@pytest.mark.slow
def test_jax_parity_one_step_fp32(smallm4air_checkpoint):
    """JAX/Linen ↔ mlx_pure must agree at fp32 for one streaming step:
    encoder output, temporal-decoder output, and every codebook's
    pre-soft-cap depth logits."""
    pytest.importorskip("jax")
    pytest.importorskip("sequence_layers.jax")

    j = _run_jax_fp32(smallm4air_checkpoint)
    p = _run_pure_fp32(smallm4air_checkpoint)

    # fp32 cross-framework (XLA vs Metal) round-off is ~1e-5 even after
    # the 24-block temporal stack; 1e-3 gives ~40x margin while still
    # catching any genuine implementation divergence (a wrong op /
    # transposed weight / missing scale lands orders of magnitude above
    # this).
    atol = 1e-3

    def _check(name, jx, px):
        jx = np.asarray(jx, np.float32)
        px = np.asarray(px, np.float32)
        assert jx.shape == px.shape, f"{name}: shape {jx.shape} vs {px.shape}"
        diff = float(np.abs(jx - px).max())
        assert diff < atol, (
            f"{name}: max|diff|={diff:.6f} >= atol={atol} "
            f"(|val|max={np.abs(jx).max():.2f})"
        )
        return diff

    _check("encoded_source", j["encoded_source"], p["encoded_source"])
    _check("temporal_outputs", j["temporal_outputs"], p["temporal_outputs"])

    assert len(j["depth_logits"]) == len(p["depth_logits"]), (
        f"codebook count mismatch: jax={len(j['depth_logits'])} "
        f"pure={len(p['depth_logits'])}"
    )
    worst = 0.0
    for q, (jl, pl) in enumerate(zip(j["depth_logits"], p["depth_logits"])):
        worst = max(worst, _check(f"depth_logits[codebook {q}]", jl, pl))
    # Sanity: we actually compared the full set, not an empty zip.
    assert len(p["depth_logits"]) >= 12, (
        f"expected >= 12 codebooks captured, got {len(p['depth_logits'])}"
    )
