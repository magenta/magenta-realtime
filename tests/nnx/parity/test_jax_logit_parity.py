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

"""Direct JAX (Linen) ↔ flax.nnx parity on the real checkpoint, at fp32.

``magenta_rt.nnx`` is a flax.nnx port of the JAX/Linen reference
(``magenta_rt.jax``). The other parity tests here pin individual nnx
leaf layers against ``flax.linen`` primitives; this one pins the *whole
depthformer* against the JAX/Linen system end-to-end.

Both pipelines run at **fp32 compute** so the comparison is a clean
implementation-vs-implementation check with no bf16 quantization noise:

* JAX: the model spec's ``compute_dtype`` is overridden to fp32 (the
  shipped checkpoint params are already fp32).
* nnx: the spec is rebuilt with fp32 compute (``dtype``) by subclassing,
  and the depthformer is bridged from the same Linen safetensors checkpoint.

For one streaming generation step it checks the deterministic signals —
``encoded_source`` (encoder output), ``temporal_outputs`` (24-block
temporal stack), and all 12 codebooks of pre-soft-cap depth logits. At
fp32 the two backends agree to ~1e-5; the argmax never flips and the
autoregressive depth loop stays in lock-step, so every codebook is
directly comparable.

This test was written after it caught two real nnx bugs:
  * streaming cross-attention used an all-ones mask instead of
    ``cross_cache.make_mask`` — attending to unfilled cache slots
    (temporal output off by ~10);
  * the depth-body ``to_logits`` Linear was built ``use_bias=False``,
    so the loader silently dropped the checkpoint's logit bias (every
    logit shifted by up to ~0.65).

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``; auto-skips
when JAX is unavailable or the checkpoint file is absent.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest
from flax import nnx


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_NAME = "mrt2_small.safetensors"

# CFG / sampling settings — shared by both pipelines. Greedy (temp=0)
# keeps the depth loop deterministic. CFG is required: the JAX no-CFG
# path is broken (``_sample_categorical_with_temperature`` calls
# ``interleave_sequences`` with too few args when arity collapses to 1).
_CFG_MUSICCOCA = 3.0
_CFG_NOTES = 1.0
# 12 mv3 MusicCoCa tokens (matches scripts/generate_test_reference.py).
_MUSICCOCA = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
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
    """One JAX/Linen streaming step at fp32 compute. Returns sown
    intermediates as fp32 numpy: ``encoded_source`` ``[3,1,De]``,
    ``temporal_outputs`` ``[3,1,Dt]``, ``depth_logits`` (list of
    ``[3,1,V]`` per codebook)."""
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


def _run_nnx_fp32(checkpoint_path: Path) -> dict:
    """One ``magenta_rt.nnx`` streaming step at fp32 compute. Bridges the
    same Linen checkpoint into an fp32-built depthformer and returns the
    matching intermediates as fp32 numpy.

    The depth loop is replayed eagerly here (rather than calling
    ``decoder.step``) so each codebook's pre-soft-cap logits can be
    captured — ``step`` runs the loop inside ``nnx.scan`` and only
    returns sampled tokens. The eager replay mirrors ``step``'s
    ``depth_body`` exactly: ``soft_reset_caches`` then, per codebook,
    ``depth`` → ``_logits`` → soft-cap → greedy sample → embed → adapt.
    """
    import jax
    import jax.numpy as jnp
    from magenta_rt.nnx import depthformer as nnx_depthformer
    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.sample_utils import sample_categorical_with_temperature

    # nnx specs are plain (class-attribute) classes, not dataclasses, so
    # subclass to override ``dtype`` to fp32 compute. ``param_dtype`` stays
    # fp32 and the checkpoint params are fp32, so the depthformer runs in fp32.
    _base = nnx_model.get_model_class("mrt2_small")

    class _Fp32Spec(_base):
        dtype = jnp.float32

    spec = _Fp32Spec()
    enc_dec = nnx_depthformer.EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    target_cfg = spec.target_tokens_config
    # Only the depthformer is needed; pass codes_to_waveform so the
    # system builds without a SpectroStream and the loader skips the
    # codec subtree.
    mrt = nnx_model.MagentaRT2Sampler(
        depthformer_model=enc_dec,
        spectrostream=None,
        num_reserved_tokens=target_cfg.num_extra_tokens,
        codebook_size=target_cfg.codebook_size,
        int16_outputs=False,
    )
    mrt.load_checkpoint(checkpoint_path)

    n_in = sum(c.rvq_truncation_level for c in spec.input_configs)
    pos, neg_musiccoca, neg_notes = _cond_blocks(n_in)
    src = jnp.asarray(
        np.stack([pos, neg_musiccoca, neg_notes], axis=0).reshape(3, 1, -1).astype(np.int32)
    )

    decoder = mrt.depthformer.decoder
    encoded = mrt.depthformer.encoder(src)
    mrt.init_streaming(batch_size=3, rngs=nnx.Rngs(0), codec_streaming=True)

    # --- temporal stack (deterministic) ---
    temporal_inputs = decoder._temporal_input(
        decoder._embed_tokens(decoder.previous_frame[...])
    )
    temporal_out = decoder.temporal(temporal_inputs, source=encoded)

    # --- depth loop, eager replay of step()'s depth_body ---
    decoder.depth.soft_reset_caches()
    depth_input = decoder._adapt_depth(temporal_out)
    if decoder.dtype is not None:
        depth_input = depth_input.astype(decoder.dtype)
    depth_logits: list = []
    key = jax.random.key(0)
    for q in range(decoder.num_active_codebooks):
        depth_out = decoder.depth(depth_input)
        logits = decoder._logits(depth_out)  # pre-soft-cap
        depth_logits.append(np.asarray(logits, np.float32))
        cap = decoder.soft_cap_logits
        capped = jnp.tanh(logits / cap) * cap if cap is not None else logits
        min_v = decoder.num_reserved_tokens + q * decoder.codebook_size
        key, step_key = jax.random.split(key)
        sample_q = sample_categorical_with_temperature(
            capped.astype(jnp.float32), rng_key=step_key,
            temperature=0.0, top_k=1,
            cfg_scales=[_CFG_MUSICCOCA, _CFG_NOTES], cfg_arity=2,
            valid_range=(min_v, min_v + decoder.codebook_size),
        )
        depth_input = decoder._adapt_depth(
            decoder.embedder(sample_q[..., None]).squeeze(-2)
        )

    return {
        "encoded_source": np.asarray(encoded, np.float32),
        "temporal_outputs": np.asarray(temporal_out, np.float32),
        "depth_logits": depth_logits,
    }


@pytest.mark.checkpoint
@pytest.mark.slow
def test_nnx_parity_one_step_fp32(smallm4air_checkpoint):
    """JAX/Linen ↔ flax.nnx must agree at fp32 for one streaming step:
    encoder output, temporal-decoder output, and every codebook's
    pre-soft-cap depth logits."""
    pytest.importorskip("jax")
    pytest.importorskip("sequence_layers.jax")

    j = _run_jax_fp32(smallm4air_checkpoint)
    n = _run_nnx_fp32(smallm4air_checkpoint)

    # fp32 cross-implementation round-off is ~1e-5 even after the
    # 24-block temporal stack; 1e-3 gives ~30x margin while still
    # catching any genuine divergence (a wrong op / dropped bias /
    # transposed weight lands orders of magnitude above this).
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

    _check("encoded_source", j["encoded_source"], n["encoded_source"])
    _check("temporal_outputs", j["temporal_outputs"], n["temporal_outputs"])

    assert len(j["depth_logits"]) == len(n["depth_logits"]), (
        f"codebook count mismatch: jax={len(j['depth_logits'])} "
        f"nnx={len(n['depth_logits'])}"
    )
    for q, (jl, nl) in enumerate(zip(j["depth_logits"], n["depth_logits"])):
        _check(f"depth_logits[codebook {q}]", jl, nl)
    assert len(n["depth_logits"]) >= 12, (
        f"expected >= 12 codebooks captured, got {len(n['depth_logits'])}"
    )
