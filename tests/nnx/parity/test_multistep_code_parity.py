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

"""Multi-step greedy code parity: ``magenta_rt.nnx`` vs JAX/Linen.

``test_jax_logit_parity.py`` pins *one* streaming step against the
Linen reference. This pins **many** steps — enough to push past the
cross-attention window (``max_past_horizon = 41``) so the KV-cache
rollover, the sliding-window mask, and the per-step streaming state
are all exercised. A bug that only manifests once the window fills
(off-by-one in the rolling buffer, a stale-slot mask — exactly the
class of the cross-attention mask bug found earlier) is invisible to a
single-step test; this catches it.

Both pipelines run at **fp32 compute, greedy** (``temperature=0`` /
``top_k=1``). Greedy makes sampling a deterministic argmax, and at fp32
the logits agree to ~1e-5 — far inside any logit gap — so the sampled
codes are *bit-exact* integers, step after step. (At bf16 the logit
noise flips argmaxes and the two pipelines desync after a few
codebooks — that's expected numerical behaviour, not a bug, which is
why this test is fp32.)

The codec is covered separately by ``test_spectrostream_jax_parity.py``;
bit-exact codes + a parity-tested codec means the end-to-end waveform
is covered transitively without the cross-framework int16-scaling /
batch-layout / lookahead-offset headaches of comparing waveforms
directly.

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
# > the cross-attention max_past_horizon (41) so window rollover is hit.
_NUM_STEPS = 45
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


def _jax_codes_fp32(checkpoint_path: Path, n_steps: int) -> np.ndarray:
    """Run ``n_steps`` of the JAX/Linen system at fp32 greedy; return the
    raw depthformer codes (pre ``convert_from_unique_codes``) sampled at
    each step, shape ``[n_steps, num_codebooks]`` (batch element 0)."""
    import jax
    import jax.numpy as jnp
    from jax import random
    import sequence_layers.jax as sl
    from magenta_rt.jax import model as jm, system as jsys, spectrostream as jss
    from magenta_rt.jax.system import _load_jax_weights as load_jax_weights

    spec = jm.get_model_class("mrt2_small")()
    spec.compute_dtype = jnp.float32
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
    state = mrt.apply(
        params, 1, jax.ShapeDtypeStruct([n_in], jnp.int32),
        constants=constants, training=False, rngs=rngs,
        method=mrt.get_initial_state,
    )
    codes = []
    for _ in range(n_steps):
        out, mutated = mrt.apply(
            params, x=block, state=state, constants=constants, training=False,
            rngs=rngs, method=mrt.step_with_emits, mutable=["intermediates"],
        )
        _, state, _ = out
        sown = _find_sown(mutated["intermediates"], "depth_samples")
        assert sown is not None, "JAX did not sow 'depth_samples'"
        step_codes = np.stack([np.asarray(a)[0, 0] for a in sown])
        codes.append(step_codes.astype(np.int64))
    return np.stack(codes)  # [n_steps, num_codebooks]


@pytest.mark.checkpoint
@pytest.mark.slow
def test_multistep_codes_match_jax_fp32(smallm4air_checkpoint):
    """45 streaming steps, fp32 greedy: every sampled code must match
    the JAX/Linen reference bit-exactly — through and past the
    cross-attention window rollover."""
    pytest.importorskip("jax")
    pytest.importorskip("sequence_layers.jax")
    import jax.numpy as jnp
    from magenta_rt.nnx import configs as nnx_configs
    from magenta_rt.nnx import depthformer as nnx_depthformer
    from magenta_rt.nnx import model as nnx_model

    jax_codes = _jax_codes_fp32(smallm4air_checkpoint, _NUM_STEPS)

    # fp32 nnx depthformer (spec rebuilt with dtype=float32 by subclassing —
    # nnx specs are plain class-attribute classes, not dataclasses);
    # only the depthformer is needed, so spectrostream is omitted.
    _base = nnx_configs.get_model_class("mrt2_small")

    class _Fp32Spec(_base):
        dtype = jnp.float32

    spec = _Fp32Spec()
    enc_dec = nnx_depthformer.EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    target_cfg = spec.target_tokens_config
    mrt = nnx_model.MagentaRT2Sampler(
        depthformer_model=enc_dec,
        spectrostream=None,
        num_reserved_tokens=target_cfg.num_extra_tokens,
        codebook_size=target_cfg.codebook_size,
        int16_outputs=False,
    )
    mrt.load_checkpoint(smallm4air_checkpoint)
    mrt.init_streaming(batch_size=3, codec_streaming=False, rngs=nnx.Rngs(0))

    n_in = sum(c.rvq_truncation_level for c in spec.input_configs)
    pos, neg_musiccoca, neg_notes = _cond_blocks(n_in)
    src = jnp.asarray(
        np.stack([pos, neg_musiccoca, neg_notes], axis=0).reshape(3, 1, -1).astype(np.int32)
    )

    nnx_codes = []
    for _ in range(_NUM_STEPS):
        codes = mrt.depthformer.step(
            source_tokens=src,
            temperature=0.0, top_k=1,
            cfg_scales=[_CFG_MUSICCOCA, _CFG_NOTES], cfg_arity=2,
        )
        nnx_codes.append(np.asarray(codes)[0, 0].astype(np.int64))
    nnx_codes = np.stack(nnx_codes)

    assert nnx_codes.shape == jax_codes.shape, (
        f"shape mismatch: jax {jax_codes.shape} vs nnx {nnx_codes.shape}"
    )
    mism = np.argwhere(nnx_codes != jax_codes)
    assert mism.size == 0, (
        f"{len(mism)}/{jax_codes.size} codes diverge from JAX/Linen "
        f"(fp32 greedy, {_NUM_STEPS} steps). First mismatches "
        f"(step, codebook): {[tuple(m) for m in mism[:8]]}"
    )
