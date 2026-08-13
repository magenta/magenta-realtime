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

"""Decoupled-AdamW weight-decay tests (DoRA-safe masking) for both SFT backends.

The decay policy (shared :func:`trainer_common._decays_weight`): decay every
trainable leaf EXCEPT the DoRA ``magnitude``, biases, and norm scales. Each
backend test runs N identical steps twice — once ``weight_decay=0``, once
``weight_decay>0`` (same seed/data, fixed LR) — and asserts:

  (a) the DoRA ``magnitude`` params are IDENTICAL between the two runs (excluded
      from decay), and
  (b) at least one decayed param (``lora_a``/``lora_b``) DIFFERS (the wd run is
      shrunk).

Plus a direct unit test of the predicate against the real leaf paths.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "notebooks", "sft"))

from magenta_rt.sft.trainer_common import _decays_weight  # noqa: E402


# ---- Predicate ------------------------------------------------------------

def test_decays_weight_policy():
    # Decayed: lora_a/lora_b, kernels, linear weights, embeddings.
    for p in (
        "decoder.depth.layers.ffn.ffn_layer1.lora_a..value",            # nnx
        "decoder.depth.layers.ffn.ffn_layer1.lora_b..value",            # nnx
        "decoder.depth.layers.ffn.ffn_layer1.base.kernel..value",       # nnx
        "decoder.embedder.embedding.embedding..value",                  # nnx
        "decoder.depth.layers.0.ffn.ffn_layer1.linear.lora_a",          # mlx
        "decoder.temporal.layers.0.ffn.ffn_layer1.linear.linear.weight",  # mlx linear
        "decoder.embedder.embedding.weight",                            # mlx embed
    ):
        assert _decays_weight(p) is True, p
    # Excluded: magnitude, biases, norm scales (nnx ``scale`` / mlx norm ``weight``).
    for p in (
        "decoder.depth.layers.ffn.ffn_layer1.magnitude..value",         # nnx
        "decoder.depth.layers.ffn.ffn_layer1.base.bias..value",         # nnx
        "decoder.depth.layers.ffn.pre_norm.scale..value",              # nnx norm
        "decoder.final_ln.scale..value",                               # nnx norm
        "decoder.depth.layers.0.ffn.ffn_layer1.linear.magnitude",       # mlx
        "decoder.temporal.layers.0.ffn.ffn_layer1.linear.linear.bias",  # mlx bias
        "decoder.temporal.layers.0.self_attn.pre_norm.weight",          # mlx norm
        "encoder.encoder_ln.weight",                                    # mlx norm
        "encoder.encoder_ln.bias",                                      # mlx bias
    ):
        assert _decays_weight(p) is False, p


# ---- MLX backend ----------------------------------------------------------

def test_weight_decay_dora_masking_mlx():
    """Two single steps on a FIXED batch, sharing the exact same grads, one with
    weight_decay=0 and one with weight_decay>0; assert the decay shrinks lora but
    leaves the DoRA magnitude untouched.

    We drive the trainer's own optimizer + ``_apply_decoupled_decay`` on a fixed
    ``mx.array`` batch rather than going through ``train()``/the grain pipeline:
    the pipeline's MusicCoCa-sticky / CFG-dropout RNG is not bit-reproducible
    across two iterator constructions, so two ``train()`` runs would not share
    grads and a magnitude change couldn't be attributed to decay vs grad drift.
    """
    # MLX is Apple-Silicon-only; skip on Windows / WSL2 / Linux.
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    import train_mlx as sft_mlx
    from magenta_rt.sft import lora_mlx as L

    spec = sft_mlx.TinyPOCSpecMLX()
    Q = spec.target_tokens_config.rvq_truncation_level
    mx.random.seed(0)
    src = mx.random.randint(0, 4, (2, 8, spec.input_num_channels))
    tgt = mx.random.randint(0, spec.target_tokens_config.codebook_size, (2, 8, Q))
    lr, wd = 1e-2, 0.5

    # build_model draws fresh random base weights each call, so snapshot ONE
    # adapter-injected model's params and restore them before each step — both
    # runs then start from identical weights and see identical grads.
    base = sft_mlx.build_model(spec)
    L.inject_lora(base, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=True, seed=0)
    L.mark_lora_trainable(base)
    snapshot = mlx.utils.tree_flatten(base.parameters())

    def step(weight_decay):
        model = sft_mlx.build_model(spec)
        L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                      dora=True, seed=0)
        L.mark_lora_trainable(model)
        model.update(mlx.utils.tree_unflatten(snapshot))  # identical start
        model.train()
        optimizer = optim.AdamW(learning_rate=lr, betas=[0.9, 0.95],
                                weight_decay=0.0)
        loss_fn = sft_mlx.make_loss_fn(model)
        loss, grads = nn.value_and_grad(model, loss_fn)(src, tgt)
        optimizer.update(model, grads)
        if weight_decay > 0:
            mask = sft_mlx._build_decay_mask(model)
            sft_mlx._apply_decoupled_decay(model, mask, 1.0 - lr * weight_decay)
        mx.eval(model.state, optimizer.state)
        flat = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
        return {k: np.array(v) for k, v in flat.items()}

    p0 = step(0.0)
    p1 = step(wd)

    mag_keys = [k for k in p0 if k.endswith("magnitude")]
    lora_keys = [k for k in p0 if k.endswith(("lora_a", "lora_b"))]
    assert mag_keys, "expected DoRA magnitude params"
    assert lora_keys, "expected lora_a/lora_b params"

    # (a) magnitude identical (excluded from decay; grads are identical).
    for k in mag_keys:
        assert np.array_equal(p0[k], p1[k]), (
            f"magnitude {k} changed under weight decay (should be excluded)"
        )
    # (b) decayed lora params are shrunk by exactly (1 - lr*wd).
    factor = 1.0 - lr * wd
    diffs = [k for k in lora_keys if not np.array_equal(p0[k], p1[k])]
    assert diffs, "no lora param differed under weight decay"
    for k in diffs:
        assert np.allclose(p1[k], p0[k] * factor, atol=1e-6), (
            f"lora {k} not shrunk by exactly (1 - lr*wd)"
        )


# ---- NNX backend ----------------------------------------------------------

def test_weight_decay_dora_masking_nnx():
    """NNX twin of the MLX test: one ``make_train_step`` on a FIXED batch with
    weight_decay=0 vs >0 (same model init + grads), built through the trainer's
    real ``build_optimizer`` (so the masked ``optax.add_decayed_weights`` lives in
    the chain exactly as in training).

    The decay is composed INTO the optax update here (decoupled AdamW), so the
    decayed lora params satisfy ``p1 = p0 - lr·wd·p0_init`` relative to the wd=0
    update — we don't assert the exact factor (that's the MLX path's manual
    form); we assert magnitude is bit-identical and lora differs.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import train_nnx as sft
    from magenta_rt.sft.configs import SFTConfig, TinyPOCSpec
    from magenta_rt.sft import lora_nnx as L

    spec = TinyPOCSpec()
    object.__setattr__(spec, "dtype", jnp.float32)
    Q = spec.target_tokens_config.rvq_truncation_level
    rng = np.random.default_rng(0)
    src = jnp.asarray(rng.integers(0, 4, (2, 8, spec.input_num_channels)), jnp.int32)
    tgt = jnp.asarray(
        rng.integers(0, spec.target_tokens_config.codebook_size, (2, 8, Q)), jnp.int32,
    )

    def run(weight_decay):
        config = SFTConfig(
            learning_rate=1e-2, warmup_steps=0, rsqrt_timescale=10**9,
            lora_rank=4, lora_alpha=8.0, lora_all_linears=True, lora_dora=True,
            weight_decay=weight_decay, seed=0, max_grad_norm=1e9,
        )
        model = sft.build_model(spec, seed=config.seed, param_dtype=jnp.float32)
        L.inject_lora(model, rank=config.lora_rank, alpha=config.lora_alpha,
                      dora=True, targets=L.all_linear_targets, seed=config.seed)
        model.train()
        optimizer = sft.build_optimizer(model, config, wrt=L.MRTLoRAParam)
        train_step = sft.make_train_step(diff_filter=L.MRTLoRAParam)
        train_step(model, optimizer, src, tgt)
        st = nnx.state(model, L.MRTLoRAParam)
        out = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(st):
            key = ".".join(str(getattr(k, "key", k)) for k in path)
            out[key] = np.array(leaf)
        return out

    p0 = run(0.0)
    p1 = run(0.5)

    mag_keys = [k for k in p0 if "magnitude" in k]
    lora_keys = [k for k in p0 if "lora_a" in k or "lora_b" in k]
    assert mag_keys, "expected DoRA magnitude params"
    assert lora_keys, "expected lora_a/lora_b params"

    # (a) magnitude identical between runs (excluded from add_decayed_weights).
    for k in mag_keys:
        assert np.array_equal(p0[k], p1[k]), (
            f"magnitude {k} changed under weight decay (should be excluded)"
        )
    # (b) at least one decayed (lora) param differs.
    diffs = [k for k in lora_keys if not np.array_equal(p0[k], p1[k])]
    assert diffs, "no lora param differed under weight decay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
