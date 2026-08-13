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

"""End-to-end POC test for the NNX SFT trainer.

Asserts:
  * grain pipeline yields the right shapes/dtypes
  * encoder freeze retypes Param → Frozen and removes them from `nnx.Param`
  * a few train_step iterations decrease the loss
  * orbax CheckpointManager round-trip restores identical model state
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

from magenta_rt.sft import create_audiotree_dataset, freeze_module, to_source_target
from magenta_rt.sft.configs import SFTConfig, TinyPOCSpec
from .test_utils import write_fake_tree_dataset
from magenta_rt.sft.freeze import Frozen
from magenta_rt.sft.lora_nnx import (
    LoRAAdapter,
    MRTLoRAParam,
    inject_lora,
    merge_lora_into_base,
)


def _make_dataset(tmpdir, spec):
    write_fake_tree_dataset(
        tmpdir,
        num_files=4,
        frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    return create_audiotree_dataset(
        tmpdir,
        batch_size=2,
        crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config,
        seed=0,
    )


def test_pipeline_shapes_and_dtypes(tmp_path):
    spec = TinyPOCSpec()
    ds = _make_dataset(str(tmp_path), spec)
    batch = next(iter(ds))
    source, target = to_source_target(batch, spec.target_tokens_config)
    crop_frames = int(2 * 25)
    assert source.shape == (2, crop_frames, spec.input_num_channels)
    assert target.shape == (
        2, crop_frames, spec.target_tokens_config.rvq_truncation_level,
    )
    assert source.dtype == np.int16
    assert target.dtype == np.int16
    # source min ≥ 0 (offsets + dropout sentinel applied)
    assert source.min() >= 0
    assert target.min() >= spec.target_tokens_config.num_extra_tokens - 1


def test_freeze_retypes_params(tmp_path):
    spec = TinyPOCSpec()
    model = sft.build_model(spec, seed=0)

    n_param_before = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    n_frozen_before = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, Frozen))
    )
    assert n_frozen_before == 0

    n_retyped = freeze_module(model.encoder)
    assert n_retyped > 0

    n_param_after = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    n_frozen_after = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, Frozen))
    )
    assert n_frozen_after > 0
    assert n_param_after == n_param_before - n_frozen_after


def test_trainstep_decreases_loss(tmp_path):
    spec = TinyPOCSpec()
    ds = _make_dataset(str(tmp_path), spec)
    config = SFTConfig(
        data_dir=str(tmp_path),
        crop_length_seconds=2, batch_size=2,
        total_steps=20, learning_rate=1e-3,
        warmup_steps=0, freeze_encoder=True,
    )
    model = sft.build_model(spec, seed=0)
    freeze_module(model.encoder)
    optimizer = sft.build_optimizer(model, config)
    train_step = sft.make_train_step()

    it = iter(ds)
    losses = []
    for _ in range(config.total_steps):
        source, target = to_source_target(
            next(it), spec.target_tokens_config, asarray=jnp.asarray,
        )
        m = train_step(model, optimizer, source, target)
        losses.append(float(m["loss"]))

    # First-five mean strictly greater than last-five mean.
    assert np.mean(losses[:5]) > np.mean(losses[-5:]), (
        f"loss did not decrease: first5={np.mean(losses[:5]):.3f} "
        f"last5={np.mean(losses[-5:]):.3f}"
    )


def test_embed_style_prompt_none_without_prompt():
    """embed_style_prompt is a no-op (returns None, spawns no subprocess) when
    config.style_prompt is unset — the common case."""
    from magenta_rt.sft import trainer_common

    assert trainer_common.embed_style_prompt(SFTConfig(), "mrt2_small") is None


def test_dropout_respects_train_eval_mode(tmp_path):
    """dropout_prob>0: a forward is stochastic under model.train() (dropout
    active, advancing the live RngState each call) and deterministic under
    model.eval() (nnx.Dropout's `deterministic` flag disables it). Tested on the
    model forward directly — make_eval_step is always deterministic (it flips to
    eval() internally), so it is not the right probe for train-mode stochasticity.
    """
    spec = TinyPOCSpec()
    ds = _make_dataset(str(tmp_path), spec)
    model = sft.build_model(spec, seed=0, dropout_prob=0.5)
    source, target = to_source_target(
        next(iter(ds)), spec.target_tokens_config, asarray=jnp.asarray,
    )
    loss_fn = sft.make_loss_fn()
    fwd = lambda: float(loss_fn(model, source, target)[0])

    model.eval()
    e1, e2 = fwd(), fwd()
    model.train()
    t1, t2 = fwd(), fwd()

    assert e1 == e2, "eval() must be deterministic (dropout disabled)"
    assert t1 != t2, "train() must be stochastic (dropout active)"


def test_dropout_off_by_default_is_deterministic(tmp_path):
    """The default build (dropout_prob=0) has no Dropout modules, so even under
    model.train() the forward is deterministic — the inference path is
    unaffected."""
    spec = TinyPOCSpec()
    ds = _make_dataset(str(tmp_path), spec)
    model = sft.build_model(spec, seed=0)  # dropout_prob defaults to 0.0
    source, target = to_source_target(
        next(iter(ds)), spec.target_tokens_config, asarray=jnp.asarray,
    )
    model.train()
    step = sft.make_eval_step()
    assert float(step(model, source, target)["eval/loss"]) == \
        float(step(model, source, target)["eval/loss"])


def test_orbax_save_restore(tmp_path):
    spec = TinyPOCSpec()
    config = SFTConfig(
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "ckpt"),
        crop_length_seconds=2, batch_size=2,
        total_steps=1, max_to_keep=2,
    )
    os.makedirs(config.data_dir, exist_ok=True)
    write_fake_tree_dataset(
        config.data_dir,
        num_files=4, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )

    model = sft.build_model(spec, seed=0)
    freeze_module(model.encoder)
    optimizer = sft.build_optimizer(model, config)
    ckpt_mgr = sft.open_ckpt_manager(config.output_dir, config.max_to_keep)

    train_step = sft.make_train_step()
    ds = create_audiotree_dataset(
        config.data_dir, batch_size=2, crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config, seed=0,
    )
    source, target = to_source_target(
        next(iter(ds)), spec.target_tokens_config, asarray=jnp.asarray,
    )
    train_step(model, optimizer, source, target)

    # Snapshot the FULL model + optimizer state (numpy copies — no aliasing),
    # save, corrupt both to zeros, restore, then assert every leaf round-trips
    # bit-exactly. The headline guarantee of this path is *exact resume*
    # including optimizer moments — not just one weight, and exactly (not
    # ``allclose``), since save/restore is a lossless serialization round-trip.
    def _to_np(x):
        x = jax.device_get(x)
        try:
            return np.asarray(x)
        except TypeError:  # PRNGKey-dtype leaves -> compare underlying ints
            return np.asarray(jax.random.key_data(x))

    def _state_leaves(*objs):
        return [_to_np(x) for o in objs for x in jax.tree.leaves(nnx.state(o))]

    def _corrupt(obj):
        def _zero(x):
            if not hasattr(x, "shape"):
                return x
            try:
                return jnp.zeros_like(x)
            except TypeError:  # leave PRNGKey leaves untouched
                return x
        nnx.update(obj, jax.tree.map(_zero, nnx.state(obj)))

    pre_model = _state_leaves(model)
    pre_opt = _state_leaves(optimizer)
    assert pre_opt, "optimizer should carry state after a train step"
    assert any(np.any(a) for a in pre_model), "model state is trivially all-zero"

    sft.save_state(ckpt_mgr, 1, model, optimizer, {"loss": 0.0})
    ckpt_mgr.wait_until_finished()

    # Corrupt both model and optimizer, proving restore actually overwrites them.
    _corrupt(model)
    _corrupt(optimizer)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(pre_model, _state_leaves(model))
    ), "corruption did not change model state"

    sft.restore_state(ckpt_mgr, 1, model, optimizer)

    post_model = _state_leaves(model)
    post_opt = _state_leaves(optimizer)
    assert len(post_model) == len(pre_model) and len(post_opt) == len(pre_opt)
    for a, b in zip(pre_model, post_model):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(pre_opt, post_opt):
        np.testing.assert_array_equal(a, b)
    ckpt_mgr.close()


def test_linen_safetensors_round_trip(tmp_path):
    """Build → export → load-into-fresh-model → bit-exact forward output."""
    from magenta_rt.nnx.depthformer import EncoderDecoder
    from magenta_rt.sft.checkpoint import (
        export_nnx_to_linen_safetensors,
        load_nnx_depthformer_from_safetensors,
    )

    spec = TinyPOCSpec()
    src = jnp.zeros((1, 4, spec.input_num_channels), jnp.int32)
    tgt = jnp.zeros((1, 4, spec.target_tokens_config.rvq_truncation_level), jnp.int32)

    ma = EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    ma.decoder.init_streaming(1, rngs=nnx.Rngs(0))
    ma.decoder(tgt, encoded_source=ma.encoder(src))
    out_a = ma.decoder(tgt, encoded_source=ma.encoder(src))

    path = str(tmp_path / "exported.safetensors")
    n = export_nnx_to_linen_safetensors(ma, path)
    assert n > 0

    mb = EncoderDecoder.from_config(spec, rngs=nnx.Rngs(99))  # different init
    mb.decoder.init_streaming(1, rngs=nnx.Rngs(99))
    mb.decoder(tgt, encoded_source=mb.encoder(src))
    load_nnx_depthformer_from_safetensors(mb, path)
    out_b = mb.decoder(tgt, encoded_source=mb.encoder(src))

    assert jnp.array_equal(out_a, out_b), (
        f"round-trip diverged: max diff = {float(jnp.abs(out_a - out_b).max()):.6f}"
    )


def test_lora_inject_is_identity_at_init():
    """B is zero-initialized → LoRA contribution is exactly zero at step 0,
    so injection (and a subsequent merge) must preserve forward outputs
    bit-exactly."""
    spec = TinyPOCSpec()
    model = sft.build_model(spec, seed=0)
    model.eval()
    src = jnp.zeros((1, 4, spec.input_num_channels), jnp.int32)
    tgt = jnp.zeros((1, 4, spec.target_tokens_config.rvq_truncation_level), jnp.int32)
    pre = model.decoder(tgt, encoded_source=model.encoder(src))

    n_wrapped = inject_lora(model, rank=4, alpha=0.0, seed=0)
    assert n_wrapped > 0
    post_inject = model.decoder(tgt, encoded_source=model.encoder(src))
    assert jnp.array_equal(pre, post_inject), "B=0 LoRA changed forward"

    n_merged = merge_lora_into_base(model)
    assert n_merged == n_wrapped
    post_merge = model.decoder(tgt, encoded_source=model.encoder(src))
    assert jnp.array_equal(post_inject, post_merge), "merge changed forward"


def test_lora_only_adapter_params_get_grads(tmp_path):
    """`wrt=MRTLoRAParam` must update only LoRA params; every other Param is
    byte-identical pre/post-train."""
    spec = TinyPOCSpec()
    write_fake_tree_dataset(
        str(tmp_path), num_files=4, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    config = SFTConfig(
        data_dir=str(tmp_path),
        crop_length_seconds=2, batch_size=2,
        total_steps=5, learning_rate=1e-2,
        lora_rank=4, lora_alpha=8.0,
    )

    model = sft.build_model(spec, seed=0)
    # Multiset of base-weight (shape, byte-hash) snapshots; inject_lora moves
    # the wrapped Linear under .base so paths change — we compare values
    # via multiset equality rather than path-keyed dicts.
    def _fingerprint(state):
        fps = []
        for leaf in jax.tree_util.tree_leaves(state):
            if hasattr(leaf, "shape"):
                fps.append((tuple(leaf.shape), bytes(np.asarray(leaf)).hex()))
        return sorted(fps)

    base_before = _fingerprint(nnx.state(model, nnx.Param))

    inject_lora(model, rank=config.lora_rank, alpha=config.lora_alpha, seed=0)
    optimizer = sft.build_optimizer(model, config, wrt=MRTLoRAParam)
    train_step = sft.make_train_step(diff_filter=MRTLoRAParam)

    ds = create_audiotree_dataset(
        str(tmp_path), batch_size=2, crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config, seed=0,
    )
    it = iter(ds)
    for _ in range(config.total_steps):
        source, target = to_source_target(
            next(it), spec.target_tokens_config, asarray=jnp.asarray,
        )
        train_step(model, optimizer, source, target)

    # Every non-LoRA Param post-train must match a pre-train Param (multiset).
    def _is_non_lora_param(path, leaf):
        return isinstance(leaf, nnx.Param) and not isinstance(leaf, MRTLoRAParam)
    base_after = _fingerprint(nnx.state(model, _is_non_lora_param))
    base_before_multiset = sorted(base_before)
    base_after_multiset = sorted(base_after)
    assert base_after_multiset == base_before_multiset, (
        "base params drifted under LoRA training"
    )

    # At least one LoRA param must have moved off zero.
    lora_after = jax.tree_util.tree_leaves(nnx.state(model, MRTLoRAParam))
    assert any(bool(jnp.any(l != 0)) for l in lora_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- tunix-inspired trainer features ----------------------------------------


def test_gradient_accumulation_updates_every_k_steps(tmp_path):
    """With optax.MultiSteps(k), params change only on every k-th micro-batch."""
    spec = TinyPOCSpec()
    ds = _make_dataset(str(tmp_path), spec)
    config = SFTConfig(
        crop_length_seconds=2, batch_size=2, learning_rate=1e-3,
        warmup_steps=0, gradient_accumulation_steps=3,
    )
    model = sft.build_model(spec, seed=0)
    freeze_module(model.encoder)
    optimizer = sft.build_optimizer(model, config)
    train_step = sft.make_train_step()

    def snapshot():
        return jax.tree.map(
            lambda x: np.asarray(x).copy(), nnx.state(model, nnx.Param)
        )

    it = iter(ds)
    before = snapshot()
    for micro_step in range(1, 7):
        source, target = to_source_target(
            next(it), spec.target_tokens_config, asarray=jnp.asarray,
        )
        train_step(model, optimizer, source, target)
        after = snapshot()
        changed = any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))
        )
        if micro_step % config.gradient_accumulation_steps == 0:
            assert changed, f"no update at accumulation boundary {micro_step}"
        else:
            assert not changed, f"params moved mid-accumulation at {micro_step}"
        before = after


def test_current_learning_rate_visible(tmp_path):
    """inject_hyperparams exposes the schedule's LR, with and without MultiSteps."""
    from magenta_rt.sft import trainer_common as sft_utils

    spec = TinyPOCSpec()
    for accum in (1, 2):
        config = SFTConfig(
            learning_rate=1e-3, warmup_steps=10,
            gradient_accumulation_steps=accum,
        )
        model = sft.build_model(spec, seed=0)
        optimizer = sft.build_optimizer(model, config)
        lr = sft.current_learning_rate(optimizer)
        assert lr is not None
        np.testing.assert_allclose(lr, sft_utils.lr_at_step(0, config), rtol=1e-6)


def test_lora_only_checkpoint_roundtrip(tmp_path):
    """lora_only checkpoints store just the adapters; restore leaves base alone."""
    spec = TinyPOCSpec()
    config = SFTConfig(output_dir=str(tmp_path / "ckpt"), learning_rate=1e-3)
    model = sft.build_model(spec, seed=0)
    inject_lora(model, rank=2, alpha=4.0)
    optimizer = sft.build_optimizer(model, config, wrt=MRTLoRAParam)

    # Give the adapters non-trivial values so the round trip is meaningful.
    rng = np.random.default_rng(0)
    for _, var in nnx.iter_graph(model):
        if isinstance(var, MRTLoRAParam):
            var.value = jnp.asarray(
                rng.standard_normal(var.value.shape).astype(np.float32)
            )

    adapter_before = jax.tree.map(
        lambda x: np.asarray(x).copy(), nnx.state(model, MRTLoRAParam)
    )
    base_before = jax.tree.map(
        lambda x: np.asarray(x).copy(), nnx.state(model, nnx.Param)
    )
    n_adapter_leaves = len(jax.tree.leaves(adapter_before))
    n_total_leaves = len(jax.tree.leaves(nnx.state(model)))
    assert n_adapter_leaves < n_total_leaves  # adapters are a strict subset

    ckpt_mgr = sft.open_ckpt_manager(config.output_dir, max_to_keep=1)
    try:
        sft.save_state(ckpt_mgr, 1, model, optimizer, {"loss": 0.0}, lora_only=True)
        ckpt_mgr.wait_until_finished()

        # Perturb the adapters, then restore.
        for _, var in nnx.iter_graph(model):
            if isinstance(var, MRTLoRAParam):
                var.value = var.value + 1.0
        sft.restore_state(ckpt_mgr, 1, model, optimizer, lora_only=True)
    finally:
        ckpt_mgr.close()

    adapter_after = nnx.state(model, MRTLoRAParam)
    for a, b in zip(jax.tree.leaves(adapter_before), jax.tree.leaves(adapter_after)):
        np.testing.assert_array_equal(a, np.asarray(b))
    base_after = nnx.state(model, nnx.Param)
    for a, b in zip(jax.tree.leaves(base_before), jax.tree.leaves(base_after)):
        np.testing.assert_array_equal(a, np.asarray(b))


def test_exact_resume_continues_data_stream(tmp_path):
    """Interrupted-and-resumed training matches an uninterrupted run exactly:
    orbax restores model + optimizer, and the grain iterator state restores
    the position in the shuffled/repeated data stream."""
    spec = TinyPOCSpec()
    write_fake_tree_dataset(
        str(tmp_path / "data"), num_files=6, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    config = SFTConfig(
        data_dir=str(tmp_path / "data"), crop_length_seconds=2, batch_size=2,
        learning_rate=1e-3, warmup_steps=0,
    )

    def make_dataset():
        return create_audiotree_dataset(
            config.data_dir, batch_size=2, crop_length_seconds=2,
            input_configs=spec.input_configs,
            target_config=spec.target_tokens_config, seed=0,
        )

    def fresh():
        model = sft.build_model(spec, seed=0)
        freeze_module(model.encoder)
        optimizer = sft.build_optimizer(model, config)
        return model, optimizer, sft.make_train_step()

    def run_steps(model, optimizer, train_step, it, n):
        losses = []
        for _ in range(n):
            source, target = to_source_target(
                next(it), spec.target_tokens_config, asarray=jnp.asarray,
            )
            losses.append(float(train_step(model, optimizer, source, target)["loss"]))
        return losses

    # Uninterrupted reference: 6 steps.
    model, optimizer, step_fn = fresh()
    ref = run_steps(model, optimizer, step_fn, iter(make_dataset()), 6)

    # Interrupted run: 3 steps, checkpoint (with iterator state), discard.
    model, optimizer, step_fn = fresh()
    it = iter(make_dataset())
    first = run_steps(model, optimizer, step_fn, it, 3)
    ckpt_mgr = sft.open_ckpt_manager(str(tmp_path / "ckpt"), max_to_keep=1)
    sft.save_state(ckpt_mgr, 3, model, optimizer, {"loss": first[-1]}, data_iter=it)
    ckpt_mgr.wait_until_finished()
    ckpt_mgr.close()
    del model, optimizer, step_fn, it

    # Fresh process-equivalent: rebuild everything, restore, run 3 more.
    model, optimizer, step_fn = fresh()
    it = iter(make_dataset())
    ckpt_mgr = sft.open_ckpt_manager(str(tmp_path / "ckpt"), max_to_keep=1)
    assert ckpt_mgr.latest_step() == 3
    sft.restore_state(ckpt_mgr, 3, model, optimizer, data_iter=it)
    ckpt_mgr.close()
    resumed = run_steps(model, optimizer, step_fn, it, 3)

    # Tolerance, not bitwise equality: GPU kernel selection/reduction order is
    # not reproducible across processes, so an identical computation can drift a
    # few 1e-4 in the loss (measured: 4e-4 absolute, 7.6e-5 relative, with the
    # first step matching exactly). What this test exists to catch — a data
    # stream that restarts or skips, or optimizer state that failed to restore —
    # moves the loss by O(0.1-1), three orders of magnitude above this bound.
    np.testing.assert_allclose(first, ref[:3], rtol=1e-3, atol=0)
    np.testing.assert_allclose(resumed, ref[3:], rtol=1e-3, atol=0)


def test_latest_checkpoint_survives_retention(tmp_path):
    """Retention must keep the newest checkpoint, not just the metric-best ones.

    Auto-resume keys off `latest_step()`. Ranking checkpoints by `best_fn`
    alone maps to a `BestN` policy with no latest-checkpoint protection, so the
    newest is garbage-collected on any step where it is not among the best —
    and the next launch silently rolls back to an older step and re-trains.
    Training loss is noisy, so "newest is not best" is the common case, not a
    corner one.
    """
    import orbax.checkpoint as ocp

    mgr = sft.open_ckpt_manager(str(tmp_path / "ckpt"), max_to_keep=2)
    # Loss gets monotonically worse, so the latest step is never metric-best.
    for step, loss in [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]:
        mgr.save(
            step,
            args=ocp.args.StandardSave({"x": np.array([float(step)])}),
            metrics={"loss": loss},
        )
    mgr.wait_until_finished()
    kept = sorted(mgr.all_steps())
    mgr.close()

    assert 4 in kept, f"latest checkpoint was collected (kept {kept}); resume would roll back"
    assert kept == [1, 2, 4], f"expected the 2 best + the latest, got {kept}"
