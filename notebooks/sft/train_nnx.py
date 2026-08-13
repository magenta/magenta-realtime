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

# %% [markdown]
# # SFT Training (NNX) — POC
#
# Proof-of-concept supervised fine-tuning of the Magenta-RT V2 depthformer
# in `flax.nnx`. Pipeline:
#
# 1. `magenta_rt.sft.create_audiotree_dataset(...)` — `grain` pipeline of pure-numpy
#    AudioTree transforms (random crop, MusicCoCa sticky repeat, CFG dropout)
#    yielding batched AudioTrees; `to_source_target(...)` builds source/target at
#    the trainer boundary (encoding audio->tokens on device when needed).
# 2. Model: `EncoderDecoder.from_config(spec)` (random weights by default;
#    pass `--checkpoint` to load pretrained Linen safetensors).
# 3. Encoder freeze via `nnx.Param` → `Frozen` Variable retype, so
#    `nnx.Optimizer(model, tx, wrt=nnx.Param)` and
#    `nnx.value_and_grad(..., argnums=nnx.DiffState(0, nnx.Param))` both skip
#    encoder state entirely.
# 4. Loss: cross-entropy over `[B, T, Q, V]` logits.
# 5. Checkpointing via `orbax.checkpoint.CheckpointManager` (async + best-key;
#    adapter-only model state when LoRA is active).
# 6. Metric logging via `clu.metric_writers` + `periodic_actions.ReportProgress`.

# %%
from __future__ import annotations

import contextlib
import dataclasses
import functools
import glob
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from orbax import checkpoint as ocp

from magenta_rt.nnx.model import MODEL_REGISTRY
from magenta_rt.nnx.depthformer import EncoderDecoder
from magenta_rt.sft import (
    EarlyStopper,
    freeze_module,
    to_source_target,
)
from magenta_rt.sft.checkpoint import (
    export_nnx_to_linen_safetensors,
    load_nnx_depthformer_from_safetensors,
)
from magenta_rt.sft.configs import SFTConfig, TinyPOCSpec
from magenta_rt.sft.freeze import Frozen
from magenta_rt.sft.lora_nnx import (
    MRTLoRAParam,
    all_linear_targets,
    default_targets,
    inject_lora,
    set_lora_strength,
)

from magenta_rt.sft import trainer_common as utils  # shared, backend-neutral trainer glue

# Window-averaged metrics use `flax.nnx.MultiMetric` (via `utils.make_*_metrics`,
# shared with the MLX trainer). TensorBoard writing still uses the vendored,
# TF-free `magenta_rt.metric_writers` / `magenta_rt.periodic_actions` (a clu
# replacement that avoids clu's eager TensorFlow import); both stay lazy
# (imported in `_make_writer` / `train`).


# %% [markdown]
# ## Loss
#
# The warmup→rsqrt LR schedule lives in `utils.lr_at_step` (shared with
# `train_mlx`); it's threaded through `optax.inject_hyperparams` below so the
# current LR is readable from optimizer state (`current_learning_rate`).

# %%
def make_loss_fn():
    def loss_fn(model: EncoderDecoder, source: jax.Array, target: jax.Array):
        encoded = model.encode(source)
        logits = model.decoder(target, encoded_source=encoded)  # [B, T, Q, V]
        # Softmax in fp32: with a bf16 base (param_dtype=bfloat16 / config.bf16)
        # the logits are bf16, and the cross-entropy's log_softmax over the
        # multi-thousand-token vocab loses precision in bf16 — the cast keeps
        # the loss/gradients numerically clean (a no-op for the fp32 path).
        nll = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), target
        )
        loss = nll.mean()
        return loss, {"loss": loss}
    return loss_fn


# %% [markdown]
# ## Train / eval / generate steps

# %%
def mutable_filter(diff_filter):
    """The state that CHANGES each train step — threaded through the jitted step
    and donated: the trainable params (``diff_filter``), the optimizer state, and
    the dropout rng. Everything else — the frozen base params and the streaming
    ``DecodeState`` slots — is the held-out *constant* the step never copies.
    Relies on ``Frozen`` / non-``diff_filter`` params NOT matching ``diff_filter``
    (the same property the gradient filter already needs)."""
    return nnx.Any(diff_filter, nnx.OptState, nnx.RngState)


def make_train_step(diff_filter=nnx.Param):
    """A functional ``jax.jit`` train step behind the plain ``(model, optimizer)``
    API (the ``MagentaRT2System`` pattern — the split is internal).

    Each call splits the live ``(model, optimizer)`` into a *constant* partition
    (graphdef + frozen base + streaming ``DecodeState``) and a *mutable* partition
    (trainable params + optimizer state + dropout rng — see
    :func:`mutable_filter`). A jitted body — captured once, on the first call —
    merges them, runs fwd/bwd + ``optimizer.update``, and splits the updated
    mutable back out: ``constant`` is a non-donated arg it never returns (so the
    frozen base is NOT copied in+out each step) and ``mutable`` (arg 1) is
    donated. The wrapper writes the updated mutable into the live objects,
    preserving the in-place-update contract callers expect. Differentiates w.r.t.
    ``diff_filter`` (``MRTLoRAParam`` for LoRA, else ``nnx.Param`` with the
    encoder ``Frozen``).
    """
    loss_fn = make_loss_fn()
    mut = mutable_filter(diff_filter)
    jitted = None

    def _build(graphdef):
        @jax.jit(donate_argnums=1)
        def step(constant, mutable, source, target):
            model, optimizer = nnx.merge(graphdef, mutable, constant)
            (loss, aux), grads = nnx.value_and_grad(
                loss_fn, argnums=nnx.DiffState(0, diff_filter), has_aux=True,
            )(model, source, target)
            # Global grad norm for logging only — the actual clipping is done by
            # optax.clip_by_global_norm in build_optimizer's tx chain. Filter to
            # Array leaves so Field metadata is skipped.
            leaves = [
                x for x in jax.tree_util.tree_leaves(grads)
                if isinstance(x, jax.Array)
            ]
            grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
            optimizer.update(model, grads)
            _, new_mutable, _ = nnx.split((model, optimizer), mut, ...)
            return {"loss": loss, "grad_norm": grad_norm}, new_mutable
        return step

    def train_step(model, optimizer, source, target):
        nonlocal jitted
        graphdef, mutable, constant = nnx.split((model, optimizer), mut, ...)
        if jitted is None:  # capture the (train-mode) graphdef once
            jitted = _build(graphdef)
        metrics, mutable = jitted(constant, mutable, source, target)
        nnx.update((model, optimizer), mutable)
        return metrics
    return train_step


def make_eval_step():
    """Functional ``jax.jit`` eval step behind the plain ``(model, source,
    target)`` API. Splits the model, merges under the captured graphdef +
    ``model.eval()`` (a deterministic forward — dropout off — without toggling
    the live, train-mode model), and scores one batch. No optimizer, no
    donation."""
    jitted = None

    def _build(graphdef):
        @jax.jit
        def step(state, source, target):
            model = nnx.merge(graphdef, state)
            model.eval()  # deterministic forward (disables dropout); no-op at p=0
            # Mirror loss_fn, but keep the per-codebook (RVQ-depth) axis so eval
            # can report a per-codebook breakdown — the EnCodec/DAC diagnostic.
            encoded = model.encode(source)
            logits = model.decoder(target, encoded_source=encoded)  # [B, T, Q, V]
            logits_f = logits.astype(jnp.float32)
            nll = optax.softmax_cross_entropy_with_integer_labels(
                logits_f, target)                                    # [B, T, Q]
            # Per-codebook cross-entropy = the training objective restricted to
            # each RVQ level (full-vocab softmax); coarse codebooks should be
            # low, fine ones higher, a flat/degenerate codebook stands out. Top-1
            # is the full-vocab argmax (consistent with the CE objective).
            ce_cb = nll.mean(axis=(0, 1))                            # [Q]
            acc_cb = (jnp.argmax(logits_f, axis=-1) == target).mean(
                axis=(0, 1)).astype(jnp.float32)                     # [Q]
            return {"eval/loss": nll.mean(), "ce_cb": ce_cb, "acc_cb": acc_cb}
        return step

    def eval_step(model, source, target):
        nonlocal jitted
        graphdef, state = nnx.split(model)
        if jitted is None:
            jitted = _build(graphdef)
        return jitted(state, source, target)
    return eval_step


def make_adapter_update_fn():
    """Build a jitted fn returning the **relative effective-update** of the LoRA
    adapters: ``(mean, max)`` of ``‖ΔW‖_F / ‖W₀‖_F`` over every wrapped layer,
    where ``ΔW = scale·strength·(A @ B)`` is the actual weight perturbation the
    adapter applies to the frozen base kernel ``W₀``.

    This is the formal, scale-free, count-invariant health metric for LoRA:
    interpretable as "the adapter shifts this layer's weights by X%", it
    accounts for the α/r scaling AND the base magnitude (unlike a raw or RMS
    adapter-param norm), and a runaway value is the energy-collapse signature.

    ``‖ΔW‖_F`` is computed via the rank-r Gram identity
    ``‖A@B‖_F² = ⟨AᵀA, BBᵀ⟩`` (both ``r×r``) so the full ``(in, out)`` product is
    NEVER materialized — keeping the metric cheap and memory-flat even for
    mrt2_base (a materialized ΔW would be GBs). Layers are stacked on leading
    axes (nnx.scan), so the Frobenius reduction is over the last two axes and
    the relative norm is per-layer before aggregating. (DoRA folds strength
    inside a norm, so for DoRA this is the pre-normalization directional delta —
    a proxy; the nnx trainer uses plain LoRA, where it is the exact ΔW.)
    """
    from magenta_rt.sft.lora_nnx import LoRAAdapter

    @nnx.jit
    def fn(model):
        rels = []

        def _walk(node):
            for attr in list(vars(node)):
                if attr.startswith("_"):
                    continue
                child = getattr(node, attr)
                if isinstance(child, LoRAAdapter):
                    a = child.lora_a[...].astype(jnp.float32)        # (..., in, r)
                    b = child.lora_b[...].astype(jnp.float32)        # (..., r, out)
                    w0 = child.base.kernel[...].astype(jnp.float32)  # (..., in, out)
                    gram_a = jnp.einsum("...ir,...is->...rs", a, a)  # (..., r, r)
                    gram_b = jnp.einsum("...ro,...so->...rs", b, b)  # (..., r, r)
                    s = float(child.scale) * float(child.lora_strength)
                    dw_fro = jnp.sqrt(
                        (s * s) * jnp.sum(gram_a * gram_b, axis=(-2, -1)))
                    w0_fro = jnp.sqrt(jnp.sum(w0 * w0, axis=(-2, -1)))
                    rels.append((dw_fro / (w0_fro + 1e-8)).reshape(-1))
                elif isinstance(child, nnx.Module):
                    _walk(child)

        _walk(model)
        if not rels:
            z = jnp.array(0.0, jnp.float32)
            return z, z
        allrel = jnp.concatenate(rels)  # one entry per (wrapped layer)
        return jnp.mean(allrel), jnp.max(allrel)

    return fn


# %% [markdown]
# ## Orbax checkpoint plumbing
#
# Saves both model and optimizer state via `nnx.state((model, optimizer))`
# so resume is exact. The pure-Linen safetensors export for inference is a
# separate post-training step (TODO).

# %%
_ADAPTER_GLOB = "sft_nnx_adapters_step_*.safetensors"


def _prune_adapter_files(output_dir: str, max_to_keep: int) -> None:
    """Keep only the newest ``max_to_keep`` portable adapter files.

    ``max_to_keep`` bounds the orbax *resume* checkpoints, but the distributable
    adapter safetensors written alongside them had no retention policy at all —
    one per ``save_every_steps`` accumulated for the whole run (a 3840-step run
    at the default cadence left 153 files / 3.5 GB). Retention is the same knob
    for both: the newest few are what anyone wants to ship or evaluate.

    ``max_to_keep <= 0`` means "keep everything", matching orbax's convention.
    Files are ordered by the step parsed from the name, not lexically, so
    ``step_960`` is correctly older than ``step_1000``.
    """
    if max_to_keep is None or max_to_keep <= 0:
        return
    paths = glob.glob(os.path.join(output_dir, _ADAPTER_GLOB))
    if len(paths) <= max_to_keep:
        return

    def _step_of(path: str) -> int:
        stem = os.path.basename(path).removesuffix(".safetensors")
        return int(stem.rsplit("_", 1)[1])

    for stale in sorted(paths, key=_step_of)[:-max_to_keep]:
        os.remove(stale)


def open_ckpt_manager(ckpt_dir: str, max_to_keep: int) -> ocp.CheckpointManager:
    """Checkpoint manager whose retention keeps both the best *and* the latest.

    Retention here serves two masters. `best_fn` selection wants the
    metric-best checkpoints; auto-resume wants the newest one, because it keys
    off `latest_step()`. Passing orbax's `best_fn`/`max_to_keep` options alone
    maps to a `BestN` policy with no latest-checkpoint protection, so the newest
    checkpoint is garbage-collected on any step where it is not among the best —
    and since `resume` defaults to True, the next launch silently rolls back to
    an older step and re-trains. Ranking is on training loss, which is noisy, so
    the newest checkpoint frequently is not best.

    `AnyPreservationPolicy` keeps a checkpoint if *either* policy wants it, so
    the latest always survives alongside the `max_to_keep` best.
    """
    preservation_policy = ocp.checkpoint_managers.AnyPreservationPolicy([
        ocp.checkpoint_managers.LatestN(n=1),
        ocp.checkpoint_managers.BestN(
            get_metric_fn=lambda m: m.get("loss", float("inf")),
            # Matches orbax's own best_fn -> BestN mapping: checkpoints sort
            # ascending and the last n are kept, so "min" flips the order.
            reverse=True,
            n=max_to_keep,
        ),
    ])
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        preservation_policy=preservation_policy,
        enable_async_checkpointing=True,
    )
    return ocp.CheckpointManager(directory=os.path.abspath(ckpt_dir), options=options)


def save_state(
    ckpt_mgr, step, model, optimizer, metrics, *, lora_only=False, data_iter=None
):
    """Save model + optimizer (and optionally data-iterator) state.

    With ``lora_only=True`` (the tunix pattern) only ``MRTLoRAParam``
    Variables are saved on the model side — KBs–MBs instead of the full
    model. The optimizer state is already adapter-only in that mode
    (``wrt=MRTLoRAParam``), so it is saved as-is either way.

    ``data_iter`` is a grain ``DatasetIterator``; its ``get_state()`` (a
    JSON-able dict) is stored alongside the arrays so a resumed run
    continues from the *exact* position in the (shuffled, repeated) data
    stream instead of replaying it from the start.
    """
    model_filter = MRTLoRAParam if lora_only else ...
    model_state = nnx.state(jax.device_get(model), model_filter)
    optimizer_state = nnx.state(jax.device_get(optimizer))
    composite = {
        "model": ocp.args.PyTreeSave(model_state),
        "optimizer": ocp.args.PyTreeSave(optimizer_state),
    }
    if data_iter is not None:
        composite["data"] = ocp.args.JsonSave(data_iter.get_state())
    ckpt_mgr.save(step, args=ocp.args.Composite(**composite), metrics=metrics)


def _shape_template(state):
    """Build an abstract template (ShapeDtypeStruct leaves) for orbax restore."""
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "shape") else x,
        state,
    )


def restore_state(ckpt_mgr, step, model, optimizer, *, lora_only=False, data_iter=None):
    """In-place restore of model + optimizer (+ data iterator) state.

    ``lora_only`` must match how the checkpoint was saved: the restore
    template is then the model's adapter sub-state, and base weights are
    left untouched (load them from the pretrained safetensors as usual).
    Pass ``data_iter`` to also restore the grain iterator position —
    requires the checkpoint to have been saved with one.
    """
    model_filter = MRTLoRAParam if lora_only else ...
    composite = {
        "model": ocp.args.PyTreeRestore(_shape_template(nnx.state(model, model_filter))),
        "optimizer": ocp.args.PyTreeRestore(_shape_template(nnx.state(optimizer))),
    }
    if data_iter is not None:
        composite["data"] = ocp.args.JsonRestore()
    restored = ckpt_mgr.restore(step, args=ocp.args.Composite(**composite))
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    if data_iter is not None:
        data_iter.set_state(restored["data"])


# %% [markdown]
# ## Build model + optimizer
#
# `TinyPOCSpec` is a shrunk model (random weights, ~hundreds of KB). For real
# runs, pass `--model_name mrt2_small --checkpoint <safetensors>` and use
# `magenta_rt.nnx.load_weights.load_from_jax_safetensors`.

# %%
def build_model(spec, seed: int, *, checkpoint_path: str | None = None,
                param_dtype=None, dropout_prob: float = 0.0,
                remat: bool = False,
                whole_source_dropout_rate: float = 0.0,
                temporal_input_dropout_prob: float = 0.0,
                temporal_self_attention_dropout_prob=None) -> EncoderDecoder:
    # ``param_dtype=jnp.bfloat16`` (via SFTConfig.bf16) stores the base weights
    # in bf16, mirroring the MLX trainer's bf16 path. The spec carries
    # ``param_dtype`` as a class attribute read by ``from_config``; set it on
    # the instance before building. The tiny POC spec is a frozen
    # ``flax.struct.dataclass`` (and ``param_dtype`` isn't a declared field, so
    # ``dataclasses.replace`` can't reach it), so bypass the freeze with
    # ``object.__setattr__`` — this only mutates this instance, not the class.
    # CAVEAT: on CPU JAX (this Mac's NNX trainer) bf16 params don't reliably cut
    # memory the way MLX/Metal does — CPU matmuls often upcast bf16 — so the
    # value here is consistent plumbing + correctness, not CPU speed/memory.
    if param_dtype is not None:
        object.__setattr__(spec, "param_dtype", param_dtype)
    # The sl-derived spec-level dropout knobs are read by from_config off the
    # spec, so set them on this instance (same object.__setattr__ bypass as
    # param_dtype). Defaults (0.0 / None) are no-ops.
    object.__setattr__(spec, "whole_source_dropout_rate", whole_source_dropout_rate)
    object.__setattr__(spec, "temporal_input_dropout_prob", temporal_input_dropout_prob)
    object.__setattr__(spec, "temporal_self_attention_dropout_prob",
                       temporal_self_attention_dropout_prob)
    # dropout_prob>0 adds transformer dropout (see magenta_rt/nnx/transformer.py).
    # The model is built under train() in train(); eval_step flips its merged
    # model to deterministic via model.eval(). The AudioSampleWriter uses a
    # SEPARATE inference sampler built with the default dropout_prob=0, so it has
    # no Dropout modules and is always deterministic.
    rngs = nnx.Rngs(seed)
    model = EncoderDecoder.from_config(
        spec, dropout_prob=dropout_prob, remat=remat, rngs=rngs)
    # The decoder declares streaming-only DecodeState slots (rng_state /
    # previous_frame) that come up with `nnx.data(None)` placeholders.
    # nnx.jit can't trace placeholders, so seed them once with init_streaming —
    # they're inert during the full-sequence training forward.
    model.decoder.init_streaming(batch_size=1, rngs=rngs)
    # Materialize via a tiny forward — keeps subsequent shape errors loud.
    B, T, Q = 1, 4, spec.target_tokens_config.rvq_truncation_level
    dummy_source = jnp.zeros((B, T, spec.input_num_channels), jnp.int32)
    dummy_target = jnp.zeros((B, T, Q), jnp.int32)
    model.eval()
    _ = model.decoder(dummy_target, encoded_source=model.encode(dummy_source))

    if checkpoint_path:
        print(f"[sft] loading pretrained depthformer from {checkpoint_path}")
        load_nnx_depthformer_from_safetensors(model, checkpoint_path)

    return model


def build_optimizer(model, config: SFTConfig, *, wrt=nnx.Param) -> nnx.Optimizer:
    """clip → adam → schedule, with the LR exposed and optional accumulation.

    The schedule is threaded through ``optax.inject_hyperparams`` so the
    current learning rate is readable from optimizer state
    (:func:`current_learning_rate`) instead of recomputing it host-side.
    ``config.gradient_accumulation_steps > 1`` wraps the whole transform in
    ``optax.MultiSteps`` (the tunix recipe): gradients accumulate across that
    many micro-batches and parameters/schedule advance once per accumulated
    update.
    """
    # Decoupled AdamW decay: add_decayed_weights BETWEEN scale_by_adam and
    # scale_by_learning_rate is exactly how optax.adamw is composed (the decay
    # adds ``wd·p`` to the post-Adam update, then scale_by_learning_rate scales
    # the whole thing by -lr → ``p ← p − lr·(adam_update + wd·p)``). The mask is
    # a callable params→bool-pytree so it adapts to whatever wrt-filtered
    # structure nnx.Optimizer passes; it keys off the joined leaf path via the
    # shared ``_decays_weight`` policy (magnitude / bias / norm-scale excluded).
    # With wd=0 add_decayed_weights is a harmless no-op, so it's always present.
    def _decay_mask(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: utils._decays_weight(
                ".".join(str(getattr(k, "key", k)) for k in path)
            ),
            params,
        )

    @optax.inject_hyperparams
    def make_tx(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.scale_by_adam(b1=config.adam_b1, b2=config.adam_b2, eps=1e-8),
            optax.add_decayed_weights(config.weight_decay, mask=_decay_mask),
            optax.scale_by_learning_rate(learning_rate),  # scale(-lr)
        )

    tx = make_tx(learning_rate=lambda step: utils.lr_at_step(step, config))
    if config.gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, config.gradient_accumulation_steps)
    if config.skip_nonfinite_steps > 0:
        # A step with non-finite gradients is skipped — parameters and
        # optimizer state stay untouched — instead of writing NaNs into the
        # weights, which no amount of later recovery undoes. The NaN still
        # propagates after this many *consecutive* bad steps, so persistent
        # divergence surfaces rather than being papered over.
        tx = optax.apply_if_finite(tx, max_consecutive_errors=config.skip_nonfinite_steps)
    return nnx.Optimizer(model, tx, wrt=wrt)


def current_learning_rate(optimizer: nnx.Optimizer) -> float | None:
    """Reads the schedule's current LR out of the optimizer state.

    Works for the plain ``inject_hyperparams`` chain and the
    ``optax.MultiSteps``-wrapped variant (whose inner state holds the
    hyperparams). Returns None when no ``learning_rate`` hyperparam exists.
    """
    state = optimizer.opt_state
    if hasattr(state, "inner_opt_state"):  # optax.MultiSteps
        state = state.inner_opt_state
    hyperparams = getattr(state, "hyperparams", None)
    if hyperparams is None or "learning_rate" not in hyperparams:
        return None
    lr = hyperparams["learning_rate"]
    return float(getattr(lr, "value", lr))


def _setup_model_and_optimizer(config: SFTConfig, spec, *, model):
    """Finalize the trainable model and build its optimizer.

    Builds the model if not supplied, then either injects LoRA/DoRA adapters
    (training only those) or freezes the encoder for full SFT, and constructs the
    optimizer over the resulting trainable filter. Prints a parameter census.
    Returns ``(model, optimizer, diff_filter)`` where ``diff_filter`` selects the
    Variables that receive gradients (``MRTLoRAParam`` for LoRA, else
    ``nnx.Param``).
    """
    if model is None:
        model = build_model(
            spec, seed=config.seed,
            param_dtype=jnp.bfloat16 if config.bf16 else None,
            dropout_prob=config.dropout_prob,  # usually 0; see build_model
            remat=config.remat,                # gradient checkpointing (memory)
            whole_source_dropout_rate=config.whole_source_dropout_rate,
            temporal_input_dropout_prob=config.temporal_input_dropout_prob,
            temporal_self_attention_dropout_prob=config.temporal_self_attention_dropout_prob,
        )
    n_params_total = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"[sft] base params: {n_params_total:,}")

    diff_filter = nnx.Param
    if config.lora_rank > 0:
        from magenta_rt.sft.lora_io import _nnx_target_fn
        targets_name = config.lora_targets or (
            "all_linears" if config.lora_all_linears else "default")
        n_lora = inject_lora(
            model, rank=config.lora_rank, alpha=config.lora_alpha,
            dora=config.lora_dora, targets=_nnx_target_fn(targets_name),
            seed=config.seed,
        )
        if n_lora == 0:
            raise ValueError(
                "inject_lora wrapped 0 Linears — target predicate matched "
                "nothing; with wrt=MRTLoRAParam the model would not train."
            )
        diff_filter = MRTLoRAParam
        n_lora_size = sum(
            v.size for v in jax.tree_util.tree_leaves(nnx.state(model, MRTLoRAParam))
        )
        print(
            f"[sft] {'DoRA' if config.lora_dora else 'LoRA'}: wrapped {n_lora} "
            f"Linears, rank={config.lora_rank}, alpha={config.lora_alpha}, "
            f"adapter params={n_lora_size:,}"
        )
    elif config.freeze_encoder:
        n_frozen = freeze_module(model.encoder)
        print(f"[sft] froze {n_frozen} encoder Variables → Frozen")

    optimizer = build_optimizer(model, config, wrt=diff_filter)
    n_trainable = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, diff_filter))
    )
    n_frozen_size = sum(
        v.size for v in jax.tree_util.tree_leaves(nnx.state(model, Frozen))
    )
    print(
        f"[sft] trainable ({diff_filter.__name__}): {n_trainable:,}  "
        f"frozen: {n_frozen_size:,}"
    )
    return model, optimizer, diff_filter


def _make_writer(config: SFTConfig):
    """Build the run's metric writer: a TensorBoard writer (TF-free tensorboardX)
    plus an optional W&B mirror, combined into one MultiWriter."""
    from magenta_rt import metric_writers
    from magenta_rt.sft.tb_writer import resolve_tb_dir
    from magenta_rt.sft.wandb_writer import maybe_make_wandb_writer

    tb_dir = resolve_tb_dir(config)
    os.makedirs(tb_dir, exist_ok=True)
    writers = [metric_writers.create_default_writer(logdir=tb_dir)]
    print(f"[sft] TensorBoard logdir: {tb_dir}  "
          f"(tensorboard --logdir {config.output_dir} "
          f"--samples_per_plugin audio=200)")
    wb = maybe_make_wandb_writer(config)
    if wb is not None:
        writers.append(wb)
        print("[sft] W&B logging enabled")
    return metric_writers.MultiWriter(writers)


def _materialize_eval_batches(config: SFTConfig, spec, *, style_tokens):
    """Draw a FIXED list of held-out eval batches once (not re-drawn each eval).

    Held-out excerpts vary widely in note density, so rotating batches make a
    short eval-loss trend unreadable; a fixed set gives a clean curve. Returns
    ``None`` when no eval set / frequency is configured.
    """
    if not (config.valid_dir and config.valid_freq > 0):
        return None
    eval_ds = utils.make_eval_dataset(config.valid_dir, config, spec,
                                      seed=config.seed + 1,
                                      style_tokens=style_tokens)
    eval_it = iter(eval_ds)
    return [
        to_source_target(next(eval_it), spec.target_tokens_config,
                         asarray=jnp.asarray)
        for _ in range(config.valid_batches)
    ]


# %% [markdown]
# ## Main training loop

# %%
def train(config: SFTConfig, spec, *, writer=None, model=None,
          model_name=None, checkpoint_path=None):
    from magenta_rt import periodic_actions

    print(f"[sft] devices: {jax.devices()}")
    print(f"[sft] spec: {type(spec).__name__}")
    print(f"[sft] config: {config}")
    # Make the run location unmistakable — checkpoints, adapter .safetensors,
    # audio samples, and TensorBoard events all live under this directory
    # (often a WSL/native home path, not the repo).
    print(f"[sft] OUTPUT DIR (checkpoints + adapters + samples + tb): "
          f"{os.path.abspath(config.output_dir)}")

    # ---- Model + optimizer ----------------------------------------------
    model, optimizer, diff_filter = _setup_model_and_optimizer(
        config, spec, model=model)

    # ---- Data -----------------------------------------------------------
    # Optional single-style-prompt recipe: one fixed prompt's MusicCoCa tokens
    # overlaid on every example (None when unset). The held-out eval set, if any,
    # is a fixed batch list for a clean eval-loss curve.
    style_tokens = utils.embed_style_prompt(config, model_name)
    ds = utils.make_dataset(config.data_dir, config, spec,
                            seed=config.seed, num_workers=config.num_workers,
                            style_tokens=style_tokens)
    it = iter(ds)
    eval_batches = _materialize_eval_batches(config, spec, style_tokens=style_tokens)

    # ---- Logging + checkpoint ------------------------------------------
    if writer is None:
        writer = _make_writer(config)
    report = periodic_actions.ReportProgress(
        num_train_steps=config.total_steps,
        every_steps=max(1, config.log_every_steps),
        writer=writer,
    )
    ckpt_mgr = open_ckpt_manager(config.output_dir, config.max_to_keep)

    # ---- Auto-resume ------------------------------------------------------
    # Restores model + optimizer + the grain iterator position, so the run
    # continues with the exact data stream it would have seen uninterrupted.
    # Must happen after inject_lora/freeze (the saved state matches the
    # final module structure) and after ``it`` exists.
    start_step = 0
    latest = ckpt_mgr.latest_step() if config.resume else None
    if latest is not None:
        restore_state(
            ckpt_mgr, latest, model, optimizer,
            lora_only=diff_filter is MRTLoRAParam, data_iter=it,
        )
        start_step = latest
        print(f"[sft] resumed from step {latest} in {config.output_dir}")

    # Functional jax.jit steps (the MagentaRT2System pattern, internal to the
    # steps): each call splits the live (model, optimizer) into a constant
    # partition (graphdef + frozen base + streaming DecodeState) and a donated
    # mutable partition (trainable params + optimizer state + dropout rng), so the
    # frozen base is not copied in+out each step, then writes the updated mutable
    # back — keeping the live objects the source of truth for logging/eval/ckpt.
    # The train step captures its graphdef in train() mode on its first call;
    # eval_step flips its merged model to deterministic via model.eval(). dropout
    # rng is live RngState in the mutable partition, so masks advance per step.
    model.train()
    train_step = make_train_step(diff_filter=diff_filter)
    eval_step = make_eval_step()
    adapter_update_fn = (
        make_adapter_update_fn() if diff_filter is MRTLoRAParam else None)

    sample_writer = AudioSampleWriter(
        model=model, diff_filter=diff_filter, config=config,
        model_name=model_name, checkpoint_path=checkpoint_path,
    )
    if config.sample_every_steps and not sample_writer.available:
        print("[sft] sample_every_steps set but no preset/checkpoint — "
              "audio samples disabled")

    # Optionally condition audio samples on a held-out EVAL clip's source (its
    # own per-clip MusicCoCa / pianoroll), instead of a training batch. Captured
    # ONCE before the loop, so the in-loop set_source() (which only fills when
    # empty) becomes a no-op and the samples track a fixed held-out prompt.
    if config.sample_from_eval and config.valid_dir and config.valid_freq > 0:
        eval_rec = next(iter(utils.make_eval_dataset(
            config.valid_dir, config, spec, seed=config.seed + 2,
            style_tokens=style_tokens)))
        eval_src, _ = to_source_target(
            eval_rec, spec.target_tokens_config, asarray=jnp.asarray)
        sample_writer.set_source(eval_src, record=eval_rec)
        print("[sft] audio samples condition on a held-out EVAL clip's MusicCoCa")

    # Early stopping on validation loss.
    stopper = EarlyStopper(
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
    )

    # ---- Loop -----------------------------------------------------------
    losses = []
    train_metrics = utils.make_train_metrics()  # window-averaged between log steps
    t0 = time.time()
    # Tracks the last step actually executed, so the summary below is honest
    # when early stopping (or an exception) ends the loop before total_steps.
    last_step = start_step
    consecutive_nans = 0
    try:
        for step in range(start_step + 1, config.total_steps + 1):
            last_step = step
            with report.timed("data") if step > 1 else contextlib.nullcontext():
                # Pre-tokenized data -> no codec. For audio data, pass a
                # SpectroStream via ``codec=`` to encode samples->tokens on device.
                batch = next(it)
                source, target = to_source_target(
                    batch, spec.target_tokens_config, asarray=jnp.asarray,
                )
                # Capture the source tokens AND their AudioTree provenance
                # (filepath/offset) so a generated sample is traceable to the
                # track its conditioning came from.
                sample_writer.set_source(source, record=batch)

            with report.timed("train") if step > 1 else contextlib.nullcontext():
                metrics = train_step(model, optimizer, source, target)

            loss = float(metrics["loss"])
            losses.append(loss)
            # Accumulate into the MultiMetric — compute() at log time gives the
            # window average (vs the single noisy step value).
            train_metrics.update(loss=metrics["loss"], grad_norm=metrics["grad_norm"])

            # NaN / Inf short-circuit before we corrupt the checkpoint. With
            # `nan_patience > 0` an isolated bad batch is tolerated (pair it
            # with `skip_nonfinite_steps` so that step does not move the
            # weights); a run that keeps producing them still stops.
            if config.nan_check and not jnp.isfinite(metrics["loss"]):
                consecutive_nans += 1
                if consecutive_nans > config.nan_patience:
                    print(f"[sft] non-finite loss at step {step} "
                          f"({consecutive_nans} consecutive) — stopping.")
                    break
                print(f"[sft] non-finite loss at step {step} "
                      f"({consecutive_nans}/{config.nan_patience}) — continuing.")
            else:
                consecutive_nans = 0

            if step % config.log_every_steps == 0:
                summary = train_metrics.compute()  # window-averaged loss + grad_norm
                train_metrics.reset()
                avg_loss, gn = float(summary["loss"]), float(summary["grad_norm"])
                # Adapter magnitude — how hard the trained adapters push the base
                # (a runaway value is the energy-collapse signature). The raw L2
                # norm scales with sqrt(param count), so it is NOT comparable
                # across ranks / target sets / model sizes; ``adapter_rms`` =
                # ||theta|| / sqrt(N) is the count-normalized mean per-weight
                # magnitude and IS comparable (and still spikes on runaway). Both
                # are logged; rms is the one to watch across runs. Cheap (the
                # adapter sub-state is small); computed outside the jit step.
                adapter_leaves = jax.tree.leaves(nnx.state(model, diff_filter))
                if adapter_leaves:
                    asq = sum(jnp.vdot(x, x) for x in adapter_leaves)
                    aN = sum(int(x.size) for x in adapter_leaves)
                    anorm = float(jnp.sqrt(asq))
                    arms = float(jnp.sqrt(asq / aN))
                else:
                    anorm = arms = 0.0
                scalars = {"train/loss": avg_loss, "train/grad_norm": gn,
                           "train/adapter_norm": anorm, "train/adapter_rms": arms}
                # Relative effective-update ‖ΔW‖/‖W₀‖ — the formal LoRA-health
                # metric ("adapter shifts each layer's weights by X%"). Jitted +
                # memory-flat (rank-r Gram trick, no ΔW materialization).
                rel_update = 0.0
                if adapter_update_fn is not None:
                    rel_mean, rel_max = adapter_update_fn(model)
                    rel_update = float(rel_mean)
                    scalars["train/adapter_rel_update"] = rel_update
                    scalars["train/adapter_rel_update_max"] = float(rel_max)
                lr = current_learning_rate(optimizer)
                if lr is not None:
                    scalars["train/learning_rate"] = lr
                if config.log_memory:
                    scalars.update(_jax_memory_metrics())
                writer.write_scalars(step, scalars)
                mem = f"  {scalars.get('mem/bytes_in_use_gb', 0):.2f}GB" if config.log_memory else ""
                print(f"  step {step:4d}  loss={avg_loss:.4f}  grad_norm={gn:.3f}  "
                      f"adapter_rms={arms:.4f}  rel_dW={rel_update:.4f}{mem}")
            report(step)

            # Periodic validation + early stopping. eval_step flips its merged
            # model to deterministic via model.eval() internally, so the live
            # (train-mode) model is never toggled.
            if eval_batches is not None and step % config.valid_freq == 0:
                eval_metrics = utils.make_eval_metrics()
                ce_sum = acc_sum = None  # accumulate per-codebook over eval batches
                n_eval = 0
                for vsource, vtarget in eval_batches:
                    vm = eval_step(model, vsource, vtarget)
                    eval_metrics.update(loss=vm["eval/loss"])
                    ce = np.asarray(vm["ce_cb"]); ac = np.asarray(vm["acc_cb"])
                    ce_sum = ce if ce_sum is None else ce_sum + ce
                    acc_sum = ac if acc_sum is None else acc_sum + ac
                    n_eval += 1
                vmean = float(eval_metrics.compute()["loss"])
                scalars = {"eval/loss": vmean}
                summary = ""
                if n_eval:
                    ce_cb, acc_cb = ce_sum / n_eval, acc_sum / n_eval
                    for q in range(len(ce_cb)):
                        scalars[f"eval/ce_cb/{q:02d}"] = float(ce_cb[q])
                        scalars[f"eval/acc_cb/{q:02d}"] = float(acc_cb[q])
                    last = len(ce_cb) - 1
                    summary = (f"  cb0 ce/acc={ce_cb[0]:.3f}/{acc_cb[0]:.2f}"
                               f"  cb{last} ce/acc={ce_cb[last]:.3f}/{acc_cb[last]:.2f}")
                writer.write_scalars(step, scalars)
                print(f"  step {step:4d}  eval/loss={vmean:.4f}{summary}")
                if stopper.update(vmean):
                    print(f"[sft] early stop at step {step} (best eval={stopper.best:.4f})")
                    break

            if (
                config.save_every_steps
                and step % config.save_every_steps == 0
            ):
                save_state(
                    ckpt_mgr, step, model, optimizer, {"loss": loss},
                    lora_only=diff_filter is MRTLoRAParam, data_iter=it,
                )
                # Also emit a portable, self-describing adapter file: one small
                # safetensors with the LoRA recipe (rank/alpha/DoRA/targets/base)
                # in its header. The orbax checkpoint above is for *resume*; this
                # is the artifact to *distribute* — the base model is never bundled.
                if diff_filter is MRTLoRAParam:
                    from magenta_rt.sft.lora_io import save_lora_adapters

                    adapter_path = os.path.join(
                        config.output_dir,
                        f"sft_nnx_adapters_step_{step}.safetensors",
                    )
                    save_lora_adapters(
                        model, adapter_path,
                        base_model=model_name or "unknown",
                        targets=(config.lora_targets or
                                 ("all_linears" if config.lora_all_linears
                                  else "default")),
                        base_checkpoint=checkpoint_path,
                    )
                    _prune_adapter_files(config.output_dir, config.max_to_keep)

            # Periodic generated-audio sample for qualitative monitoring.
            if config.sample_every_steps and step % config.sample_every_steps == 0:
                sample_writer(writer, step)
    finally:
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()
        writer.flush()
        if hasattr(writer, "close"):
            writer.close()

    ran = last_step - start_step
    stopped_early = (
        "" if last_step >= config.total_steps
        else f" (stopped at step {last_step} of {config.total_steps})"
    )
    print(f"[sft] {ran} steps in {time.time()-t0:.1f}s{stopped_early}")
    return losses


# ---- Optional sample-generation + memory telemetry -----------------------

def _jax_memory_metrics() -> dict[str, float]:
    """Best-effort device memory in GB. Returns {} when unavailable
    (CPU JAX exposes no `memory_stats`)."""
    try:
        s = jax.devices()[0].memory_stats()
        return {
            "mem/bytes_in_use_gb": s.get("bytes_in_use", 0) / 1024**3,
            "mem/peak_bytes_gb": s.get("peak_bytes_in_use", 0) / 1024**3,
        }
    except Exception:
        return {}


class AudioSampleWriter:
    """Periodic audio samples from the training model (opt-in).

    Generating audio needs the SpectroStream codec and the depthformer's
    *streaming* machinery — and arming streaming on the live training
    model would replace cache variables and flip static attributes,
    changing the graphdef the functional train step captured (and its
    constant/mutable split). So this keeps a **separate sampler**: a full
    ``MagentaRT2Sampler`` (depthformer +
    codec) built lazily from the same preset/checkpoint, mirrored with
    the run's LoRA structure, whose *trainable* state is synced from the
    training model right before each sample (adapter-only in LoRA mode —
    KBs; the trainable ``nnx.Param`` set in full-SFT mode). Training is
    never touched.

    Conditioning comes from a held-out batch of the run's own prepared
    source tokens, so the sample shows what the model does on
    training-like conditioning. Cost: a second model instance in memory —
    only paid when ``config.sample_every_steps > 0`` and the run has a
    real preset + checkpoint (the tiny POC has no codec weights).
    """

    SAMPLE_SECONDS = 4
    TEMPERATURE = 1.0
    TOP_K = 40

    def __init__(self, *, model, diff_filter, config: SFTConfig,
                 model_name: str | None, checkpoint_path: str | None):
        self._model = model
        self._diff_filter = diff_filter
        self._config = config
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._sample_mrt = None
        self._source = None
        self._provenance = None

    @property
    def available(self) -> bool:
        return bool(
            self._config.sample_every_steps
            and self._checkpoint_path
            and self._model_name in MODEL_REGISTRY
        )

    def set_source(self, source, *, record=None) -> None:
        """Capture one prepared source batch ``[B, T, C]`` as conditioning.

        ``record`` is the batched ``AudioTree`` the source was built from; its
        public ``.filepath`` (``List[str]``, empty when the export carried no
        provenance) and ``.offset`` (``None`` likewise) are stashed so the
        generated sample can be traced to the track its conditioning came from.
        Both live in AudioTree's own provenance container, not in ``extras``.
        """
        if self._source is None:
            frames = min(source.shape[1], self.SAMPLE_SECONDS * 25)
            self._source = jnp.asarray(source[:1, :frames])
            if record is not None:
                filepaths = record.filepath  # List[str]; [] if no provenance
                offsets = record.offset  # None if no provenance
                self._provenance = {
                    "filepath": filepaths[0] if filepaths else None,
                    "offset": (float(np.asarray(offsets)[0])
                               if offsets is not None else None),
                }

    def _ensure_sampler(self):
        if self._sample_mrt is not None:
            return self._sample_mrt
        from magenta_rt.nnx.checkpoint_utils import load_into_abstract
        from magenta_rt.nnx.model import MagentaRT2Sampler

        print("[sft] building audio sampler (depthformer + codec)…")
        # Match the run's storage dtype. For mrt2_base a fp32 sampler (9.6 GB)
        # cannot co-reside with the bf16 training model on a 16 GB GPU, so build
        # the sampler in bf16 too when the run is bf16. Building abstractly keeps
        # the discarded random init off the GPU entirely, which is the same
        # pressure this dtype match exists to relieve.
        #
        # host=True for bf16 runs: avoids an on-device fp32 checkpoint load that
        # would OOM beside the resident (bf16 mrt2_base) training model.
        mrt = load_into_abstract(
            lambda: MagentaRT2Sampler.from_preset(
                self._model_name, int16_outputs=False,
                param_dtype=jnp.bfloat16 if self._config.bf16 else None,
                rngs=nnx.Rngs(0),
            ),
            lambda m: m.load_checkpoint(
                self._checkpoint_path, host=self._config.bf16
            ),
        )
        if self._diff_filter is MRTLoRAParam:
            # Mirror the run's adapter structure so trainable-state paths
            # match for the per-sample sync below.
            from magenta_rt.sft.lora_io import _nnx_target_fn
            sampler_targets = self._config.lora_targets or (
                "all_linears" if self._config.lora_all_linears else "default")
            inject_lora(
                mrt.depthformer,
                rank=self._config.lora_rank,
                alpha=self._config.lora_alpha,
                dora=self._config.lora_dora,
                targets=_nnx_target_fn(sampler_targets),
                seed=self._config.seed,
            )
        self._sample_mrt = mrt
        return mrt

    def __call__(self, writer, step: int) -> None:
        if not self.available or self._source is None:
            return
        mrt = self._ensure_sampler()
        # Sync exactly the trainable set (adapters in LoRA mode).
        nnx.update(
            mrt.depthformer, nnx.state(self._model, self._diff_filter)
        )
        # Blend the adapter toward base for the preview (LoRA/DoRA only). The
        # strength is a static wrapper attr, untouched by nnx.update, so set it
        # after each sync. 1.0 = full adapter (no-op).
        if self._diff_filter is MRTLoRAParam:
            n = set_lora_strength(
                mrt.depthformer, self._config.sample_lora_strength
            )
            if step == self._config.sample_every_steps:  # first sample only
                print(f"  [sft] audio sampler LoRA strength="
                      f"{self._config.sample_lora_strength} on {n} wrappers")
        # Fresh streaming caches per sample event.
        mrt.init_streaming(batch_size=1, rngs=nnx.Rngs(self._config.seed))
        chunks = []
        for t in range(self._source.shape[1]):
            tree = mrt.step(
                source_tokens=self._source[:, t : t + 1],
                temperature=self.TEMPERATURE,
                top_k=self.TOP_K,
            )
            chunks.append(np.asarray(tree.waveform))
        audio = np.concatenate(chunks, axis=-1)  # [1, 2, T]
        peak = np.abs(audio).max() or 1.0
        audio = np.transpose(audio, (0, 2, 1)) / max(1.0, peak)  # [N, T, C]
        writer.write_audios(step, {"sample/audio": audio}, sample_rate=tree.sample_rate)
        # Generated-audio energy diagnostics (matches the MLX trainer): a rising
        # gen/frac_silent flags the energy-collapse failure mode even while the
        # loss keeps falling. audio is [1, T, C] → pass the single clip [T, C].
        m = utils.audio_energy_metrics(audio[0], sample_rate=tree.sample_rate)
        writer.write_scalars(step, {"gen/rms": m["rms"],
                                    "gen/frac_silent": m["frac_silent"]})
        prov = ""
        if self._provenance and self._provenance.get("filepath"):
            fp = self._provenance["filepath"]
            off = self._provenance.get("offset")
            prov = (f"  <- {os.path.basename(fp)}"
                    + (f" @ {off:.0f}s" if off is not None else ""))
        print(f"  [sft] wrote {audio.shape[1] / tree.sample_rate:.1f}s audio sample "
              f"@ step {step}{prov}")


# %% [markdown]
# ## CLI

# %%
@dataclasses.dataclass
class NNXTrainCLI(utils.TrainCLI):
    """TrainCLI + the NNX-only Linen export flag."""

    export_linen: Optional[str] = None
    """If set, export the final model as Linen-format safetensors to this path
    (for inference-path consumption on any backend)."""


def _parse_args(argv=None):
    cli = utils.parse_train_cli(NNXTrainCLI, argv)
    return (utils.to_sft_config(cli),
            cli.model_name, cli.checkpoint, cli.export_linen)


def _resolve_spec(model_name: str):
    return utils.resolve_spec(
        model_name, tiny_cls=TinyPOCSpec, lookup=MODEL_REGISTRY.__getitem__,
    )


# %%
def main(argv=None):
    config, model_name, checkpoint_path, export_linen = _parse_args(argv)
    spec = _resolve_spec(model_name)

    if not config.data_dir:
        raise ValueError("Must specify `--data_dir` containing the SFT dataset.")
    # Build (with optional pretrained load), train, optionally export.
    model = build_model(
        spec, seed=config.seed, checkpoint_path=checkpoint_path,
        param_dtype=jnp.bfloat16 if config.bf16 else None,
        dropout_prob=config.dropout_prob,
        remat=config.remat,                # gradient checkpointing (memory)
        whole_source_dropout_rate=config.whole_source_dropout_rate,
        temporal_input_dropout_prob=config.temporal_input_dropout_prob,
        temporal_self_attention_dropout_prob=config.temporal_self_attention_dropout_prob,
    )
    train(config, spec, model=model,
          model_name=model_name, checkpoint_path=checkpoint_path)
    if export_linen:
        n = export_nnx_to_linen_safetensors(
            model, export_linen,
            source_checkpoint_path=checkpoint_path,  # pass through SpectroStream
        )
        print(f"[sft] exported {n} tensors to {export_linen}")


if __name__ == "__main__":
    main()
