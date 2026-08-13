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
# # SFT Training (MLX) — Apple Silicon GPU driver
#
# Supervised fine-tuning of the Magenta-RT V2 depthformer on Metal via
# `magenta_rt.mlx_pure` (pure-MLX, no `sequence_layers` dependency). The NNX
# trainer (`train_nnx.py`) runs on CPU JAX; this MLX driver is the path to
# **GPU** fine-tuning on a Mac. It shares the grain data pipeline and the
# `SFTConfig` knobs, and supports **LoRA**, **pretrained-checkpoint loading**,
# **gradient accumulation** (`gradient_accumulation_steps>1`: the compiled
# step returns grads, the host accumulates k micro-batches, and the optimizer
# updates once — LR advances per update), **loss-curve logging** (JSONL per
# step + a matplotlib PNG at exit), and **periodic audio sampling**
# (`sample_every_steps>0`: a separate sampler — depthformer + codec, LoRA
# adapters synced from the run — generates a few seconds of audio conditioned
# on a held-out source batch and writes a wav under `output_dir/samples/`).
# Exact resume (optimizer + data-iterator state), Linen-interchange export,
# and QKV-LoRA remain NNX-only follow-ups.
#
# Pipeline:
#
# 1. `magenta_rt.sft.create_audiotree_dataset(...)` — same grain pipeline as the
#    NNX trainer (pure numpy in, AudioTree out). `to_source_target(...)` builds
#    source/target and we cast `mx.array(...)` at the consumption boundary.
# 2. Model: `MagentaRT2ModelBase.build_decoder()` (mlx_pure path). Pass
#    `--checkpoint` to load a pretrained depthformer via
#    `mlx_pure.load_sft_depthformer_from_safetensors` (codec skipped).
# 3. Adaptation (one of):
#    * LoRA (`lora_rank>0`) — `lora_mlx.inject_lora` wraps the FFN linears
#      (+ attention output projections with `lora_all_linears`); then
#      `mark_lora_trainable` freezes the base and trains only the adapters.
#    * Encoder freeze (`freeze_encoder`) — `model.encoder.freeze()`.
#    Either way `nn.value_and_grad(model, fn)` + `optimizer.update(model, grads)`
#    touch only the trainable (unfrozen) params.
# 4. Loss: cross-entropy over `[B, T, Q, V]` logits.
# 5. Checkpoints: `model.save_weights(path)` per N steps (MLX-native
#    safetensors). For LoRA, the adapters can be merged back into the base
#    (`lora_mlx.merge_lora_into_base`) and exported to Linen format — a
#    follow-up for the inverse-weight-bridge interchange.
# 6. `mx.eval(...)` after each step to materialize and free the lazy compute
#    graph.

# %%
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import numpy as np
from functools import partial

from absl import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from magenta_rt.mlx_pure.configs import (
    MagentaRT2ModelBase,
    ModelSpec,
    SPECTROSTREAM,
    TokensConfig,
    get_model_class,
)
from magenta_rt.mlx_pure.load_weights import (
    load_depthformer_from_safetensors_direct,
    load_sft_depthformer_from_safetensors,
)
from magenta_rt.sft import EarlyStopper, to_source_target
from magenta_rt.sft.configs import SFTConfig
from magenta_rt.sft.lora_mlx import (
    all_linear_targets,
    default_targets,
    inject_lora,
    mark_lora_trainable,
    merge_lora_into_base,
)

from magenta_rt.sft import trainer_common as utils  # shared, backend-neutral trainer glue
# NOTE: `wandb_writer` pulls in TensorFlow, which aborts when co-resident with
# MLX/JAX in the combined pytest process. It's only used by train()'s logging,
# so it's imported lazily there — keeping this module (and the POC tests that
# import it) TF-free.


# %% [markdown]
# ## Tiny POC spec (random weights, fits trivially on a laptop)

# %%
_TINY = ModelSpec(
    num_layers=2,
    model_dims=64,
    hidden_dims=128,
    num_heads=2,
    dim_per_head=32,
    ffn_use_gated_activation=False,
)


class TinyPOCSpecMLX(MagentaRT2ModelBase):
    """Tiny mrt2-shaped spec for the MLX trainer POC.

    Plain (non-pretrained-MusicCoCa) encoder embedder so the tiny model
    doesn't carry the full 12k-row dequantizer table; mirrors the NNX
    ``magenta_rt.sft.configs.TinyPOCSpec``.
    """

    encoder_size: ModelSpec = _TINY
    decoder_temporal_size: ModelSpec = _TINY
    decoder_depth_size: ModelSpec = _TINY

    use_pretrained_musiccoca_embedder: bool = False

    # Smaller target vocab (4 codebooks × 32 entries + 6 reserved).
    spectrostream: TokensConfig = dataclasses.replace(
        SPECTROSTREAM, rvq_truncation_level=4, codebook_size=32,
    )

    encoder_max_past_horizon: int = 8
    decoder_temporal_self_attention_max_past_horizon: int = 8
    decoder_temporal_cross_attention_max_past_horizon: int = 8

    crop_length_seconds: int = 2


# %% [markdown]
# ## Loss
#
# The warmup→rsqrt LR schedule lives in `utils.lr_at_step` (shared with
# `train_nnx`); the loop sets `optimizer.learning_rate` from it each step.

# %%
def make_loss_fn(model):
    """Cross-entropy over `[B, T, Q, V]` logits.

    Captures `model` in a closure — `nn.value_and_grad(model, fn)` reads
    trainable params off the model itself, so the fn signature stays
    `(source, target) → loss`.
    """
    def loss_fn(source: mx.array, target: mx.array) -> mx.array:
        encoded = model.encode(source)
        logits = model.decoder(target, encoded_source=encoded)  # [B,T,Q,V]
        # Softmax in fp32: with a bf16 base (mrt2_base) the logits are bf16,
        # and log_softmax over the multi-thousand-token vocab loses precision
        # in bf16 — the cast keeps the loss/gradients numerically clean.
        log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        nll = -mx.take_along_axis(
            log_probs, target[..., None].astype(mx.int32), axis=-1,
        ).squeeze(-1)                                            # [B,T,Q]
        return nll.mean()
    return loss_fn


# %% [markdown]
# ## Gradient utilities + checkpoint

# %%
def clip_grad_norm(grads, max_norm: float):
    """Clip the global L2 norm of `grads`. Returns (clipped_grads, total_norm)."""
    leaves = [g for _, g in mlx.utils.tree_flatten(grads) if g is not None]
    total = sum(mx.sum(g * g) for g in leaves) if leaves else mx.array(0.0)
    gnorm = mx.sqrt(total)
    scale = mx.minimum(mx.array(1.0), mx.array(max_norm) / (gnorm + 1e-6))
    clipped = mlx.utils.tree_map(
        lambda g: g * scale if g is not None else g, grads,
    )
    return clipped, gnorm


def save_checkpoint(model, step: int, output_dir: str,
                    *, lora_only: bool = False) -> str:
    """MLX-native safetensors. Use the JAX-Linen inverse weight bridge for
    interchange with the NNX / JAX / MLX-pure inference paths (follow-up).

    With ``lora_only`` (LoRA runs), saves just the trainable adapter params
    (KBs–MBs instead of the full model) — restore by ``inject_lora`` with the
    same config, then ``model.load_weights(path, strict=False)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    if lora_only:
        path = os.path.join(output_dir, f"sft_mlx_adapters_step_{step}.safetensors")
        flat = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(path, flat)
    else:
        path = os.path.join(output_dir, f"sft_mlx_step_{step}.safetensors")
        model.save_weights(path)
    return path


def count_params(params_tree) -> int:
    return sum(g.size for _, g in mlx.utils.tree_flatten(params_tree))


def adapter_l2_norm(params_tree) -> float:
    """Global L2 norm of a param tree (the trained adapters).

    A health metric distinct from grad_norm: how far the adapter has moved from
    its zero init / how hard its delta pushes the base. A runaway value is the
    direct signature of the energy-collapse failure mode (delta overpowering the
    base). Cheap — the adapter tree is small."""
    leaves = [p for _, p in mlx.utils.tree_flatten(params_tree) if p is not None]
    if not leaves:
        return 0.0
    return float(mx.sqrt(sum(mx.sum(p.astype(mx.float32) ** 2) for p in leaves)))


def adapter_rms(params_tree) -> float:
    """Global RMS norm of a param tree (the trained adapters)."""
    leaves = [p for _, p in mlx.utils.tree_flatten(params_tree) if p is not None]
    if not leaves:
        return 0.0
    asq = sum(mx.sum(p.astype(mx.float32) ** 2) for p in leaves)
    aN = sum(p.size for p in leaves)
    return float(mx.sqrt(asq / aN))


def make_adapter_update_fn(model):
    """Build a fn returning the relative effective-update of the LoRA adapters:
    (mean, max) of ‖ΔW‖_F / ‖W₀‖_F over every wrapped layer, where
    ΔW = scale·strength·(A @ B) is the actual weight perturbation.
    """
    from magenta_rt.sft.lora_mlx import LoRALinear, LoRAEinsumDense

    def fn():
        rels = []

        def _walk(node):
            for attr in list(vars(node)):
                if attr.startswith("_"):
                    continue
                child = getattr(node, attr)
                if isinstance(child, (LoRALinear, LoRAEinsumDense)):
                    a = child.lora_a.astype(mx.float32)        # [in, r]
                    b = child.lora_b.astype(mx.float32)        # [r, out]
                    if isinstance(child, LoRALinear):
                        w0 = child.linear.weight.astype(mx.float32) # [out, in]
                    else:
                        w0 = child._kernel_2d().astype(mx.float32)  # [out, in]
                    
                    gram_a = a.T @ a                           # [r, r]
                    gram_b = b @ b.T                           # [r, r]
                    s = float(child.scale) * float(child.lora_strength)
                    dw_fro = mx.sqrt((s * s) * mx.sum(gram_a * gram_b))
                    w0_fro = mx.sqrt(mx.sum(w0 * w0))
                    rels.append(mx.reshape(dw_fro / (w0_fro + 1e-8), (-1,)))
                elif isinstance(child, nn.Module):
                    _walk(child)

        _walk(model)
        if not rels:
            z = mx.array(0.0, mx.float32)
            return z, z
        allrel = mx.concatenate(rels)
        return float(mx.mean(allrel)), float(mx.max(allrel))

    return fn


# %% [markdown]
# ## Periodic audio samples (opt-in: `sample_every_steps > 0`)

# %%
class AudioSampleWriter:
    """Periodic audio samples from the training model (MLX twin of the NNX
    ``AudioSampleWriter``).

    Keeps a **separate** ``MagentaRT2Sampler`` (depthformer + SpectroStream
    codec) built lazily from the same preset/checkpoint and mirrored with the
    run's LoRA structure; right before each sample the run's *trainable*
    params (the adapters, in LoRA mode) are synced in via ``Module.update``.
    The training model and its compiled step are never touched. Conditioning
    is a held-out batch of the run's own prepared source tokens, tiled along
    time to ``sample_seconds`` (our conditioning rows are constant per frame
    for style / CFG / dropout-filled piano-roll streams, so tiling is sound).
    Wavs land in ``<output_dir>/samples/step_<n>.wav``.
    """

    TEMPERATURE = 1.0
    TOP_K = 40

    def __init__(self, *, model, config: SFTConfig,
                 model_name: str | None, checkpoint_path: str | None,
                 sample_seconds: float = 8.0):
        self._model = model
        self._config = config
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._sample_frames = int(sample_seconds * 25)
        self._sample_mrt = None
        self._source = None

    @property
    def available(self) -> bool:
        if not (self._config.sample_every_steps and self._checkpoint_path):
            return False
        try:
            get_model_class(self._model_name)
        except (KeyError, ValueError):
            return False
        return True

    def set_source(self, source: mx.array) -> None:
        """Capture one prepared source batch ``[B, T, C]`` as conditioning."""
        if self._source is None:
            row = source[:1]
            reps = -(-self._sample_frames // row.shape[1])  # ceil
            tiled = mx.concatenate([row] * reps, axis=1)
            self._source = tiled[:, : self._sample_frames]

    def _ensure_sampler(self):
        if self._sample_mrt is not None:
            return self._sample_mrt
        from magenta_rt.mlx_pure.model import MagentaRT2Sampler

        logging.info("[sft-mlx] building audio sampler (depthformer + codec)…")
        mrt = MagentaRT2Sampler.from_preset(
            self._model_name, int16_outputs=False
        )
        mrt.load_from_safetensors(
            self._checkpoint_path, model_name=self._model_name
        )
        if self._config.lora_rank > 0:
            # Mirror the run's adapter structure so trainable-param paths
            # match for the per-sample sync below.
            inject_lora(
                mrt.depthformer,
                rank=self._config.lora_rank,
                alpha=self._config.lora_alpha,
                dora=self._config.lora_dora,
                targets=(all_linear_targets if self._config.lora_all_linears
                         else default_targets),
                seed=self._config.seed,
            )
        self._sample_mrt = mrt
        return mrt

    def __call__(self, step: int):
        """Generate one clip from the captured conditioning at the current
        (adapter) weights. Returns ``(path, audio[T, 2])`` or ``None``.

        ``step`` may be ``0`` to capture the pre-SFT *baseline* before any
        update — the reference the later steps are judged against by ear.
        """
        if not self.available or self._source is None:
            return None
        import numpy as np
        import soundfile

        mrt = self._ensure_sampler()
        # Sync exactly the trainable set (adapters in LoRA mode). At step 0 the
        # adapters are still zero-init (lora_b=0), so this clip is the unmodified
        # base model — the baseline.
        mrt.depthformer.update(self._model.trainable_parameters())
        state = mrt.make_initial_state(batch_size=1, seed=self._config.seed)
        chunks = []
        for t in range(self._source.shape[1]):
            waveform, state = mrt.step(
                state,
                source_tokens=self._source[:, t : t + 1],
                temperature=self.TEMPERATURE,
                top_k=self.TOP_K,
            )
            mx.eval(waveform)
            chunks.append(np.asarray(waveform))
        audio = np.concatenate(chunks, axis=-1)[0]  # [2, T]
        audio = audio.T / max(1.0, float(np.abs(audio).max()))  # [T, 2]
        out_dir = os.path.join(self._config.output_dir, "samples")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"step_{step:05d}.wav")
        soundfile.write(path, audio, 48_000)
        return path, audio


# %% [markdown]
# ## Build model + optimizer

# %%
def build_model(spec, *, checkpoint_path: str | None = None,
                model_name: str = "mrt2_small", param_dtype=None,
                dropout_prob: float = 0.0, direct_load: bool = True):
    """Construct the mlx_pure depthformer and force a tiny forward pass to
    materialize any lazy-init weights.

    `EinsumDense` (used for attention output projections) allocates its
    `kernel` on the first call rather than in `__init__`, so the optimizer
    must see a fully-shaped param tree before training begins — otherwise
    a later `optimizer.update(model, grads)` errors with a `KeyError`
    when the grad tree contains keys the optimizer state never saw.

    `param_dtype=mx.bfloat16` stores the base weights in bf16 — **required to
    fit mrt2_base (2.4 B params) on 16 GB**: fp32 weights are 9.6 GB and would
    OOM against the fp32 sl-bridge loader; bf16 weights (4.8 GB) keep the load
    peak ~15 GB (same as inference). The LoRA adapters inherit this dtype
    (`from_base`); the loss casts logits to fp32 for a clean softmax.

    `direct_load=True` (default) loads the pretrained depthformer via the
    DIRECT Linen→pure loader (`load_depthformer_from_safetensors_direct`),
    which reads the safetensors flat dict and writes the pure params with the
    same composed transform the sl bridge produces — but **without** building
    the fp32 sl model in the middle. It is bit-identical to the sl bridge
    (verified by `tests/mlx_pure/parity/test_direct_loader_parity.py`) and
    peaks at ~model size (~5 GB for base) instead of ~model + fp32-sl (~20 GB,
    which thrashes a 16 GB Mac). Set `direct_load=False` to fall back to the
    reference sl-bridge `load_sft_depthformer_from_safetensors`.
    """
    if param_dtype is not None:
        spec.param_dtype = param_dtype  # instance attr shadows the class default
    if dropout_prob:
        # Residual dropout for SFT regularization (GUESS placement — see the
        # todos in mlx_pure/transformer.py). Set on the spec before building;
        # the trainer calls model.train() so it's active during training and
        # off under model.eval(). Inference rebuilds with the 0.0 default.
        spec.dropout_prob = dropout_prob
    model = spec.build_decoder()
    Q = spec.target_tokens_config.rvq_truncation_level
    dummy_src = mx.zeros((1, 4, spec.input_num_channels), mx.int32)
    dummy_tgt = mx.zeros((1, 4, Q), mx.int32)
    model.decoder(dummy_tgt, encoded_source=model.encode(dummy_src))
    mx.eval(model.parameters())

    if checkpoint_path:
        logging.info("[sft-mlx] loading pretrained depthformer from %s",
                     checkpoint_path)
        if direct_load:
            # DIRECT Linen→pure load: no fp32 sl twin in memory (peaks ~model
            # size, the only path that fits mrt2_base on 16 GB). Bit-identical
            # to the sl bridge — see test_direct_loader_parity.py.
            load_depthformer_from_safetensors_direct(model, checkpoint_path)
        else:
            # Reference sl-bridge load: builds a structurally-matching sl
            # sampler, loads the Linen checkpoint into it, and mirrors the
            # depthformer weights into our EncoderDecoder (codec skipped —
            # training never needs it). ``model_name`` selects the sl shape.
            load_sft_depthformer_from_safetensors(
                model, checkpoint_path, model_name=model_name,
            )
        mx.eval(model.parameters())
    return model


def build_optimizer(config: SFTConfig) -> optim.Optimizer:
    # weight_decay stays 0.0 on the AdamW itself — MLX's AdamW can't mask which
    # leaves get decayed, so the DoRA-safe decoupled decay is applied MANUALLY
    # after each optimizer.update (see _decay_mask / _apply_decay below). Keeping
    # AdamW with wd=0 minimises the diff vs the original step.
    return optim.AdamW(
        learning_rate=config.learning_rate,
        betas=[config.adam_b1, config.adam_b2],
        weight_decay=0.0,
    )


def _build_decay_mask(model):
    """Set of trainable leaf-paths that should be decayed.

    A leaf is decayed (``lora_a``/``lora_b``, kernels, weights, embeddings)
    unless it's a DoRA magnitude / bias / norm scale — the shared
    :func:`trainer_common._decays_weight` policy keyed off each leaf's dotted
    path. Built ONCE (the trainable structure is fixed after injection). Returned
    as a path set rather than a mask *tree* so it lines up by exact leaf path
    even when ``trainable_parameters()`` carries empty subtrees (e.g. the
    adapter-free ``encoder`` in LoRA mode), which a parallel ``tree_map`` would
    trip over.
    """
    flat = mlx.utils.tree_flatten(model.trainable_parameters())
    return {path for path, _ in flat if utils._decays_weight(path)}


def _apply_decoupled_decay(model, decay_paths, factor: float) -> None:
    """In-place decoupled weight decay: ``p ← p·(1 − lr·wd)`` on decay-eligible
    trainable leaves only (``factor`` is ``1 − lr·wd``).

    Applied AFTER ``optimizer.update`` (outside the compiled grad fn) so it never
    perturbs ``mx.compile``'s traced graph; the next ``mx.eval(model.state, ...)``
    materialises it together with the optimizer step. No-op when wd==0 (the
    caller guards on that, so these ops only run when decay is on). Operates on
    the flattened trainable tree (path-keyed) so it survives empty subtrees.
    """
    flat = mlx.utils.tree_flatten(model.trainable_parameters())
    decayed = [
        (path, (p * factor) if path in decay_paths else p)
        for path, p in flat
    ]
    model.update(mlx.utils.tree_unflatten(decayed))


def _setup_model(config: SFTConfig, spec, *, model, model_name, checkpoint_path):
    """Build the model (if not supplied) and apply adaptation, returning it.

    LoRA/DoRA injects adapters and freezes everything else (``mark_lora_trainable``
    then unfreezes just the adapters); otherwise ``freeze_encoder`` does full SFT
    of the decoder. Mirrors the NNX trainer's ``lora_rank>0`` elif
    ``freeze_encoder`` structure and logs a parameter census.
    """
    if model is None:
        model = build_model(spec, checkpoint_path=checkpoint_path,
                            model_name=model_name,
                            param_dtype=mx.bfloat16 if config.bf16 else None,
                            dropout_prob=config.dropout_prob)
    n_total = count_params(model.parameters())
    logging.info("[sft-mlx] base params: %s", f"{n_total:,}")

    if config.lora_rank > 0:
        targets = all_linear_targets if config.lora_all_linears else default_targets
        n_lora = inject_lora(
            model, rank=config.lora_rank, alpha=config.lora_alpha,
            dora=config.lora_dora, targets=targets, seed=config.seed,
        )
        if n_lora == 0:
            raise ValueError(
                "inject_lora wrapped 0 layers — target predicate matched "
                "nothing; the model would not train."
            )
        mark_lora_trainable(model)
        n_trainable = count_params(model.trainable_parameters())
        logging.info(
            "[sft-mlx] %s: wrapped %d layers, rank=%d, alpha=%s, targets=%s. "
            "adapter params: %s  frozen: %s",
            "DoRA" if config.lora_dora else "LoRA",
            n_lora, config.lora_rank, config.lora_alpha,
            "all_linears" if config.lora_all_linears else "ffn",
            f"{n_trainable:,}", f"{n_total:,}")
    elif config.freeze_encoder:
        model.encoder.freeze()
        n_trainable = count_params(model.trainable_parameters())
        logging.info("[sft-mlx] froze encoder. trainable: %s  frozen: %s",
                     f"{n_trainable:,}", f"{n_total - n_trainable:,}")
    return model


def _materialize_eval_batches(config: SFTConfig, spec, *, style_tokens):
    """Draw a FIXED list of held-out eval batches once (not re-drawn each eval).

    Held-out excerpts vary widely in note density, so a rotating sample makes the
    eval-loss curve noisy; a fixed set + the deterministic eval conditioning
    (``make_eval_dataset``) gives a clean, comparable signal. Returns ``None``
    when no eval set / frequency is configured.
    """
    if not (config.valid_dir and config.valid_freq > 0):
        return None
    eval_ds = utils.make_eval_dataset(config.valid_dir, config, spec,
                                      seed=config.seed + 1,
                                      style_tokens=style_tokens)
    eval_it = iter(eval_ds)
    return [
        to_source_target(next(eval_it), spec.target_tokens_config,
                         asarray=mx.array)
        for _ in range(config.valid_batches)
    ]


def _make_writers(config: SFTConfig):
    """Build the run's metric writers: TensorBoard (TF-free tensorboardX, safe
    co-resident with MLX/Metal) plus an optional W&B mirror, hparams logged to
    each. Returns ``(tb, wb)``, either of which may be ``None``."""
    # `wandb_writer` is a lazy (TF-pulling) import — see the top-of-file note;
    # `tb_writer` is TF-free but imported here for locality.
    from magenta_rt.sft.tb_writer import maybe_make_tb_writer
    from magenta_rt.sft.wandb_writer import (
        dataclasses_to_dict,
        maybe_make_wandb_writer,
    )

    hparams = dataclasses_to_dict(config)
    tb = maybe_make_tb_writer(config)
    if tb is not None:
        tb.write_hparams(hparams)
        logging.info("[sft-mlx] TensorBoard logdir: %s  "
                     "(tensorboard --logdir %s --samples_per_plugin audio=200)",
                     tb.logdir, config.output_dir)
    wb = maybe_make_wandb_writer(config)
    if wb is not None:
        wb.write_hparams(hparams)
        logging.info("[sft-mlx] W&B logging enabled")
    return tb, wb


def _make_step_fns(model, optimizer, loss_and_grad_fn, config: SFTConfig):
    """Build the ``mx.compile``'d training kernels.

    ``mx.compile(..., inputs=state, outputs=state)`` collapses forward +
    backward + optimizer update into a single Metal graph rather than ~hundreds
    of eager kernel launches; ``inputs``/``outputs`` are mandatory — without them
    ``mx.compile`` snapshots ``model.state`` / ``optimizer.state`` at trace time
    and per-step updates never propagate. Returns
    ``(step_fn, accum_grad_fn, apply_grads_fn)``:

    * ``step_fn`` — the fused single-step kernel (``accum=1``).
    * ``accum_grad_fn`` / ``apply_grads_fn`` — the gradient-accumulation pair.
      ``accum_grad_fn`` fuses forward + backward + the running-sum add (only the
      adapter-sized accumulator crosses the Python boundary); ``apply_grads_fn``
      fuses scale + clip + optimizer update. The LR schedule advances per
      *update*, matching the NNX ``optax.MultiSteps`` semantics.
    """
    state = [model.state, optimizer.state]
    inv_accum = 1.0 / max(1, config.gradient_accumulation_steps)

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(source, target):
        loss, grads = loss_and_grad_fn(source, target)
        if config.max_grad_norm > 0:
            grads, gnorm = clip_grad_norm(grads, config.max_grad_norm)
        else:
            leaves = [g for _, g in mlx.utils.tree_flatten(grads) if g is not None]
            gnorm = mx.sqrt(sum(mx.sum(g * g) for g in leaves))
        optimizer.update(model, grads)
        return loss, gnorm

    @partial(mx.compile, inputs=[model.state], outputs=[model.state])
    def accum_grad_fn(accum, source, target):
        loss, grads = loss_and_grad_fn(source, target)
        accum = mlx.utils.tree_map(lambda a, g: a + g, accum, grads)
        return loss, accum

    @partial(mx.compile, inputs=state, outputs=state)
    def apply_grads_fn(accum):
        grads = mlx.utils.tree_map(lambda g: g * inv_accum, accum)
        if config.max_grad_norm > 0:
            grads, gnorm = clip_grad_norm(grads, config.max_grad_norm)
        else:
            leaves = [g for _, g in mlx.utils.tree_flatten(grads) if g is not None]
            gnorm = mx.sqrt(sum(mx.sum(g * g) for g in leaves))
        optimizer.update(model, grads)
        return gnorm

    return step_fn, accum_grad_fn, apply_grads_fn


# %% [markdown]
# ## Main training loop
#
# `mx.eval(...)` is the MLX equivalent of forcing materialization on JAX's
# lazy compute graph. We eval after each step so the per-step graph doesn't
# accumulate across iterations — without this, MLX would hold every step's
# compute graph in memory simultaneously.

# %%
def train(config: SFTConfig, spec, *, model=None,
          model_name: str = "mrt2_small", checkpoint_path: str | None = None):
    utils.setup_logging()
    logging.info("[sft-mlx] device: %s   spec: %s", mx.default_device(),
                 type(spec).__name__)
    logging.info("[sft-mlx] model=%s  steps=%d  batch=%d x accum=%d (effective %d)"
                 "  lr=%.2e  bf16=%s  dropout=%.2f",
                 model_name, config.total_steps, config.batch_size,
                 config.gradient_accumulation_steps,
                 config.batch_size * config.gradient_accumulation_steps,
                 config.learning_rate, config.bf16, config.dropout_prob)
    cfg_path = utils.dump_config(config, config.output_dir)
    logging.info("[sft-mlx] wrote config: %s", cfg_path)

    # ---- Model + optimizer ----------------------------------------------
    model = _setup_model(config, spec, model=model, model_name=model_name,
                         checkpoint_path=checkpoint_path)

    optimizer = build_optimizer(config)
    loss_fn = make_loss_fn(model)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Decoupled AdamW weight decay (DoRA-safe). MLX's AdamW can't mask leaves, so
    # the decay is applied manually after each optimizer.update: a boolean mask
    # (built once over the trainable tree) selects the decay-eligible leaves
    # (magnitude / bias / norm-scale excluded — the shared _decays_weight policy)
    # and they're scaled by (1 − lr·wd). Built only when wd>0 so the wd=0 path is
    # byte-for-byte the original step (no extra ops, no extra eval).
    use_weight_decay = config.weight_decay > 0
    decay_mask = _build_decay_mask(model) if use_weight_decay else None

    # Compiled training kernels (single-step + gradient-accumulation pair).
    step_fn, accum_grad_fn, apply_grads_fn = _make_step_fns(
        model, optimizer, loss_and_grad_fn, config)
    accum_steps = max(1, config.gradient_accumulation_steps)
    adapter_update_fn = (
        make_adapter_update_fn(model) if config.lora_rank > 0 else None)

    # ---- Data -----------------------------------------------------------
    # Optional single-style-prompt recipe: one fixed prompt's MusicCoCa tokens
    # overlaid on every example (None when unset; shared with the NNX trainer).
    style_tokens = utils.embed_style_prompt(config, model_name)
    ds = utils.make_dataset(config.data_dir, config, spec,
                            seed=config.seed, num_workers=config.num_workers,
                            style_tokens=style_tokens)
    it = iter(ds)

    # Optional eval + early stopping. eval_compiled / stopper are only needed
    # when an eval set exists. `inputs=[model.state]` is mandatory: a bare
    # `mx.compile(loss_fn)` snapshots the params at trace time, so every eval
    # would re-score the *initial* model and ignore training (the tell is a
    # perfectly constant eval/loss). No `outputs=` — eval doesn't mutate state.
    eval_batches = _materialize_eval_batches(config, spec, style_tokens=style_tokens)
    if eval_batches is not None:
        @partial(mx.compile, inputs=[model.state])
        def eval_compiled(source, target):
            encoded = model.encode(source)
            logits = model.decoder(target, encoded_source=encoded)  # [B, T, Q, V]
            logits_f = logits.astype(mx.float32)
            log_probs = nn.log_softmax(logits_f, axis=-1)
            nll = -mx.take_along_axis(
                log_probs, target[..., None].astype(mx.int32), axis=-1,
            ).squeeze(-1)                                            # [B, T, Q]
            loss = nll.mean()
            ce_cb = nll.mean(axis=(0, 1))                            # [Q]
            acc_cb = (mx.argmax(logits_f, axis=-1) == target).mean(axis=(0, 1)) # [Q]
            return loss, ce_cb, acc_cb

        stopper = EarlyStopper(
            min_delta=config.early_stop_min_delta,
            patience=config.early_stop_patience,
        )

    tb, wb = _make_writers(config)

    # Periodic audio sampling (opt-in). Conditioning is drawn from the held-out
    # eval set when available (a fairer "did the sound improve" probe than a
    # memorized training excerpt); falls back to a training batch otherwise.
    sample_writer = AudioSampleWriter(
        model=model, config=config,
        model_name=model_name, checkpoint_path=checkpoint_path,
    )
    if config.sample_every_steps:
        if eval_batches:
            sample_writer.set_source(eval_batches[0][0])
        logging.info("[sft-mlx] audio sampling every %d steps: %s  (conditioning: %s)",
                     config.sample_every_steps,
                     "enabled" if sample_writer.available else "UNAVAILABLE",
                     "held-out eval" if eval_batches else "training batch")

    # Per-step JSONL loss log (plotted to loss_curve.png at exit).
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    def _emit_sample(step: int) -> None:
        """Generate a clip, log its energy diagnostics, and (if W&B) the audio.

        The ``gen/rms`` and ``gen/frac_silent`` scalars make the energy-collapse
        failure mode (low-energy / mostly-silent free-running output despite a
        falling loss) a curve you can watch instead of a by-ear surprise. With
        W&B, the clip itself lands in the audio panel keyed by ``step`` so you
        can scrub the slider and *hear* the sound evolve across training.
        """
        result = sample_writer(step)
        if result is None:
            return
        path, audio = result
        m = utils.audio_energy_metrics(audio, sample_rate=48_000)
        tag = "baseline" if step == 0 else f"step {step}"
        logging.info("  [sft-mlx] sample %-9s rms=%.4f frac_silent=%.2f -> %s",
                     tag, m["rms"], m["frac_silent"], path)
        log_file.write(json.dumps({
            "step": step, "gen_rms": m["rms"], "gen_frac_silent": m["frac_silent"],
        }) + "\n")
        log_file.flush()
        gen_scalars = {"gen/rms": m["rms"], "gen/frac_silent": m["frac_silent"]}
        if tb is not None:
            tb.write_scalars(step, gen_scalars)
            tb.write_audios(step, {"gen/audio": audio}, sample_rate=48_000)
            tb.flush()
        if wb is not None:
            wb.write_scalars(step, gen_scalars)
            wb.write_audios(step, {"sample": audio[None]}, sample_rate=48_000)

    # ---- Loop -----------------------------------------------------------
    model.train()
    t0 = time.time()
    losses = []
    last_saved_step = 0
    step_dt = 0.0  # smoothed seconds/step for ETA
    # Windowed-average accumulator: the console + W&B `train/loss`/`train/grad_norm`
    # report the mean over `log_every_steps` rather than the single noisy step
    # value. Shared with the NNX trainer via `flax.nnx.MultiMetric` (TF-free, so
    # safe in this Metal process); it accumulates host floats in tiny jax CPU ops.
    # Per-step raw values still land in train_log.jsonl for the fine-grained curve.
    train_metrics = utils.make_train_metrics()

    # Baseline clip (step 0): the pre-SFT reference for the audio panel / by-ear
    # comparison. Adapters are still zero-init, so this is the untouched base.
    if sample_writer.available:
        _emit_sample(0)
        model.train()

    for step in range(1, config.total_steps + 1):
        step_start = time.time()
        optimizer.learning_rate = utils.lr_at_step(step, config)

        if accum_steps == 1:
            # Pre-tokenized data -> no codec. For audio data, pass a
            # SpectroStream via ``codec=`` to encode samples->tokens on device.
            source, target = to_source_target(
                next(it), spec.target_tokens_config, asarray=mx.array,
            )
            sample_writer.set_source(source)
            loss, gnorm = step_fn(source, target)
            # Decoupled weight decay AFTER the optimizer update, OUTSIDE the
            # compiled step (so mx.compile's traced graph is untouched). lr was
            # set on the optimizer above, so factor reflects this step's lr.
            if use_weight_decay:
                _apply_decoupled_decay(
                    model, decay_mask,
                    1.0 - float(optimizer.learning_rate) * config.weight_decay,
                )
            # Force materialization so the lazy graph from this step is freed
            # before the next forward kicks off (no-op cost when compiled,
            # since the compiled graph already produces concrete outputs).
            mx.eval(model.state, optimizer.state, loss, gnorm)
            loss_val = float(loss)
        else:
            accum = mlx.utils.tree_map(
                mx.zeros_like, model.trainable_parameters()
            )
            loss_sum = 0.0
            for _ in range(accum_steps):
                source, target = to_source_target(
                    next(it), spec.target_tokens_config, asarray=mx.array,
                )
                sample_writer.set_source(source)
                loss, accum = accum_grad_fn(accum, source, target)
                # Materialize per micro-batch so only the (small) accumulator
                # is held, not accum_steps full backward graphs.
                mx.eval(accum, loss)
                loss_sum += float(loss)
            gnorm = apply_grads_fn(accum)
            # Decoupled weight decay AFTER the (accumulated) optimizer update,
            # OUTSIDE the compiled apply_grads_fn — same policy as the accum=1
            # path. One decay per parameter update (matching MultiSteps).
            if use_weight_decay:
                _apply_decoupled_decay(
                    model, decay_mask,
                    1.0 - float(optimizer.learning_rate) * config.weight_decay,
                )
            mx.eval(model.state, optimizer.state, gnorm)
            loss_val = loss_sum / accum_steps
        losses.append(loss_val)

        gnorm_val = float(gnorm)
        train_metrics.update(loss=loss_val, grad_norm=gnorm_val)
        log_file.write(json.dumps({
            "step": step, "loss": loss_val, "grad_norm": gnorm_val,
            "lr": float(optimizer.learning_rate),
        }) + "\n")
        log_file.flush()

        # Smoothed seconds/step (EMA) → ETA for the remaining steps.
        dt = time.time() - step_start
        step_dt = dt if step_dt == 0.0 else 0.8 * step_dt + 0.2 * dt
        eta = utils.format_eta(step_dt * (config.total_steps - step))

        if config.nan_check and not math.isfinite(loss_val):
            logging.error("[sft-mlx] non-finite loss at step %d — stopping.", step)
            break

        if step % config.log_every_steps == 0:
            summary = train_metrics.compute()
            train_metrics.reset()
            avg_loss = float(summary["loss"])
            avg_gnorm = float(summary["grad_norm"])
            anorm = adapter_l2_norm(model.trainable_parameters())
            arms = adapter_rms(model.trainable_parameters())
            
            scalars = {"train/loss": avg_loss, "train/grad_norm": avg_gnorm,
                       "train/adapter_norm": anorm, "train/adapter_rms": arms,
                       "train/lr": optimizer.learning_rate,
                       "perf/steps_per_sec": 1.0 / max(1e-6, step_dt)}
            
            rel_update = 0.0
            if adapter_update_fn is not None:
                rel_mean, rel_max = adapter_update_fn()
                rel_update = float(rel_mean)
                scalars["train/adapter_rel_update"] = rel_update
                scalars["train/adapter_rel_update_max"] = float(rel_max)

            log_file.write(json.dumps({
                "step": step,
                "adapter_norm": anorm,
                "adapter_rms": arms,
                "adapter_rel_update": rel_update
            }) + "\n")
            
            mem_str = ""
            if config.log_memory:
                mem_gb = mx.get_active_memory() / 1024**3
                peak_gb = mx.get_peak_memory() / 1024**3
                mem_str = f"  mem={mem_gb:.2f}GB peak={peak_gb:.2f}GB"
                scalars["mem/active_gb"] = mem_gb
                scalars["mem/peak_gb"] = peak_gb
                
            logging.info(
                "  step %4d/%d  loss=%.4f  grad_norm=%.3f  adapter_rms=%.4f  "
                "rel_dW=%.4f  lr=%.2e  %.1f steps/s  eta %s%s",
                step, config.total_steps, avg_loss, avg_gnorm, arms,
                rel_update, optimizer.learning_rate, 1.0 / max(1e-6, step_dt), eta, mem_str)
                
            if tb is not None:
                tb.write_scalars(step, scalars)
            if wb is not None:
                wb.write_scalars(step, scalars)

        # Periodic eval + early stopping (over the fixed eval batch set).
        if eval_batches is not None and step % config.valid_freq == 0:
            model.eval()
            eval_metrics = utils.make_eval_metrics()
            ce_sum = None
            acc_sum = None
            n_eval = 0
            for vsource, vtarget in eval_batches:
                vloss, vce, vacc = eval_compiled(vsource, vtarget)
                mx.eval(vloss, vce, vacc)
                eval_metrics.update(loss=float(vloss))
                
                vce_np = np.asarray(vce)
                vacc_np = np.asarray(vacc)
                ce_sum = vce_np if ce_sum is None else ce_sum + vce_np
                acc_sum = vacc_np if acc_sum is None else acc_sum + vacc_np
                n_eval += 1
                
            vmean = float(eval_metrics.compute()["loss"])
            scalars = {"eval/loss": vmean}
            summary_str = ""
            if n_eval > 0:
                ce_cb = ce_sum / n_eval
                acc_cb = acc_sum / n_eval
                for q in range(len(ce_cb)):
                    scalars[f"eval/ce_cb/{q:02d}"] = float(ce_cb[q])
                    scalars[f"eval/acc_cb/{q:02d}"] = float(acc_cb[q])
                last = len(ce_cb) - 1
                summary_str = (f"  cb0 ce/acc={ce_cb[0]:.3f}/{acc_cb[0]:.2f}"
                               f"  cb{last} ce/acc={ce_cb[last]:.3f}/{acc_cb[last]:.2f}")
            
            logging.info("  step %4d/%d  eval/loss=%.4f%s", step,
                         config.total_steps, vmean, summary_str)
            log_file.write(json.dumps({"step": step, "eval_loss": vmean}) + "\n")
            log_file.flush()
            if tb is not None:
                tb.write_scalars(step, scalars)
            if wb is not None:
                wb.write_scalars(step, scalars)
            if stopper.update(vmean):
                logging.info("[sft-mlx] early stop at step %d (best=%.4f)",
                             step, stopper.best)
                break
            model.train()

        if (sample_writer.available
                and step % config.sample_every_steps == 0):
            _emit_sample(step)
            model.train()

        if config.save_every_steps and step % config.save_every_steps == 0:
            p = save_checkpoint(model, step, config.output_dir,
                                lora_only=config.lora_rank > 0)
            logging.info("  [sft-mlx] saved %s", p)
            last_saved_step = step

    # Always persist the final (most-trained) state — the periodic save is
    # missed when the loop ends on a non-save step, or when early-stop breaks
    # right at a save-step boundary (before the save block runs).
    if config.save_every_steps and step != last_saved_step:
        p = save_checkpoint(model, step, config.output_dir,
                            lora_only=config.lora_rank > 0)
        logging.info("  [sft-mlx] saved final %s", p)

    log_file.close()
    curve = _plot_loss_curve(log_path, config.output_dir)
    if curve:
        logging.info("[sft-mlx] loss curve: %s", curve)
    if tb is not None:
        tb.close()
    if wb is not None:
        wb.close()
    elapsed = time.time() - t0
    logging.info("[sft-mlx] %d steps in %s (%.1fs/step)", step,
                 utils.format_eta(elapsed), elapsed / max(1, step))
    return losses


def _plot_loss_curve(log_path: str, output_dir: str) -> str | None:
    """Render train_log.jsonl to loss_curve.png (skipped without matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    steps, losses = [], []
    eval_steps, eval_losses = [], []
    gen_steps, gen_silent = [], []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec:
                steps.append(rec["step"])
                losses.append(rec["loss"])
            elif "eval_loss" in rec:
                eval_steps.append(rec["step"])
                eval_losses.append(rec["eval_loss"])
            elif "gen_frac_silent" in rec:
                gen_steps.append(rec["step"])
                gen_silent.append(rec["gen_frac_silent"])
    if not steps:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=0.8, alpha=0.5, label="train loss")
    if len(losses) >= 20:
        k = max(5, len(losses) // 50)
        smoothed = [
            sum(losses[max(0, i - k + 1): i + 1])
            / (i - max(0, i - k + 1) + 1)
            for i in range(len(losses))
        ]
        ax.plot(steps, smoothed, linewidth=1.6, label=f"train loss (avg {k})")
    if eval_steps:
        ax.plot(eval_steps, eval_losses, "o-", linewidth=1.4, color="crimson",
                label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    # Generated-audio energy on a twin axis: a rising frac-silent line flags the
    # energy-collapse failure mode even while the loss keeps falling.
    if gen_steps:
        ax2 = ax.twinx()
        ax2.plot(gen_steps, gen_silent, "s--", linewidth=1.2, color="darkorange",
                 label="gen frac_silent")
        ax2.set_ylabel("generated frac_silent")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# %% [markdown]
# ## CLI

# %%
def _parse_args(argv=None):
    cli = utils.parse_train_cli(utils.TrainCLI, argv)
    return utils.to_sft_config(cli), cli.model_name, cli.checkpoint


def _resolve_spec(model_name: str):
    return utils.resolve_spec(
        model_name, tiny_cls=TinyPOCSpecMLX, lookup=get_model_class,
    )


# %%
if __name__ == "__main__":
    config, model_name, checkpoint_path = _parse_args()
    spec = _resolve_spec(model_name)

    if not config.data_dir:
        raise ValueError("Must specify `--data_dir` containing the SFT dataset.")
    train(config, spec, model_name=model_name, checkpoint_path=checkpoint_path)
