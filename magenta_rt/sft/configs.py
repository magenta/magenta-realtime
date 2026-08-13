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

"""Training configuration + a tiny model spec for the POC.

``SFTConfig`` is framework-neutral. ``TinyPOCSpec`` is a shrunk
``MagentaRT2ModelBase`` that fits a few MB of params so the trainer can run
end-to-end on a laptop with random weights.
"""

# TODO(pretraining): Add nnx.Dropout to TransformerBlock attention/FFN
# and LocalSelfAttention. Wire dropout_prob from ModelSpec. The trainer
# must call model.train() before the loop and model.eval() for sample/
# eval/tabulate forwards (NNX cascades `deterministic` automatically).
# If the encoder is frozen, also call model.encoder.eval() once at
# startup so encoder dropout stays off even under model.train(). For
# SFT-from-pretrained, dropout_prob=0 is the current default and this
# TODO can stay deferred.

from __future__ import annotations

import dataclasses
from typing import Optional

from flax.struct import dataclass

from magenta_rt.nnx.model import (
    MagentaRT2ModelBase,
    ModelSpec,
    SPECTROSTREAM,
    TokensConfig,
)


@dataclasses.dataclass
class SFTConfig:
    """Hyperparameters for SFT runs (mirrors the JAX/MLX trainer configs)."""

    # Data

    data_dir: str = ""
    valid_dir: str = ""           # optional held-out dataset dir; "" → no eval
    # Training crop length in seconds. None (the default) = no crop: use each
    # record at its full stored length (right for fixed-length exports, e.g. the
    # trimmed 2 s records). Set an int to randomly crop/zero-pad to N seconds.
    crop_length_seconds: Optional[int] = None
    # Single-style-prompt recipe (training time): condition every example on one
    # fixed text prompt's MusicCoCa tokens instead of per-clip style. Right for a
    # codec-only export (no per-clip style) — anchors the LoRA to one style with
    # no re-export. Embedded once in an isolated subprocess; condition inference
    # on the same prompt. None = use whatever style the dataset carries.
    style_prompt: Optional[str] = None
    musiccoca_sticky_prob: float = 0.995
    num_workers: int = 0          # 0 → no mp_prefetch (POC-friendly)
    per_worker_buffer_size: int = 2
    # Load the whole dataset into RAM at init (workers read from arrays, no disk
    # I/O) vs the default memory-mapped read (low RAM, small per-read cost — right
    # for large exports / limited host memory). For a SMALL dataset that fits in
    # RAM, prefer ``--load_into_memory --num_workers 0``: in-RAM data removes the
    # disk I/O the prefetch workers exist to hide, so a single in-process reader
    # is simplest and fastest (and sidesteps grain's mp shared-memory overhead).
    load_into_memory: bool = False
    # Mask ALL content conditioning (MusicCoCa style, note pianoroll, drum
    # pianoroll): exclude those leaves at load so ``prepare_source_tokens`` falls
    # back to each stream's learned unconditional (-1 → dropout) token, every
    # example. Trains only the SpectroStream-target relationship under fully
    # unconditional content conditioning (the CFG-strength channels are still
    # synthesized as usual). Leaves the conditioning code/data intact — this is a
    # run-time switch, no re-export. Condition inference the same way (all -1).
    mask_conditioning: bool = False
    # Mask content conditioning for the EVAL pass only, independent of training.
    # ``None`` (default) → eval matches ``mask_conditioning``. Set ``True`` to
    # measure UNCONDITIONAL eval loss (notes/drums → dropout token, style still
    # overlaid by ``--style_prompt``) while training conditions on notes — i.e.
    # train with the per-clip notes as an auxiliary signal but score the model in
    # the regime you actually generate in (text-style only, masked notes, like the
    # jam app), comparable to the masked baseline. Set ``False`` to force a
    # conditioned eval even when training is masked.
    eval_mask_conditioning: Optional[bool] = None

    # Optimizer
    #
    # Defaults are the best config from the 2026-06-17 overnight mrt2_base LoRA
    # sweep (electronic-music recipe, ~1000 updates): lr 3e-4 with a short linear
    # warmup then an rsqrt decay that actually engages over a ~1000-update run
    # (``rsqrt_timescale=300`` decays to ~0.5x peak by the end; the old default
    # ``10_000`` never decayed in that regime, i.e. a constant LR). lr >= 5e-4 was
    # worse. Tuned for the ~1000-update sweet spot; raise ``rsqrt_timescale`` for
    # much longer runs so the decay isn't too aggressive.
    learning_rate: float = 3e-4
    warmup_steps: int = 50
    rsqrt_timescale: int = 300
    max_grad_norm: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    # Decoupled AdamW weight decay (``p ← p − lr·wd·p``, applied alongside the
    # Adam step — NOT coupled L2 through the gradient). DoRA magnitude, biases,
    # and norm scales are EXCLUDED from decay (decaying the magnitude shrinks
    # ‖W‖ and re-introduces energy collapse); lora_a/lora_b, kernels/weights,
    # and embeddings ARE decayed. Default 0.0 = off (no behaviour change), and
    # both trainers skip the decay ops entirely at 0.0.
    weight_decay: float = 0.0
    # >1 wraps the optimizer in optax.MultiSteps: gradients accumulate over
    # this many micro-batches before one parameter update (the LR schedule
    # advances per *update*, not per micro-batch).
    gradient_accumulation_steps: int = 1

    # Training

    # Default 2; mrt2_base LoRA on a 16 GB card (e.g. RTX 4080) needs
    # batch_size 1 — batch 2 tries to allocate ~19.5 GB and OOMs even with bf16
    # + remat, because the backward pass still materializes the frozen base's
    # activations to reach the LoRA adapters. Smaller models / bigger cards can
    # raise it. (See the bf16 + remat notes below: all three are required
    # together to train mrt2_base on the 4080.)
    batch_size: int = 2
    total_steps: int = 50
    seed: int = 0
    freeze_encoder: bool = True
    # Store base weights in bfloat16 (MLX trainer). Required to fit mrt2_base
    # (2.4 B params) on 16 GB — fp32 weights (9.6 GB) OOM against the fp32
    # sl-bridge loader; bf16 (4.8 GB) keeps the load peak ~15 GB. LoRA adapters
    # inherit the dtype; the loss casts logits to fp32 for a clean softmax.
    bf16: bool = False
    # Gradient checkpointing (nnx.remat per decoder layer): recompute each
    # layer's activations in the backward pass instead of storing them, cutting
    # training activation memory from O(num_layers) to ~one layer. Numerically
    # identical (just recompute), ~20-30% slower steps. Off by default; required
    # to fit full-length mrt2_base (20 temporal layers) LoRA on a 16 GB GPU. NNX
    # path (wired through build_model → EncoderDecoder.from_config).
    remat: bool = False
    # Transformer dropout during SFT: FFN (act + output), attention-sublayer
    # residual, and (via the fallback below) attention-probability dropout.
    # Placement reconciled against the sl ground truth (commit 4c0a72b). 0.0 =
    # off; build_model passes it to EncoderDecoder.from_config; inference/eval
    # keep it off. NOTE: enabling attention-probability dropout switches the
    # attention to a manual (non-flash) path — more memory + slower.
    dropout_prob: float = 0.0
    # The newly-reconciled (sl-derived) dropout knobs, set on the model spec by
    # build_model so EncoderDecoder.from_config reads them:
    #   * whole_source_dropout_rate — per-example zeroing of the ENTIRE source
    #     conditioning (CFG-style; zeros to token 0, no rescale).
    #   * temporal_input_dropout_prob — per-example zeroing of the decoder's own
    #     past (temporal input), encouraging reliance on the encoder.
    #   * temporal_self_attention_dropout_prob — attention-probability dropout on
    #     the temporal self-attention (None → falls back to dropout_prob).
    whole_source_dropout_rate: float = 0.0
    temporal_input_dropout_prob: float = 0.0
    temporal_self_attention_dropout_prob: Optional[float] = None
    # Audio samples condition on a held-out EVAL clip's source (its per-clip
    # MusicCoCa etc.) instead of a training batch's. Needs valid_dir + valid_freq.
    sample_from_eval: bool = False

    # LoRA. lora_rank=0 → full SFT (freeze_encoder still applies).
    # lora_rank>0 → inject adapters and train only those (the base weights
    # stay frozen by selection, so `freeze_encoder` is implicitly true).
    # Default vs all-linears targets differ per backend, because the two model
    # trees expose different wrap points:
    #   * NNX  — default: attention QKV projections; all_linears: + FFN.
    #   * MLX  — default: FFN linears; all_linears: + attention output proj.
    #     (mlx_pure stores q/kv as bare arrays, not modules — see
    #     magenta_rt.sft.lora_mlx for the rationale.)
    # The adapter math and merge are identical, so a merged checkpoint
    # round-trips across backends regardless of which layers were adapted.
    lora_rank: int = 0
    lora_alpha: float = 0.0       # 0 → no scaling; common values 8/16/32
    lora_all_linears: bool = False  # True → widen the target set (see above)
    # Override the LoRA target preset by name: "default" (attn QKV), "all_linears"
    # (attn QKV + FFN), or "all_plus" (every nnx.Linear incl the depth-input
    # adapter + logits head). None → fall back to lora_all_linears. Recorded in
    # the adapter file so it round-trips at inference.
    lora_targets: Optional[str] = None
    # DoRA (Liu et al.): add a learned per-output magnitude so the adapter sets
    # weight direction and magnitude independently. Recommended over plain LoRA —
    # the magnitude/direction split resists the effective-weight-norm runaway
    # that collapses plain LoRA's free-running output at full strength. MLX path
    # only (mlx_pure backend); inference can still blend via set_lora_strength.
    lora_dora: bool = False

    # Style augmentation in embedding space (StyleEmbeddingJitter): perturb
    # the stored MusicCoCa embedding and re-quantize per example. The 12 RVQ
    # style tokens nearly uniquely fingerprint an excerpt, an easy shortcut
    # for SFT to memorize (style -> exact codes); jitter breaks it. Needs an
    # export that kept ``musiccoca_embedding`` (the default).
    style_jitter_std: float = 0.0   # 0 → disabled; try 0.03–0.08
    style_jitter_prob: float = 1.0

    # Eval / early-stop

    valid_freq: int = 0           # 0 → no eval mid-training
    valid_batches: int = 8        # how many eval batches to average over
    # Eval micro-batch size, decoupled from the training ``batch_size``. None →
    # reuse ``batch_size``. Lets eval run at a different size (e.g. a larger eval
    # batch for a smoother eval-loss estimate, or a smaller one to fit a bigger
    # model's eval forward). Total eval examples per round =
    # ``valid_batches * (eval_batch_size or batch_size)``.
    eval_batch_size: Optional[int] = None
    # Eval conditioning is deterministic (fixed CFG, no input dropout) so the
    # eval-loss curve is a clean generalization signal — the training pipeline's
    # stochastic CFG + 15% feature dropout would otherwise make it too noisy to
    # read. These pin the eval CFG strengths to the inference defaults of the
    # generation systems (e.g. ``MagentaRT2System.generate`` cfg_musiccoca=3.0,
    # cfg_notes=1.0, cfg_drums=1.0), so eval loss is read at the strengths you
    # actually sample at. (Training always samples CFG uniformly per example.)
    eval_cfg_musiccoca: float = 3.0
    eval_cfg_notes: float = 1.0
    eval_cfg_drums: float = 1.0
    early_stop_min_delta: float = 1e-4
    early_stop_patience: int = 5  # # validation rounds without improvement
    nan_check: bool = True        # short-circuit on loss==NaN/Inf
    # How many *consecutive* non-finite losses to tolerate before stopping.
    # 0 (the default) stops on the first, which is safe but lets a single bad
    # batch end a multi-hour run. Pair with `skip_nonfinite_steps` so the bad
    # step is skipped rather than survived-but-corrupting; persistent
    # divergence still terminates.
    nan_patience: int = 0
    # If > 0, wrap the optimizer in `optax.apply_if_finite`: a step whose
    # gradients are non-finite leaves the parameters and optimizer state
    # untouched, and only after this many consecutive such steps does the NaN
    # propagate. 0 (the default) leaves the transform unwrapped — the wrapper
    # adds state, so turning it on changes the optimizer-state structure and
    # existing checkpoints will not restore into it.
    skip_nonfinite_steps: int = 0

    # I/O

    # Auto-resume: when output_dir already holds checkpoints, restore the
    # latest one (model + optimizer + grain data-iterator position) and
    # continue from that step. Disable for a fresh run into the same dir.
    resume: bool = True
    output_dir: str = "./sft_checkpoints"
    # TensorBoard event dir. "" → "<output_dir>/tb" (events live inside the run,
    # so `tensorboard --logdir <output_dir>` shows one run and `--logdir <parent>`
    # compares many). Both trainers write here: NNX via clu.metric_writers, MLX
    # via the TF-free tensorboardX shim (magenta_rt.sft.tb_writer).
    tensorboard_dir: str = ""
    log_every_steps: int = 1
    save_every_steps: int = 25
    sample_every_steps: int = 0   # 0 → disabled
    # LoRA/DoRA strength applied to the periodic audio sampler (set_lora_strength
    # on the separate generation model — 1.0 = full adapter, 0.0 = base). A
    # strongly-trained adapter often sounds best blended toward base (~0.6–0.8);
    # this lets the monitored samples preview that without affecting training.
    # No effect in full-SFT (non-LoRA) mode.
    sample_lora_strength: float = 1.0
    max_to_keep: int = 2

    # Telemetry

    log_memory: bool = False      # log host/device memory each step
    use_wandb: bool = False       # mirror TB writes to W&B
    wandb_project: str = ""       # "" → derived from output_dir basename
    wandb_name: str = ""          # "" → derived from output_dir leaf
    wandb_entity: str = ""


_TINY = ModelSpec(
    num_layers=2,
    model_dims=64,
    hidden_dims=128,
    num_heads=2,
    dim_per_head=32,
    ffn_use_gated_activation=False,
)


@dataclass
class TinyPOCSpec(MagentaRT2ModelBase):
    """Tiny mrt2-shaped spec for end-to-end POC training on a laptop.

    Uses a plain (non-pretrained-MusicCoCa) encoder embedder so the tiny
    model doesn't carry the full 12k-row dequantizer table.
    """

    encoder_size: ModelSpec = _TINY
    decoder_temporal_size: ModelSpec = _TINY
    decoder_depth_size: ModelSpec = _TINY

    use_pretrained_musiccoca_embedder: bool = False

    # Smaller target vocab (4 codebooks × 32 entries + 6 reserved = 134).
    spectrostream: TokensConfig = dataclasses.replace(
        SPECTROSTREAM, rvq_truncation_level=4, codebook_size=32,
    )

    encoder_max_past_horizon: int = 8
    decoder_temporal_self_attention_max_past_horizon: int = 8
    decoder_temporal_cross_attention_max_past_horizon: int = 8

    crop_length_seconds: int = 2
