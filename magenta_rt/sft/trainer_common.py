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

"""Shared, framework-neutral glue for the SFT trainers (``train_nnx`` / ``train_mlx``).

Keeps the duplicated bits in one place: CLI parsing over ``SFTConfig`` fields,
model-spec resolution, the warmup→rsqrt learning-rate schedule, and the dataset
factory. Everything here is backend-agnostic.

``lr_at_step`` dispatches on the *type* of ``step`` — a jax tracer (inside an
optax schedule) uses ``jax.numpy`` and returns a jax scalar; a python/numpy step
(the MLX host-set path) uses numpy and returns a ``float``. ``jax.numpy`` is
imported lazily, only on the jax path, so the MLX trainer never *computes* the
schedule with jax.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from typing import Optional

import numpy as np

from magenta_rt.sft import create_audiotree_dataset
from magenta_rt.sft.configs import SFTConfig


# ---------------------------------------------------------------------------
# CLI parsing (argbind over SFTConfig) + model-spec resolution
# ---------------------------------------------------------------------------
#
# The CLI is the config dataclass itself. ``argbind.bind(TrainCLI)`` derives one
# flag per field — ``--batch_size 4``, ``--lora_dora``, ``--checkpoint path`` —
# so there is no flag list to keep in sync with ``SFTConfig`` (the failure mode
# of the old hand-rolled argparse). ``without_prefix=True`` keeps the flags bare
# rather than argbind's default ``--TrainCLI.batch_size`` namespacing.
#
# Bools accept both the bare ``--lora_dora`` and an explicit ``--lora_dora False``;
# there is no ``--no-flag`` spelling (use ``--freeze_encoder False``).
#
# argbind also adds ``--args.save out.yml`` / ``--args.load in.yml``, which is
# the reason to prefer it here: a run's exact configuration can be written out
# and replayed verbatim, and it is the same config mechanism ``audiotree`` uses.


@dataclasses.dataclass
class TrainCLI(SFTConfig):
    """``SFTConfig`` plus the two run-level flags every trainer needs.

    Inherits every ``SFTConfig`` field (so they all become CLI flags) and adds
    the model selector + optional pretrained checkpoint. Backend-specific
    trainers subclass this to add their own flags (e.g. nnx ``export_linen``).
    """

    model_name: str = "tiny_poc"
    """'tiny_poc' or a key in the backend's model registry (e.g. mrt2_small)."""
    checkpoint: Optional[str] = None
    """Optional safetensors checkpoint to fine-tune from (omit for random init)."""


def to_sft_config(cli: SFTConfig) -> SFTConfig:
    """Project a ``TrainCLI`` (or any subclass) back to a plain ``SFTConfig``."""
    return SFTConfig(
        **{f.name: getattr(cli, f.name) for f in dataclasses.fields(SFTConfig)}
    )


def parse_train_cli(cli_cls=TrainCLI, argv=None):
    """argbind over ``cli_cls`` (a ``TrainCLI`` subclass); returns the instance.

    argbind is imported lazily so merely importing this module (e.g. in tests
    that only want the LR schedule or dataset factory) doesn't require it.

    ``argbind.parse_args`` reads ``sys.argv`` directly, so an explicit ``argv``
    is applied by swapping it for the duration of the parse (keeping the
    program name in slot 0) and restoring it afterwards.
    """
    import argbind

    bound = argbind.bind(cli_cls, without_prefix=True)
    if argv is None:
        args = argbind.parse_args()
    else:
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], *argv]
        try:
            args = argbind.parse_args()
        finally:
            sys.argv = saved_argv
    with argbind.scope(args):
        return bound()


def resolve_spec(model_name: str, *, tiny_cls, lookup):
    """``'tiny_poc'`` -> ``tiny_cls()``; otherwise ``lookup(model_name)()``.

    ``lookup`` maps a name to a model-spec *class* (e.g. ``MODEL_REGISTRY.__getitem__``
    for nnx, ``get_model_class`` for mlx_pure).
    """
    if model_name == "tiny_poc":
        return tiny_cls()
    return lookup(model_name)()


# ---------------------------------------------------------------------------
# Weight-decay policy (shared by both trainers)
# ---------------------------------------------------------------------------
#
# Decoupled AdamW decay (``p ← p − lr·wd·p``) is applied to a trainable leaf
# UNLESS the leaf is one of these — which are EXCLUDED from decay:
#   * the DoRA ``magnitude`` (critical: decaying it shrinks ‖W‖ and
#     re-introduces the energy-collapse failure mode),
#   * biases,
#   * norm scales (RMSNorm / LayerNorm scale/gain).
# So ``lora_a`` / ``lora_b``, kernels/weights, and embeddings ARE decayed.
#
# A single predicate keyed off the dotted leaf-path string is shared by both
# backends so the policy can't drift. The path strings differ slightly per
# backend (verified against the live trees):
#   * NNX  — ``...ffn_layer1.magnitude..value`` (trailing ``.value`` from the
#     Variable wrapper); norm scale is ``...pre_norm.scale..value``; bias is
#     ``...base.bias..value``.
#   * MLX  — ``...ffn_layer1.linear.magnitude``; norm scale is
#     ``...pre_norm.weight`` (MLX RMSNorm/LayerNorm name the gain ``weight``,
#     the SAME name a Linear uses — so a bare ``weight`` is only treated as a
#     norm scale when its parent module segment ends in ``norm``/``ln``).
# Empty segments (NNX's ``..value``) and a trailing ``value`` segment are
# stripped so the meaningful leaf name and its parent are inspected directly.

# Parent-module segment suffixes that mark a norm module (RMSNorm/LayerNorm),
# so its ``weight``/``scale`` leaf is a norm gain and must NOT be decayed.
_NORM_PARENT_SUFFIXES = ("norm", "_ln", "ln")


def _decays_weight(path: str) -> bool:
    """Decoupled-AdamW decay predicate for a trainable leaf at ``path``.

    ``path`` is the dotted/joined leaf-path string (NNX or MLX flavour).
    Returns ``False`` for the DoRA ``magnitude``, biases, and norm scales
    (these are EXCLUDED from decay); ``True`` for everything else
    (``lora_a``/``lora_b``, kernels/weights, embeddings).
    """
    # Normalise: drop empty segments (NNX ``..value`` → ``['..','magnitude','value']``)
    # and a trailing NNX ``value`` wrapper segment so ``leaf`` is the param name.
    segs = [s for s in path.split(".") if s]
    if segs and segs[-1] == "value":
        segs = segs[:-1]
    if not segs:
        return True
    leaf = segs[-1]
    parent = segs[-2] if len(segs) >= 2 else ""

    # DoRA magnitude — never decay (decaying it shrinks ‖W‖ → energy collapse).
    if leaf == "magnitude":
        return False
    # Biases — never decay.
    if leaf == "bias":
        return False
    # Norm scales — never decay. NNX names the gain ``scale``; MLX names it
    # ``weight`` (same as a Linear weight), so a ``weight`` leaf is only a norm
    # scale when its parent module is a norm (``*norm`` / ``*_ln`` / ``*ln``).
    if leaf == "scale":
        return False
    if leaf == "weight" and parent.endswith(_NORM_PARENT_SUFFIXES):
        return False
    # Everything else (lora_a/lora_b, kernels, linear weights, embeddings).
    return True



# ---------------------------------------------------------------------------
# Logging / progress helpers (shared by both trainers)
# ---------------------------------------------------------------------------


def setup_logging(level: str = "info") -> None:
    """Route training logs through a terse single-line ``absl.logging`` format.

    absl's default record prefix is noisy (``I0613 14:23:01.123456 ...``); swap
    in ``HH:MM:SS L message`` so a run reads like a clean timeline. Idempotent —
    safe to call once at the top of ``train``.
    """
    import logging as _py
    from absl import logging as _absl

    _absl.use_absl_handler()
    _absl.set_verbosity(getattr(_absl, level.upper(), _absl.INFO))
    _absl.get_absl_handler().setFormatter(
        _py.Formatter("%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    )


def dump_config(config: SFTConfig, output_dir: str) -> str:
    """Persist the resolved ``SFTConfig`` to ``<output_dir>/config.yaml``.

    A stdout dump is lost the moment the terminal scrolls or the log isn't
    captured; a file in the run dir is the durable record of exactly what was
    trained (and pairs with the checkpoints + loss curve already written there).
    """

    from magenta_rt.sft.wandb_writer import dataclasses_to_dict

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses_to_dict(config), f, sort_keys=True)
    return path


def format_eta(seconds: float) -> str:
    """``H:MM:SS`` for a duration in seconds (compact ETA / wall-clock)."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def audio_energy_metrics(audio, *, sample_rate: int = 48_000,
                         window: int = 2048, silent_thresh: float = 0.02) -> dict:
    """Cheap loudness diagnostics for a generated clip ``[T, C]`` (or ``[T]``).

    Returns ``{rms, peak, frac_silent}``. ``frac_silent`` is the fraction of
    ``window``-sample frames whose RMS falls below ``silent_thresh`` — a direct
    readout of the energy-collapse failure mode (a healthy clip sits near 0;
    a collapsed one approaches 1). Logging this per audio-sample step turns a
    by-ear check into a curve you can watch.
    """


    a = np.asarray(audio, dtype=np.float32)
    mono = a.mean(axis=-1) if a.ndim > 1 else a
    rms = float(np.sqrt((a ** 2).mean())) if a.size else 0.0
    peak = float(np.abs(a).max()) if a.size else 0.0
    n = len(mono) // window
    if n:
        frames = mono[: n * window].reshape(n, window)
        we = np.sqrt((frames ** 2).mean(axis=1))
        frac_silent = float((we < silent_thresh).mean())
    else:
        frac_silent = 1.0 if rms < silent_thresh else 0.0
    return {"rms": rms, "peak": peak, "frac_silent": frac_silent}


# ---------------------------------------------------------------------------
# Learning-rate schedule (backend-dispatched) + dataset factory
# ---------------------------------------------------------------------------


def lr_at_step(step, config: SFTConfig):
    """Linear warmup → rsqrt decay, shared by both trainers.

    Pass a python ``int`` (MLX host path) to get a ``float``; pass a jax tracer
    (inside ``optax.scale_by_schedule``) to get a jax scalar. ``jax.numpy`` is
    imported only on the jax branch.
    """
    if type(step).__module__.split(".", 1)[0] in ("jax", "jaxlib"):
        import jax.numpy as xp

        is_jax, fdtype = True, xp.float32   # jax default; matches the old nnx schedule
    else:
        import numpy as xp

        is_jax, fdtype = False, xp.float64  # matches the old (pure-python) mlx schedule

    step = xp.asarray(step, fdtype)
    if config.warmup_steps == 0:
        warmup = xp.asarray(1.0, fdtype)
    else:
        warmup = xp.minimum(step / config.warmup_steps, 1.0)
    s = xp.maximum(step - config.warmup_steps, 0.0)
    t0 = xp.asarray(config.rsqrt_timescale, fdtype)
    decay = xp.sqrt(t0) / xp.sqrt(s + t0)
    lr = config.learning_rate * warmup * decay
    return lr if is_jax else float(lr)


# Content-conditioning leaves excluded at load when ``SFTConfig.mask_conditioning``
# is set, so ``prepare_source_tokens`` falls back to each stream's learned
# unconditional (-1 → dropout) token. The CFG-strength channels are NOT here:
# they are synthesized per example by ``PrepareCFG`` and stay part of the recipe.
_MASKED_CONDITIONING_PREFIXES = (
    "extras.mulan_tokens_25hz",          # MusicCoCa style
    "extras.pianoroll_with_onsets_tokens",  # note pianoroll
    "extras.drum_pianoroll_tokens",      # drum pianoroll (usually already absent)
    "extras.musiccoca_embedding",        # static embedding (only used by jitter)
)


def _masked_exclude_prefixes(config: SFTConfig) -> tuple:
    return _MASKED_CONDITIONING_PREFIXES if config.mask_conditioning else ()


def _eval_exclude_prefixes(config: SFTConfig) -> tuple:
    """Eval-pass content masking — ``eval_mask_conditioning`` if set, else the
    training ``mask_conditioning``. Lets eval score the model unconditionally
    (notes/drums masked, style still overlaid) even when training conditions on
    the per-clip notes."""
    masked = (config.mask_conditioning if config.eval_mask_conditioning is None
              else config.eval_mask_conditioning)
    return _MASKED_CONDITIONING_PREFIXES if masked else ()


def make_dataset(data_dir, config: SFTConfig, spec, *, seed, num_workers=0,
                 style_tokens=None):
    """``create_audiotree_dataset`` with the ``SFTConfig`` / spec wiring both trainers share.

    ``style_tokens`` (one RVQ row) overlays a fixed MusicCoCa style on every
    example — the training-time single-style-prompt recipe (``SFTConfig.style_prompt``).
    ``config.mask_conditioning`` excludes the content-conditioning leaves so every
    stream falls back to its unconditional token (SpectroStream-target-only SFT).
    """
    return create_audiotree_dataset(
        data_dir,
        batch_size=config.batch_size,
        crop_length_seconds=config.crop_length_seconds,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config,
        seed=seed,
        musiccoca_sticky_prob=config.musiccoca_sticky_prob,
        num_workers=num_workers,
        per_worker_buffer_size=config.per_worker_buffer_size,
        tree_exclude_prefixes=_masked_exclude_prefixes(config),
        load_into_memory=config.load_into_memory,
        style_jitter_std=config.style_jitter_std,
        style_jitter_prob=config.style_jitter_prob,
        style_tokens=style_tokens,
    )


def make_eval_dataset(data_dir, config: SFTConfig, spec, *, seed,
                      style_tokens=None):
    """Deterministic eval pipeline: **fixed CFG, no input dropout, no style
    augmentation**, so eval loss is a clean, comparable generalization signal.

    The training pipeline applies stochastic CFG dropout (15% per feature) and
    a randomly-sampled CFG strength per example — both swing teacher-forced
    loss by several nats, so a stochastic eval is too noisy to read a few-step
    trend off of. Here every conditioning feature is kept (dropout rate forced
    to 0, but the dropout-token *offset* preserved by leaving ``dropout_prob``
    non-None) and CFG is pinned to the inference defaults.
    """
    from magenta_rt import config as _cfg

    eval_configs = tuple(
        dataclasses.replace(c, dropout_prob=0.0)
        if c.dropout_prob is not None else c
        for c in spec.input_configs
    )
    cfg_fixed_scales = {
        _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key: (
            config.eval_cfg_musiccoca, config.eval_cfg_notes),
        _cfg.CFG_CONDITIONING_DRUMS.key: config.eval_cfg_drums,
    }
    return create_audiotree_dataset(
        data_dir,
        batch_size=config.eval_batch_size or config.batch_size,
        crop_length_seconds=config.crop_length_seconds,
        input_configs=eval_configs,
        target_config=spec.target_tokens_config,
        seed=seed,
        musiccoca_sticky_prob=config.musiccoca_sticky_prob,
        cfg_fixed_scales=cfg_fixed_scales,
        tree_exclude_prefixes=_eval_exclude_prefixes(config),
        load_into_memory=config.load_into_memory,
        style_jitter_std=0.0,
        style_tokens=style_tokens,
    )





def embed_style_prompt(config: SFTConfig, model_name: Optional[str]):
    """Embed ``config.style_prompt`` into fixed MusicCoCa RVQ tokens, once.

    The single-style-prompt SFT recipe conditions every training example on one
    fixed text prompt instead of per-clip style (right for a codec-only export
    with no per-clip style — it anchors the adapter to one style with no
    re-export). MusicCoCa's text path loads SentencePiece, a C++ runtime that
    deadlocks once grain/JAX are live in this process, so the embedding runs in
    an isolated CPU subprocess (``python -m magenta_rt.sft.embed_prompt``) using
    the shipped TFLite MusicCoCa — whose tokens match ``mrt ... generate``
    inference, so train- and inference-time style agree.

    Backend-neutral (subprocess + numpy only). Returns the token array to overlay
    on every example, or ``None`` when ``config.style_prompt`` is unset.
    """
    if not config.style_prompt:
        return None


    print(f"[sft] embedding style prompt (isolated subprocess): "
          f"{config.style_prompt!r}")
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        subprocess.run(
            [sys.executable, "-m", "magenta_rt.sft.embed_prompt",
             "--prompt", config.style_prompt,
             "--backend", "tflite", "--use-mapper", "--out", tmp],
            check=True,
            # Pin the child to CPU: it only needs TFLite, and the training
            # process already holds the GPU — a second CUDA context would contend.
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        with np.load(tmp) as d:
            style_tokens = d["tokens"]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[sft] fixed style tokens ({config.style_prompt!r}): "
          f"{style_tokens.tolist()}")
    return style_tokens


# ---------------------------------------------------------------------------
# Metric accumulators (shared by both trainers)
# ---------------------------------------------------------------------------
#
# ``flax.nnx.MultiMetric`` is the windowed-average accumulator for both trainers
# (a first-party, TensorFlow-free nnx API — so it is safe even in the MLX/Metal
# process, where TensorFlow aborts). It holds ``Average`` children whose state
# lives in jax arrays; the values fed in are host scalars (``float(loss)``), so
# the per-step accumulation is a couple of tiny jax CPU ops on either backend.
# Each child names the ``update`` kwarg it reads (``update`` broadcasts every
# kwarg to every child), so the argnames below must match the ``update(...)``
# call sites. Usage: ``m.update(loss=..., grad_norm=...)`` per step, ``m.compute()``
# (returns a name->jax-scalar dict) at log/eval time, ``m.reset()`` per window.


def make_train_metrics():
    """A ``MultiMetric`` tracking window-averaged train ``loss`` + ``grad_norm``."""
    from flax import nnx

    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"),
                           grad_norm=nnx.metrics.Average("grad_norm"))


def make_eval_metrics():
    """A ``MultiMetric`` tracking the mean eval ``loss`` over a batch set."""
    from flax import nnx

    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
