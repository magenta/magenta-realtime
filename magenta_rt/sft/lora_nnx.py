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

"""LoRA adapters for SFT.

Two pieces:

* :class:`MRTLoRAParam` — project-local ``nnx.LoRAParam`` subclass so
  ``wrt=MRTLoRAParam`` filters select only the adapters we inject (and not
  any third-party LoRA params that might wander in), while ecosystem
  filters on ``nnx.LoRAParam`` (or plain ``nnx.Param``) still match them.
* :func:`inject_lora` — walks the model and wraps target ``nnx.Linear``
  submodules in :class:`LoRAAdapter`, sharing the base layer rather than
  duplicating its weights.
* :func:`merge_lora_into_base` — folds ``A @ B`` into ``base.kernel`` and
  replaces each wrapper with the underlying ``Linear`` so the model is a
  plain inference module again (saves to the existing Linen-format export).

Why a custom adapter instead of ``flax.nnx.LoRA``? The decoder transformers
in this codebase are built under ``nnx.vmap``, so ``q_proj.kernel`` has shape
``(num_layers, in, out)``. ``flax.nnx.LoRA`` hardcodes its adapter shapes to
``(in, rank)`` / ``(rank, out)``, which breaks inside the ``nnx.scan`` body.
:class:`LoRAAdapter` automatically matches whatever leading axes the base
kernel carries, so the same code wraps vmapped and non-vmapped Linears.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx


_DORA_EPS = 1e-6  # column-norm floor in the DoRA forward (matches lora_mlx).


class MRTLoRAParam(nnx.LoRAParam):
    """LoRA-adapter Param. Project-local so ``wrt=MRTLoRAParam`` is exact.

    Subclasses ``nnx.LoRAParam`` (itself an ``nnx.Param``), so generic
    ``nnx.LoRAParam`` / ``nnx.Param`` filters also match these adapters.
    """


def _kernel_in_norm(kernel: jax.Array) -> jax.Array:
    """Per-output-row norm of an ``(..., in, out)`` kernel over the ``in`` axis.

    Returns shape ``(..., out)`` — the DoRA ``magnitude`` init. eps is **inside**
    the sqrt (``sqrt(Σv²+eps²)``): the alternative ``sqrt(Σv²)+eps`` leaves the
    sqrt gradient (∝1/‖v‖) singular at a zero kernel row and NaNs the backward
    pass (the bug that bit the MLX EinsumDense path). Computed in fp32 for a
    stable norm, cast back to the kernel dtype by the caller.
    """
    k = kernel.astype(jnp.float32)
    return jnp.sqrt((k * k).sum(axis=-2) + _DORA_EPS * _DORA_EPS)


def _exact_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """``a @ b`` at full fp32 precision, regardless of the global default.

    Merging is a one-time weight-space fold whose result is written into an
    exported checkpoint, so it must not inherit the accelerator's fast-matmul
    default: on Ampere-class GPUs (and TPU) JAX contracts fp32 matmuls in TF32
    /bf16 unless told otherwise, which would bake a truncated ``A@B`` into the
    merged kernel and break bit-exactness against the unmerged forward pass.
    The adapter matrices are tiny (rank r), so the exact contraction is free.
    """
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _dora_kernel(kernel: jax.Array, delta: jax.Array, magnitude: jax.Array,
                 *, strength: float, scale: float, out_dtype) -> jax.Array:
    """DoRA effective kernel for an ``(..., in, out)`` NNX kernel.

    ``W' = magnitude · (W + scale·strength·Δ) / ‖W + scale·strength·Δ‖_in``
    — the adapter updates the weight *direction* (per output, norm over the
    ``in`` axis), and the learned per-output ``magnitude`` (init ``‖W‖_in``)
    sets the scale separately. Decoupling the two keeps the effective weight
    norm from running away the way plain LoRA's does (the energy-collapse
    failure mode). ``kernel``/``delta`` are ``(..., in, out)``; ``magnitude`` is
    ``(..., out)``. Computed in fp32 for a stable norm, then cast back.

    NOTE the orientation difference vs ``lora_mlx._dora_weight``: MLX weights are
    ``[out, in]`` (norm over axis=1, magnitude broadcast as ``[:, None]``); NNX
    kernels are ``[..., in, out]`` (norm over axis=-2, magnitude broadcast on a
    new ``in`` axis at -2).
    """
    v = kernel.astype(jnp.float32) + (scale * strength) * delta.astype(jnp.float32)
    # eps INSIDE the sqrt — see _kernel_in_norm. keepdims so it broadcasts over
    # the contracted ``in`` axis: norm is (..., 1, out), v is (..., in, out).
    col_norm = jnp.sqrt((v * v).sum(axis=-2, keepdims=True) + _DORA_EPS * _DORA_EPS)
    mag = jnp.expand_dims(magnitude.astype(jnp.float32), axis=-2)  # (..., 1, out)
    w = mag * (v / col_norm)
    return w.astype(out_dtype)


class LoRAAdapter(nnx.Module):
    """LoRA wrapper around an ``nnx.Linear``.

    Forward: ``base(x) + (alpha/rank) * (x @ A @ B)``. ``B`` is zero-initialized
    so the wrapped layer is identical to the base at step 0.

    Shape note: ``A`` and ``B`` carry the same leading axes as ``base.kernel``
    (e.g. ``(num_layers, in, rank)`` when the host transformer was built
    under ``nnx.vmap``). Inside an ``nnx.scan`` body each slice strips the
    leading axis and the matmul ``x @ A @ B`` has the expected 2-D shape.

    ``dora=True`` switches to DoRA: the effective kernel is
    ``magnitude · (W + scale·strength·AB) / ‖W + scale·strength·AB‖_in`` with a
    learned per-output ``magnitude`` (init ``‖W‖_in``, the per-output-row norm
    over the ``in`` axis), so the adapter sets direction and magnitude
    independently — a better-behaved fit that resists the weight-norm runaway
    that collapses plain LoRA at full strength. Identical to the base at step 0
    (``B=0`` ⇒ ``W' = magnitude·W/‖W‖ = W``). ``magnitude`` is an
    :class:`MRTLoRAParam` (shape ``leading + (out,)``) so it joins the trainable
    adapter set and the adapter-only checkpoint automatically.

    ``lora_strength`` is a runtime multiplier (plain float, default 1.0; *not* an
    ``nnx.Param``) on the adapter delta — set it <1 at inference to blend the
    adapter toward the base (:func:`set_lora_strength`).
    """

    def __init__(
        self,
        base: nnx.Linear,
        *,
        rank: int,
        alpha: float = 0.0,
        dora: bool = False,
        rngs: nnx.Rngs = None,
    ):
        kshape = base.kernel[...].shape           # (..., in, out)
        leading, in_f, out_f = kshape[:-2], kshape[-2], kshape[-1]
        a_shape = leading + (in_f, rank)
        b_shape = leading + (rank, out_f)
        a_init = nnx.initializers.he_uniform()(
            rngs.params(), a_shape, base.param_dtype,
        )
        b_init = jnp.zeros(b_shape, base.param_dtype)

        self.base = base
        self.lora_a = MRTLoRAParam(a_init)
        self.lora_b = MRTLoRAParam(b_init)
        # alpha == 0 → "no rescaling" (raw A @ B); otherwise use alpha/rank
        # (Hu et al. convention). Stored as a Python float so it stays static.
        self.scale = (alpha / rank) if alpha else 1.0
        self.dora = bool(dora)
        # Runtime adapter-blend knob (plain python float — NOT an nnx.Param, so
        # it stays static through nnx.jit and never enters the param tree).
        self.lora_strength = 1.0
        if self.dora:
            # Magnitude init = base per-output-row norm over the ``in`` axis, so
            # step 0 == base (B=0 ⇒ W'=magnitude·W/‖W‖=W). Shape leading+(out,).
            mag_init = _kernel_in_norm(base.kernel[...]).astype(base.param_dtype)
            self.magnitude = MRTLoRAParam(mag_init)

    def __call__(self, x, **kwargs):
        if self.dora:
            # Effective-kernel forward — the same op merge_lora_into_base() bakes
            # into base.kernel, so the wrapped and merged forwards are
            # numerically equal (up to fp32 contraction order). The base
            # kernel/bias are read by hand (no self.base(x) call) so we never
            # double-apply the bias. _dora_kernel builds W' in the *param* dtype
            # (matching the kernel merge writes), then we promote (x, W', bias)
            # to the base's *compute* dtype exactly as nnx.Linear does — so a
            # bf16-compute / fp32-param run tracks the merged path's precision.
            kernel = self.base.kernel[...]
            delta = self.lora_a[...] @ self.lora_b[...]  # (..., in, out)
            w = _dora_kernel(
                kernel, delta, self.magnitude[...],
                strength=self.lora_strength, scale=self.scale,
                out_dtype=kernel.dtype,
            )
            cdt = self.base.dtype if self.base.dtype is not None else kernel.dtype
            out = jnp.matmul(x.astype(cdt), w.astype(cdt))
            if self.base.bias is not None:
                bias = self.base.bias[...].astype(cdt)
                out = out + jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))
            return out
        base_out = self.base(x, **kwargs)
        # Cast adapter weights into the base's compute dtype so the LoRA
        # branch tracks the same precision as the surrounding stack
        # (otherwise fp32 params would silently upcast the residual sum).
        a = self.lora_a[...].astype(base_out.dtype)
        b = self.lora_b[...].astype(base_out.dtype)
        delta = (x.astype(base_out.dtype) @ a) @ b
        return base_out + (self.scale * self.lora_strength) * delta


TargetFn = Callable[[tuple, nnx.Module], bool]


_DEFAULT_ATTN_TARGETS = {"q_proj", "kv_proj"}
_DEFAULT_FFN_TARGETS = {"ffn_layer1", "ffn_layer2"}
# Linears that ``all_linears`` skips: the depth transformer's input projection
# and the codebook logits head. ``all_plus`` adds these — the logits head in
# particular sits right at the output distribution, a strong SFT target.
_PLUS_TARGETS = {"depth_input_adapter", "to_logits"}


def default_targets(path: tuple, module: nnx.Module) -> bool:
    """Wrap attention QKV projections by default. Skips FFN."""
    return isinstance(module, nnx.Linear) and path and path[-1] in _DEFAULT_ATTN_TARGETS


def all_linear_targets(path: tuple, module: nnx.Module) -> bool:
    """Wrap attention QKV + FFN Linears (skips the depth-input adapter and the
    logits head — see :func:`all_plus_targets` for those)."""
    return isinstance(module, nnx.Linear) and path and path[-1] in (
        _DEFAULT_ATTN_TARGETS | _DEFAULT_FFN_TARGETS
    )


def all_plus_targets(path: tuple, module: nnx.Module) -> bool:
    """Wrap EVERY ``nnx.Linear``: attention QKV + FFN + the depth-input adapter
    + the codebook logits head (the attention output projection is an
    ``nnx.Einsum``, not a Linear, so it is not included)."""
    return isinstance(module, nnx.Linear) and path and path[-1] in (
        _DEFAULT_ATTN_TARGETS | _DEFAULT_FFN_TARGETS | _PLUS_TARGETS
    )


def inject_lora(
    model: nnx.Module,
    *,
    rank: int,
    alpha: float = 0.0,
    dora: bool = False,
    targets: Optional[TargetFn] = None,
    seed: int = 0,
) -> int:
    """Wrap matching ``nnx.Linear`` submodules in :class:`LoRAAdapter`.

    Returns the number of layers wrapped. The wrapper shares the original
    ``Linear`` instance via ``self.base`` — no weight duplication.

    Note: the walk only visits direct module attributes (``vars``), not
    modules stored inside lists/dicts — fine for this codebase, where
    transformer layers are vmap-stacked rather than held in containers.
    Callers should check the returned count (the trainers raise on 0).

    ``dora=True`` adds a learned per-output ``magnitude`` (DoRA) to every wrapper.
    """
    if targets is None:
        targets = default_targets

    rngs = nnx.Rngs(seed)
    count = 0

    def _walk(node: nnx.Module, path: tuple):
        nonlocal count
        for attr in list(vars(node)):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr)
            if isinstance(child, nnx.Linear):
                if targets(path + (attr,), child):
                    setattr(
                        node, attr,
                        LoRAAdapter(
                            child, rank=rank, alpha=alpha, dora=dora, rngs=rngs,
                        ),
                    )
                    count += 1
            elif isinstance(child, nnx.Module):
                _walk(child, path + (attr,))

    _walk(model, ())
    return count


def set_lora_strength(model: nnx.Module, strength: float) -> int:
    """Set the runtime adapter blend on every :class:`LoRAAdapter` (1.0 = full,
    0.0 = base). Returns the count set.

    A first-class inference knob (the NNX twin of ``lora_mlx.set_lora_strength``):
    a strongly-trained adapter often sounds best blended toward the base
    (``strength`` ~0.6–0.8) rather than at full strength. Multiplies the adapter
    delta (LoRA and DoRA alike), so it composes with the trained ``alpha/rank``
    scale without retraining. ``lora_strength`` is a plain python float, so this
    must run *outside* a traced ``nnx.jit`` step (it changes a static attr).
    """
    n = 0

    def _walk(node: nnx.Module):
        nonlocal n
        for attr in list(vars(node)):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr)
            if isinstance(child, LoRAAdapter):
                child.lora_strength = float(strength)
                n += 1
            elif isinstance(child, nnx.Module):
                _walk(child)

    _walk(model)
    return n


def merge_lora_into_base(model: nnx.Module) -> int:
    """Fold the adapter into each base ``Linear.kernel`` and unwrap the adapter.

    Honors ``lora_strength`` (so merging a strength-dialed adapter bakes that
    blend in). For plain LoRA this folds ``scale·strength·A@B`` into the kernel;
    for DoRA it writes the full direction+magnitude effective kernel.

    After this call the model has the original structure (every adapter is
    replaced with the underlying ``Linear``), so the standard Linen-format
    safetensors export works without LoRA-aware code anywhere downstream.
    """
    count = 0

    def _walk(node: nnx.Module):
        nonlocal count
        for attr in list(vars(node)):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr)
            if isinstance(child, LoRAAdapter):
                base = child.base
                kdtype = base.kernel[...].dtype
                if child.dora:
                    delta = _exact_matmul(child.lora_a[...], child.lora_b[...])  # (..., in, out)
                    base.kernel[...] = _dora_kernel(
                        base.kernel[...], delta, child.magnitude[...],
                        strength=child.lora_strength, scale=child.scale,
                        out_dtype=kdtype,
                    )
                else:
                    delta = (
                        (child.scale * child.lora_strength)
                        * _exact_matmul(child.lora_a[...], child.lora_b[...])
                    ).astype(kdtype)
                    # Broadcast handles vmapped (leading L axis) and plain shapes.
                    base.kernel[...] = base.kernel[...] + delta
                setattr(node, attr, base)
                count += 1
            elif isinstance(child, nnx.Module):
                _walk(child)

    _walk(model)
    return count
