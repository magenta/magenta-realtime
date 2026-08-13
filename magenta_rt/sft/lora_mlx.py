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

"""Idiomatic-MLX LoRA adapters for SFT (``mlx_pure`` backend).

The MLX twin of :mod:`magenta_rt.sft.lora`. It mirrors the NNX adapter *math*
and merge semantics exactly — ``base(x) + (alpha/rank) * (x @ A) @ B`` with
``B`` zero-initialised, and a ``fuse`` that folds ``A @ B`` back into the base
kernel — so a merged checkpoint round-trips to every backend. But the plumbing
follows the MLX / ``mlx-lm`` idiom: wrap a base module via :meth:`from_base`,
train only the adapter params through ``model.freeze()`` + selective
``unfreeze``, and :meth:`fuse` back to a plain module for export.

Target divergence vs the NNX trainer
-------------------------------------
The NNX depthformer stores attention ``q``/``kv`` as vmap-stacked
``nnx.Linear``, so its default LoRA target is ``{q_proj, kv_proj}``. In
``mlx_pure`` those projections are bare ``mx.array`` attributes applied by a
matmul helper (``attention._apply_proj``) — not modules — so they can't be
wrapped without perturbing the (parity-sensitive) inference hot path. The clean,
idiomatic wrap points here are:

* **FFN** ``Dense`` layers — each holds a plain ``nn.Linear`` at ``.linear``.
  We wrap that inner linear so the adapter sits *inside* the FFN activation.
* **attention** ``output_projection`` — an :class:`~magenta_rt.mlx_pure.layers.EinsumDense`
  with equation ``"...nh,dnh->...d"``, i.e. a flattenable ``(n*h) -> d`` map.

So :func:`default_targets` selects the FFN linears and :func:`all_linear_targets`
adds the attention output projections. (QKV LoRA on the bare arrays — matching
the NNX default exactly — is a documented follow-up.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from magenta_rt.mlx_pure.layers import EinsumDense

# The attention output projection's einsum: input ``[..., n, h]``, kernel
# ``[d, n, h]``, output ``[..., d]``. The LoRA wrapper flattens ``(n, h)``.
_OUTPUT_PROJ_EQUATION = "...nh,dnh->...d"


_DORA_EPS = 1e-6  # column-norm floor in the DoRA forward (matches the SA3 impl)


def _lora_scale(rank: int, alpha: float) -> float:
    """NNX convention: ``alpha/rank`` (Hu et al.), or ``1.0`` when ``alpha==0``
    ("no rescaling", raw ``A @ B``). Stored as a Python float so it stays
    static through ``mx.compile``."""
    return (alpha / rank) if alpha else 1.0


def _dora_weight(base_w: mx.array, delta: mx.array, magnitude: mx.array,
                 *, strength: float, scale: float, out_dtype: mx.Dtype) -> mx.array:
    """DoRA effective weight for a 2-D ``[out, in]`` base.

    ``W' = magnitude · (W + scale·strength·Δ) / ‖W + scale·strength·Δ‖_row``
    — the adapter updates the weight *direction* (per output row), and the
    learned per-row ``magnitude`` (init ``‖W‖_row``) sets the scale separately.
    Decoupling the two is what keeps the effective weight norm from running away
    the way plain LoRA's does (the energy-collapse failure mode). Computed in
    fp32 for a stable row norm, then cast back. ``base_w`` and ``delta`` are
    ``[out, in]``; ``magnitude`` is ``[out]``.
    """
    # Compute the effective weight in ``out_dtype`` (bf16 for real runs), NOT a
    # full fp32 upcast of every base weight: materializing 98 fp32 weight copies
    # and holding them through the backward thrashes a 16 GB Mac into swap. Only
    # the small per-row norm reduction is promoted to fp32 for stability (its
    # result is just ``[out, 1]``). For an fp32 base (the parity tests) this is
    # all fp32 and bit-identical to the old path.
    v = base_w.astype(out_dtype) + mx.array(scale * strength, out_dtype) * delta.astype(out_dtype)
    # eps INSIDE the sqrt: ``sqrt(Σv²)+eps`` leaves the sqrt gradient (∝1/‖v‖)
    # singular at a zero row, which NaNs the backward pass; ``sqrt(Σv²+eps²)``
    # is finite everywhere and ≈ identical in the forward.
    row_sq = (v * v).sum(axis=1, keepdims=True).astype(mx.float32)
    row_norm = mx.sqrt(row_sq + _DORA_EPS * _DORA_EPS).astype(out_dtype)
    w = magnitude.astype(out_dtype)[:, None] * (v / row_norm)
    return w


def _uniform_a(in_features: int, rank: int, dtype: mx.Dtype, key) -> mx.array:
    """``lora_a`` init: uniform in ``±1/sqrt(in)`` (the mlx-lm initialisation).
    Cross-framework RNG parity with NNX's he_uniform is impossible, so we use
    the well-tested MLX init and seed it deterministically per adapter."""
    scale = 1.0 / math.sqrt(in_features)
    a = mx.random.uniform(low=-scale, high=scale, shape=(in_features, rank), key=key)
    return a.astype(dtype)


class LoRALinear(nn.Module):
    """LoRA / DoRA wrapper around an ``nn.Linear`` (mlx-lm idiom).

    LoRA forward: ``base(x) + scale * strength * (dropout(x) @ A) @ B``. ``B`` is
    zero-init so the wrapped layer is identical to the base at step 0.

    ``dora=True`` switches to DoRA: the effective weight is
    ``magnitude · (W + scale·strength·BA) / ‖W + scale·strength·BA‖_row`` with a
    learned per-output-row ``magnitude`` (init ``‖W‖_row``), so the adapter sets
    direction and magnitude independently — a better-behaved fit that resists the
    weight-norm runaway that collapses plain LoRA at full strength. Identical to
    the base at step 0 (``B=0`` ⇒ ``W' = magnitude·W/‖W‖ = W``).

    ``lora_strength`` is a runtime multiplier (default 1.0): set it <1 at
    inference to blend the adapter toward the base (``set_lora_strength``).
    """

    @staticmethod
    def from_base(
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float = 0.0,
        dropout: float = 0.0,
        dora: bool = False,
        key=None,
    ) -> "LoRALinear":
        out_features, in_features = linear.weight.shape
        lora = LoRALinear(
            in_features, out_features, rank=rank, alpha=alpha, dropout=dropout,
            dora=dora, param_dtype=linear.weight.dtype, key=key,
        )
        lora.linear = linear  # share the (pretrained) base — no weight copy
        if dora:
            # Magnitude init = base per-output-row norm, so step 0 == base.
            w = linear.weight.astype(mx.float32)
            lora.magnitude = mx.sqrt((w * w).sum(axis=1)).astype(linear.weight.dtype)
        return lora

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float = 0.0,
        dropout: float = 0.0,
        dora: bool = False,
        param_dtype: mx.Dtype = mx.float32,
        key=None,
    ):
        super().__init__()
        # Placeholder; ``from_base`` overwrites with the real (pretrained) base.
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = _lora_scale(rank, alpha)
        self.dora = bool(dora)
        self.lora_strength = 1.0  # runtime blend knob (not a trained param)
        if key is None:
            key = mx.random.key(0)
        self.lora_a = _uniform_a(in_features, rank, param_dtype, key)
        self.lora_b = mx.zeros((rank, out_features), dtype=param_dtype)
        if self.dora:
            self.magnitude = mx.ones((out_features,), dtype=param_dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if self.dora:
            # Effective-weight forward — same op as fuse() (``x @ W'.T``), so the
            # two are bit-exact. ``W'`` is built in the base weight dtype; the
            # matmul then promotes exactly like the base ``nn.Linear`` would.
            # dropout gates the adapter factor only (not the base direction).
            wdt = self.linear.weight.dtype
            delta = self.lora_b.T @ self.dropout(self.lora_a).T  # [out, in]
            w = _dora_weight(self.linear.weight, delta, self.magnitude,
                             strength=self.lora_strength, scale=self.scale,
                             out_dtype=wdt)
            out = x.astype(wdt) @ w.T
            if "bias" in self.linear:
                out = out + self.linear.bias
            return out
        y = self.linear(x)
        # Track the base *output* dtype (NNX casts to ``base_out.dtype``), not
        # the input dtype: with ``compute_dtype=bf16`` the input is bf16 but the
        # fp32 base weight promotes the matmul to fp32, and ``fuse`` folds the
        # delta in fp32 — computing the adapter in fp32 keeps the two exactly
        # equal and matches the surrounding precision.
        dt = y.dtype
        a = self.lora_a.astype(dt)
        b = self.lora_b.astype(dt)
        z = (self.dropout(x).astype(dt) @ a) @ b
        return y + (self.scale * self.lora_strength * z).astype(dt)

    def fuse(self) -> nn.Linear:
        """Fold the adapter into the base weight; return a plain ``nn.Linear``.

        Uses ``lora_strength`` (so fusing a strength-dialed adapter bakes that
        blend in). For DoRA this folds the full direction+magnitude weight.
        """
        linear = self.linear
        out_features, in_features = linear.weight.shape
        has_bias = "bias" in linear
        fused = nn.Linear(in_features, out_features, bias=has_bias)
        # nn.Linear: ``y = x @ W.T`` with ``W`` [out, in]; LoRA adds
        # ``x @ (A @ B)`` → fused ``W' = W + scale * strength * (A @ B).T``.
        delta = self.lora_b.T @ self.lora_a.T  # [out, in]
        if self.dora:
            fused.weight = _dora_weight(
                linear.weight, delta, self.magnitude, strength=self.lora_strength,
                scale=self.scale, out_dtype=linear.weight.dtype)
        else:
            fused.weight = linear.weight + (
                self.scale * self.lora_strength * delta).astype(linear.weight.dtype)
        if has_bias:
            fused.bias = linear.bias
        return fused


class LoRAEinsumDense(nn.Module):
    """LoRA wrapper around the attention ``output_projection`` EinsumDense.

    Specific to the ``"...nh,dnh->...d"`` equation (kernel ``[d, n, h]``): the
    map is linear in the flattened ``(n*h)`` input, so the adapter is a plain
    ``(n*h) -> d`` low-rank pair. The base EinsumDense must already be
    materialised (run one forward) before wrapping.
    """

    @staticmethod
    def from_base(
        einsum: EinsumDense,
        *,
        rank: int,
        alpha: float = 0.0,
        dropout: float = 0.0,
        dora: bool = False,
        key=None,
    ) -> "LoRAEinsumDense":
        if einsum.equation != _OUTPUT_PROJ_EQUATION:
            raise ValueError(
                f"LoRAEinsumDense only supports {_OUTPUT_PROJ_EQUATION!r}, "
                f"got {einsum.equation!r}"
            )
        if einsum.kernel is None:
            raise ValueError(
                "EinsumDense kernel is lazy-uninitialised — run a forward pass "
                "(build_model does this) before injecting LoRA."
            )
        d, n, h = einsum.kernel.shape
        lora = LoRAEinsumDense(
            n, h, d, rank=rank, alpha=alpha, dropout=dropout, dora=dora,
            param_dtype=einsum.kernel.dtype, key=key,
        )
        lora.einsum = einsum  # share the (pretrained) base — no weight copy
        if dora:
            # Magnitude init = base per-output-row norm of the [d, n*h] kernel.
            k = einsum.kernel.reshape(d, n * h).astype(mx.float32)
            lora.magnitude = mx.sqrt((k * k).sum(axis=1)).astype(einsum.kernel.dtype)
        return lora

    def __init__(
        self,
        n: int,
        h: int,
        d: int,
        *,
        rank: int,
        alpha: float = 0.0,
        dropout: float = 0.0,
        dora: bool = False,
        param_dtype: mx.Dtype = mx.float32,
        key=None,
    ):
        super().__init__()
        self._n, self._h, self._d = int(n), int(h), int(d)
        # Placeholder; ``from_base`` overwrites with the real (pretrained) base.
        self.einsum = EinsumDense(
            equation=_OUTPUT_PROJ_EQUATION, output_shape=(d,), param_dtype=param_dtype,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.scale = _lora_scale(rank, alpha)
        self.dora = bool(dora)
        self.lora_strength = 1.0  # runtime blend knob (not a trained param)
        if key is None:
            key = mx.random.key(0)
        self.lora_a = _uniform_a(n * h, rank, param_dtype, key)  # [n*h, r]
        self.lora_b = mx.zeros((rank, d), dtype=param_dtype)     # [r, d]
        if self.dora:
            self.magnitude = mx.ones((d,), dtype=param_dtype)

    def _kernel_2d(self) -> mx.array:
        """Base kernel ``[d, n, h]`` viewed as the 2-D ``[d, n*h]`` weight."""
        return self.einsum.kernel.reshape(self._d, self._n * self._h)

    def __call__(self, x: mx.array) -> mx.array:
        if self.dora:
            # Effective-weight forward (matches fuse()); dropout gates the adapter
            # factor only (not the base direction). No base einsum() call needed.
            dt = self.einsum.kernel.dtype
            delta = self.lora_b.T @ self.dropout(self.lora_a).T  # [d, n*h]
            w = _dora_weight(self._kernel_2d(), delta, self.magnitude,
                             strength=self.lora_strength, scale=self.scale,
                             out_dtype=dt)  # [d, n*h]
            vb = x.reshape(*x.shape[:-2], self._n * self._h).astype(dt)
            out = vb @ w.T  # [..., d]
            if self.einsum.bias is not None:
                out = out + self.einsum.bias.astype(dt)
            return out
        y = self.einsum(x)  # [..., d]
        dt = y.dtype  # match the base output precision (see LoRALinear.__call__)
        v = self.dropout(x).reshape(*x.shape[:-2], self._n * self._h).astype(dt)
        a = self.lora_a.astype(dt)
        b = self.lora_b.astype(dt)
        z = (v @ a) @ b  # [..., d]
        return y + (self.scale * self.lora_strength * z).astype(dt)

    def fuse(self) -> EinsumDense:
        """Fold the adapter into the base kernel; return a plain EinsumDense.

        Honors ``lora_strength``; for DoRA folds the full direction+magnitude
        kernel (computed in the ``[d, n*h]`` view, then reshaped to ``[d,n,h]``).
        """
        e = self.einsum
        fused = EinsumDense(
            equation=e.equation,
            output_shape=e.output_shape_spec,
            bias_axes=e.bias_axes,
            activation=e.activation,
            compute_dtype=e.compute_dtype,
            param_dtype=e.param_dtype,
        )
        if self.dora:
            delta = self.lora_b.T @ self.lora_a.T  # [d, n*h]
            w = _dora_weight(self._kernel_2d(), delta, self.magnitude,
                             strength=self.lora_strength, scale=self.scale,
                             out_dtype=e.kernel.dtype)  # [d, n*h]
            fused.kernel = w.reshape(self._d, self._n, self._h)
        else:
            delta = (self.scale * self.lora_strength) * (self.lora_a @ self.lora_b)  # [n*h, d]
            delta_k = delta.reshape(self._n, self._h, self._d).transpose(2, 0, 1)  # [d,n,h]
            fused.kernel = e.kernel + delta_k.astype(e.kernel.dtype)
        fused.bias = e.bias
        fused._initialized = True
        return fused


# ---- Target predicates -----------------------------------------------------

TargetFn = Callable[[str, nn.Module], bool]

_FFN_LINEAR_LEAVES = ("ffn_layer1.linear", "ffn_layer2.linear")


def default_targets(path: str, module: nn.Module) -> bool:
    """Wrap the FFN linears (``ffn_layer1``/``ffn_layer2`` inner ``nn.Linear``)."""
    return isinstance(module, nn.Linear) and path.endswith(_FFN_LINEAR_LEAVES)


def all_linear_targets(path: str, module: nn.Module) -> bool:
    """FFN linears + every attention ``output_projection`` EinsumDense."""
    if default_targets(path, module):
        return True
    return isinstance(module, EinsumDense) and path.endswith("output_projection")


# ---- Injection / freeze / merge -------------------------------------------

def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float = 0.0,
    dropout: float = 0.0,
    dora: bool = False,
    targets: Optional[TargetFn] = None,
    seed: int = 0,
) -> int:
    """Wrap matching submodules in a LoRA/DoRA adapter (sharing the base weights).

    Returns the number of layers wrapped. Walks ``model.named_modules()`` and
    swaps matches in via ``update_modules`` — the ``mlx-lm`` pattern. Adapter
    init is seeded deterministically (per-adapter key split from ``seed``).
    ``dora=True`` adds a learned per-output magnitude (DoRA) to every wrapper.
    """
    if targets is None:
        targets = default_targets

    matches = sorted(
        ((p, m) for p, m in model.named_modules() if targets(p, m)),
        key=lambda pm: pm[0],
    )
    if not matches:
        return 0

    keys = mx.random.split(mx.random.key(seed), len(matches))
    replacements = []
    for i, (path, module) in enumerate(matches):
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear.from_base(
                module, rank=rank, alpha=alpha, dropout=dropout, dora=dora,
                key=keys[i],
            )
        else:  # EinsumDense
            wrapped = LoRAEinsumDense.from_base(
                module, rank=rank, alpha=alpha, dropout=dropout, dora=dora,
                key=keys[i],
            )
        replacements.append((path, wrapped))

    model.update_modules(tree_unflatten(replacements))
    return len(replacements)


def mark_lora_trainable(model: nn.Module) -> None:
    """Freeze everything, then unfreeze only the adapter params.

    After this, ``model.trainable_parameters()`` is exactly ``lora_a``/``lora_b``
    (plus ``magnitude`` for DoRA wrappers) across all wrappers — the base weights
    stay frozen, so ``nn.value_and_grad(model, ...)`` and
    ``optimizer.update(model, grads)`` touch only the adapters (the mlx-lm recipe).
    """
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b", "magnitude"], recurse=True)


def set_lora_strength(model: nn.Module, strength: float) -> int:
    """Set the runtime adapter blend on every wrapper (1.0 = full, 0.0 = base).

    A first-class inference knob: a strongly-trained adapter often sounds best
    blended toward the base (``strength`` ~0.6–0.8) rather than at full strength.
    Multiplies the adapter delta (LoRA and DoRA alike), so it composes with the
    trained ``alpha/rank`` scale without retraining. Returns the count set.
    """
    n = 0
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAEinsumDense)):
            module.lora_strength = float(strength)
            n += 1
    return n


def merge_lora_into_base(model: nn.Module) -> int:
    """Fuse every adapter back into its base module (``fuse``) and unwrap.

    After this the model is a plain inference module again — the standard
    Linen-format export and the mlx_pure inference path work with no
    LoRA-aware code downstream. Returns the number of adapters merged.
    """
    fused = [
        (path, module.fuse())
        for path, module in model.named_modules()
        if isinstance(module, (LoRALinear, LoRAEinsumDense))
    ]
    if fused:
        model.update_modules(tree_unflatten(fused))
    return len(fused)
