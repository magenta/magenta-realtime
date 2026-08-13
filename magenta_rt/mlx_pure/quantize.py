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

"""Quantization workflows for ``mlx_pure`` models.

Two entry points:

* :func:`quantize_in_place` — naive nearest-rounding int4 / int8.
  Walks the module tree, calls ``.to_quantized(...)`` on every
  :class:`mlx_pure.layers.Dense` and :class:`mlx_pure.layers.EinsumDense`
  it finds. Suitable when the model was already produced by GPTQ-aware
  training, or for quick inference-time compression where some
  accuracy loss is acceptable.

* :func:`gptq_calibrate_and_quantize` — full GPTQ calibration loop
  (Hessian-aware rounding, Frantar et al., 2022). The math is a port
  of ``magenta_rt.mlx.gptq``; the activation-capture machinery is
  mlx-pure-flavored (instance ``__class__`` swap rather than wrapping
  ``Sequence.layer``).

Both APIs operate on a fully-built pure-MLX system (e.g.,
:class:`mlx_pure.system.MagentaRT2Sampler`); the quantization is
in-place, after which the system runs the quantized matmul path
transparently via :func:`mx.quantized_matmul`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn

from .attention import LocalSelfAttention, StreamingCrossAttention
from .layers import Dense, EinsumDense


def _walk(module: nn.Module, path: str = "") -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(path, module)`` for every nn.Module in ``module``'s tree.

    ``mlx.nn.Module`` extends ``dict``, so child modules live as
    dict-items rather than ``__dict__`` entries. We iterate both:
    dict-items (for child modules registered the standard way) and
    ``vars()`` (for python-side attributes like plain lists).
    """
    yield path.rstrip("."), module
    seen: set[int] = set()
    # Dict-items: standard MLX child registration.
    for k, v in module.items():
        seen.add(id(v))
        if isinstance(v, nn.Module):
            yield from _walk(v, f"{path}{k}.")
        elif isinstance(v, (list, tuple)):
            for i, c in enumerate(v):
                if isinstance(c, nn.Module):
                    yield from _walk(c, f"{path}{k}.{i}.")
    # Plain python attrs: catches lists of modules stored as `self.foo = [...]`.
    for k, v in vars(module).items():
        if k.startswith("_") and k not in ("_main", "_post", "_grouped", "_ungrouped"):
            continue
        if id(v) in seen:
            continue
        if isinstance(v, nn.Module):
            yield from _walk(v, f"{path}{k}.")
        elif isinstance(v, (list, tuple)):
            for i, c in enumerate(v):
                if isinstance(c, nn.Module):
                    yield from _walk(c, f"{path}{k}.{i}.")


def quantize_in_place(
    root: nn.Module,
    *,
    group_size: int = 64,
    bits: int = 4,
    skip: Callable[[str, nn.Module], bool] | None = None,
    verbose: bool = False,
) -> list[str]:
    """Quantize every layer in ``root`` that defines ``to_quantized``.

    Thin wrapper over :func:`mlx.nn.quantize`: MLX walks the tree and
    calls ``module.to_quantized(group_size, bits, mode)`` on any layer
    that exposes it. We add an optional ``skip`` predicate (path-aware)
    plus a ``verbose`` log to surface what got quantized.

    Args:
        root: the model tree (e.g., ``MagentaRT2Sampler`` or any sub-module).
        group_size: passed to ``.to_quantized``.
        bits: int4 (default) or int8.
        skip: optional callable ``(path, module) -> bool``; return
            True to leave a layer at full precision (e.g., to keep
            the final ``to_logits`` Dense un-quantized).
        verbose: if True, prints each layer it touches.

    Returns:
        List of fully-qualified paths that were quantized.
    """
    quantized: list[str] = []

    # Pre-pass: ``mlx.nn.quantize`` only walks leaf modules
    # (``model.leaf_modules()``), so it never reaches ``to_quantized``
    # on non-leaf modules like ``LocalSelfAttention`` (which has an
    # ``output_projection`` child). Invoke them explicitly first so
    # their ``q_proj`` / ``kv_proj`` raw arrays get quantized too;
    # the child ``output_projection`` is reached by nn.quantize below.
    for path, m in _walk(root):
        if isinstance(m, (LocalSelfAttention, StreamingCrossAttention)):
            if skip is not None and skip(path, m):
                continue
            m.to_quantized(group_size=group_size, bits=bits)
            if getattr(m, "_qkv_quantized", False):
                quantized.append(path)
                if verbose:
                    print(f"quantized {type(m).__name__} {path!r}")

    def _predicate(path: str, m: nn.Module) -> bool:
        if not hasattr(m, "to_quantized"):
            return False
        if skip is not None and skip(path, m):
            return False
        quantized.append(path)
        if verbose:
            print(f"quantized {type(m).__name__} {path!r}")
        return True

    nn.quantize(root, group_size=group_size, bits=bits, class_predicate=_predicate)
    return quantized


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------


def _find_quantizable_layers(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """Walk ``model`` and return ``[(path, layer, kind)]`` for every
    quantizable Dense / EinsumDense.

    ``kind`` is ``"dense"`` or ``"einsum"``. EinsumDense layers whose
    equation isn't the supported ``'...nh,dnh->...d'`` (output-projection)
    pattern are skipped — the same shapes that ``EinsumDense.to_quantized``
    accepts.
    """
    out: list[tuple[str, nn.Module, str]] = []
    for path, m in _walk(model):
        if isinstance(m, Dense):
            if m.linear is None or not hasattr(m.linear, "weight"):
                continue
            if m.linear.weight is None:
                continue
            out.append((path, m, "dense"))
        elif isinstance(m, EinsumDense):
            if m.kernel is None:
                continue
            if m.equation != "...nh,dnh->...d":
                continue
            out.append((path, m, "einsum"))
    return out


class _ActivationCapture:
    """Context manager: instrument every Dense / EinsumDense in
    ``model`` so that one or more forward passes record their input
    activations into ``self.captured[path]``.

    MLX dispatches ``__call__`` via ``type(instance).__call__`` (special
    method lookup happens on the class), so we can't just reassign
    ``instance.__call__``. Instead we swap each instance's ``__class__``
    to a one-off subclass with an overriding ``__call__``; the swap is
    reverted on context exit.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_samples: int = 2048,
        layers: list[tuple[str, nn.Module, str]] | None = None,
    ):
        self.model = model
        self.max_samples = max_samples
        self.captured: dict[str, list[mx.array]] = defaultdict(list)
        self._patched: list[tuple[nn.Module, type]] = []
        # Caller may pre-filter the layer list (e.g. honoring the ``skip``
        # predicate) so we only patch the layers we intend to quantize.
        # Patching extras would (a) waste compute capturing their activations
        # and (b) leak the class swap on the skipped layers if the caller
        # tried to drop them from ``_patched`` post-hoc.
        self._layers = layers if layers is not None else _find_quantizable_layers(model)

    def __enter__(self) -> "_ActivationCapture":
        for path, layer, kind in self._layers:
            self._patch(path, layer, kind)
        return self

    def __exit__(self, *args) -> None:
        for instance, orig_cls in self._patched:
            instance.__class__ = orig_cls
            for attr in ("_gptq_path", "_gptq_kind", "_gptq_capture"):
                if attr in instance.__dict__:
                    del instance.__dict__[attr]
        self._patched.clear()

    def _patch(self, path: str, layer: nn.Module, kind: str) -> None:
        captured = self.captured
        max_samples = self.max_samples

        if kind == "dense":
            def _capture(name, x):
                # Dense input is [..., in_features]; flatten leading dims.
                if isinstance(x, mx.array):
                    v = x.reshape(-1, x.shape[-1])
                    if len(captured[name]) * 128 < max_samples:
                        captured[name].append(v)
        elif kind == "einsum":
            def _capture(name, x):
                # EinsumDense for '...nh,dnh->...d': input is [..., n, h];
                # flatten to [..., n*h].
                if isinstance(x, mx.array):
                    if x.ndim >= 2:
                        v = x.reshape(-1, x.shape[-2] * x.shape[-1])
                    else:
                        v = x.reshape(-1, x.shape[-1])
                    if len(captured[name]) * 128 < max_samples:
                        captured[name].append(v)
        else:
            return

        orig_cls = layer.__class__
        new_cls = type(
            orig_cls.__name__ + "_GPTQWrapped", (orig_cls,), {},
        )

        def hooked_call(self, x, *args, **kwargs):
            self._gptq_capture(self._gptq_path, x)
            return orig_cls.__call__(self, x, *args, **kwargs)

        new_cls.__call__ = hooked_call
        layer._gptq_path = path
        layer._gptq_kind = kind
        layer._gptq_capture = _capture
        layer.__class__ = new_cls
        self._patched.append((layer, orig_cls))

    def get_hessians(self, *, debug_identity_hessian: bool = False
                     ) -> dict[str, mx.array]:
        """Compute ``H = X^T @ X / n`` (with diagonal damping) per layer.

        With ``debug_identity_hessian=True`` returns identity matrices
        — handy as a smoke test (GPTQ then degenerates to nearest-
        rounding and should be bit-equal to ``quantize_in_place``).
        """
        hessians: dict[str, mx.array] = {}
        for name, chunks in self.captured.items():
            if not chunks:
                continue
            X = mx.concatenate(chunks, axis=0).astype(mx.float32)
            n_samples, dim = X.shape
            if debug_identity_hessian:
                H = mx.eye(dim)
            else:
                H = (X.T @ X) / n_samples
                diag_mean = mx.mean(mx.diag(H))
                H = H + 0.01 * diag_mean * mx.eye(dim)
            mx.eval(H)
            hessians[name] = H
        return hessians


def _pack_int4(Q: mx.array, bits: int = 4) -> mx.array:
    """Pack int values into ``uint32`` (8 nibbles per ``uint32`` for
    bits=4, LSB first). Matches MLX's :func:`mx.quantize` convention.
    """
    elems_per_int = 32 // bits
    Q = Q.astype(mx.uint32)
    rows, cols = Q.shape
    packed = mx.zeros((rows, cols // elems_per_int), dtype=mx.uint32)
    for k in range(elems_per_int):
        packed = packed | (Q[:, k::elems_per_int] << (k * bits))
    return packed


def _gptq_quantize_weight(
    W: mx.array, H: mx.array,
    *, bits: int = 4, group_size: int = 64, block_size: int = 128,
) -> tuple[mx.array, mx.array, mx.array]:
    """GPTQ error compensation for a single weight matrix.

    Returns ``(packed, scales, biases)`` in the same format as
    :func:`mx.quantize`, ready to set directly on the layer's
    ``q_weight`` / ``q_scales`` / ``q_biases`` attributes.

    With ``H = I`` (identity), produces bit-identical output to
    :func:`mx.quantize` on ``W``.
    """
    orig_dtype = W.dtype
    W = W.astype(mx.float32)
    rows, cols = W.shape
    if cols % group_size != 0:
        raise ValueError(
            f"weight last dim {cols} not divisible by group_size {group_size}"
        )

    # mx.linalg.inv is currently CPU-only.
    H = H.astype(mx.float32)
    cpu = mx.cpu
    try:
        H_inv = mx.linalg.inv(H, stream=cpu)
        mx.eval(H_inv)
    except Exception:
        diag_mean = mx.mean(mx.diag(H))
        H = H + 0.1 * diag_mean * mx.eye(H.shape[0])
        H_inv = mx.linalg.inv(H, stream=cpu)
        mx.eval(H_inv)

    W_work = mx.array(W)
    max_val = 2 ** bits - 1
    num_groups = cols // group_size
    elems_per_int = 32 // bits

    Q_all = mx.zeros((rows, cols), dtype=mx.int32)
    out_scales = mx.zeros((rows, num_groups), dtype=orig_dtype)
    out_biases = mx.zeros((rows, num_groups), dtype=orig_dtype)

    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        H_inv_block = H_inv[block_start:block_end, block_start:block_end]
        err_block = mx.zeros((rows, block_end - block_start), dtype=mx.float32)

        cur_scale_f32 = None
        cur_bias_f32 = None
        group_q = None
        group_modified: list[bool] | None = None

        for j_local in range(block_end - block_start):
            j = block_start + j_local
            g = j // group_size
            j_in_group = j % group_size

            if j_in_group == 0:
                g_start = g * group_size
                group_cols = W_work[:, g_start:g_start + group_size]
                g_packed, g_scales, g_biases = mx.quantize(
                    group_cols.astype(orig_dtype),
                    group_size=group_size, bits=bits,
                )
                mx.eval(g_packed, g_scales, g_biases)
                out_scales[:, g] = g_scales.squeeze(1)
                out_biases[:, g] = g_biases.squeeze(1)
                cur_scale_f32 = g_scales.astype(mx.float32)
                cur_bias_f32 = g_biases.astype(mx.float32)

                # Unpack int4 nibbles.
                group_q = mx.zeros((rows, group_size), dtype=mx.int32)
                for k in range(elems_per_int):
                    nibbles = ((g_packed >> (k * bits)) & max_val).astype(mx.int32)
                    for p in range(nibbles.shape[1]):
                        col_in_group = k + p * elems_per_int
                        if col_in_group < group_size:
                            group_q[:, col_in_group] = nibbles[:, p]
                mx.eval(group_q)
                group_modified = [False] * group_size

            if group_modified[j_in_group]:
                w_col_dt = W_work[:, j].astype(orig_dtype)[:, None]
                q_val = mx.clip(
                    mx.floor(
                        (w_col_dt.astype(mx.float32) - cur_bias_f32) / cur_scale_f32 + 0.5
                    ),
                    0, max_val,
                ).squeeze(1).astype(mx.int32)
            else:
                q_val = group_q[:, j_in_group]

            w_col = W_work[:, j]
            d = H_inv_block[j_local, j_local]
            if d < 1e-10:
                d = mx.array(1e-10)
            q_col = q_val[:, None].astype(mx.float32)
            w_hat_col = (cur_scale_f32 * q_col + cur_bias_f32).squeeze(1)
            Q_all[:, j] = q_val

            err = (w_col - w_hat_col) / d
            err_block[:, j_local] = err

            if j_local < block_end - block_start - 1:
                h_compensation = H_inv_block[j_local, j_local + 1:block_end - block_start]
                W_work[:, j + 1:block_end] -= (
                    err[:, None] * h_compensation[None, :]
                )
                if mx.any(h_compensation != 0).item():
                    for jj in range(j + 1, min(block_end, (g + 1) * group_size)):
                        group_modified[jj % group_size] = True

        if block_end < cols:
            W_work[:, block_end:] -= (
                err_block @ H_inv[block_start:block_end, block_end:]
            )
        mx.eval(W_work, err_block, Q_all)

    Q_all = mx.clip(Q_all, 0, max_val)
    packed = _pack_int4(Q_all, bits)
    mx.eval(packed, out_scales, out_biases)
    return packed, out_scales, out_biases


def _set_dense_quantized(layer: Dense, packed: mx.array,
                         scales: mx.array, biases: mx.array,
                         *, bits: int, group_size: int) -> None:
    """Install GPTQ-packed weights on a :class:`Dense`, mirroring what
    :meth:`Dense.to_quantized` does (bypassing its
    nearest-rounding ``mx.quantize`` call)."""
    layer.q_weight = packed
    layer.q_scales = scales
    layer.q_biases = biases
    layer._q_group_size = group_size
    layer._q_bits = bits
    layer._q_bias = layer.linear.bias if hasattr(layer.linear, "bias") else None
    layer.linear = nn.Identity()
    layer._quantized = True


def _set_einsum_quantized(layer: EinsumDense, packed: mx.array,
                          scales: mx.array, biases: mx.array,
                          *, bits: int, group_size: int,
                          n: int, h: int) -> None:
    """Install GPTQ-packed weights on an :class:`EinsumDense`."""
    layer.q_weight = packed
    layer.q_scales = scales
    layer.q_biases = biases
    layer._q_group_size = group_size
    layer._q_bits = bits
    layer._q_n, layer._q_h = n, h
    layer.kernel = None
    layer._quantized = True


def gptq_calibrate_and_quantize(
    root: nn.Module,
    calibrate_fn: Callable[[nn.Module], None],
    *,
    group_size: int = 64,
    bits: int = 4,
    block_size: int = 128,
    max_samples: int = 2048,
    debug_identity_hessian: bool = False,
    skip: Callable[[str, nn.Module], bool] | None = None,
    verbose: bool = False,
) -> list[str]:
    """End-to-end GPTQ: capture activations → Hessians → packed int4.

    Drop-in replacement for :func:`quantize_in_place`. Same model
    structure and tensor shapes after, just with GPTQ-error-compensated
    rounding instead of nearest-rounding.

    Args:
        root: model tree to quantize in place.
        calibrate_fn: callable that drives ``root`` for one or more
            forward passes (typically a closure over a small batch of
            representative inputs). Activations are captured during
            this call.
        group_size / bits / block_size: standard GPTQ knobs.
        max_samples: cap on per-layer activation rows captured.
        debug_identity_hessian: force ``H = I`` for every layer.
            With this on, GPTQ degenerates to nearest-rounding and
            output is bit-identical to :func:`quantize_in_place`.
        skip: optional ``(path, module) -> bool`` to leave specific
            layers full-precision (e.g. ``to_logits``).
        verbose: progress prints.

    Returns:
        List of fully-qualified paths that were quantized.
    """
    layers = _find_quantizable_layers(root)
    if skip is not None:
        layers = [(p, m, k) for p, m, k in layers if not skip(p, m)]

    if verbose:
        print(f"Capturing activations for {len(layers)} GPTQ layers…")
    with _ActivationCapture(root, max_samples=max_samples, layers=layers) as cap:
        calibrate_fn(root)
        if verbose:
            print(f"  captured {len(cap.captured)} layers' activations.")
        hessians = cap.get_hessians(debug_identity_hessian=debug_identity_hessian)
    if verbose:
        print(f"Computed Hessians for {len(hessians)} layers.")

    quantized: list[str] = []
    for path, layer, kind in layers:
        if path not in hessians:
            if verbose:
                print(f"  [skip] {path}: no captured activations.")
            continue
        H = hessians[path]
        if kind == "dense":
            W = layer.linear.weight  # [out, in]
            if W.shape[-1] % group_size != 0:
                if verbose:
                    print(f"  [skip] {path}: in-dim not divisible by group_size.")
                continue
            packed, scales, biases = _gptq_quantize_weight(
                W, H, bits=bits, group_size=group_size, block_size=block_size,
            )
            _set_dense_quantized(layer, packed, scales, biases,
                                 bits=bits, group_size=group_size)
            quantized.append(path)
            if verbose:
                print(f"  [GPTQ] {path} (Dense) {tuple(W.shape)} → packed.")
        elif kind == "einsum":
            kernel = layer.kernel  # [d, n, h]
            d, n, h = kernel.shape
            if (n * h) % group_size != 0:
                if verbose:
                    print(f"  [skip] {path}: n*h not divisible by group_size.")
                continue
            W2d = kernel.reshape(d, n * h)
            packed, scales, biases = _gptq_quantize_weight(
                W2d, H, bits=bits, group_size=group_size, block_size=block_size,
            )
            _set_einsum_quantized(layer, packed, scales, biases,
                                  bits=bits, group_size=group_size,
                                  n=n, h=h)
            quantized.append(path)
            if verbose:
                print(f"  [GPTQ] {path} (EinsumDense) {tuple(kernel.shape)} → packed.")
    if verbose:
        print(f"GPTQ done: {len(quantized)} layers quantized.")
    return quantized
