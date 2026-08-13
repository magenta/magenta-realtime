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

"""Fixtures and helpers for `mlx_pure` ↔ `sequence_layers.mlx` parity tests.

Tests build the `sl`-backed module first (initializing its parameters
deterministically), bridge the parameters into a `mlx_pure` module via
:func:`mlx_pure.load_weights.mirror_params`, run identical inputs through
both, and assert numerical equality via :func:`np.testing.assert_allclose`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _to_np(x: mx.array) -> np.ndarray:
    """Bring an mx.array to numpy fp32 for assertion."""
    return np.array(x.astype(mx.float32))


def assert_close(a: mx.array, b: mx.array, *, atol: float = 1e-6, rtol: float = 1e-6, name: str = ""):
    """Thin wrapper around :func:`np.testing.assert_allclose`.

    Casts both arrays to numpy fp32, then delegates. Keeps the
    ``name=`` kwarg for backwards-compat with existing tests.
    """
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    np.testing.assert_allclose(
        _to_np(a), _to_np(b), atol=atol, rtol=rtol,
        err_msg=name or "arrays differ", verbose=True,
    )


def tol(dtype, level: str = "leaf") -> tuple[float, float]:
    """Tolerance presets keyed by dtype and depth-level."""
    if dtype == mx.float32:
        return {
            "leaf": (1e-6, 1e-6),
            "block": (1e-5, 1e-5),
            "stack": (1e-4, 1e-4),
        }[level]
    if dtype == mx.bfloat16:
        return {
            "leaf": (5e-3, 5e-3),
            "block": (2e-2, 2e-2),
            "stack": (5e-2, 5e-2),
        }[level]
    return (1e-4, 1e-4)


@pytest.fixture
def rng_key():
    return mx.random.key(20260507)


def make_normal(key, shape, *, dtype=mx.float32, scale: float = 0.02):
    return mx.random.normal(shape, dtype=dtype, key=key) * scale


def randomize_module(module, key, *, scale: float = 0.02):
    """Re-init every leaf parameter of ``module`` with N(0, scale)."""
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(module.parameters()))
    new = {}
    for i, (name, arr) in enumerate(flat.items()):
        sub = mx.random.split(key, len(flat))[i]
        new[name] = (mx.random.normal(arr.shape, dtype=arr.dtype, key=sub) * scale).astype(arr.dtype)
    module.update(tree_unflatten(list(new.items())))
