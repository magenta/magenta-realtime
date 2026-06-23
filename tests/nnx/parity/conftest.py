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

"""Fixtures and helpers for `magenta_rt.nnx` parity tests.

Tests build the nnx module, optionally bridge weights from a
reference implementation (``flax.linen`` for leaf-layer parity,
``magenta_rt.jax`` for full-system parity), run identical inputs
through both, and assert numerical equality via
:func:`np.testing.assert_allclose`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Force true fp32 matmuls for the whole parity suite. On Ampere/Ada GPUs JAX
# defaults fp32 matmuls to TF32 (~10-bit mantissa), which rounds two equivalent
# but differently-structured computations (sl/linen vs nnx; full-seq vs
# streaming) apart by ~1e-3 — enough to fail the fp32 parity tolerances even
# though both implementations are correct. "highest" keeps the comparison
# deterministic across backends and hardware.
jax.config.update("jax_default_matmul_precision", "highest")


def _to_np(x: jnp.ndarray) -> np.ndarray:
    """Bring a JAX array to numpy fp32 for assertion."""
    return np.asarray(jnp.asarray(x).astype(jnp.float32))


def assert_close(a, b, *, atol: float = 1e-6, rtol: float = 1e-6, name: str = ""):
    """Thin wrapper around :func:`np.testing.assert_allclose`."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    np.testing.assert_allclose(
        _to_np(a), _to_np(b), atol=atol, rtol=rtol,
        err_msg=name or "arrays differ", verbose=True,
    )


@pytest.fixture
def rng_key():
    return jax.random.key(20260509)
