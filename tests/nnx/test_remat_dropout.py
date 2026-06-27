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

"""Gradient checkpointing (``nnx.remat``) must be consistent with the manual
attention-dropout path.

``Transformer(remat=True)`` wraps each scanned layer in ``nnx.remat`` so the
backward pass *recomputes* the forward. The attention-probability dropout
(``attention_dropout_prob``) draws its mask from a forked ``RngStream`` inside
the scanned body. If remat's recompute drew a *different* mask than the forward,
the gradient would be inconsistent with the value (silently wrong training). This
pins:

* with attention dropout active, the remat and non-remat **forwards are identical**
  (same mask), and
* their **gradients are identical** (the recompute replays the same mask),

so combining attention dropout with the memory-saving remat path is safe.
Checkpoint-free (tiny config) → runs in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from magenta_rt.nnx.transformer import Transformer


def _build(remat: bool) -> Transformer:
    # dropout_prob=0 isolates the manual ATTENTION-probability dropout path
    # (the FFN/residual nnx.Dropout stay rate-0 no-ops, drawing no rng).
    m = Transformer(
        num_layers=3, model_dim=64, num_heads=2, units_per_head=32,
        ffn_dim=128, max_past_horizon=8,
        dropout_prob=0.0, attention_dropout_prob=0.3,
        remat=remat, rngs=nnx.Rngs(0),
    )
    m.train()  # deterministic=False → attention dropout active
    return m


def _loss(model, x):
    return jnp.mean(model(x) ** 2)


def test_attention_dropout_is_active():
    """Sanity: with attention dropout on, train != eval (dropout does something)."""
    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 64), jnp.float32)
    m = _build(remat=False)
    y_train = np.asarray(m(x))
    m.eval()
    y_eval = np.asarray(m(x))
    assert float(np.abs(y_train - y_eval).max()) > 1e-4, "attention dropout had no effect"


def test_remat_replays_attention_dropout_mask():
    """remat forward AND gradient must match the non-remat path (same mask)."""
    x = jnp.asarray(np.random.RandomState(1).randn(2, 12, 64), jnp.float32)

    # Fresh, identically-seeded models → identical params and dropout streams.
    m0, m1 = _build(remat=False), _build(remat=True)

    grad_fn = nnx.value_and_grad(_loss, argnums=nnx.DiffState(0, nnx.Param))
    l0, g0 = grad_fn(m0, x)
    l1, g1 = grad_fn(m1, x)

    # Forward: same dropout mask ⇒ identical loss.
    np.testing.assert_allclose(
        float(l0), float(l1), rtol=0, atol=0,
        err_msg="remat forward differs from non-remat (mask mismatch)",
    )
    # Backward: remat must recompute with the SAME mask. A *different* mask would
    # diverge the gradient by O(1); only fp32 recompute round-off (~1e-7) is
    # expected, so this tolerance still catches a real mask mismatch.
    leaves0 = jax.tree_util.tree_leaves(nnx.state(g0))
    leaves1 = jax.tree_util.tree_leaves(nnx.state(g1))
    assert leaves0 and len(leaves0) == len(leaves1)
    max_diff = max(
        float(jnp.abs(jnp.asarray(a) - jnp.asarray(b)).max())
        for a, b in zip(leaves0, leaves1)
    )
    assert max_diff < 1e-4, (
        f"remat gradient diverges from non-remat by {max_diff} — the backward "
        f"recompute drew a different attention-dropout mask than the forward "
        f"(O(1) divergence = mask mismatch; ~1e-7 = benign recompute round-off)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
