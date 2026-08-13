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

"""Helpers for loading weights into abstractly-constructed nnx models.

A loader can build its model with ``nnx.eval_shape(lambda: Model(...))`` instead
of calling the constructor directly. That skips allocating — and RNG-computing —
a full set of throwaway random-init weights that the checkpoint is about to
overwrite anyway, which matters once a model is large enough that the discarded
init is measured in gigabytes.

The cost is that *everything* the constructor would have produced starts out as
a ``jax.ShapeDtypeStruct`` placeholder, and only what the loader explicitly
assigns becomes real. Three kinds of leaf need attention afterwards:

* **Parameters** — restored from the checkpoint. :func:`assert_fully_loaded`
  verifies none were missed, turning a silently-abstract weight into a loud
  error at load time rather than an obscure failure inside a later ``jit``.
* **RNG state** — never in a checkpoint. :func:`materialize_abstract_rngs`
  gives it concrete keys. Required whenever the model is later passed through
  ``nnx.split``/``jit``, since an abstract key is not a valid JAX type.
* **Computed constants** — tables a constructor derives rather than loads (a
  sinusoidal position embedding, an STFT synthesis window). Nothing generic can
  restore these; the owning loader must recompute them explicitly.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax
from flax import nnx
from jax import numpy as jnp

_M = TypeVar("_M", bound=nnx.Module)


def assert_fully_loaded(model: nnx.Module) -> None:
    """Raise if any parameter is still an abstract placeholder.

    Only ``Param`` and ``BatchStat`` are checked — the leaves that come from a
    pretrained checkpoint. RNG state and inference caches are intentionally
    ignored, since an ``eval_shape`` load leaves those abstract by design (see
    :func:`materialize_abstract_rngs` and the module docstring).

    Args:
        model: A model whose weights should all be concrete arrays by now.

    Raises:
        RuntimeError: If one or more parameters were never assigned.
    """
    weights = nnx.state(model, nnx.Any(nnx.Param, nnx.BatchStat))
    flat = jax.tree_util.tree_flatten_with_path(weights)[0]
    missing = [
        jax.tree_util.keystr(path)
        for path, leaf in flat
        if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise RuntimeError(
            f"{len(missing)} parameter(s) were never loaded (still abstract): "
            f"{shown}. The weight loader must assign every parameter."
        )


def materialize_abstract_rngs(model: nnx.Module, seed: int = 0) -> None:
    """Replace abstract RNG state with concrete keys and counts, in place.

    An ``eval_shape`` build leaves RNG state abstract; it is not in the
    checkpoint and :func:`assert_fully_loaded` deliberately skips it. That is
    harmless right up until the model is split and handed to a jitted function,
    where an abstract key fails as "not a valid JAX type" — so call this before
    any such use.

    Args:
        model: A model whose RNG state may still be abstract after loading.
        seed: Seed used to build the concrete keys.
    """
    _, rng_state, _ = nnx.split(model, nnx.RngState, ...)

    def _concretize(leaf):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            return leaf
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            if leaf.shape == ():
                return jax.random.key(seed)
            return jax.random.split(jax.random.key(seed), leaf.shape)
        return jnp.zeros(leaf.shape, leaf.dtype)  # RNG counts

    rng_state = jax.tree.map(
        _concretize, rng_state, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)
    )
    nnx.update(model, rng_state)


def load_into_abstract(
    build: Callable[[], _M],
    load: Callable[[_M], None],
    *,
    seed: int = 0,
) -> _M:
    """Build a model abstractly, load weights into it, and make it usable.

    Runs the full sequence an ``nnx.eval_shape`` load requires, in the order it
    requires. ``materialize_constants`` comes first because a derived constant
    may not be an abstract *state* leaf at all: when the object holding it is a
    plain Python object rather than an ``nnx.Module``, nnx keeps it statically
    and the abstract build leaves behind a tracer that escaped its trace. That
    tracer must not survive to be traversed.

    Args:
        build: Zero-argument constructor for the model. Must not load weights.
        load: Callable that assigns every parameter from a checkpoint.
        seed: Seed for the RNG state materialized after loading.

    Returns:
        The loaded model, with no leaf left abstract.
    """
    model = nnx.eval_shape(build)
    model.materialize_constants()
    load(model)
    materialize_abstract_rngs(model, seed=seed)
    assert_fully_loaded(model)
    return model
