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

"""Freeze NNX submodules by retyping their Params.

Idiom: convert every ``nnx.Param`` under a submodule to a ``Frozen`` Variable
subclass. ``nnx.Optimizer(model, tx, wrt=nnx.Param)`` then skips them — no
Adam moments allocated — and ``nnx.value_and_grad(loss, argnums=
nnx.DiffState(0, nnx.Param))`` doesn't differentiate through them.

(Optax has ``optax.transforms.freeze(mask)`` / ``selective_transform(...)`` for
the same effect at the optimizer layer; we prefer the NNX-level approach
because the freeze decision is then encoded in the model state itself and
shows up directly in ``nnx.tabulate`` output.)

Tradeoff vs. path filters: flax's surgery guide would express this as
``wrt=nnx.All(nnx.Param, nnx.PathContains('decoder'))`` — no retyping, fully
reversible, and the checkpoint state keeps plain ``Param`` types. We accept
the retype because it keeps ``diff_filter=nnx.Param`` simple and makes the
freeze introspectable; the cost is that state-walking code (e.g. the Linen
safetensors export) must extract ``(nnx.Param, Frozen)`` rather than just
``nnx.Param``.
"""

from __future__ import annotations

from flax import nnx


class Frozen(nnx.Variable):
    """Param-shaped state that ``wrt=nnx.Param`` filters skip."""


def freeze_module(module: nnx.Module, *, kind: type = Frozen) -> int:
    """Retype every ``nnx.Param`` under ``module`` to ``kind``. Returns count."""
    n = 0
    for _path, sub in nnx.iter_modules(module):
        for attr_name in list(vars(sub)):
            v = getattr(sub, attr_name)
            if isinstance(v, nnx.Param):
                setattr(sub, attr_name, kind(v[...]))
                n += 1
    return n
