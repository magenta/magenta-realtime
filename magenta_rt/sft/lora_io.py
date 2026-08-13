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

"""Portable, self-describing LoRA adapter files (a single ``.safetensors``).

A fine-tune should be distributable as **one small file** — only the trained
adapter tensors (KBs–MBs), never the ~230M base — that is also **self-contained**:
the safetensors header's ``__metadata__`` records the full *recipe* (rank, alpha,
DoRA on/off, the wrapped target set, the base model name and its checkpoint hash),
so the consumer reconstructs the adapter with no out-of-band flags to remember.

* :func:`save_lora_adapters` — write the ``MRTLoRAParam`` leaves of an NNX model
  (``lora_a`` / ``lora_b`` / DoRA ``magnitude``) plus the recipe metadata. The
  recipe is *inferred from the model* (rank/alpha/DoRA), so it always matches the
  saved weights.
* :func:`load_lora_adapters` — read the recipe, :func:`inject_lora` matching
  adapters into a fresh base model, load the weights, apply the stored (or an
  override) ``lora_strength``. The inverse of save.
* :func:`read_metadata` — just the recipe dict (no weights), e.g. for a CLI to
  print or to drive ``inject_lora`` itself.

The on-disk math is identical across backends (the LoRA/DoRA merge is bit-exact
between NNX and MLX), so an adapter saved by the MLX trainer loads here and vice
versa; ``backend`` in the metadata is informational. The MLX trainer writes the
same metadata schema (:func:`build_metadata`) into its
``mx.save_safetensors(..., metadata=...)`` call.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

FORMAT = "mrt-lora"
FORMAT_VERSION = "1"

_TRUE = {"true", "1", "yes"}


def _as_bool(s: str) -> bool:
    return str(s).strip().lower() in _TRUE


def file_sha256(path: str | Path, *, bufsize: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (used to fingerprint the base checkpoint)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(
    *,
    backend: str,
    base_model: str,
    rank: int,
    alpha: float,
    dora: bool,
    targets: str,
    base_checkpoint_sha256: str = "",
    lora_strength: float = 1.0,
) -> dict[str, str]:
    """Build the safetensors ``__metadata__`` recipe dict (all values strings).

    ``targets`` is a preset *name* (``"default"`` / ``"all_linears"``) so the
    wrapped set round-trips without serializing a predicate function.
    """
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "backend": str(backend),
        "base_model": str(base_model),
        "rank": str(int(rank)),
        "alpha": repr(float(alpha)),
        "dora": str(bool(dora)),
        "targets": str(targets),
        "base_checkpoint_sha256": str(base_checkpoint_sha256),
        "lora_strength": repr(float(lora_strength)),
    }


def read_metadata(path: str | Path) -> dict[str, str]:
    """Return the adapter recipe dict from a ``.safetensors`` header.

    Raises if the file carries no ``mrt-lora`` recipe (so a plain weights
    safetensors is rejected loudly rather than mis-loaded).
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
    if not meta or meta.get("format") != FORMAT:
        raise ValueError(
            f"{path} has no '{FORMAT}' recipe metadata — not a portable LoRA "
            "adapter file (was it saved by save_lora_adapters / the trainer?)."
        )
    return meta


# ---------------------------------------------------------------------------
# NNX target-preset registry (maps the metadata name <-> the predicate fn)
# ---------------------------------------------------------------------------


def _nnx_target_fn(name: str):
    from . import lora_nnx

    table = {
        "default": lora_nnx.default_targets,
        "all_linears": lora_nnx.all_linear_targets,
        "all_plus": lora_nnx.all_plus_targets,
    }
    if name not in table:
        raise ValueError(
            f"unknown targets preset {name!r}; expected one of {sorted(table)}."
        )
    return table[name]


def _find_nnx_adapter(model):
    """Return any :class:`LoRAAdapter` in the model, or None if none injected."""
    from flax import nnx

    from .lora_nnx import LoRAAdapter

    for _, module in nnx.iter_graph(model):
        if isinstance(module, LoRAAdapter):
            return module
    return None


def infer_nnx_recipe(model) -> dict:
    """Recover ``{rank, alpha, dora}`` from an injected NNX model.

    ``alpha`` is reconstructed as ``scale * rank`` so feeding it back to
    :func:`inject_lora` reproduces the exact ``scale`` (including the
    ``alpha == 0 -> scale == 1`` case).
    """
    adapter = _find_nnx_adapter(model)
    if adapter is None:
        raise ValueError("model has no LoRAAdapter — call inject_lora first.")
    rank = int(adapter.lora_a[...].shape[-1])
    scale = float(adapter.scale)
    return {"rank": rank, "alpha": scale * rank, "dora": bool(adapter.dora)}


# ---------------------------------------------------------------------------
# NNX save / load
# ---------------------------------------------------------------------------


def save_lora_adapters(
    model,
    out_path: str | Path,
    *,
    base_model: str,
    targets: str = "all_linears",
    base_checkpoint: Optional[str | Path] = None,
    lora_strength: float = 1.0,
    backend: str = "nnx",
) -> str:
    """Write an NNX model's adapters to a self-describing ``.safetensors``.

    Only the ``MRTLoRAParam`` leaves are written (``lora_a`` / ``lora_b`` /
    DoRA ``magnitude``), with the leading vmapped ``num_layers`` axis kept as a
    stacked tensor. The recipe (rank/alpha/DoRA) is inferred from the model;
    ``targets`` (a preset name) and ``base_model`` are recorded so
    :func:`load_lora_adapters` can rebuild the wrappers. If ``base_checkpoint``
    is given, its SHA-256 is stored so a consumer can detect a base mismatch.

    Returns the output path.
    """
    import jax
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from safetensors.numpy import save_file

    from .lora_nnx import MRTLoRAParam

    recipe = infer_nnx_recipe(model)
    pure = nnx.to_pure_dict(nnx.state(model, MRTLoRAParam))
    flat = flatten_dict(pure, sep="/")
    if not flat:
        raise ValueError("no MRTLoRAParam adapters found to save.")

    tensors: dict[str, np.ndarray] = {}
    for key, value in flat.items():
        arr = np.asarray(jax.device_get(value))
        # safetensors.numpy cannot store bfloat16; persist as fp32 and let the
        # loader cast back to the target model's adapter dtype.
        if str(arr.dtype) == "bfloat16":
            arr = arr.astype(np.float32)
        tensors[key] = np.ascontiguousarray(arr)

    sha = file_sha256(base_checkpoint) if base_checkpoint else ""
    meta = build_metadata(
        backend=backend, base_model=base_model, targets=targets,
        base_checkpoint_sha256=sha, lora_strength=lora_strength, **recipe,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=meta)
    return str(out_path)


def load_lora_adapters(
    model,
    path: str | Path,
    *,
    strength: Optional[float] = None,
) -> dict[str, str]:
    """Inject + load adapters into a fresh NNX base model from its recipe.

    Reads the embedded recipe, :func:`inject_lora` with the matching
    rank/alpha/DoRA/targets, overwrites the adapter leaves with the stored
    weights, and applies the stored ``lora_strength`` (or ``strength`` if
    given). Returns the recipe metadata.
    """
    from flax import nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    from safetensors import safe_open

    from .lora_nnx import MRTLoRAParam, inject_lora, set_lora_strength

    meta = read_metadata(path)
    n = inject_lora(
        model,
        rank=int(meta["rank"]),
        alpha=float(meta["alpha"]),
        dora=_as_bool(meta["dora"]),
        targets=_nnx_target_fn(meta["targets"]),
        seed=0,  # irrelevant: every adapter leaf is overwritten below
    )
    if n == 0:
        raise ValueError(
            "inject_lora wrapped 0 layers — base model or targets preset does "
            f"not match this adapter (base_model={meta['base_model']!r})."
        )
    return load_lora_weights(model, path, strength=strength)


def load_lora_weights(
    model,
    path: str | Path,
    *,
    strength: Optional[float] = None,
) -> dict[str, str]:
    """Overwrite the adapter weights of an **already-injected** model in place.

    Unlike :func:`load_lora_adapters` this does NOT call ``inject_lora`` — it
    assumes ``model`` already carries matching LoRA wrappers (same rank / alpha /
    targets) and just replaces their ``lora_a`` / ``lora_b`` (and DoRA
    ``magnitude``) leaves with the file's. This is the cheap path for *switching*
    between adapter checkpoints of the same run: no re-inject, no base reload,
    and (since only leaf *values* change, not the graph) no ``nnx.jit``
    recompile. Returns the recipe metadata.
    """
    from flax import nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    from safetensors import safe_open

    from .lora_nnx import MRTLoRAParam, set_lora_strength

    meta = read_metadata(path)
    with safe_open(str(path), framework="numpy") as f:
        loaded = {k: f.get_tensor(k) for k in f.keys()}

    state = nnx.state(model, MRTLoRAParam)
    template = flatten_dict(nnx.to_pure_dict(state), sep="/")
    if set(loaded) != set(template):
        missing = sorted(set(template) - set(loaded))[:3]
        extra = sorted(set(loaded) - set(template))[:3]
        raise ValueError(
            "adapter tensor keys do not match the injected model "
            f"(missing={missing}, unexpected={extra}) — recipe/base mismatch."
        )
    # Cast each tensor to the freshly-injected leaf's dtype (e.g. a bf16 base).
    casted = {
        k: np.asarray(loaded[k]).astype(np.asarray(template[k]).dtype)
        for k in template
    }
    nnx.replace_by_pure_dict(state, unflatten_dict(casted, sep="/"))
    nnx.update(model, state)

    s = strength if strength is not None else float(meta["lora_strength"])
    if s != 1.0:
        set_lora_strength(model, s)
    return meta
