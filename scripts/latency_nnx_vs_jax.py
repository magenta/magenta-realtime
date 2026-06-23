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

"""Streaming-generation latency benchmark: ``jax`` vs ``nnx`` backend.

Measures steady-state per-frame latency of the public ``System.generate``
streaming path. One ``(backend, size, param_dtype, compute_dtype)`` config is
benchmarked per process invocation because the 16 GB RTX 4080 only fits one
model at a time; sweep the matrix with ``scripts/latency_nnx_vs_jax.sh``.

Timing model: 1 frame = 40 ms of audio @ 48 kHz, so the real-time factor is
``rtf = ms_per_step / 40`` (``rtf <= 1.0`` means the backend keeps up with
real-time streaming). The first ``--warmup-frames`` are discarded so the
reported numbers exclude JIT/AOT compilation and reflect a full attention
rolling window; the measured window is repeated ``--reps`` times and the median
ms/step is reported.

Generation is unconditional by default (``style=None`` feeds masked style
tokens), which exercises the full depthformer + codec per frame without loading
MusicCoCa. Pass ``--prompt`` to condition on a text style instead.

The nnx ``System.generate`` path donates the model to its jitted step (in-place
cache updates), which is what lets ``mrt2_base`` bf16 stream on a 16 GB card.
That needs slightly more of the GPU than XLA's 75% default (peak ~10.9 GB; the
default pool fragments just over it), so this script bumps
``XLA_PYTHON_CLIENT_MEM_FRACTION`` to 0.85 before importing JAX — harmless for
the smaller configs, and the reason a re-run shows base fitting. Override by
exporting the env var yourself.
"""

from __future__ import annotations

import os

# Must be set before JAX initializes; setdefault respects a user override.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from magenta_rt import paths

_DTYPES = {"fp32": jnp.float32, "bf16": jnp.bfloat16}


def _sync(tree) -> None:
    """Force device->host completion so timing brackets real GPU work."""
    jax.block_until_ready(np.asarray(tree.waveform))


def _build_jax_system(size: str, param_dtype, compute_dtype, prompt: str | None):
    """Build a ``jax`` system with the requested storage / compute dtypes.

    The jax system reads its dtypes off the model spec class, so a thin spec
    subclass is registered and selected by name; the real checkpoint is passed
    explicitly. ``style_model`` is stubbed for unconditional runs to skip the
    (otherwise eager) MusicCoCa load.
    """
    from magenta_rt.jax import model as jm
    from magenta_rt.jax import system as jsys

    base_cls = jm.get_model_class(size)
    _pdt, _cdt = _DTYPES[param_dtype], _DTYPES[compute_dtype]

    class _BenchSpec(base_cls):
        param_dtype = _pdt
        compute_dtype = _cdt

    jm.MODEL_REGISTRY["_bench"] = _BenchSpec
    style_model = None if prompt is not None else object()
    system = jsys.MagentaRT2System(
        size="_bench",
        checkpoint=f"{size}.safetensors",
        style_model=style_model,
    )
    return system


def _build_nnx_system(size: str, param_dtype, compute_dtype, prompt: str | None):
    """Build an ``nnx`` system with the requested storage / compute dtypes.

    The model is built and weight-loaded directly (``param_dtype=bf16`` halves
    the resident parameter tree and loads through host RAM to avoid an on-device
    fp32 spike), then wrapped by the system with ``restore=False``. The system's
    streaming step is the functional ``jax.jit`` state-threading path (split
    params/stream once, thread the stream pytree).
    """
    from flax import nnx

    from magenta_rt.nnx import model as nm
    from magenta_rt.nnx import system as nsys

    sampler = nm.MagentaRT2Sampler.from_preset(
        size,
        int16_outputs=False,
        param_dtype=_DTYPES[param_dtype],
        dtype=_DTYPES[compute_dtype],
        rngs=nnx.Rngs(0),
    )
    ckpt = paths.checkpoints_dir() / f"{size}.safetensors"
    sampler.load_checkpoint(ckpt, host=(param_dtype == "bf16"))
    system = nsys.MagentaRT2System(
        size=size, model=sampler, restore=False, jit=True, style_model=None,
    )
    return system


def benchmark(
    backend: str,
    size: str,
    param_dtype: str,
    compute_dtype: str,
    warmup_frames: int,
    measure_frames: int,
    reps: int,
    prompt: str | None,
) -> dict:
    """Build one system, warm it up, and time steady-state per-frame latency."""
    if backend == "jax":
        system = _build_jax_system(size, param_dtype, compute_dtype, prompt)
    elif backend == "nnx":
        system = _build_nnx_system(size, param_dtype, compute_dtype, prompt)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    style = system.embed_style(prompt) if prompt is not None else None

    # Warmup: triggers compilation and fills the attention rolling window so the
    # measured steps run in genuine steady state.
    tree, state = system.generate(style=style, frames=warmup_frames)
    _sync(tree)

    per_step_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        tree, state = system.generate(style=style, frames=measure_frames, state=state)
        _sync(tree)
        elapsed = time.perf_counter() - t0
        per_step_ms.append(elapsed / measure_frames * 1000.0)

    ms = statistics.median(per_step_ms)
    result = {
        "backend": backend,
        "size": size,
        "param_dtype": param_dtype,
        "compute_dtype": compute_dtype,
        "ms_per_step": round(ms, 3),
        "steps_per_s": round(1000.0 / ms, 2),
        "rtf": round(ms / 40.0, 4),
        "best_ms_per_step": round(min(per_step_ms), 3),
        "measure_frames": measure_frames,
        "reps": reps,
        "device": str(jax.devices()[0]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser("latency_nnx_vs_jax")
    parser.add_argument("--backend", required=True, choices=["jax", "nnx"])
    parser.add_argument("--size", default="mrt2_small")
    parser.add_argument("--param-dtype", default="fp32", choices=list(_DTYPES))
    parser.add_argument("--compute-dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--warmup-frames", default=50, type=int)
    parser.add_argument("--measure-frames", default=25, type=int)
    parser.add_argument("--reps", default=4, type=int)
    parser.add_argument(
        "--prompt", default=None, type=str,
        help="Text style prompt; omit for unconditional (no MusicCoCa load).",
    )
    parser.add_argument(
        "--out", default=None, type=str,
        help="Append the result as a JSON line to this path.",
    )
    args = parser.parse_args()

    result = benchmark(
        backend=args.backend,
        size=args.size,
        param_dtype=args.param_dtype,
        compute_dtype=args.compute_dtype,
        warmup_frames=args.warmup_frames,
        measure_frames=args.measure_frames,
        reps=args.reps,
        prompt=args.prompt,
    )

    label = f"{args.backend:>3} {args.size:<11} p={args.param_dtype} c={args.compute_dtype}"
    print(
        f"{label} | {result['ms_per_step']:7.2f} ms/step | "
        f"{result['steps_per_s']:6.2f} steps/s | RTF {result['rtf']:.3f} "
        f"(best {result['best_ms_per_step']:.2f} ms)"
    )
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
