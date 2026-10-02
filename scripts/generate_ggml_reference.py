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

"""Generate JAX fp32 reference tensors for the ggml port.

Runs the JAX MagentaRT-2 sampler with compute_dtype=float32 and captures
per-submodule intermediates via flax's `capture_intermediates`, then writes each
selected tensor as a flat little-endian fp32 `.bin` plus a `manifest.json`
describing shapes. The C++ side reads these with a plain fread, mirroring the
parity harness used for the ggml Llama block port.

Unlike tests/testdata/jax_fp32_step1_*.npy (which only covers the four sow()
points), this dumps per-layer outputs so a mismatch can be localized to a single
transformer layer instead of bisecting twelve of them.

Usage:
  python scripts/generate_ggml_reference.py --model=mrt2_small --num-steps=2
"""

import argparse
import json
import re
from pathlib import Path

import flax.traverse_util as flaxtu
import jax
import numpy as np
import safetensors.flax as safetensors_flax
from jax import numpy as jnp, random

import magenta_rt  # noqa: F401 — activates vendored sequence_layers
import sequence_layers.jax as sl
from magenta_rt import paths
from magenta_rt.jax import model, spectrostream, system

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must match scripts/generate_test_reference.py so both harnesses agree.
_MUSICCOCA = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
_NOTES = [0] * 127 + [1]
_DRUMS = [-1]
_CFG_CONDITIONING = [4, 4, 4]
NUM_RESERVED_TOKENS = 7  # system.NUM_RESERVED_TOKENS(6) + 1 dropout

# Intermediates worth dumping. Anything else (dropout masks, setup markers,
# get_initial_state) is noise for parity purposes.
KEEP = [
    # end-to-end sow() points
    r"decoder/temporal_inputs$",
    r"decoder/temporal_outputs$",
    r"decoder/depth_logits$",
    r"decoder/depth_samples$",
    r"encoder/body/step$",
    # per-layer outputs, so a mismatch localizes to one layer
    r"decoder/(temporal|depth)_body/transformer/x_layers_\d+/step$",
    # per-sublayer (residual block) outputs
    r"decoder/(temporal|depth)_body/transformer/x_layers_\d+/(self_attention|cross_attention|ffn)/step$",
    # attention internals, for localizing within a sublayer
    r"x_layers_\d+/\w+/attention/step_with_emits$",
    r"x_layers_\d+/\w+/attention/(query|key|value)_projection/project_sequence$",
    r"x_layers_\d+/\w+/attention/(query|key|value)_network/step$",
    r"x_layers_\d+/\w+/attention/(query|key|value)_network/layers_\d+/step$",
    # FFN + norm internals
    r"x_layers_\d+/ffn/ffn_layer\d+/step$",
    r"x_layers_\d+/\w+/(pre_norm|post_norm)/step$",
    # depth head
    r"decoder/depth_body/(depth_input_adapter|final_ln|to_logits)/step$",
    # SpectroStream codec: per-block outputs and the final waveform
    r"soundstream/decoder/decoder_\d+/conv2d_3x3/layers_0/layer$",
    r"soundstream/decoder/decoder_\d+/conv2dtranspose_\d+x\d+/layers_0/layer$",
    r"soundstream/decoder/decoder_\d+/shortcut_layer/upsample2d/layer$",
    r"soundstream/decoder/decoder_\d+/residual_function$",
    r"soundstream/decoder/input_layer/.*/layer$",
    r"soundstream/decoder/input_layers_residual_unit/.*/layer$",
    r"soundstream/decoder/output_layer/.*/layer$",
    r"soundstream/.*/conv/_layer$",
    r"soundstream/decoder/.*/upsample2d/layer$",
    r"intermediates/layers_\d+/layer$",
]


def _keep(name: str) -> bool:
    return any(re.search(p, name) for p in KEEP)


def _to_arrays(value):
    """Flatten a captured intermediate into a list of numpy arrays."""
    out = []

    def walk(v, depth=0):
        if depth > 4:
            return
        if isinstance(v, sl.Sequence):
            out.append(np.asarray(jax.device_get(v.values)))
        elif isinstance(v, (jnp.ndarray, np.ndarray)):
            arr = np.asarray(jax.device_get(v))
            if arr.dtype.kind in "fiu" and arr.size:
                out.append(arr)
        elif isinstance(v, (tuple, list)):
            for item in v:
                walk(item, depth + 1)

    walk(value)
    return out


def sanitize(name: str) -> str:
    name = name.replace("intermediates/", "").replace("/", ".")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--num-steps", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--minimal", action="store_true",
                    help="Dump only the sow() points and audio (for long runs).")
    args = ap.parse_args()

    global KEEP
    if args.minimal:
        KEEP = [r"decoder/temporal_inputs$", r"decoder/temporal_outputs$",
                r"decoder/depth_logits$", r"decoder/depth_samples$",
                r"encoder/body/step$"]

    outdir = Path(args.outdir) if args.outdir else (
        REPO_ROOT / "tests" / "testdata" / "ggml" / args.model
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("JAX devices:", jax.devices())
    exp = model.get_model_class(args.model)()
    exp.compute_dtype = jnp.float32  # fp32 reference
    depthformer_config = exp.depthformer_config()
    rvq_truncation = exp.spectrostream.rvq_truncation_level
    spectrostream_config = (
        spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq_truncation, use_unique_codes=False,
        )
    )
    sampler = system.MagentaRT2Sampler.Config(
        depthformer=depthformer_config, spectrostream=spectrostream_config,
    ).make()

    checkpoint = paths.resolve_checkpoint(f"{args.model}.safetensors")
    flat_weights = safetensors_flax.load_file(str(checkpoint))
    params = flaxtu.unflatten_dict(
        {tuple(k.split("/")): v for k, v in flat_weights.items()}
    )
    print(f"Loaded params from {checkpoint.name}")

    cond_tokens = np.concatenate(
        [_MUSICCOCA, _NOTES, _DRUMS, _CFG_CONDITIONING], axis=0
    ) + NUM_RESERVED_TOKENS
    block = sl.Sequence.from_values(cond_tokens.reshape(1, 1, -1).astype(np.int32))
    input_spec = jax.ShapeDtypeStruct([len(cond_tokens)], jnp.int32)
    rngs = {"params": random.PRNGKey(42), "random": random.PRNGKey(0)}
    constants = {
        "temperature": jnp.array([1.3]),
        "top_k": jnp.array([40]),
    }

    state = sampler.apply(
        params, 1, input_spec, constants=constants, training=False, rngs=rngs,
        method=sampler.get_initial_state,
    )

    manifest: dict[str, dict] = {}

    def dump(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        arr.tofile(outdir / f"{name}.bin")
        manifest[name] = {"shape": list(arr.shape), "file": f"{name}.bin"}

    for step_idx in range(args.num_steps):
        (y, state, _), intermediates = sampler.apply(
            params, x=block, state=state, constants=constants, training=False,
            rngs=rngs, method=sampler.step_with_emits,
            capture_intermediates=lambda mdl, name: True,
        )
        flat = flaxtu.flatten_dict(intermediates, sep="/")
        kept = 0
        for key in sorted(flat):
            if not _keep(key):
                continue
            arrays = _to_arrays(flat[key])
            base = f"step{step_idx}.{sanitize(key)}"
            for i, arr in enumerate(arrays):
                dump(base if len(arrays) == 1 else f"{base}.{i}", arr)
                kept += 1
        dump(f"step{step_idx}.audio", np.asarray(jax.device_get(y.values)))
        print(f"step {step_idx}: dumped {kept + 1} tensors")

    # Also record the conditioning tokens so C++ feeds identical input.
    tokens = cond_tokens.astype(np.int32)
    tokens.tofile(outdir / "cond_tokens.i32.bin")
    manifest["cond_tokens"] = {
        "shape": list(tokens.shape), "file": "cond_tokens.i32.bin", "dtype": "int32",
    }

    meta = {
        "model": args.model,
        "num_steps": args.num_steps,
        "temperature": 1.3,
        "top_k": 40,
        "num_reserved_tokens": NUM_RESERVED_TOKENS,
        "tensors": manifest,
    }
    (outdir / "manifest.json").write_text(json.dumps(meta, indent=2))

    # Flat TSV mirror of the manifest: "name<TAB>ndim<TAB>d0 d1 ...<TAB>file".
    # The C++ parity harness parses this instead of JSON to stay dependency-free.
    lines = []
    for name in sorted(manifest):
        entry = manifest[name]
        dims = " ".join(str(d) for d in entry["shape"])
        lines.append(f"{name}\t{len(entry['shape'])}\t{dims}\t{entry['file']}")
    (outdir / "manifest.txt").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {len(manifest)} tensors + manifest.json/.txt to {outdir}")


if __name__ == "__main__":
    main()
