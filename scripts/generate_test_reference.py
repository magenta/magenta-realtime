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

"""Generate JAX fp32 reference data for MLX bit-level parity tests.

Runs the JAX depthformer model with compute_dtype=float32, captures
intermediate values (temporal_inputs, temporal_outputs, depth_logits,
depth_samples for all 16 RVQ iterations), and saves them as .npy files
in mlx/testdata/.

The MLX test (test_bitlevel_parity.py) uses these golden references to
verify that the MLX implementation produces matching results. For the
depth decoder, the test uses "teacher forcing" — feeding JAX's sampled
tokens back as input at each RVQ step — so that logits can be compared
deterministically across all 16 iterations.

Usage:
  python scripts/generate_test_reference.py

Prerequisites:
  - JAX and sequence-layers (JAX backend) must be installed
  - The checkpoint file must exist at the expected path
"""

import jax
from jax import numpy as jnp, random
import safetensors.flax as safetensors_flax
import flax.traverse_util as flaxtu
import numpy as np
import magenta_rt  # noqa: F401 — activates vendored sequence_layers
import sequence_layers.jax as sl

from pathlib import Path
from magenta_rt.jax import model, system, spectrostream
from magenta_rt import paths

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants — must match test_bitlevel_parity.py
# ---------------------------------------------------------------------------
_MUSICCOCA = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
_NOTES = [0] * 127 + [1]
_DRUMS = [-1]
_CFG_CONDITIONING = [4, 4, 4]
NUM_RESERVED_TOKENS = 7  # system.NUM_RESERVED_TOKENS(6) + 1 dropout


def main():
    print("GPUs Available:", jax.devices())

    # Float32 compute for parity testing.
    exp = model.MagentaRT2ModelBase()
    exp.compute_dtype = jnp.float32
    depthformer_config = exp.depthformer_config()
    rvq_truncation = exp.spectrostream.rvq_truncation_level
    spectrostream_config = (
        spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq_truncation, use_unique_codes=False,
        )
    )
    mrt_sampler = system.MagentaRT2Sampler.Config(
        depthformer=depthformer_config, spectrostream=spectrostream_config,
    ).make()

    # Load weights.
    checkpoint = paths.resolve_checkpoint(paths.DEFAULT_CHECKPOINT)
    flat_weights = safetensors_flax.load_file(str(checkpoint))
    nested_dict = {tuple(k.split('/')): v for k, v in flat_weights.items()}
    all_params = flaxtu.unflatten_dict(nested_dict)
    print(f"Loaded params from {checkpoint.name}")

    rngs = {"params": random.PRNGKey(42), "random": random.PRNGKey(0)}
    input_channel_spec = jax.ShapeDtypeStruct([len(_MUSICCOCA) + len(_NOTES) + len(_DRUMS) + len(_CFG_CONDITIONING)], jnp.int32)

    # Conditioning inputs.
    cond_tokens = np.concatenate([_MUSICCOCA, _NOTES, _DRUMS, _CFG_CONDITIONING], axis=0) + NUM_RESERVED_TOKENS
    block = sl.Sequence.from_values(
        cond_tokens.reshape(1, 1, -1).astype(np.int32)
    )
    constants = {
        "temperature": jnp.array([1.3]),
        "top_k": jnp.array([40]),
    }

    def _init_state(params, constants):
        return mrt_sampler.apply(
            params, 1, input_channel_spec,
            constants=constants, training=False, rngs=rngs,
            method=mrt_sampler.get_initial_state,
        )

    def _streaming_step(params, x, constants, state):
        return mrt_sampler.apply(
            params, x=x, state=state,
            constants=constants, training=False, rngs=rngs,
            method=mrt_sampler.step_with_emits,
            capture_intermediates=True,
        )

    # Run one step.
    state = _init_state(all_params, constants)
    (y, state, _), intermediates = _streaming_step(
        all_params, block, constants, state
    )

    outdir = REPO_ROOT / 'tests' / 'testdata'
    outdir.mkdir(exist_ok=True)
    decoder_intermediates = intermediates['intermediates']['depthformer']['decoder']

    # Temporal intermediates.
    temporal_inputs = jax.device_get(decoder_intermediates['temporal_inputs'][0])
    temporal_outputs = jax.device_get(decoder_intermediates['temporal_outputs'][0])
    np.save(outdir / 'jax_fp32_step1_temporal_inputs.npy', temporal_inputs)
    np.save(outdir / 'jax_fp32_step1_temporal_outputs.npy', temporal_outputs)
    print(f"temporal_inputs:  shape={temporal_inputs.shape}, dtype={temporal_inputs.dtype}")
    print(f"temporal_outputs: shape={temporal_outputs.shape}, dtype={temporal_outputs.dtype}")

    # Depth logits & samples (one per RVQ index, for teacher forcing).
    depth_logits_list = decoder_intermediates['depth_logits']
    depth_samples_list = decoder_intermediates['depth_samples']
    depth_logits_all = np.stack(
        [jax.device_get(dl) for dl in depth_logits_list], axis=0
    )
    depth_samples_all = np.stack(
        [jax.device_get(ds) for ds in depth_samples_list], axis=0
    )
    np.save(outdir / 'jax_fp32_step1_depth_logits.npy', depth_logits_all)
    np.save(outdir / 'jax_fp32_step1_depth_samples.npy', depth_samples_all)
    print(f"depth_logits:  shape={depth_logits_all.shape}, dtype={depth_logits_all.dtype}")
    print(f"depth_samples: shape={depth_samples_all.shape}, dtype={depth_samples_all.dtype}")

    # Sanity print first 3 RVQ levels.
    for rvq_idx in range(min(3, len(depth_logits_list))):
        logits = depth_logits_all[rvq_idx]
        sample = depth_samples_all[rvq_idx]
        print(f"  rvq={rvq_idx}: logits[0,0,:5]={logits[0, 0, :5]}, sample[0,0]={sample[0, 0]}")

    print(f"\nSaved all reference data to {outdir}")


if __name__ == '__main__':
    main()
