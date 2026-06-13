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

"""Tests for prefill and state save/load correctness.

This file verifies that the MLX model state can be correctly saved and restored,
leading to bit-exact identical continuation. It simulates the prefill behavior
of the C++ engine by feeding a sequence of tokens using the `forced_tokens`
feature to build the KV cache state, and then checks that saving and restoring
this state produces deterministic outputs.

Note: This file validates that the prefill codepath functions correctly for state
serialization, though it does not explicitly verify the generative quality of
the prefilled state itself.
"""

import unittest
import mlx.core as mx
import numpy as np
import numpy.testing as npt

import magenta_rt  # noqa: F401 — activates vendored sequence_layers
import sequence_layers.mlx as sl
from magenta_rt.mlx import model
from magenta_rt.mlx import system
from magenta_rt.mlx import spectrostream
from magenta_rt.mlx import load_weights as lw
from magenta_rt.mlx.export import _flatten_state, _unflatten_state
from magenta_rt import paths

# CI checkpoint: mrt2_small.safetensors (downloaded by the CI workflow).
_CI_CHECKPOINT = 'mrt2_small.safetensors'

class TestPrefillCorrectness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        checkpoint_path = paths.resolve_checkpoint(_CI_CHECKPOINT)
        if not checkpoint_path.exists():
            raise unittest.SkipTest(f"Checkpoint not found: {checkpoint_path}")

        print('Building MLX model (compute_dtype=bfloat16)...')
        mrt_model = model.MagentaRT2ModelSmall()
        mrt_model.compute_dtype = mx.bfloat16

        depthformer_config = mrt_model.depthformer_config()
        spectrostream_config = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=16, use_unique_codes=False,
        )
        mrt_config = system.MagentaRT2Sampler.Config(
            depthformer=depthformer_config, spectrostream=spectrostream_config,
        )
        cls.mrt_sampler = mrt_config.make()

        lw.load_weights(cls.mrt_sampler, checkpoint_path, num_input_channels=mrt_model.input_num_channels)

        # Dummy conditioning inputs
        musiccoca = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
        notes = [0] * 127 + [1]
        drums = [0]
        cfg_conditioning = [0, 0, 0]
        cond_tokens = np.concatenate([musiccoca, notes, drums, cfg_conditioning], axis=0) + 6 + 1
        cls.block = sl.Sequence(
            mx.array(cond_tokens.reshape(1, 1, -1), dtype=mx.int32),
            mx.array([[True]], dtype=mx.bool_),
        )

        cls.constants = {
            'temperature': mx.array([0.0]),  # Greedy sampling for bit-exact test
            'top_k': mx.array([40]),
        }

    def test_save_load_state_bit_exact(self):
        """Test that saving and loading state produces identical continuation after prefill."""
        input_spec = sl.ChannelSpec(shape=[12 + 128 + 1 + 3], dtype=mx.int32)

        # Use the sampler layer directly to handle forced_tokens
        sampler = self.mrt_sampler.layers[0]
        state = sampler.get_initial_state(1, input_spec, constants=self.constants)

        codebook_size = self.mrt_sampler.cfg.spectrostream.quantizer.num_embeddings
        num_codebooks = self.mrt_sampler.cfg.depthformer.decoder.num_codebooks

        # Generate a sequence of dummy tokens for prefill
        num_prefill_frames = 10
        token_offsets = mx.arange(num_codebooks) * codebook_size + 6
        token_offsets = mx.reshape(token_offsets, (1, 1, num_codebooks))

        prefill_seq = mx.zeros((1, num_prefill_frames, num_codebooks), dtype=mx.int32)
        prefill_seq = prefill_seq + token_offsets

        # Prefill loop: process N-1 frames
        print(f"Prefilling {num_prefill_frames} frames...")
        for step in range(num_prefill_frames - 1):
            prev_token = prefill_seq[:, step:step+1, :]
            y, state = sampler.step(x=self.block, state=state, forced_tokens=prev_token, constants=self.constants)
            mx.eval(y.values, state)

        # Seed previous_frame with the last token without calling the model
        enc_state, prev_out, samp_state, delay = state
        final_token = prefill_seq[:, num_prefill_frames-1:num_prefill_frames, :]
        new_prev_out = sl.Sequence.from_values(final_token)
        state = (enc_state, new_prev_out, samp_state, delay)
        mx.eval(state)

        # Capture state
        flat_state_saved, structure = _flatten_state(state)
        flat_state_saved = [mx.array(arr) for arr in flat_state_saved]

        # Run 5 more steps (autoregressive)
        outputs_1 = []
        for _ in range(5):
            y, state = sampler.step(x=self.block, state=state, constants=self.constants)
            mx.eval(y.values, state)
            outputs_1.append(np.array(y.values))

        # Restore state
        restored_state = _unflatten_state(flat_state_saved, structure)

        # Run 5 steps again from restored state
        outputs_2 = []
        for _ in range(5):
            y, restored_state = sampler.step(x=self.block, state=restored_state, constants=self.constants)
            mx.eval(y.values, restored_state)
            outputs_2.append(np.array(y.values))

        # Verify outputs are identical
        for o1, o2 in zip(outputs_1, outputs_2):
            npt.assert_array_equal(o1, o2, err_msg="Outputs diverged after state restore")

        print("  Save/Load state bit-exactness verified. ✓")


if __name__ == '__main__':
    unittest.main(verbosity=2)
