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

"""Compare .mlxfn output between Python and C++ by running one step in Python
and saving the raw audio output for comparison."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from magenta_rt.mlx import model
from magenta_rt import paths

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
# system.NUM_RESERVED_TOKENS (6) + 1 dropout token. Sourced from model.py so
# this stays in sync with export.py / mlx_engine.cpp's kNumReservedTokens.
NUM_RESERVED_TOKENS = model.NUM_RESERVED_TOKENS + 1


def main():
    # ── Load exported function ──
    mlxfn_path = str(RESOURCE_DIR / "MagentaRT_transformer.mlxfn")
    state_path = str(RESOURCE_DIR / "MagentaRT_transformer_state.safetensors")


    print(f"Loading {mlxfn_path}")
    fn = mx.import_function(mlxfn_path)

    # ── Load state ──
    state_dict = mx.load(state_path)
    state = []
    for i in range(len(state_dict)):
        key = f"state_{i}"
        if key not in state_dict:
            break
        state.append(state_dict[key])
    mx.eval(state)
    print(f"Loaded {len(state)} state arrays")

    # ── Derive the conditioning layout from the model spec (matches export.py
    #    and the C++ engine; avoids hardcoding the channel count). ──
    exp = model.get_model_class(paths.DEFAULT_MODEL_NAME)()
    num_musiccoca = exp.input_configs[0].rvq_truncation_level
    num_pitches = exp.input_configs[1].rvq_truncation_level
    cond_len = exp.input_num_channels

    rvq_depth = 12

    # ── Build conditioning input (must match the C++ test exactly) ──
    # Layout: [num_musiccoca | 128 notes | 1 drum | 3 cfg], every slot offset by
    # NUM_RESERVED_TOKENS (see export.py / mlx_engine.cpp populate_condition_tokens).
    musiccoca = [309, 73, 885, 825, 615, 672]
    musiccoca = (musiccoca + [0] * num_musiccoca)[:num_musiccoca]
    notes = [0] * (num_pitches - 1) + [1]   # last pitch ON
    drums = [-1]                            # -1 -> let model decide
    cfg_cond = [0, 0, 0]                     # musiccoca / notes / drums cfg tokens
    masked_musiccoca = [-1] * num_musiccoca
    masked_notes = [-1] * num_pitches

    def to_cond(values):
        arr = np.array(values, dtype=np.int32) + NUM_RESERVED_TOKENS
        return mx.array(arr.reshape(1, 1, cond_len), dtype=mx.int32)

    cond_array = to_cond(musiccoca + notes + drums + cfg_cond)
    neg_musiccoca_array = to_cond(masked_musiccoca + notes + drums + cfg_cond)
    neg_notes_array = to_cond(musiccoca + masked_notes + drums + cfg_cond)

    # Parameters (match C++ defaults)
    temperature = mx.array([1.3])
    top_k = mx.array([40], dtype=mx.int32)
    cfg_musiccoca = mx.array([3.0])
    cfg_notes = mx.array([0.1])

    # forced_tokens sits at index 7 (before state), matching export.py and the
    # C++ engine. An empty time dim (shape [1, 0, rvq_depth]) selects the normal
    # streaming path (the C++ generate_frame uses the same empty tensor).
    forced_tokens = mx.zeros((1, 0, rvq_depth), dtype=mx.int32)

    # ── Run one step (no mx.compile, matching raw import_function) ──
    args = [cond_array, temperature, top_k, cfg_musiccoca, cfg_notes,
            neg_musiccoca_array, neg_notes_array, forced_tokens] + state
    outputs = fn(args)
    mx.eval(outputs)

    audio = outputs[0]
    new_state = outputs[1:]

    print(f"\n{'='*60}")
    print(f"  Python .mlxfn output (step 1)")
    print(f"{'='*60}")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio dtype: {audio.dtype}")
    print(f"  Audio min:   {audio.min().item():.8f}")
    print(f"  Audio max:   {audio.max().item():.8f}")
    print(f"  Audio mean:  {mx.mean(audio).item():.8f}")

    # Print first 20 values of L channel
    audio_np = np.array(audio)
    L = audio_np[0, 0, :]  # L channel
    R = audio_np[0, 1, :]  # R channel
    print(f"\n  L channel first 20 samples:")
    for i in range(20):
        print(f"    L[{i:4d}] = {L[i]:.8f}")
    print(f"\n  R channel first 20 samples:")
    for i in range(20):
        print(f"    R[{i:4d}] = {R[i]:.8f}")

    # Save raw output for binary comparison
    np.save(str(RESOURCE_DIR / "python_step1_audio.npy"), audio_np)
    print(f"\n  Saved raw output to {RESOURCE_DIR / 'python_step1_audio.npy'}")

    # ── Run step 2 to check state propagation ──
    args2 = [cond_array, temperature, top_k, cfg_musiccoca, cfg_notes,
             neg_musiccoca_array, neg_notes_array, forced_tokens] + list(new_state)
    outputs2 = fn(args2)
    mx.eval(outputs2)
    audio2 = outputs2[0]

    print(f"\n{'='*60}")
    print(f"  Python .mlxfn output (step 2)")
    print(f"{'='*60}")
    print(f"  Audio shape: {audio2.shape}")
    print(f"  Audio min:   {audio2.min().item():.8f}")
    print(f"  Audio max:   {audio2.max().item():.8f}")
    print(f"  Audio mean:  {mx.mean(audio2).item():.8f}")

    L2 = np.array(audio2)[0, 0, :]
    print(f"\n  L channel first 20 samples:")
    for i in range(20):
        print(f"    L[{i:4d}] = {L2[i]:.8f}")

if __name__ == "__main__":
    main()
