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

"""Export a lightweight sinusoid "transformer" `.mlxfn`.

This is a synthetic stand-in for the depthformer transformer exported by
`export.py`: instead of sampling SpectroStream tokens it just renders simple
per-MIDI-note sinusoids. It exists so the C++ MLX runtime / AUv3 plugin can be
exercised without the heavy real model.

The C++ engine calls both the real model and this stand-in through the *same*
imported function, so the exported `.mlxfn` here MUST accept exactly the same
positional inputs (and return audio + state in the same layout) as `export.py`:

    streaming_step(cond, temperature, top_k, cfg_musiccoca, cfg_notes, cfg_drums,
                   neg_musiccoca, neg_notes, forced_tokens, *state)
      -> (audio[1, 2, 1920], *new_state)

Only the model body differs. To keep that contract in sync automatically, the
conditioning layout (channel count, note offset, reserved tokens) is derived
from the same model spec `export.py` uses rather than hardcoded.
"""

import os
from pathlib import Path

import mlx.core as mx

from . import model
from magenta_rt import paths

# When no --output-name is given, the sinusoid model is written into the AUv3
# plugin's bundled resources under these canonical names. (The C++ engine
# derives the state path from the `.mlxfn` path.)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCE_DIR = REPO_ROOT / 'examples' / 'auv3' / 'resources'
TRANSFORMER_MLXFN = 'MagentaRT_transformer.mlxfn'
TRANSFORMER_STATE = 'MagentaRT_transformer_state.safetensors'

FRAME_SAMPLES = 1920  # 48 kHz / 25 Hz
SAMPLE_RATE = 48000


def get_initial_state():
    """Initial streaming state: [sample_index, per-pitch envelope]."""
    sample_index = mx.array([0], dtype=mx.int32)
    envelopes = mx.zeros((128,), dtype=mx.float32)
    return [sample_index, envelopes]


def main(
    model_name: str = paths.DEFAULT_MODEL_NAME,
    # control (ignored by the sinusoid body, kept for API parity):
    temperature: float = 1.3,
    top_k: int = 40,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 0.1,
    cfg_drums: float = 1.0,
    # utils:
    output_name: str | None = None,
    output_dir: str = paths.models_dir(),
):
    """Export the sinusoid stand-in transformer (.mlxfn + state)."""
    # Derive the conditioning layout from the same model spec export.py uses, so
    # the exported function's input shapes track the real model automatically.
    exp = model.get_model_class(model_name)()
    num_musiccoca_tokens = exp.input_configs[0].rvq_truncation_level
    num_pitches = exp.input_configs[1].rvq_truncation_level
    cond_len = sum(
        cfg.rvq_truncation_level
        for cfg in exp.input_configs
        if not cfg.key.startswith('cfg_conditioning')
    )
    rvq_truncation = exp.spectrostream.rvq_truncation_level
    # +1 for the dropout token (matches export.py and the C++ kNumReservedTokens).
    num_reserved_tokens = model.NUM_RESERVED_TOKENS + 1
    note_offset = num_musiccoca_tokens          # 128 pitch slots follow MusicCoCa
    note_on_min = num_reserved_tokens + 1        # token value of an active note
    print(f"Using model: {model_name} ({type(exp).__name__})")
    print(f"  cond_len={cond_len} (musiccoca={num_musiccoca_tokens}, "
          f"pitches={num_pitches}), rvq_depth={rvq_truncation}")

    # --- Dummy inputs for tracing (mirrors export.py's positional layout) ---
    cond_values = mx.zeros((1, 1, cond_len), dtype=mx.int32)
    t_val = mx.array([temperature], dtype=mx.float32)
    k_val = mx.array([top_k], dtype=mx.int32)
    cfg_musiccoca_val = mx.array([cfg_musiccoca], dtype=mx.float32)
    cfg_notes_val = mx.array([cfg_notes], dtype=mx.float32)
    cfg_drums_val = mx.array([cfg_drums], dtype=mx.float32)
    neg_musiccoca_values = mx.zeros((1, 1, cond_len), dtype=mx.int32)
    neg_notes_values = mx.zeros((1, 1, cond_len), dtype=mx.int32)
    empty_forced_tokens = mx.zeros((1, 0, rvq_truncation), dtype=mx.int32)

    state = get_initial_state()

    def streaming_step(x_values, temperature_arg, top_k_arg, cfg_musiccoca_arg,
                       cfg_notes_arg, cfg_drums_arg, neg_musiccoca_values,
                       neg_notes_values, forced_tokens, *state_flat):
        # The sinusoid stand-in ignores the sampling / CFG / forced-token inputs;
        # they exist only so the exported API matches the real model (export.py).
        del (temperature_arg, top_k_arg, cfg_musiccoca_arg, cfg_notes_arg,
             cfg_drums_arg, neg_musiccoca_values, neg_notes_values, forced_tokens)
        sample_index, env = state_flat

        # Active MIDI notes: the 128 pitch slots sit right after the MusicCoCa
        # tokens; an active note has a token value >= note_on_min.
        notes = (
            x_values[0, 0, note_offset:note_offset + num_pitches] >= note_on_min
        ).astype(mx.float32)  # (128,)

        # Frequencies for MIDI 0..127
        pitches = mx.arange(num_pitches, dtype=mx.float32)
        freqs = 440.0 * (2.0 ** ((pitches - 69.0) / 12.0))

        N = FRAME_SAMPLES
        samples = sample_index[0] + mx.arange(N, dtype=mx.float32)  # (N,)

        # phase [128, N]
        phase = 2.0 * mx.pi * freqs[:, None] * samples[None, :] / SAMPLE_RATE
        sinusoids = mx.sin(phase)

        # Envelopes (10-sample linear attack/release => step = 0.1)
        step = 0.1
        direction = mx.sign(notes - env)  # [128]
        t_steps = mx.arange(1, N + 1, dtype=mx.float32)  # [N]
        change = direction[:, None] * t_steps[None, :] * step  # [128, N]
        env_full = mx.clip(env[:, None] + change, 0.0, 1.0)  # [128, N]

        # Next state
        next_env = env_full[:, -1]
        next_sample_index = sample_index + mx.array([N], dtype=mx.int32)

        # Audio mix -> stereo [1, 2, N]
        audio = sinusoids * env_full
        mix = mx.sum(audio, axis=0) * 0.05
        stereo = mx.stack([mix, mix], axis=0)  # [2, N]
        stereo = mx.reshape(mx.flatten(stereo), (1, 2, N))

        return (stereo, next_sample_index, next_env)

    # --- Warm up (mirrors export.py) ---
    print("warming up for export...")
    for _ in range(5):
        y, *state = streaming_step(
            cond_values, t_val, k_val, cfg_musiccoca_val, cfg_notes_val,
            cfg_drums_val, neg_musiccoca_values, neg_notes_values,
            empty_forced_tokens, *state,
        )
        mx.eval(y, *state)
    print("warmed up.")

    # --- Resolve output paths ---
    if output_name:
        export_dir = Path(output_dir).resolve() / output_name
        os.makedirs(export_dir, exist_ok=True)
        mlxfn_path = str(export_dir / f"{output_name}.mlxfn")
        state_path = str(export_dir / f"{output_name}_state.safetensors")
    else:
        os.makedirs(RESOURCE_DIR, exist_ok=True)
        mlxfn_path = str(RESOURCE_DIR / TRANSFORMER_MLXFN)
        state_path = str(RESOURCE_DIR / TRANSFORMER_STATE)

    print('Exporting sinusoids transformer...')

    # Trace two forced_tokens shapes so the exported function handles both
    # streaming (time dim 0) and prefill (time dim >= 1), exactly like export.py.
    with mx.exporter(mlxfn_path, streaming_step, shapeless=False) as exporter:
        exporter(
            cond_values, t_val, k_val, cfg_musiccoca_val, cfg_notes_val,
            cfg_drums_val, neg_musiccoca_values, neg_notes_values,
            empty_forced_tokens, *state,
        )
        forced_tokens = mx.zeros((1, 1, rvq_truncation), dtype=mx.int32)
        exporter(
            cond_values, t_val, k_val, cfg_musiccoca_val, cfg_notes_val,
            cfg_drums_val, neg_musiccoca_values, neg_notes_values,
            forced_tokens, *state,
        )

    print(f'Exported transformer to {mlxfn_path}')

    # Re-initialize a clean initial state before saving. Cast each array to the
    # dtype the function was traced with (the warm-up loop can promote dtypes,
    # e.g. int32 -> int64).
    traced_dtypes = [arr.dtype for arr in state]
    clean_state = get_initial_state()
    clean_state = [arr.astype(dt) for arr, dt in zip(clean_state, traced_dtypes)]
    mx.eval(*clean_state) if clean_state else None

    state_dict = {f'state_{i}': arr for i, arr in enumerate(clean_state)}
    mx.save_safetensors(state_path, state_dict)
    print(f'Exported {len(clean_state)} state arrays to {state_path}')

    print("All done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("export_sinusoids")
    parser.add_argument(
        '--model', default=paths.DEFAULT_MODEL_NAME, type=str,
        help=f'Model variant whose conditioning layout to mirror (default: {paths.DEFAULT_MODEL_NAME}).')
    parser.add_argument('--temperature', default=1.3, type=float)
    parser.add_argument('--top-k', default=40, type=int)
    parser.add_argument('--cfg-musiccoca', default=3.0, type=float)
    parser.add_argument('--cfg-notes', default=0.1, type=float)
    parser.add_argument('--cfg-drums', default=1.0, type=float)
    parser.add_argument(
        '--output-name',
        default=None,
        type=str,
        help='Name of the exported model. If set, exports to <output-dir>/<name>/; '
             'otherwise writes the bundled AUv3 resource files.')
    parser.add_argument(
        '--output-dir',
        default=paths.models_dir(),
        help=f'Directory for exported models (default: {paths.models_dir()}).')
    args = parser.parse_args()
    main(
        model_name=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        output_name=args.output_name,
        output_dir=args.output_dir,
    )
