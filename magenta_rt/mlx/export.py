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

import os
import os.path
from pathlib import Path

import mlx.core as mx
import numpy as np

import sequence_layers.mlx as sl
from sequence_layers.mlx import export

from . import model, system
from . import spectrostream
from .load_weights import load_weights, convert_to_bf16
from magenta_rt import paths


def _flatten_state(state):
  """Flatten a nested pytree state into a list of mx.array.

  Handles tuples, lists, and mx.array leaves. Empty tuples contribute
  zero arrays.

  Args:
    state: Nested tuple/list of mx.array.

  Returns:
    (flat_arrays, structure) where structure encodes the nesting.
  """
  flat = []

  def _record(node):
    if isinstance(node, mx.array):
      flat.append(node)
      return 'array'
    elif isinstance(node, tuple):
      children = [_record(child) for child in node]
      return ('tuple', children)
    elif isinstance(node, list):
      children = [_record(child) for child in node]
      return ('list', children)
    elif isinstance(node, dict):
      # We shouldn't hit dicts, if we do we need to handle keys properly.
      raise TypeError(f'Unsupported state node type: {type(node)}')
    elif isinstance(node, sl.MaskedSequence):
      children = [_record(child) for child in [node.values, node.mask]]
      return ('MaskedSequence', children)
    elif isinstance(node, sl.Sequence):
      children = [_record(child) for child in [node.values, node.mask]]
      return ('Sequence', children)
    else:
      raise TypeError(f'Unsupported state node type: {type(node)}')

  structure = _record(state)
  return flat, structure


def _unflatten_state(flat, structure):
  """Reconstruct a nested state from a flat array list and structure.

  Args:
    flat: List of mx.array.
    structure: Structure descriptor from _flatten_state.

  Returns:
    Nested tuple/list matching the original structure.
  """
  idx = [0]

  def _rebuild(struct):
    if struct == 'array':
      result = flat[idx[0]]
      idx[0] += 1
      return result
    elif isinstance(struct, tuple) and struct[0] == 'tuple':
      return tuple(_rebuild(s) for s in struct[1])
    elif isinstance(struct, tuple) and struct[0] == 'list':
      return [_rebuild(s) for s in struct[1]]
    elif struct[0] == 'MaskedSequence':
       return sl.MaskedSequence(_rebuild(struct[1][0]), _rebuild(struct[1][1]))
    elif struct[0] == 'Sequence':
       return sl.Sequence(_rebuild(struct[1][0]), _rebuild(struct[1][1]))
    else:
      raise ValueError(f'Unknown structure node: {struct}')

  result = _rebuild(structure)
  if idx[0] != len(flat):
    raise ValueError(f'Not all arrays consumed: used {idx[0]} of {len(flat)}')
  return result


def _discretize_cfg_token(value, step, max_bin, offset):
  """MLX-op version of ``system.discretize_cfg`` plus the reserved-token offset.

  Bins a float CFG scale in [-1.0, 7.0] to a conditioning token index, then
  shifts it by ``offset`` (the conditioning vocab's reserved-token count). This
  is the MLX-op equivalent of ``magenta_rt.mlx.system.discretize_cfg`` (and of
  the binning that used to live in ``core/src/mlx_engine.cpp``, now removed so
  the C++ runtime passes raw float scales instead). The two agree except at
  exact bin boundaries, where this function's float32 math can round one bin
  differently than the float64 Python version. Operates on a traced
  ``mx.array`` scalar of shape ``[1]`` and returns an int32 token of shape
  ``[1]``.

  Args:
    value: CFG scale; clamped to [-1.0, 7.0].
    step: Quantization step (0.2 for musiccoca/notes, 1.0 for drums).
    max_bin: Largest valid token index (40 for musiccoca/notes, 8 for drums).
    offset: Reserved-token offset added to the binned index.

  Returns:
    int32 ``mx.array`` of shape ``[1]`` holding ``bin + offset``.
  """
  clamped = mx.clip(value, -1.0, 7.0)
  bin_index = mx.round((clamped - (-1.0)) / step)
  bin_index = mx.clip(bin_index, 0.0, float(max_bin))
  return bin_index.astype(mx.int32) + offset


def main(
    restore: bool = True,
    model_name: str = paths.DEFAULT_MODEL_NAME,
    # Wide depthformer (temporal decoder):
    num_layers: int | None = None,
    model_dims: int | None = None,
    hidden_dims: int | None = None,
    # L_SHALLOW_TPU_OPTIMIZED (depth decoder):
    depth_num_layers: int | None = None,
    depth_model_dims: int | None = None,
    depth_hidden_dims: int | None = None,
    # speed/performance:
    bits: int | None = None,  # 0/None/32 means no bit quantization
    quantize_group_size: int | None = None,
    quantize_method: str = 'default',  # 'default' or 'gptq'
    gptq_cal_steps: int = 128,  # number of calibration steps for GPTQ
    gptq_debug_identity: bool = False,  # force H=I for sanity check
    num_codebooks: int | None = None,  # 0/None means no reduction in codebooks
    # control
    temperature: float = 1.3,
    top_k: int = 40,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 0.1,
    cfg_drums: float = 1.0,
    num_cfgs: int = 2,  # 0 = disabled, 1 = musiccoca only, 2 = all (musiccoca+notes)
    # utils
    output_name: str | None = None,
    output_dir: str = paths.models_dir(),
    checkpoint: str | None = None,
):
    """Export the MLX model to a format suitable for the MLX runtime."""
    if not output_name:
        raise ValueError("output_name must be specified")
    exp_cls = model.get_model_class(model_name)
    exp = exp_cls()
    musiccoca_tokens_cfg = exp.input_configs[0]
    pianoroll_tokens_cfg = exp.input_configs[1]
    num_musiccoca_tokens = musiccoca_tokens_cfg.rvq_truncation_level
    num_pitches = pianoroll_tokens_cfg.rvq_truncation_level
    print(f"Using model: {model_name} ({exp_cls.__name__})")

    # Build depthformer_config kwargs — only override what was explicitly set,
    # so the model class's own architecture spec (encoder_size, etc.) is used
    # by default.
    df_kwargs = dict(num_active_codebooks=num_codebooks)
    if num_layers is not None:
        df_kwargs["num_layers"] = num_layers
    if model_dims is not None:
        df_kwargs["model_dims"] = model_dims
    if hidden_dims is not None:
        df_kwargs["hidden_dims"] = hidden_dims
    if depth_num_layers is not None:
        df_kwargs["depth_num_layers"] = depth_num_layers
    if depth_model_dims is not None:
        df_kwargs["depth_model_dims"] = depth_model_dims
    if depth_hidden_dims is not None:
        df_kwargs["depth_hidden_dims"] = depth_hidden_dims

    # Resolve actual num_layers for context size computation.
    # Must set max_past_horizon before depthformer_config() since it reads
    # these values from the exp instance.
    #
    # The `20*25` literal is the design target for the model's *effective*
    # receptive field, in frames (frame rate is 25 Hz, so this is 20 s).
    # Local-attention windows of size W per layer compound across N layers
    # to give an effective receptive field of roughly N·W frames, so we
    # divide the budget evenly: W = 20*25 // N. For divisible N (4, 5, 10,
    # 20, 25) this is exactly 500 frames; for N=12 (XXL_SHALLOWEST_WIDE)
    # it rounds down to W=41 → N·W=492 ≈ 19.7 s.
    #
    # This is a model design choice baked into the exported .mlxfn (the
    # attention-mask shapes use W). Inference *cannot* see context older
    # than ~20 s, and prefill should target ~N·W frames of clean tokens
    # to fully saturate every layer's KV cache. Anything more is wasted;
    # the receptive field is a hard ceiling.
    actual_num_layers = num_layers if num_layers is not None else exp.decoder_temporal_size.num_layers
    max_past_horizon = 20*25 // actual_num_layers
    exp.encoder_max_past_horizon = max_past_horizon
    exp.decoder_temporal_self_attention_max_past_horizon = max_past_horizon
    exp.decoder_temporal_cross_attention_max_past_horizon = max_past_horizon

    depthformer_config = exp.depthformer_config(**df_kwargs)

    rvq_truncation = exp.spectrostream.rvq_truncation_level
    spectrostream_config = (
        spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq_truncation,
            use_unique_codes=False,
        )
    )

    mrt_config = system.MagentaRT2Sampler.Config(
        depthformer=depthformer_config,
        spectrostream=spectrostream_config,
        int16_outputs=False,
    )
    mrt_sampler = mrt_config.make()

    NUM_RESERVED_TOKENS = system.NUM_RESERVED_TOKENS + 1  # + 1 for dropout token

    # Load weights from checkpoint
    if restore:
        print("Restoring weights.")
        if checkpoint is None:
            if model_name not in system._CHECKPOINT_REGISTRY:
                raise ValueError(
                    f"No default checkpoint for model '{model_name}'. "
                    f"Available: {list(system._CHECKPOINT_REGISTRY.keys())}. "
                    f"Pass --checkpoint explicitly."
                )
            checkpoint = system._CHECKPOINT_REGISTRY[model_name]
        checkpoint_path = paths.resolve_checkpoint(checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        load_weights(mrt_sampler, checkpoint_path, num_input_channels=exp.input_num_channels)
    else:
        print("Materializing deferred layers (random initialization)...")
        input_spec = sl.ChannelSpec(shape=(exp.input_num_channels,), dtype=mx.int32)
        dummy_constants = {
            'classifier_free_guidance_scale_musiccoca': mx.array([1.0]),
            'classifier_free_guidance_scale_notes': mx.array([1.0]),
            'temperature': mx.array([1.0]),
            'top_k': mx.array([40]),
        }
        export._materialize_deferred(mrt_sampler, batch_size=1, input_spec=input_spec, constants=dummy_constants)
        convert_to_bf16(mrt_sampler.depthformer)

    if bits and bits < 32:
        # Quantize weights to int4/int8 for reduced memory bandwidth
        if quantize_group_size is None:
           quantize_group_size = 32 if bits == 4 else 64

        if quantize_method == 'gptq':
            print(f"GPTQ quantizing model weights to {bits}-bit "
                  f"(group_size={quantize_group_size}, cal_steps={gptq_cal_steps})...")
            from .gptq import gptq_calibrate_and_quantize
            import csv

            # Load diverse MusicCoCa prompts from eval prompt set.
            # Using real prompts (not random) ensures the Hessian reflects
            # actual activation distributions across diverse musical styles.
            prompt_csv = Path(__file__).resolve().parent.parent / 'data' / 'example_prompt_set.csv'
            from magenta_rt.musiccoca import MusicCoCa
            coca = MusicCoCa()
            musiccoca_prompts = []
            with open(prompt_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tokens = coca.tokenize(coca.embed(row['prompt'])).tolist()
                    musiccoca_prompts.append(tokens[:num_musiccoca_tokens])
            print(f"  Loaded {len(musiccoca_prompts)} calibration prompts from {prompt_csv.name}")

            # Default gptq_cal_steps=128 runs ~5s of streaming generation
            # (at 25Hz), cycling through all 128 prompts once. This gives
            # ~384 activation samples per layer (128 steps × 3 CFG batch).

            _notes = [-1] * num_pitches
            _masked_musiccoca = [-1] * num_musiccoca_tokens
            _masked_notes = [-1] * num_pitches
            _t_val = mx.array([temperature])
            _k_val = mx.array([top_k], dtype=mx.int32)
            _cfg_musiccoca_val = mx.array([cfg_musiccoca])
            _cfg_notes_val = mx.array([cfg_notes])

            def _make_cal_inputs(musiccoca_tokens):
                """Build conditioning inputs for a single MusicCoCa prompt."""
                cond = np.concatenate([musiccoca_tokens, _notes], axis=0) + NUM_RESERVED_TOKENS
                block = sl.Sequence(
                    mx.array(cond.reshape(1, 1, -1), dtype=mx.int32),
                    mx.array([[True]], dtype=mx.bool_),
                )
                neg_musiccoca = sl.Sequence(
                    mx.array([[_masked_musiccoca + _notes]], dtype=mx.int32) + NUM_RESERVED_TOKENS,
                    mx.array([[True]], dtype=mx.bool_),
                )
                neg_notes = sl.Sequence(
                    mx.array([[musiccoca_tokens + _masked_notes]], dtype=mx.int32) + NUM_RESERVED_TOKENS,
                    mx.array([[True]], dtype=mx.bool_),
                )
                constants = {
                    "temperature": _t_val,
                    "top_k": _k_val,
                    "classifier_free_guidance_scale_musiccoca": _cfg_musiccoca_val,
                    "classifier_free_guidance_scale_notes": _cfg_notes_val,
                    "classifier_free_guidance_negative_musiccoca": neg_musiccoca,
                    "classifier_free_guidance_negative_notes": neg_notes,
                }
                return block, constants

            def _calibrate_fn(model_unused):
                _input_spec = sl.ChannelSpec(shape=(num_musiccoca_tokens + num_pitches,), dtype=mx.int32)
                # Run multiple streaming steps per prompt before switching.
                # State is continuous (no reset) — captures both within-prompt
                # temporal evolution and prompt transition dynamics.
                steps_per_prompt = max(1, gptq_cal_steps // len(musiccoca_prompts))
                first_block, first_constants = _make_cal_inputs(musiccoca_prompts[0])
                _state = mrt_sampler.get_initial_state(
                    1, _input_spec, constants=first_constants, training=False)
                for step_i in range(gptq_cal_steps):
                    prompt_idx = (step_i // steps_per_prompt) % len(musiccoca_prompts)
                    cal_block, cal_constants = _make_cal_inputs(musiccoca_prompts[prompt_idx])
                    y, _state, _ = mrt_sampler.step_with_emits(
                        x=cal_block, state=_state,
                        constants=cal_constants, training=False)
                    mx.eval(y.values)
                    if (step_i + 1) % 32 == 0:
                        print(f"    Calibration step {step_i + 1}/{gptq_cal_steps}")

            hessian_save_path = Path(output_dir) / f"{output_name}_hessians.safetensors"
            gptq_calibrate_and_quantize(
                mrt_sampler, _calibrate_fn,
                bits=bits, group_size=quantize_group_size,
                debug_identity_hessian=gptq_debug_identity,
                hessian_save_path=hessian_save_path)
        else:
            print(f"Quantizing model weights to {bits}-bit (default nn.quantize).")
            import mlx.nn as nn
            nn.quantize(mrt_sampler, group_size=quantize_group_size, bits=bits)

    if num_codebooks:
        print(f"Active codebooks: {num_codebooks} / {mrt_sampler.depthformer.decoder.config.num_codebooks}")

    input_spec = sl.ChannelSpec(shape=(exp.input_num_channels,), dtype=mx.int32)

    def init_state(constants):
        return mrt_sampler.get_initial_state(1, input_spec, constants=constants, training=False)

    # --- Set up conditioning inputs ---
    musiccoca = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
    musiccoca = musiccoca[:num_musiccoca_tokens]
    notes = [-1] * num_pitches
    drums = [-1]  # -1->"let model decide"; 0->"don't play drums"; 1->"please play drums"
    masked_musiccoca = [-1] * num_musiccoca_tokens
    masked_notes = [-1] * num_pitches
    cond_tokens = np.concatenate([musiccoca, notes, drums], axis=0) + NUM_RESERVED_TOKENS
    block = sl.Sequence(
        mx.array(cond_tokens.reshape(1, 1, -1), dtype=mx.int32),
        mx.array([[True]], dtype=mx.bool_),
    )
    neg_musiccoca = sl.Sequence(
        mx.array([[masked_musiccoca + notes + drums]], dtype=mx.int32) + NUM_RESERVED_TOKENS,
        mx.array([[True]], dtype=mx.bool_),
    )
    neg_notes = sl.Sequence(
        mx.array([[musiccoca + masked_notes + drums]], dtype=mx.int32) + NUM_RESERVED_TOKENS,
        mx.array([[True]], dtype=mx.bool_),
    )

    t_val = mx.array([temperature])
    k_val = mx.array([top_k], dtype=mx.int32)
    cfg_musiccoca_val = mx.array([cfg_musiccoca])
    cfg_notes_val = mx.array([cfg_notes])
    cfg_drums_val = mx.array([cfg_drums])

    if num_cfgs == 0:
        # No CFG: batch = 1x (positive only, no guidance)
        constants = {
            "temperature": t_val,
            "top_k": k_val,
        }
        print(f"Using 0 CFGs (disabled) → batch multiplier = 1x")
    elif num_cfgs == 1:
        # Only musiccoca CFG: batch = 2x (1 positive + 1 negative)
        constants = {
            "temperature": t_val,
            "top_k": k_val,
            "classifier_free_guidance_scale_musiccoca": cfg_musiccoca_val,
            "classifier_free_guidance_negative_musiccoca": neg_musiccoca,
        }
        print(f"Using 1 CFG (musiccoca only) → batch multiplier = 2x")
    else:
        # All 2 CFGs: batch = 3x (1 positive + 2 negatives)
        constants = {
            "temperature": t_val,
            "top_k": k_val,
            "classifier_free_guidance_scale_musiccoca": cfg_musiccoca_val,
            "classifier_free_guidance_scale_notes": cfg_notes_val,
            "classifier_free_guidance_negative_musiccoca": neg_musiccoca,
            "classifier_free_guidance_negative_notes": neg_notes,
        }
        print(f"Using 2 CFGs (musiccoca+notes) → batch multiplier = 3x")

    state = init_state(constants)
    flat_state, structure = _flatten_state(state)

    # todo: this mx.compile decorator isn't helping performance for some reason.
    # @partial(mx.compile, inputs=(mx.random.state, structure), outputs=mx.random.state)
    def streaming_step(x_values, temperature_arg, top_k_arg, cfg_musiccoca_arg, cfg_notes_arg, cfg_drums_arg, neg_musiccoca_values, neg_notes_values, forced_tokens, *state_flat):
        # Discretize the float CFG scales and append them to the conditioning
        # token slots (musiccoca, notes, drums). This replaces the
        # discretize_cfg() logic that previously lived in
        # core/src/mlx_engine.cpp: the C++ runtime now passes raw float scales
        # and the exported function bins them. The same tokens are written into
        # the positive block and both CFG negative blocks, matching the old C++
        # behavior where each negative was a copy of the positive conditioning
        # with one modality masked.
        cfg_tokens = mx.concatenate([
            _discretize_cfg_token(cfg_musiccoca_arg, 0.2, 40, NUM_RESERVED_TOKENS),
            _discretize_cfg_token(cfg_notes_arg, 0.2, 40, NUM_RESERVED_TOKENS),
            _discretize_cfg_token(cfg_drums_arg, 1.0, 8, NUM_RESERVED_TOKENS),
        ], axis=-1).reshape(1, 1, 3)
        x_values = mx.concatenate([x_values, cfg_tokens], axis=-1)
        neg_musiccoca_values = mx.concatenate([neg_musiccoca_values, cfg_tokens], axis=-1)
        neg_notes_values = mx.concatenate([neg_notes_values, cfg_tokens], axis=-1)

        state = _unflatten_state(list(state_flat), structure)
        x = sl.Sequence(x_values, mx.array([[True]], dtype=mx.bool_))

        if num_cfgs == 0:
            dynamic_constants = {
                "temperature": temperature_arg,
                "top_k": top_k_arg,
            }
        elif num_cfgs == 1:
            dynamic_constants = {
                "temperature": temperature_arg,
                "top_k": top_k_arg,
                "classifier_free_guidance_scale_musiccoca": cfg_musiccoca_arg,
                "classifier_free_guidance_negative_musiccoca": sl.Sequence(neg_musiccoca_values, mx.array([[True]], dtype=mx.bool_)),
            }
        else:
            dynamic_constants = {
                "temperature": temperature_arg,
                "top_k": top_k_arg,
                "classifier_free_guidance_scale_musiccoca": cfg_musiccoca_arg,
                "classifier_free_guidance_scale_notes": cfg_notes_arg,
                "classifier_free_guidance_negative_musiccoca": sl.Sequence(neg_musiccoca_values, mx.array([[True]], dtype=mx.bool_)),
                "classifier_free_guidance_negative_notes": sl.Sequence(neg_notes_values, mx.array([[True]], dtype=mx.bool_)),
            }

        y, new_state, _ = mrt_sampler.step_with_emits(x=x, state=state, constants=dynamic_constants, forced_tokens=forced_tokens, training=False)
        new_flat, _ = _flatten_state(new_state)
        y = mx.swapaxes(y.values, -2, -1)  # [B, T, C] -> [B, C, T]
        y = mx.reshape(mx.flatten(y), (1, 2, -1)) # Force physically contiguous memory allocation
        return (y, *new_flat)

    empty_forced_tokens = mx.zeros((1, 0, rvq_truncation), dtype=mx.int32)
    print("warming up for export...")
    for _ in range(5):
        y_values, *flat_state  = streaming_step(
        block.values, t_val, k_val, cfg_musiccoca_val, cfg_notes_val, cfg_drums_val,
        neg_musiccoca.values, neg_notes.values, empty_forced_tokens, *flat_state
    )
        mx.eval(y_values, *flat_state)  # Force evaluation
    print("warmed up.")

    export_dir = Path(output_dir).resolve()
    export_dir = export_dir / output_name
    os.makedirs(export_dir, exist_ok=True)
    mlxfn_path = str(export_dir / f"{output_name}.mlxfn")
    state_path = str(export_dir / f"{output_name}_state.safetensors")

    print('Exporting transformer...')

    # todo: using shapeless=True could have performance cost, but
    # it would allow us to do parallel prefill over inputs with a dynamic time axis.
    with mx.exporter(mlxfn_path, streaming_step, shapeless=False) as exporter:

        exporter(
            block.values,
            t_val,
            k_val,
            cfg_musiccoca_val,
            cfg_notes_val,
            cfg_drums_val,
            neg_musiccoca.values,
            neg_notes.values,
            empty_forced_tokens,
            *flat_state,
        )
        forced_tokens = mx.zeros((1, 1, rvq_truncation), dtype=mx.int32)
        exporter(
            block.values,
            t_val,
            k_val,
            cfg_musiccoca_val,
            cfg_notes_val,
            cfg_drums_val,
            neg_musiccoca.values,
            neg_notes.values,
            forced_tokens,
            *flat_state,
        )

    print(f'Exported transformer to {mlxfn_path}')

    # Re-initialize to a clean initial state before saving.
    # We must cast each array to match the dtype the mlxfn was traced with
    # (from flat_state after warm-up), since the warm-up loop can promote
    # dtypes (e.g. int32 -> int64).
    # TODO: need to find out why stream_step promotes dtypes
    traced_dtypes = [arr.dtype for arr in flat_state]
    state = init_state(constants)
    fresh_flat, _ = _flatten_state(state)
    flat_state = [arr.astype(dt) for arr, dt in zip(fresh_flat, traced_dtypes)]
    mx.eval(*flat_state) if flat_state else None

    state_dict = {f'state_{i}': arr for i, arr in enumerate(flat_state)}
    mx.save_safetensors(state_path, state_dict)
    print(f'Exported {len(flat_state)} state arrays to {state_path}')




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("export_model")

    parser.add_argument("--model", default=paths.DEFAULT_MODEL_NAME, type=str,
                        help=f"Model variant name (default: {paths.DEFAULT_MODEL_NAME}).")

    parser.add_argument("--num-layers", default=None, type=int, help="Number of layers (default: from model spec)")
    parser.add_argument("--model-dims", default=None, type=int, help="Model dimension (default: from model spec)")
    parser.add_argument("--hidden-dims", default=None, type=int, help="Hidden dimension (default: from model spec)")

    parser.add_argument("--bits", default=None, type=int, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--quantize-group-size", default=None, type=int,
        help="Only matters if `--bits` is set.")
    parser.add_argument("--quantize-method", default='default',
        choices=['default', 'gptq'],
        help="Quantization method: 'default' (nn.quantize), 'gptq' (GPTQ calibrated).")
    parser.add_argument("--gptq-cal-steps", default=128, type=int,
        help="Number of calibration steps for GPTQ (default: 128).")
    parser.add_argument("--gptq-debug-identity", action="store_true", default=False,
        help="Force H=I for GPTQ (sanity check: should match nn.quantize).")

    parser.add_argument("--num-codebooks", default=None, type=int)

    parser.add_argument("--temperature", default=1.3, type=float)
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    parser.add_argument("--cfg-drums", default=1.0, type=float)
    parser.add_argument("--num-cfgs", default=2, type=int, choices=[0, 1, 2],
                        help="Number of CFGs to use: 0=disabled (1x batch), 1=musiccoca only (2x batch), 2=all (3x batch)")

    parser.add_argument(
        '--skip-restore',
        action='store_false',
        dest='restore',
        default=True,
        help='Disable restoring (default: True). Useful if benchmarking a model spec.'
    )
    parser.add_argument(
        '--output-name',
        default=None,
        type=str,
        help='Name of the exported model (default: None)'
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='Checkpoint filename (default: derived from --model)'
    )
    args = parser.parse_args()

    main(
        restore=args.restore,
        model_name=args.model,
        num_layers=args.num_layers,
        model_dims=args.model_dims,
        hidden_dims=args.hidden_dims,
        bits=args.bits,
        quantize_group_size=args.quantize_group_size,
        quantize_method=args.quantize_method,
        gptq_cal_steps=args.gptq_cal_steps,
        gptq_debug_identity=args.gptq_debug_identity,
        num_codebooks=args.num_codebooks,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        num_cfgs=args.num_cfgs,
        output_name=args.output_name,
        checkpoint=args.checkpoint,
    )
