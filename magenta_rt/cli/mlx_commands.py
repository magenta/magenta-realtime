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

"""CLI commands for the MLX backend: mrt mlx {generate,export,benchmark}."""

import click

from magenta_rt.cli import main
from magenta_rt import paths


@main.group()
def mlx():
    """MLX backend commands."""


@mlx.command()
@click.option("--prompt", default="disco funk", help="Text conditioning for MusicCoCa.")
@click.option("--model", default=paths.DEFAULT_MODEL_NAME, type=str, help="Model variant name (e.g. 'mrt2_base', 'mrt2_small').")
@click.option("--duration", default=4.0, type=float, help="Duration in seconds.")
@click.option("--bits", default=None, type=click.Choice(["2", "3", "4", "5", "6", "8"]), help="Bit quantization level.")
@click.option("--temperature", default=1.3, type=float)
@click.option("--top-k", default=40, type=int)
@click.option("--cfg-musiccoca", default=3.0, type=float)
@click.option("--cfg-notes", default=1.0, type=float)
@click.option("--checkpoint", default=None, type=str, help="Checkpoint filename in checkpoints/ directory.")
@click.option("--mlxfn/--no-mlxfn", default=True, help="Use exported .mlxfn model (default) or Python model.")
def generate(prompt, model, duration, bits, temperature, top_k,
             cfg_musiccoca, cfg_notes, checkpoint, mlxfn):
    """Generate audio with the MLX backend."""
    bits = int(bits) if bits else None

    from magenta_rt.mlx.generate import main as run
    kwargs = dict(
        prompt=prompt,
        model_name=model,
        bits=bits,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        duration=duration,
        checkpoint=checkpoint,
        use_mlxfn=mlxfn,
    )
    run(**kwargs)


@mlx.command("export")
@click.option("--output-name", required=True, help="Name for the exported model.")
@click.option("--model", default=None, type=str, help="Model variant name (e.g. 'mrt2_base', 'mrt2_small').")
@click.option("--bits", default=None, type=click.Choice(["2", "3", "4", "5", "6", "8"]), help="Bit quantization level.")
@click.option("--quantize-method", default="default", type=click.Choice(["default", "gptq"]),
              help="Quantization method: default (nn.quantize), gptq (GPTQ calibrated).")
@click.option("--gptq-cal-steps", default=128, type=int, help="Number of GPTQ calibration steps (default: 128).")
@click.option("--num-layers", default=None, type=int)
@click.option("--model-dims", default=None, type=int)
@click.option("--hidden-dims", default=None, type=int)
@click.option("--depth-num-layers", default=None, type=int)
@click.option("--depth-model-dims", default=None, type=int)
@click.option("--depth-hidden-dims", default=None, type=int)
@click.option("--num-codebooks", default=None, type=int)
@click.option("--num-cfgs", default=None, type=int, help="Number of CFGs: 0=disabled (1x batch), 1=musiccoca only (2x batch), 2=all (3x batch).")
@click.option("--output-dir", default=paths.models_dir(), help=f"Directory for exported models (default: {paths.models_dir()}).")
@click.option("--skip-restore", is_flag=True, default=False, help="Use random weights.")
@click.option("--checkpoint", default=None, type=str, help="Checkpoint filename in checkpoints/ directory.")
def export_cmd(output_name, model, bits, quantize_method, gptq_cal_steps,
               num_layers, model_dims, hidden_dims,
               depth_num_layers, depth_model_dims, depth_hidden_dims,
               num_codebooks, num_cfgs, output_dir, skip_restore, checkpoint):
    """Export .mlxfn model for the Audio Unit plugin."""
    bits = int(bits) if bits else None

    from magenta_rt.mlx.export import main as run
    kwargs = dict(
        restore=not skip_restore,
        bits=bits,
        quantize_method=quantize_method,
        gptq_cal_steps=gptq_cal_steps,
        output_name=output_name,
        output_dir=output_dir,
    )
    if model is not None:
        kwargs["model_name"] = model
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    if model_dims is not None:
        kwargs["model_dims"] = model_dims
    if hidden_dims is not None:
        kwargs["hidden_dims"] = hidden_dims
    if depth_num_layers is not None:
        kwargs["depth_num_layers"] = depth_num_layers
    if depth_model_dims is not None:
        kwargs["depth_model_dims"] = depth_model_dims
    if depth_hidden_dims is not None:
        kwargs["depth_hidden_dims"] = depth_hidden_dims
    if num_codebooks is not None:
        kwargs["num_codebooks"] = num_codebooks
    if num_cfgs is not None:
        kwargs["num_cfgs"] = num_cfgs
    if checkpoint is not None:
        kwargs["checkpoint"] = checkpoint
    run(**kwargs)


SPECTROSTREAM_OUTPUT_PATH = str(paths.spectrostream_dir() / "spectrostream_encoder.mlxfn")


@mlx.command("export-spectrostream")
@click.option("--output", default=SPECTROSTREAM_OUTPUT_PATH,
    help=f"Output path for the .mlxfn file (default: {SPECTROSTREAM_OUTPUT_PATH}).")
@click.option("--checkpoint", default=paths.DEFAULT_CHECKPOINT, type=str,
    help=f"Checkpoint filename in checkpoints directory. (default: {paths.DEFAULT_CHECKPOINT})")
def export_spectrostream_cmd(output, checkpoint):
    """Export SpectroStream encoder to .mlxfn file."""
    import os
    import mlx.core as mx
    import sequence_layers.mlx as sl
    from magenta_rt.mlx import spectrostream, model
    from magenta_rt.mlx.spectrostream.load_weights import load_spectrostream_weights
    from magenta_rt import paths

    # Only the SpectroStream sub-model is needed to export its encoder, so we
    # build it directly rather than constructing a full MagentaRT sampler.
    exp = model.get_model_class("mrt2_small")()
    rvq_truncation = exp.spectrostream.rvq_truncation_level
    spectrostream_config = (
        spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=rvq_truncation,
            use_unique_codes=False,
        )
    )
    ss = spectrostream.SpectroStream(spectrostream_config)

    checkpoint_path = paths.resolve_checkpoint(checkpoint)
    load_spectrostream_weights(ss, checkpoint_path)

    # Export the encoder (waveform -> codes) as a traced .mlxfn function.
    print('Exporting SpectroStream encoder...')
    def spectrostream_encode(waveform_values):
        x = sl.Sequence(waveform_values, mx.ones((1, waveform_values.shape[1]), dtype=mx.bool_))
        codes = ss.waveform_to_codes_layer.layer(x)
        return codes.values

    # Dummy waveform for tracing: 1 batch, 28 seconds at 48 kHz, stereo.
    #
    # We've observed that when performing continuation with low
    # CFG-MusicCoCa, low temperature, and small top-k (intending to strictly
    # imitate the prefill context), a longer prefill (up to 28 seconds) tends to
    # yield better continuation quality, even beyond the model's strict
    # receptive field (~19.7 s). To test this more concretely, try
    # `examples/hello_mrt2`.
    #
    # If you ever change this, also re-validate the trim defaults
    # (`MLXEngine::Impl::prefill_state` uses 25/25 for the mrt2_base STFT
    # config; different SpectroStream hyperparameters could shift the
    # head/tail transient zones).
    dummy_waveform = mx.zeros((1, 48_000 * 28, 2), dtype=mx.float32)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    mx.export_function(output, spectrostream_encode, dummy_waveform)
    print(f'Exported SpectroStream encoder to {output}')
