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

"""Eager vs ``.mlxfn`` streaming parity on the real mrt2_small
checkpoint.

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``. Builds
the mlx_pure system with real weights via the sl bridge (same path
as ``magenta_rt.mlx_pure.export.main``), warms it up, snapshots the
full flat state, exports a ``.mlxfn`` to ``tmp_path``, then runs the
same N=10 streaming steps both eagerly and through the imported
``.mlxfn`` from the shared snapshot. The two waveform sequences must
match bit-exactly.

This is the regression net for the codec-state-threading fix: if any
future change to ``LocalKVCache``, the spectrostream Conv2D caches, or
the OverlapAddCache breaks the eager↔traced symmetry, this test
catches it before the AUv3 plugin sees garbage audio.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_NAME = "mrt2_small.safetensors"


@pytest.fixture
def smallm4air_checkpoint():
    from magenta_rt import paths as _paths
    p = pathlib.Path(_paths.resolve_checkpoint(_CHECKPOINT_NAME))
    if not p.exists():
        pytest.skip(f"checkpoint not found: {p}")
    return p


def _build_mrt_with_weights(checkpoint_path: Path):
    """Mirror of ``magenta_rt.mlx_pure.export.main``'s build+restore path,
    minus quantization. Returns (mrt, target_cfg).
    """
    from magenta_rt.mlx_pure import configs as pure_configs
    from magenta_rt.mlx_pure import model as pure_model
    from magenta_rt.mlx_pure.load_weights import load_weights_from_combinator

    import sequence_layers.mlx as sl  # noqa: F401
    from magenta_rt.mlx import (
        model as combinator_model, spectrostream as combinator_ss, system as combinator_system,
    )
    from magenta_rt.mlx.load_weights import load_weights as combinator_load_weights

    mrt = pure_model.MagentaRT2Sampler.from_preset("mrt2_small", int16_outputs=False)
    spec = pure_configs.get_model_class("mrt2_small")()
    target_cfg = spec.target_tokens_config

    combinator_spec = combinator_model.get_model_class("mrt2_small")()
    df_config = combinator_spec.depthformer_config()
    ss_config = combinator_ss.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=target_cfg.rvq_truncation_level,
        use_unique_codes=False,
    )
    combinator_mrt = combinator_system.MagentaRT2Sampler.Config(
        depthformer=df_config, spectrostream=ss_config, int16_outputs=False,
    ).make()
    combinator_load_weights(combinator_mrt, checkpoint_path,
                            num_input_channels=combinator_spec.input_num_channels)
    load_weights_from_combinator(mrt, combinator_mrt)
    return mrt, target_cfg


@pytest.mark.checkpoint
@pytest.mark.slow
def test_export_matches_eager_across_streaming_steps(
    smallm4air_checkpoint, tmp_path,
):
    """Bit-exact: 10 streaming steps via .mlxfn must equal 10 steps
    via the eager Python path, given the same post-warmup snapshot.
    """
    from magenta_rt.mlx_pure.export import (
        _flatten_state, _flatten_codec, _install_state, _install_codec,
        _promote_cache_offsets,
    )
    from magenta_rt.mlx_pure.generate import _build_source_tokens

    mrt, target_cfg = _build_mrt_with_weights(smallm4air_checkpoint)

    batch_size = 3       # num_cfgs=2 → batch = 3
    cfg_scales = [3.0, 1.0]
    temperature = 1.3
    top_k = 40
    # 12-token "disco funk" MusicCoCa style; num_cfgs=2 makes _build_source_tokens
    # emit the batch=3 (pos / neg_musiccoca / neg_notes) source frame.
    style = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
    src = _build_source_tokens(
        style=style, num_cfgs=2,
        input_num_channels=mrt.depthformer.encoder.embedding.num_channels,
        num_reserved=target_cfg.num_extra_tokens,
    )

    # ----- 5-step eager warmup so every lazy cache is allocated -----
    state = mrt.make_initial_state(batch_size=batch_size, seed=0)
    for _ in range(5):
        _, state = mrt.step(
            state, source_tokens=src,
            temperature=temperature, top_k=top_k,
            cfg_scales=cfg_scales, cfg_arity=2,
        )
    _promote_cache_offsets(state)
    depth_size = len(_flatten_state(state))
    flat_state = _flatten_state(state) + _flatten_codec(mrt.spectrostream)
    mx.eval(*flat_state)
    snapshot = [mx.array(a) for a in flat_state]
    ref_state = state

    # ----- Build streaming_step closure (same shape as export.py) -----
    def streaming_step(x_values, *state_flat):
        flat_list = list(state_flat)
        local_state = _install_state(flat_list[:depth_size], ref_state)
        _install_codec(mrt.spectrostream, flat_list, depth_size)
        wave, new_state = mrt.step(
            local_state, source_tokens=x_values,
            temperature=temperature, top_k=top_k,
            cfg_scales=cfg_scales, cfg_arity=2,
        )
        new_flat = _flatten_state(new_state) + _flatten_codec(mrt.spectrostream)
        return (wave, *new_flat)

    # Run one dry-run cycle to settle shapes, then export.
    outputs = streaming_step(src, *flat_state)
    wave, *flat_state = outputs
    mx.eval(wave, *flat_state)

    # Re-snapshot state (so eager and .mlxfn both start from the same
    # post-dry-run state).
    snapshot = [mx.array(a) for a in flat_state]

    mlxfn_path = str(tmp_path / "parity.mlxfn")
    with mx.exporter(mlxfn_path, streaming_step, shapeless=False) as exporter:
        exporter(src, *flat_state)

    # ----- Eager: install snapshot, run 10 more steps -----
    flat = [mx.array(a) for a in snapshot]
    eager_waves = []
    for _ in range(10):
        out = streaming_step(src, *flat)
        wave, *flat = out
        mx.eval(wave)
        eager_waves.append(np.array(wave))

    # ----- Exported: load .mlxfn, install snapshot, run 10 more steps -----
    imp = mx.import_function(mlxfn_path)
    flat = [mx.array(a) for a in snapshot]
    exp_waves = []
    for _ in range(10):
        out = imp(src, *flat)
        wave, *flat = out
        mx.eval(wave)
        exp_waves.append(np.array(wave))

    # ----- Compare per-step waveforms -----
    for s in range(10):
        a, b = eager_waves[s], exp_waves[s]
        diff = float(np.abs(a - b).max())
        assert diff == 0.0, (
            f"step {s}: max|diff|={diff:.6e} "
            f"(eager_peak={np.abs(a).max():.3f} "
            f"exp_peak={np.abs(b).max():.3f})"
        )
