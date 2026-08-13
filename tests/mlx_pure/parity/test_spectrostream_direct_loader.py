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

"""Strict parity: direct Linen -> pure SpectroStream loader vs the sl bridge.

The pure-MLX SpectroStream codec can be loaded two ways:

* GOLDEN (oracle): build an sl ``SpectroStream``, load it from the main
  checkpoint via ``magenta_rt.mlx.spectrostream.load_weights`` (which now
  resolves the encoder from ``resources/spectrostream/``), then mirror it into
  a pure module via ``mlx_pure.spectrostream.load_weights.load_spectrostream_weights``.
* DIRECT (under test): build a pure ``SpectroStream`` and load it ENTIRELY
  from the two standalone Linen safetensors via
  ``load_spectrostream_from_linen`` — no sequence_layers anywhere.

Because the sl -> pure copy is the identity for conv kernels/biases, the
direct loader applies the same Linen kernel transforms inline and must produce
a bit-identical codec. We assert exact-equal int codes and tight audio
round-trip agreement.

The numeric test is gated by ``@pytest.mark.checkpoint`` (needs the main
checkpoint + the resources encoder/decoder). The no-sl import check is cheap
and always runs.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CI-cheap guard: the direct loader must not import sequence_layers.
# ---------------------------------------------------------------------------


def test_direct_loader_has_no_sequence_layers_import():
    import magenta_rt.mlx_pure.spectrostream.load_weights as mod

    src = inspect.getsource(mod)
    # The module is shared with the sl-bridge helpers, so only the direct
    # loader's own functions are checked: they must not reach sl.
    direct_fns = [
        mod.load_spectrostream_from_linen,
        mod.load_spectrostream_encoder_from_linen,
        mod.load_spectrostream_decoder_from_linen,
        mod.load_quantizer_from_linen,
        mod._read_linen_safetensors,
        mod._assign_conv2d,
        mod._assign_conv2d_transpose,
    ]
    for fn in direct_fns:
        fsrc = inspect.getsource(fn)
        assert "sequence_layers" not in fsrc, (
            f"{fn.__name__} references sequence_layers"
        )
        assert " sl." not in fsrc and "import sl" not in fsrc, (
            f"{fn.__name__} references sl"
        )


def test_export_mlx_pure_branch_has_no_sequence_layers():
    """The export's mlx_pure codec build must not bridge through sl."""
    # The export's codec builder now lives in the ``mrt sft export`` command
    # module (``magenta_rt.cli.sft_commands._build_codec_and_style``); read its
    # source and isolate the mlx_pure branch (up to the nnx branch).
    from magenta_rt.cli.sft_commands import _build_codec_and_style

    fn_src = inspect.getsource(_build_codec_and_style)
    mlx_pure_branch = fn_src.split('elif backend == "nnx"')[0]
    # Strip comment lines so prose mentioning the old path doesn't trip the
    # checks — we care about actual code.
    code_lines = [
        ln for ln in mlx_pure_branch.splitlines()
        if not ln.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)

    assert "MagentaRT2Sampler" not in code, (
        "mlx_pure branch still references MagentaRT2Sampler"
    )
    assert "sequence_layers" not in code
    assert "load_spectrostream_from_linen" in code, (
        "mlx_pure branch should use the direct Linen loader"
    )


# ---------------------------------------------------------------------------
# Numeric parity (checkpoint-gated).
# ---------------------------------------------------------------------------

_MODEL_NAME = "mrt2_small"
_SR = 48000
_DUR_S = 4


def _checkpoint_path():
    from magenta_rt import paths

    p = Path(paths.resolve_checkpoint(f"{_MODEL_NAME}.safetensors"))
    return p


def _fixed_input():
    """Deterministic [1, 2, 48000*4] stereo waveform (seeded sinusoid mix)."""
    n = _SR * _DUR_S
    t = np.arange(n, dtype=np.float64) / _SR
    rng = np.random.default_rng(1234)
    # A reproducible mix of a few tones per channel plus a touch of seeded
    # noise so the encoder sees broadband content (not a pure tone).
    left = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * rng.standard_normal(n)
    )
    right = (
        0.4 * np.sin(2 * np.pi * 330.0 * t)
        + 0.25 * np.sin(2 * np.pi * 550.0 * t)
        + 0.05 * rng.standard_normal(n)
    )
    wav = np.stack([left, right], axis=0)[None]  # [1, 2, n]
    return wav.astype(np.float32)


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.checkpoint
def test_direct_loader_matches_sl_bridge():
    import mlx.core as mx

    from magenta_rt import paths

    ckpt = _checkpoint_path()
    if not ckpt.exists():
        pytest.skip(f"checkpoint not found: {ckpt}")
    if not paths.resolve_encoder_weights().exists():
        pytest.skip("resources encoder.safetensors not found")
    if not paths.resolve_decoder_weights().exists():
        pytest.skip("resources decoder.safetensors not found")

    import magenta_rt.mlx.spectrostream as sl_ss_mod
    import magenta_rt.mlx.spectrostream.load_weights as sl_lw
    import magenta_rt.mlx_pure.load_weights as plw
    import magenta_rt.mlx_pure.spectrostream.load_weights as pslw
    from magenta_rt.mlx_pure import configs as pure_configs

    spec = pure_configs.get_model_class(_MODEL_NAME)()
    trunc = spec.spectrostream.rvq_truncation_level

    # ---- GOLDEN: sl bridge ----
    sl_ss = sl_ss_mod.SpectroStream(
        sl_ss_mod.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=trunc, use_unique_codes=False,
        )
    )
    sl_lw.load_spectrostream_weights(
        sl_ss, str(ckpt),
        soundstream_params=plw._read_soundstream_params(str(ckpt)),
    )
    pure_golden = spec.build_spectrostream()
    pslw.load_spectrostream_weights(pure_golden, sl_ss)

    # ---- DIRECT: standalone Linen ----
    pure_direct = spec.build_spectrostream()
    pslw.load_spectrostream_from_linen(pure_direct)

    pure_golden.disable_streaming()
    pure_direct.disable_streaming()

    wav = _fixed_input()
    wav_mx = mx.array(wav)

    codes_golden = np.asarray(pure_golden.waveform_to_codes(wav_mx))
    codes_direct = np.asarray(pure_direct.waveform_to_codes(wav_mx))

    print("codes_golden shape", codes_golden.shape, "dtype", codes_golden.dtype)
    print("codes_direct shape", codes_direct.shape, "dtype", codes_direct.dtype)

    # Codes must be EXACTLY equal.
    assert np.array_equal(codes_golden, codes_direct), (
        "direct-loader codes differ from sl-bridge codes: "
        f"mismatch fraction = "
        f"{float(np.mean(codes_golden != codes_direct)):.4f}"
    )

    # Round-trip audio: feed the first num_expected_input_codes codebooks.
    n_in = pure_golden.quantizer.num_expected_input_codes
    codes_in = mx.array(codes_golden[..., :n_in].astype(np.int32))

    audio_golden = np.asarray(pure_golden.codes_to_waveform(codes_in))
    audio_direct = np.asarray(pure_direct.codes_to_waveform(codes_in))

    max_diff = float(np.max(np.abs(audio_golden - audio_direct)))
    print("codes_to_waveform max-abs-diff", max_diff)
    assert max_diff < 1e-2, f"round-trip audio diff too large: {max_diff}"

    # Sanity: both round-trips should track the input reasonably.
    rt_corr_golden = _corr(wav, audio_golden)
    rt_corr_direct = _corr(wav, audio_direct)
    print("round-trip corr golden", rt_corr_golden, "direct", rt_corr_direct)
    assert rt_corr_golden > 0.5, f"golden round-trip corr too low: {rt_corr_golden}"
    assert rt_corr_direct > 0.5, f"direct round-trip corr too low: {rt_corr_direct}"
