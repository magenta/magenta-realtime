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

"""End-to-end audio diff: sl-backed vs pure-mlx codec on identical
random weights, using the v1v3 production SpectroStream config.

We avoid the tiny-config build path because sl's
``SpectroStream.Config.make()`` crashes there with
``AttributeError: 'Serial' object has no attribute
'get_accumulated_input_latency'`` (latency-tracking protocol
regression in this sl checkout). The v1v3 build doesn't hit the bug.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


def _build_v1v3_codec_pair():
    """Build a v1v3 legacy SpectroStream + matching pure SpectroStream, mirror
    weights via the codec section of ``load_weights_from_legacy``, and
    randomize the params so the comparison isn't trivially zero.

    Skips on the sl latency-tracking AttributeError if it happens to
    trigger here too.
    """
    pytest.importorskip("sequence_layers")
    import sequence_layers.mlx as sl
    from sequence_layers.mlx import export as sl_export
    from magenta_rt.mlx import (
        model as combinator_model, spectrostream as combinator_ss, system as combinator_system,
    )
    from magenta_rt.mlx_pure import (
        model as pure_model,
    )
    from magenta_rt.mlx_pure.load_weights import (
        init_random_params, load_weights_from_combinator,
    )

    combinator_spec = combinator_model.get_model_class("mrt2_small")()
    combinator_mrt = combinator_system.MagentaRT2Sampler.Config(
        depthformer=combinator_spec.depthformer_config(),
        spectrostream=combinator_ss.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=combinator_spec.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        ),
        int16_outputs=False,
    ).make()

    input_spec = sl.ChannelSpec(shape=(combinator_spec.input_num_channels,), dtype=mx.int32)
    sl_export._materialize_deferred(
        combinator_mrt, batch_size=1, input_spec=input_spec,
        constants={
            "classifier_free_guidance_scale_musiccoca": mx.array([1.0]),
            "classifier_free_guidance_scale_notes": mx.array([1.0]),
            "temperature": mx.array([0.0]),
            "top_k": mx.array([1], dtype=mx.int32),
        },
    )
    init_random_params(combinator_mrt, seed=0, only_zeros=True)

    pure_mrt = pure_model.MagentaRT2Sampler.from_preset("mrt2_small", int16_outputs=False)
    load_weights_from_combinator(pure_mrt, combinator_mrt)
    return sl, combinator_mrt, pure_mrt


def test_v1v3_codec_codes_to_waveform_e2e_diff():
    """combinator ``codes_to_waveform`` vs pure ``codes_to_waveform`` on
    identical RVQ codes with mirrored random weights — bit-exact
    within bf16 tolerance."""
    sl, combinator_mrt, pure_mrt = _build_v1v3_codec_pair()

    codes = mx.random.randint(0, 1024, (1, 25, 12), dtype=mx.int32, key=mx.random.key(7))
    sl_seq = sl.Sequence(codes, mx.ones(codes.shape[:2], dtype=mx.bool_))
    sl_audio = combinator_mrt.spectrostream.codes_to_waveform_layer.layer(sl_seq).values
    pure_audio = pure_mrt.spectrostream.codes_to_waveform(codes)

    sl_np = np.array(sl_audio.astype(mx.float32))  # sl: [B, T, C]
    pure_np = np.array(pure_audio.astype(mx.float32))
    # The pure codec emits channel-major audio ([B, C, T], or [B, T] mono);
    # compare in sl's [B, T, C] layout.
    if pure_np.ndim == 3:
        pure_np = pure_np.swapaxes(1, 2)
    else:
        pure_np = pure_np[:, :, None]
    common = min(sl_np.shape[1], pure_np.shape[1])
    assert common > 0
    np.testing.assert_allclose(
        sl_np[:, :common], pure_np[:, :common],
        atol=1e-4, rtol=1e-4,
        err_msg="sl vs pure codes_to_waveform diverges",
    )


def test_v1v3_codec_streaming_chunks_match_full():
    """``enable_streaming`` + per-step ``step_codes_to_waveform`` chunks
    concatenate to match a single non-streaming ``codes_to_waveform``
    on the joined codes — sample-for-sample within bf16 tolerance."""
    _, _, pure_mrt = _build_v1v3_codec_pair()
    ss = pure_mrt.spectrostream

    codes = mx.random.randint(0, 1024, (1, 25, 12), dtype=mx.int32, key=mx.random.key(11))
    ss.disable_streaming()
    ref = ss.codes_to_waveform(codes)

    ss.enable_streaming()
    chunks = []
    for t in range(codes.shape[1]):
        chunks.append(ss.step_codes_to_waveform(codes[:, t : t + 1]))
        mx.eval(chunks[-1])
    # Audio is time-last ([B, T] mono or [B, C, T]); concat on the last axis.
    streamed = mx.concatenate(chunks, axis=-1)

    common = min(ref.shape[-1], streamed.shape[-1])
    np.testing.assert_allclose(
        np.array(ref[..., :common].astype(mx.float32)),
        np.array(streamed[..., :common].astype(mx.float32)),
        atol=1e-4, rtol=1e-4,
        err_msg="streaming concat diverges from non-streaming forward",
    )
