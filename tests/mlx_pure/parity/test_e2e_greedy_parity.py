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

"""End-to-end greedy parity: legacy ``MagentaRT2Sampler`` vs pure on
identical inputs with mirrored random weights, ``temperature=0`` /
``top_k=1`` / no CFG.

With argmax sampling and a bit-exact codec, the two pipelines should
emit identical codes at every codebook of every step and identical
audio sample-for-sample (within bf16 tolerance).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest


def _build_pair():
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

    # sl's ``nn.Linear`` default init draws from the global ``mx.random``
    # state. Without this reset, FFN / embedding weights vary across
    # builds and ``init_random_params(only_zeros=True)`` can't normalize
    # them (they aren't zero by the time it runs).
    mx.random.seed(0)
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
    return sl, combinator_spec, combinator_mrt, pure_mrt


def test_v1v3_waveform_to_codes_bit_exact():
    """combinator ``waveform_to_codes`` vs pure ``waveform_to_codes`` on
    identical random audio with mirrored random weights — assert all
    encoded codebook values match exactly. Pins the encoder bridge
    (base_conv, encoder_0..6, bottleneck, output_convs) and the
    forward STFT padding."""
    sl, _, combinator_mrt, pure_mrt = _build_pair()
    cfg = combinator_mrt.spectrostream.config
    audio = mx.random.normal(
        (1, 24000, cfg.num_channels // 2), key=mx.random.key(13),
    ) * 0.1
    sl_seq = sl.Sequence(audio, mx.ones((1, audio.shape[1]), dtype=mx.bool_))
    sl_codes = combinator_mrt.spectrostream.waveform_to_codes_layer.layer(sl_seq).values
    # sl consumes [B, T, C]; the pure codec consumes channel-major [B, C, T].
    pure_codes = pure_mrt.spectrostream.waveform_to_codes(
        mx.transpose(audio, (0, 2, 1))
    )
    sl_np = np.array(sl_codes)
    pure_np = np.array(pure_codes)
    common = min(sl_np.shape[1], pure_np.shape[1])
    assert common > 0
    np.testing.assert_array_equal(
        sl_np[:, :common], pure_np[:, :common],
        err_msg="sl vs pure waveform_to_codes diverges",
    )


def test_greedy_codes_match_across_streaming_steps():
    """Run 3 streaming steps in each pipeline; assert all 36 codes
    (12 codebooks × 3 steps) match combinator exactly."""
    sl, combinator_spec, combinator_mrt, pure_mrt = _build_pair()

    # mrt2 144-channel source frame: musiccoca(12) + onsets(128) + drums(1) + cfgs(3).
    musiccoca = [1] * 12
    notes = [-1] * 128
    drums = [-1] * 1
    cfgs = [20, 10, 4]
    cond = np.array(musiccoca + notes + drums + cfgs, dtype=np.int32) + 7  # NUM_RESERVED+1
    src_tokens = mx.array(cond.reshape(1, 1, -1), dtype=mx.int32)
    encoder_out = combinator_mrt.depthformer.encoder.body.layer(
        sl.Sequence(src_tokens, mx.ones((1, 1), dtype=mx.bool_))
    ).values

    constants = {
        "temperature": mx.array([0.0]),
        "top_k": mx.array([1], dtype=mx.int32),
        "source": sl.Sequence(encoder_out, mx.ones((1, 1), dtype=mx.bool_)),
    }
    sl_dec = combinator_mrt.depthformer.decoder
    sos = sl_dec.get_sos(1)
    sl_state = sl_dec.get_initial_state(1, sos.channel_spec, constants=constants, training=False)
    pure_state = pure_mrt.depthformer.make_initial_state(batch_size=1, seed=0)

    for step in range(3):
        sl_out, sl_state, _ = sl_dec.step_with_emits(sos, sl_state, constants=constants)
        pure_codes, pure_state = pure_mrt.depthformer.step(
            pure_state, source_frame=encoder_out, temperature=0.0, top_k=1,
        )
        sc = np.array(sl_out.values).flatten().tolist()
        pc = np.array(pure_codes).flatten().tolist()
        assert sc == pc, f"step {step}: sl={sc} pure={pc}"
