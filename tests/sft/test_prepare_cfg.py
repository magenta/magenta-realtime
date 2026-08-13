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

"""Tests for the CFG-conditioning channel synthesis (PrepareCFG)."""

from __future__ import annotations

import numpy as np
import pytest
from audiotree import AudioTree

from magenta_rt import config as _cfg
from magenta_rt.conditioning import discretize_cfg
from magenta_rt.sft.transforms import PrepareCFG

_KEY_MC_NOTES = _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key  # [T, 2], 41 bins
_KEY_DRUMS = _cfg.CFG_CONDITIONING_DRUMS.key               # [T, 1], 9 bins


def _example(frames=50, with_cfg=False):
    rng = np.random.RandomState(0)
    metadata = {
        _cfg.MUSICCOCA.key: rng.randint(
            0, 1024, (1, frames, 12)).astype(np.int32),
    }
    if with_cfg:
        metadata[_KEY_MC_NOTES] = np.full((1, frames, 2), 7, np.int32)
        metadata[_KEY_DRUMS] = np.full((1, frames, 1), 3, np.int32)
    return AudioTree(
        waveform=None,
        sample_rate=48_000,
        codes=rng.randint(0, 1024, (1, frames, 4)).astype(np.int32),
        extras=metadata,
    )


def test_samples_missing_channels_constant_over_frames():
    audio = _example(frames=50)
    out = PrepareCFG().random_map(audio, np.random.default_rng(0))

    mc = out.extras[_KEY_MC_NOTES]
    drums = out.extras[_KEY_DRUMS]
    assert mc.shape == (1, 50, 2) and mc.dtype == np.int32
    assert drums.shape == (1, 50, 1) and drums.dtype == np.int32
    # Constant across the example's frames (inference holds CFG constant).
    np.testing.assert_array_equal(mc, np.tile(mc[:, :1], (1, 50, 1)))
    np.testing.assert_array_equal(drums, np.tile(drums[:, :1], (1, 50, 1)))
    # In the channel's token range.
    assert 0 <= mc.min() and mc.max() < 41
    assert 0 <= drums.min() and drums.max() < 9


def test_sampling_varies_across_examples():
    rng = np.random.default_rng(0)
    rows = {
        tuple(PrepareCFG().random_map(_example(), rng)
              .extras[_KEY_MC_NOTES][0, 0])
        for _ in range(16)
    }
    assert len(rows) > 1  # a fresh scale per example, not one global value


def test_present_channels_pass_through():
    audio = _example(with_cfg=True)
    out = PrepareCFG().random_map(audio, np.random.default_rng(0))
    np.testing.assert_array_equal(out.extras[_KEY_MC_NOTES], 7)
    np.testing.assert_array_equal(out.extras[_KEY_DRUMS], 3)


def test_fixed_scales_use_discretize_cfg():
    fixed = {_KEY_MC_NOTES: (3.0, 1.0), _KEY_DRUMS: 1.0}
    out = PrepareCFG(fixed_scales=fixed).random_map(
        _example(frames=10), np.random.default_rng(0)
    )
    # Same binning as the inference systems: step 0.2 / 1.0, range [-1, 7].
    assert out.extras[_KEY_MC_NOTES][0, 0, 0] == discretize_cfg(3.0, 0.2, 40)
    assert out.extras[_KEY_MC_NOTES][0, 0, 1] == discretize_cfg(1.0, 0.2, 40)
    assert out.extras[_KEY_DRUMS][0, 0, 0] == discretize_cfg(1.0, 1.0, 8)
    # token = (scale + 1) / step, so 3.0 -> 20, 1.0 -> 10 / 2.
    assert out.extras[_KEY_MC_NOTES][0, 0, 0] == 20
    assert out.extras[_KEY_MC_NOTES][0, 0, 1] == 10
    assert out.extras[_KEY_DRUMS][0, 0, 0] == 2


def test_fixed_scales_wrong_width_raises():
    with pytest.raises(ValueError, match="expected 2 scale"):
        PrepareCFG(fixed_scales={_KEY_MC_NOTES: (3.0, 1.0, 5.0)}).random_map(
            _example(), np.random.default_rng(0)
        )


def test_no_configured_channels_is_identity():
    audio = _example()
    out = PrepareCFG(cfg_configs=()).random_map(audio, np.random.default_rng(0))
    assert out is audio


def test_full_mrt2_input_configs_through_export(tmp_path):
    """An export without CFG channels trains against the full 6-channel
    mrt2 source spec once PrepareCFG fills the gap (this used to KeyError)."""
    pytest.importorskip("audiotree")
    import soundfile

    from magenta_rt.musiccoca import MockMusicCoCa
    from magenta_rt.sft import create_audiotree_dataset, to_source_target
    from tests.sft.test_export import FakeCodec, FakeTranscriber, _RVQ_DEPTH

    rng = np.random.RandomState(0)
    audio = rng.randn(4 * 48_000, 2).astype(np.float32) * 0.1
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    soundfile.write(audio_dir / "clip.wav", audio, 48_000)

    from magenta_rt.sft.export import export_tree_dataset

    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        transcriber=FakeTranscriber(),
        num_samples=2,
        duration=2.0,
    )

    input_configs = [
        _cfg.MUSICCOCA,
        _cfg.PIANOROLL_WITH_ONSETS,
        _cfg.DRUM_PIANOROLL,
        _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES,
        _cfg.CFG_CONDITIONING_DRUMS,
    ]
    target_config = _cfg.TokensConfig(
        key="spectrostream_tokens",
        codebook_size=1024,
        rvq_levels=_RVQ_DEPTH,
        rvq_truncation_level=_RVQ_DEPTH,
        frame_rate=25,
    )
    ds = create_audiotree_dataset(
        out,
        batch_size=2,
        crop_length_seconds=1,
        input_configs=input_configs,
        target_config=target_config,
        seed=0,
        cfg_fixed_scales={_KEY_MC_NOTES: (3.0, 1.0), _KEY_DRUMS: 1.0},
    )
    batch = next(iter(ds))
    source, target = to_source_target(batch, target_config)
    num_channels = sum(c.rvq_truncation_level for c in input_configs)
    assert source.shape == (2, 25, num_channels)
    assert target.shape == (2, 25, _RVQ_DEPTH)
    assert source.min() >= 0
