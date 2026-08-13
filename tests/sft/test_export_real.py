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

"""Real-model SFT export integration: nnx SpectroStream + nnx MusicCoCa.

Exports two 10-second windows from synthesized audio with real weights and
verifies the records read back through the training pipeline. Gated on the
local ``mrt2_small`` checkpoint and the converted MusicCoCa safetensors;
opt in with ``pytest -m checkpoint``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("audiotree")

from magenta_rt import paths

pytestmark = pytest.mark.checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = paths.resolve_checkpoint("mrt2_small.safetensors")
MUSICCOCA_WEIGHTS = paths.musiccoca_dir() / "musiccoca_nnx.safetensors"


def test_real_export_round_trip(tmp_path):
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT}")
    if not MUSICCOCA_WEIGHTS.exists():
        pytest.skip(f"MusicCoCa weights not found at {MUSICCOCA_WEIGHTS}")

    import soundfile
    from flax import nnx as flax_nnx

    from magenta_rt import config as _cfg
    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.musiccoca import MusicCoCa
    from magenta_rt.sft import create_audiotree_dataset, to_source_target
    from magenta_rt.sft.export import EMBEDDING_KEY, export_tree_dataset

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=flax_nnx.Rngs(0)
    )
    mrt.load_checkpoint(CHECKPOINT)

    # 20 s of band-limited noise-modulated tone -> two 10 s excerpts.
    sr = 48_000
    t = np.arange(20 * sr) / sr
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)
             * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    soundfile.write(audio_dir / "tone.wav", np.stack([audio, audio], axis=1), sr)

    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=mrt.spectrostream,
        style_model=MusicCoCa(),
        num_samples=2,
        duration=10.0,
        batch_size=2,
    )

    target_config = _cfg.SPECTROSTREAM
    ds = create_audiotree_dataset(
        out,
        batch_size=2,
        crop_length_seconds=2,
        input_configs=[_cfg.MUSICCOCA],
        target_config=target_config,
        seed=0,
    )
    batch = next(iter(ds))
    assert batch.codes.shape[0] == 2
    assert batch.codes.shape[2] == target_config.rvq_levels
    assert batch.extras[EMBEDDING_KEY].shape == (2, 768)

    source, target = to_source_target(batch, target_config)
    assert source.shape == (2, 50, _cfg.MUSICCOCA.rvq_truncation_level)
    assert target.shape == (2, 50, target_config.rvq_truncation_level)
    # Real RVQ codes: every depth should be in range and non-constant.
    assert batch.codes.min() >= 0
    assert batch.codes.max() < target_config.codebook_size
    assert len(np.unique(batch.codes)) > 8


def test_real_export_with_mt3_transcription(tmp_path):
    """Full preprocessing stack: SpectroStream + MusicCoCa + MT3 per window."""
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT}")
    if not MUSICCOCA_WEIGHTS.exists():
        pytest.skip(f"MusicCoCa weights not found at {MUSICCOCA_WEIGHTS}")
    if not (paths.mt3_dir() / "mt3_mt3.safetensors").exists():
        pytest.skip("mt3 checkpoint not downloaded")

    import soundfile
    from flax import nnx as flax_nnx

    from magenta_rt import config as _cfg
    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.musiccoca import MusicCoCa
    from magenta_rt.sft.export import export_tree_dataset, mt3_transcriber
    from magenta_rt.sft.pianoroll import PITCH_ONSET

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=flax_nnx.Rngs(0)
    )
    mrt.load_checkpoint(CHECKPOINT)

    # A harmonics-rich arpeggio (the synthetic timbre MT3 detects reliably).
    sr = 48_000

    def tone(pitch, start, dur, total):
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        t = np.arange(int(dur * sr)) / sr
        x = sum((0.6**k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        x = x * np.exp(-2.5 * t) * 0.3
        out = np.zeros(int(total * sr), np.float32)
        i = int(start * sr)
        out[i : i + len(x)] += x.astype(np.float32)
        return out

    onsets = [(60, 0.5), (64, 3.0), (67, 5.5), (72, 8.0)]
    audio = sum(tone(p, s, 1.5, 10.0) for p, s in onsets)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    soundfile.write(
        audio_dir / "arpeggio.wav", np.stack([audio, audio], axis=1), sr
    )

    from audiotree.sources import TreeDataSource

    # The file is exactly one `duration` long, so the excerpt offset is
    # forced to 0 and the onset-time assertions below stay deterministic.
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=mrt.spectrostream,
        style_model=MusicCoCa(),
        transcriber=mt3_transcriber("mt3"),
        num_samples=1,
        duration=10.0,
    )

    record = TreeDataSource(out)[0]
    roll = record.extras[_cfg.PIANOROLL_WITH_ONSETS.key][0]  # [250, 128]
    assert roll.shape == (250, 128)
    assert roll.max() <= 2 and roll.min() >= 0
    # MT3 should place an onset within ±3 frames (120 ms) of each note.
    # Exact pitch/program vary on this synthetic timbre (the original mt3
    # smoke test makes the same allowance), so assert onset *times* only.
    onset_frames = np.flatnonzero((roll == PITCH_ONSET).any(axis=1))
    for _, start in onsets:
        expected = start * 25
        assert any(abs(f - expected) <= 3 for f in onset_frames), (
            f"no onset near frame {expected}: {onset_frames}"
        )
    # The drum channel is intentionally not synthesized (it's an intent
    # directive, not an onset raster); training conditions it on the dropout
    # token, so the export carries no drum leaf.
    assert _cfg.DRUM_PIANOROLL.key not in record.extras
