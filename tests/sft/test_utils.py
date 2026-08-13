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

"""Synthesize a fake SFT TreeWriter dataset for testing."""

from __future__ import annotations

import numpy as np

from magenta_rt import config as _cfg


def _source_schema(musiccoca_codebook_size: int) -> dict[str, tuple[int, int, int]]:
    return {
        _cfg.MUSICCOCA.key: (                        # 'mulan_tokens_25hz'
            _cfg.MUSICCOCA.rvq_levels, -1, musiccoca_codebook_size),
        _cfg.PIANOROLL_WITH_ONSETS.key: (            # 'pianoroll_with_onsets_tokens'
            128, 0, _cfg.PIANOROLL_WITH_ONSETS.codebook_size),
        _cfg.DRUM_PIANOROLL.key: (                   # 'drum_pianoroll_tokens'
            1, 0, _cfg.DRUM_PIANOROLL.codebook_size),
        _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key: (  # 'cfg_conditioning_tokens'
            _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.rvq_levels, 0,
            _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.codebook_size),
        _cfg.CFG_CONDITIONING_DRUMS.key: (           # 'cfg_conditioning_drums_tokens'
            _cfg.CFG_CONDITIONING_DRUMS.rvq_levels, 0,
            _cfg.CFG_CONDITIONING_DRUMS.codebook_size),
    }


def _fake_example(
    rng: np.random.Generator,
    *,
    frames: int,
    target_rvq: int,
    target_codebook_size: int,
    musiccoca_codebook_size: int,
    with_audio: bool,
    samples_per_frame: int,
    audio_channels: int,
) -> dict[str, np.ndarray]:
    schema = _source_schema(musiccoca_codebook_size)
    arrays = {
        key: rng.integers(low, high, size=(frames, width), dtype=np.int32)
        for key, (width, low, high) in schema.items()
    }
    if with_audio:
        arrays["audio"] = rng.standard_normal(
            (frames * samples_per_frame, audio_channels),
        ).astype(np.float32)
    else:
        arrays["soundstream_tokens"] = rng.integers(
            0, target_codebook_size, size=(frames, target_rvq), dtype=np.int32,
        )
    return arrays


def write_fake_tree_dataset(
    out_dir: str,
    *,
    num_files: int = 8,
    frames_per_file: int = 75,
    target_rvq: int = 4,
    target_codebook_size: int = 32,
    musiccoca_codebook_size: int = 1024,
    seed: int = 0,
    with_audio: bool = False,
    samples_per_frame: int = 1920,
    audio_channels: int = 2,
) -> str:
    from audiotree import TreeWriter, AudioTree

    rng = np.random.default_rng(seed)
    with TreeWriter(out_dir, expected_samples=num_files) as writer:
        for _ in range(num_files):
            arrays = _fake_example(
                rng,
                frames=frames_per_file,
                target_rvq=target_rvq,
                target_codebook_size=target_codebook_size,
                musiccoca_codebook_size=musiccoca_codebook_size,
                with_audio=with_audio,
                samples_per_frame=samples_per_frame,
                audio_channels=audio_channels,
            )
            codes = arrays.pop("soundstream_tokens", None)
            audio = arrays.pop("audio", None)
            writer.write(
                AudioTree(
                    waveform=None if audio is None else audio.T[None],
                    sample_rate=48_000,
                    codes=None if codes is None else codes[None],
                    extras={k: v[None] for k, v in arrays.items()},
                )
            )
    return out_dir
