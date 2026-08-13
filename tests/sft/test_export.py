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

"""Tests for the TreeWriter SFT export pipeline (fake codec + mock MusicCoCa)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile

pytest.importorskip("audiotree")

from audiotree import AudioTree
from magenta_rt.config import MUSICCOCA as _MUSICCOCA
from magenta_rt.musiccoca import MockMusicCoCa
from magenta_rt.sft.export import (
    EMBEDDING_KEY,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    export_tree_dataset,
)

_RVQ_DEPTH = 6


class FakeCodec:
    """Duck-typed SpectroStream: deterministic codes from the audio content."""

    def waveform_to_codes(self, audio) -> np.ndarray:
        # Accept a raw [B, C, T] array or an AudioTree, like the real codecs.
        if isinstance(audio, AudioTree):
            audio = np.asarray(audio.waveform)
        # audio: [B, C, T] -> [B, frames, D]
        b, _, t = audio.shape
        frames = t // SAMPLES_PER_FRAME
        per_frame = audio[:, 0, : frames * SAMPLES_PER_FRAME].reshape(
            b, frames, SAMPLES_PER_FRAME
        )
        return (np.abs(per_frame).sum(-1, keepdims=True).astype(np.int64)
                % 1024).astype(np.int32) + np.arange(_RVQ_DEPTH, dtype=np.int32)


def _write_wavs(tmp_path, durations, seed=0):
    """Writes loud stereo noise clips into tmp_path/audio; returns the dir."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(seed)
    for i, seconds in enumerate(durations):
        audio = rng.randn(int(seconds * SAMPLE_RATE), 2).astype(np.float32) * 0.1
        soundfile.write(audio_dir / f"clip_{i}.wav", audio, SAMPLE_RATE)
    return audio_dir


def test_export_and_read_back(tmp_path):
    from audiotree.sources import TreeDataSource

    audio_dir = _write_wavs(tmp_path, [2.5, 1.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=3,
        duration=1.0,
        batch_size=2,
    )

    source = TreeDataSource(out)
    assert len(source) == 3
    frames = 25
    for i in range(len(source)):
        record = source[i]
        assert record.waveform is None  # waveform is not saved
        assert record.sample_rate == SAMPLE_RATE
        assert record.codes.shape == (1, frames, _RVQ_DEPTH)
        assert record.codes.dtype == np.uint16
        style = record.extras[_MUSICCOCA.key]
        assert style.shape == (1, frames, 12)
        assert style.dtype == np.int16
        # Style tokens are one embedding broadcast across the excerpt.
        np.testing.assert_array_equal(style[0], np.tile(style[0, 0], (frames, 1)))
        emb = record.extras[EMBEDDING_KEY]
        assert emb.shape == (1, 768)
        assert emb.dtype == np.float32
        # Provenance: the source file and the excerpt offset round-trip.
        assert "clip_" in record.filepath[0]
        assert record.offset.shape == (1,)


def test_export_mono_input_recomputes_loudness(tmp_path):
    """Mono sources pass through stereo() (mono->stereo), which rebuilds the
    waveform and drops the loudness the saliency search computed; the export
    recomputes it before the near-silence filter, so mono input exports without a
    None-loudness crash. Regression: stereo sources skip stereo() and never hit
    this — which is why _write_wavs (stereo) missed the bug."""
    from audiotree.sources import TreeDataSource

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rng = np.random.RandomState(0)
    for i, seconds in enumerate([2.5, 1.0]):
        mono = rng.randn(int(seconds * SAMPLE_RATE)).astype(np.float32) * 0.1
        soundfile.write(audio_dir / f"mono_{i}.wav", mono, SAMPLE_RATE)  # 1 channel

    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=3,
        duration=1.0,
        batch_size=2,
    )

    source = TreeDataSource(out)
    assert len(source) == 3
    for i in range(len(source)):
        assert source[i].codes.shape == (1, 25, _RVQ_DEPTH)


def test_export_with_style_prompt_is_constant_across_records(tmp_path):
    """``style_prompt`` conditions every record on one fixed text embedding:
    identical MusicCoCa tokens + embedding across the whole dataset."""
    from audiotree.sources import TreeDataSource

    audio_dir = _write_wavs(tmp_path, [2.5, 1.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=3,
        duration=1.0,
        batch_size=2,
        style_prompt="electronic dance music",
    )

    source = TreeDataSource(out)
    assert len(source) == 3
    ref_style = source[0].extras[_MUSICCOCA.key]
    ref_emb = source[0].extras[EMBEDDING_KEY]
    for i in range(len(source)):
        record = source[i]
        # Same fixed-prompt tokens/embedding on every record (and every frame).
        np.testing.assert_array_equal(record.extras[_MUSICCOCA.key], ref_style)
        np.testing.assert_array_equal(record.extras[EMBEDDING_KEY], ref_emb)
        # But the audio (codes) still differs per excerpt — only style is fixed.
    assert not np.array_equal(source[0].codes, source[1].codes)


def test_file_level_split_no_leakage(tmp_path):
    """discover_audio_files + split_audio_files + the files= export path give
    a leak-free file-level train/val split (no audio file in both)."""
    from audiotree.sources import TreeDataSource

    from magenta_rt.sft.export import (
        discover_audio_files,
        split_audio_files,
    )

    audio_dir = _write_wavs(tmp_path, [3.0] * 10)  # 10 distinct files
    all_files = discover_audio_files(audio_dir, [".wav"])
    assert len(all_files) == 10
    train_files, val_files = split_audio_files(
        all_files, val_fraction=0.3, split_seed=0
    )
    assert len(train_files) == 7 and len(val_files) == 3
    assert not (set(train_files) & set(val_files))
    # Deterministic.
    assert split_audio_files(all_files, val_fraction=0.3, split_seed=0)[1] == val_files

    common = dict(codec=FakeCodec(), style_model=MockMusicCoCa(),
                  duration=1.0, batch_size=4)
    export_tree_dataset(None, tmp_path / "train", files=train_files,
                        num_samples=12, **common)
    export_tree_dataset(None, tmp_path / "val", files=val_files,
                        num_samples=6, **common)

    def drawn_files(d):
        src = TreeDataSource(str(d))
        return {src[i].filepath[0] for i in range(len(src))}

    # The files each split actually drew excerpts from never overlap.
    assert not (drawn_files(tmp_path / "train") & drawn_files(tmp_path / "val"))


def test_export_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        export_tree_dataset(None, tmp_path / "d", codec=FakeCodec(),
                            style_model=MockMusicCoCa(), num_samples=1)


def test_export_more_samples_than_files(tmp_path):
    """num_samples can exceed the file count: files repeat with fresh
    excerpt positions, and save_embedding=False skips the embedding leaf."""
    from audiotree.sources import TreeDataSource

    audio_dir = _write_wavs(tmp_path, [3.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=5,
        duration=1.0,
        batch_size=3,
        save_embedding=False,
    )
    source = TreeDataSource(out)
    assert len(source) == 5
    assert EMBEDDING_KEY not in source[0].extras
    # Excerpts are drawn at random positions, not all identical.
    codes = np.stack([source[i].codes[0] for i in range(5)])
    assert not all(
        np.array_equal(codes[0], codes[i]) for i in range(1, 5)
    )


def test_export_bad_duration_raises(tmp_path):
    audio_dir = _write_wavs(tmp_path, [1.0])
    with pytest.raises(ValueError, match="whole number"):
        export_tree_dataset(
            audio_dir,
            tmp_path / "dataset",
            codec=FakeCodec(),
            style_model=MockMusicCoCa(),
            num_samples=1,
            duration=1.01,
        )


def test_export_feeds_training_pipeline(tmp_path):
    """The export is directly consumable by create_audiotree_dataset +
    to_source_target — the full cross-framework training read path."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset, to_source_target

    audio_dir = _write_wavs(tmp_path, [4.0, 4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=4,
        duration=2.0,
    )

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
        input_configs=[_cfg.MUSICCOCA],
        target_config=target_config,
        seed=0,
    )
    batch = next(iter(ds))
    # The static embedding passes through batching untouched.
    assert batch.extras[EMBEDDING_KEY].shape == (2, 768)
    source, target = to_source_target(batch, target_config)
    assert source.shape == (2, 25, _cfg.MUSICCOCA.rvq_truncation_level)
    assert target.shape == (2, 25, _RVQ_DEPTH)
    assert source.dtype == np.int16 and target.dtype == np.int16


def test_create_dataset_no_crop_uses_full_record(tmp_path):
    """crop_length_seconds=None (the default) trains on each record at its full
    stored length — no AudioTreeRandomCrop. A 2 s export -> 50 frames."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset, to_source_target

    audio_dir = _write_wavs(tmp_path, [4.0, 4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=4,
        duration=2.0,  # 50 frames per record
    )
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
        crop_length_seconds=None,  # default: no crop
        input_configs=[_cfg.MUSICCOCA],
        target_config=target_config,
        seed=0,
    )
    batch = next(iter(ds))
    source, target = to_source_target(batch, target_config)
    assert source.shape == (2, 50, _cfg.MUSICCOCA.rvq_truncation_level)
    assert target.shape == (2, 50, _RVQ_DEPTH)


def test_fixed_style_tokens_overlay(tmp_path):
    """style_tokens overlays one fixed MusicCoCa row on every example (the
    training-time single-style-prompt recipe): a codec-only export then
    conditions on that style, not the learned dropout token."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset, to_source_target

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir, tmp_path / "dataset", codec=FakeCodec(),
        style_model=None, num_samples=2, duration=2.0,  # codec-only (no style)
    )
    target_config = _cfg.TokensConfig(
        key="spectrostream_tokens", codebook_size=1024, rvq_levels=_RVQ_DEPTH,
        rvq_truncation_level=_RVQ_DEPTH, frame_rate=25,
    )
    n_mc = _cfg.MUSICCOCA.rvq_truncation_level
    fixed = np.arange(n_mc, dtype=np.int32)  # distinct per channel

    ds = create_audiotree_dataset(
        out, batch_size=2, crop_length_seconds=None,
        input_configs=[_cfg.MUSICCOCA], target_config=target_config,
        seed=0, style_tokens=fixed,
    )
    source, _ = to_source_target(next(iter(ds)), target_config)
    styled = source[:, :, :n_mc]  # the MusicCoCa channel block

    # One constant style: identical across frames and batch.
    assert (styled == styled[0:1, 0:1, :]).all()
    # The real fixed tokens flow through (n_mc distinct values), so this is the
    # applied style, not the single-valued learned dropout token.
    assert len(np.unique(styled[0, 0])) == n_mc
    assert not (styled == _cfg.MUSICCOCA.num_extra_tokens).all()


def test_volume_change_fixed_gain():
    """A fixed-dB ``volume_change`` (min_db==max_db) scales by a constant linear
    factor. Level normalization is supplied entirely by audiotree.transforms
    (``peak_norm`` / ``volume_norm`` / ``volume_change``); the export module
    keeps no local helper.
    """
    import numpy as _np
    from audiotree import AudioTree
    from audiotree.transforms import volume_change

    w = np.random.RandomState(0).randn(1, 2, 1000).astype(np.float32) * 0.1
    at = AudioTree(waveform=w, sample_rate=SAMPLE_RATE)

    gain = 0.5
    gain_db = 20.0 * _np.log10(gain)
    out = volume_change(min_db=gain_db, max_db=gain_db).random_map(
        at, np.random.default_rng(0)
    )
    np.testing.assert_allclose(np.asarray(out.waveform), w * gain, rtol=1e-5)


def test_export_with_normalize_map(tmp_path):
    """export_tree_dataset accepts a plain ``Map`` normalize (peak_norm)."""
    from audiotree.sources import TreeDataSource
    from audiotree.transforms import peak_norm

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir, tmp_path / "dataset", codec=FakeCodec(),
        style_model=None, num_samples=3, duration=1.0,
        normalize=peak_norm(),
    )
    assert len(TreeDataSource(out)) == 3


def test_export_with_normalize_randommap(tmp_path):
    """export_tree_dataset accepts a ``RandomMap`` normalize (volume_change),
    applied via ``.random_map`` with a pipeline-derived seed."""
    from audiotree.sources import TreeDataSource
    from audiotree.transforms import volume_change

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir, tmp_path / "dataset", codec=FakeCodec(),
        style_model=None, num_samples=3, duration=1.0,
        normalize=volume_change(min_db=-3.0, max_db=-3.0),
    )
    assert len(TreeDataSource(out)) == 3


def test_export_excluding_embedding_at_read(tmp_path):
    """tree_exclude_prefixes leaves the embedding memmap unread."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=4,
        duration=2.0,
    )
    ds = create_audiotree_dataset(
        out,
        batch_size=2,
        crop_length_seconds=1,
        input_configs=[_cfg.MUSICCOCA],
        target_config=None,
        seed=0,
        tree_exclude_prefixes=[f"extras.{EMBEDDING_KEY}"],
    )
    batch = next(iter(ds))
    assert EMBEDDING_KEY not in batch.extras


class FakeTranscriber:
    """Deterministic transcriber: one pitched note + one drum hit per excerpt."""

    def __init__(self):
        self.calls = 0

    def __call__(self, samples: np.ndarray):
        import dataclasses

        assert samples.ndim == 1  # mono 16 kHz samples

        @dataclasses.dataclass
        class Note:
            pitch: int
            start_time: float
            end_time: float
            is_drum: bool = False

        @dataclasses.dataclass
        class Transcription:
            notes: list

        self.calls += 1
        return Transcription(notes=[
            Note(pitch=60 + self.calls, start_time=0.2, end_time=0.6),
            Note(pitch=36, start_time=0.4, end_time=0.5, is_drum=True),
        ])


def test_export_with_transcriber_channels(tmp_path):
    from audiotree.sources import TreeDataSource

    from magenta_rt import config as _cfg

    audio_dir = _write_wavs(tmp_path, [2.0])
    transcriber = FakeTranscriber()
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        transcriber=transcriber,
        num_samples=2,
        duration=1.0,
    )
    assert transcriber.calls == 2  # one per excerpt

    source = TreeDataSource(out)
    for i in range(2):
        record = source[i]
        roll = record.extras[_cfg.PIANOROLL_WITH_ONSETS.key]
        assert roll.shape == (1, 25, 128) and roll.dtype == np.int8
        pitch = 60 + (i + 1)
        assert roll[0, 5, pitch] == 2          # onset at 0.2s = frame 5
        assert (roll[0, 6:15, pitch] == 1).all()  # sustain through 0.6s
        # The drum channel is intentionally not synthesized (see pianoroll.py);
        # training conditions it on the unconditional dropout token instead.
        assert _cfg.DRUM_PIANOROLL.key not in record.extras


def test_export_without_transcriber_uses_dropout_tokens(tmp_path):
    """An export with no MT3 channels still trains against the full mrt2
    source spec: the missing piano-roll streams fall back to their learned
    unconditional (dropout) token."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset, to_source_target

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=2,
        duration=2.0,
    )
    target_config = _cfg.TokensConfig(
        key="spectrostream_tokens",
        codebook_size=1024,
        rvq_levels=_RVQ_DEPTH,
        rvq_truncation_level=_RVQ_DEPTH,
        frame_rate=25,
    )
    input_configs = [
        _cfg.MUSICCOCA, _cfg.PIANOROLL_WITH_ONSETS, _cfg.DRUM_PIANOROLL,
    ]
    ds = create_audiotree_dataset(
        out,
        batch_size=2,
        crop_length_seconds=1,
        input_configs=input_configs,
        target_config=target_config,
        seed=0,
    )
    batch = next(iter(ds))
    source, _ = to_source_target(batch, target_config)
    num_channels = sum(c.rvq_truncation_level for c in input_configs)
    assert source.shape == (2, 25, num_channels)
    # The piano-roll block is exactly the dropout token (-1 + offset =
    # num_extra_tokens), i.e. the learned unconditional embedding row.
    n_mc = _cfg.MUSICCOCA.rvq_truncation_level
    roll = source[:, :, n_mc : n_mc + 128]
    drums = source[:, :, n_mc + 128 :]
    assert (roll == _cfg.PIANOROLL_WITH_ONSETS.num_extra_tokens).all()
    assert (drums == _cfg.DRUM_PIANOROLL.num_extra_tokens).all()


def test_export_codec_only_skips_musiccoca(tmp_path):
    """``style_model=None`` (and no ``style_prompt``) is a codec-only export:
    codes + provenance only, no MusicCoCa tokens/embedding written."""
    from audiotree.sources import TreeDataSource

    audio_dir = _write_wavs(tmp_path, [4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=None,
        num_samples=3,
        duration=1.0,
        batch_size=2,
    )
    source = TreeDataSource(out)
    assert len(source) == 3
    for i in range(len(source)):
        record = source[i]
        assert record.codes.shape == (1, 25, _RVQ_DEPTH)
        assert _MUSICCOCA.key not in record.extras
        assert EMBEDDING_KEY not in record.extras
        # Provenance is still recorded.
        assert "clip_" in record.filepath[0]


def test_export_style_prompt_without_model_raises(tmp_path):
    audio_dir = _write_wavs(tmp_path, [1.0])
    with pytest.raises(ValueError, match="style_prompt"):
        export_tree_dataset(
            audio_dir,
            tmp_path / "dataset",
            codec=FakeCodec(),
            style_model=None,
            num_samples=1,
            duration=1.0,
            style_prompt="techno",
        )


def test_export_trim_frames_crops_codes_and_channels(tmp_path):
    """``trim_frames`` drops the outer frames from codes AND every per-frame
    conditioning channel, keeping the central window; non-frame metadata
    (embedding, provenance) is untouched."""
    from audiotree.sources import TreeDataSource

    from magenta_rt import config as _cfg

    audio_dir = _write_wavs(tmp_path, [4.0])
    # duration=2.0 -> 50 frames; trim 10 each side -> keep the central 30.
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        transcriber=FakeTranscriber(),
        num_samples=2,
        duration=2.0,
        trim_frames=10,
    )
    source = TreeDataSource(out)
    for i in range(len(source)):
        record = source[i]
        assert record.codes.shape == (1, 30, _RVQ_DEPTH)
        assert record.extras[_MUSICCOCA.key].shape == (1, 30, 12)
        assert record.extras[_cfg.PIANOROLL_WITH_ONSETS.key].shape == (1, 30, 128)
        # The drum channel is not synthesized (see pianoroll.py).
        assert _cfg.DRUM_PIANOROLL.key not in record.extras
        # Non-frame metadata is left at full size.
        assert record.extras[EMBEDDING_KEY].shape == (1, 768)
        assert record.offset.shape == (1,)


def test_export_trim_frames_too_large_raises(tmp_path):
    audio_dir = _write_wavs(tmp_path, [2.0])
    with pytest.raises(ValueError, match="trim_frames"):
        export_tree_dataset(
            audio_dir,
            tmp_path / "dataset",
            codec=FakeCodec(),
            style_model=MockMusicCoCa(),
            num_samples=1,
            duration=1.0,  # 25 frames
            trim_frames=13,  # 2*13 >= 25
        )


def test_export_with_transcriber_feeds_full_input_configs(tmp_path):
    """All four exported channels flow through the mrt2 source prep."""
    from magenta_rt import config as _cfg
    from magenta_rt.sft import create_audiotree_dataset, to_source_target

    audio_dir = _write_wavs(tmp_path, [4.0, 4.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        transcriber=FakeTranscriber(),
        num_samples=4,
        duration=2.0,
    )
    target_config = _cfg.TokensConfig(
        key="spectrostream_tokens",
        codebook_size=1024,
        rvq_levels=_RVQ_DEPTH,
        rvq_truncation_level=_RVQ_DEPTH,
        frame_rate=25,
    )
    input_configs = [
        _cfg.MUSICCOCA, _cfg.PIANOROLL_WITH_ONSETS, _cfg.DRUM_PIANOROLL,
    ]
    ds = create_audiotree_dataset(
        out,
        batch_size=2,
        crop_length_seconds=1,
        input_configs=input_configs,
        target_config=target_config,
        seed=0,
    )
    batch = next(iter(ds))
    source, target = to_source_target(batch, target_config)
    num_channels = sum(c.rvq_truncation_level for c in input_configs)
    assert source.shape == (2, 25, num_channels)
    assert target.shape == (2, 25, _RVQ_DEPTH)
    assert source.min() >= 0
    # The transcriber no longer produces the drum channel, so its column falls
    # back to the dropout token (-1 + (num_extra_tokens + 1) = num_extra_tokens).
    n_mc = _cfg.MUSICCOCA.rvq_truncation_level
    drums = source[:, :, n_mc + 128 :]
    assert (drums == _cfg.DRUM_PIANOROLL.num_extra_tokens).all()


def test_export_time_varying_musiccoca(tmp_path):
    """``musiccoca_time_varying`` writes a per-frame (time-varying) style channel
    from LEADING windows: the drawn window is [head | TARGET | look-ahead], codes
    + pianoroll are kept on TARGET, and the style tokens differ across frames."""
    from audiotree.sources import TreeDataSource

    from magenta_rt import config as _cfg

    # duration 13 = 1 s head-trim + 2 s target + 10 s look-ahead -> 50 target frames.
    audio_dir = _write_wavs(tmp_path, [20.0, 20.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        transcriber=FakeTranscriber(),
        num_samples=2,
        duration=13.0,
        batch_size=2,
        musiccoca_time_varying=True,
        head_trim_seconds=1.0,
        musiccoca_lookahead_seconds=10.0,
        musiccoca_hop_seconds=0.04,  # per-frame
    )
    source = TreeDataSource(out)
    for i in range(len(source)):
        record = source[i]
        assert record.codes.shape == (1, 50, _RVQ_DEPTH)
        style = record.extras[_MUSICCOCA.key]
        assert style.shape == (1, 50, 12) and style.dtype == np.int16
        # Time-varying: NOT a single row broadcast across the frames.
        assert not np.array_equal(style[0], np.tile(style[0, 0], (50, 1)))
        # Pianoroll is kept on the 50 target frames (aligned to codes/style).
        assert record.extras[_cfg.PIANOROLL_WITH_ONSETS.key].shape == (1, 50, 128)
        # No single per-clip embedding is written in time-varying mode.
        assert EMBEDDING_KEY not in record.extras


def test_export_time_varying_no_target_frames_raises(tmp_path):
    """A drawn window too short to leave any target after head-trim + look-ahead
    is rejected."""
    audio_dir = _write_wavs(tmp_path, [20.0])
    with pytest.raises(ValueError, match="no target frames"):
        export_tree_dataset(
            audio_dir, tmp_path / "d", codec=FakeCodec(),
            style_model=MockMusicCoCa(), num_samples=1, duration=11.0,
            musiccoca_time_varying=True, head_trim_seconds=1.0,
            musiccoca_lookahead_seconds=10.0,  # 11 - 1 - 10 = 0 target frames
        )


def test_export_time_varying_coarse_hop(tmp_path):
    """Verify that a coarse hop (e.g. 8.0s) produces a correct piecewise-constant
    aligned style sequence over 25 Hz target frames."""
    from audiotree.sources import TreeDataSource

    # duration 31 = 1 s head-trim + 20 s target + 10 s look-ahead -> 500 target frames.
    audio_dir = _write_wavs(tmp_path, [40.0])
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=1,
        duration=31.0,
        batch_size=1,
        musiccoca_time_varying=True,
        head_trim_seconds=1.0,
        musiccoca_lookahead_seconds=10.0,
        musiccoca_hop_seconds=8.0,  # Snapped blocks: [0s-8s), [8s-16s), [16s-20s)
    )
    source = TreeDataSource(out)
    record = source[0]
    style = record.extras[_MUSICCOCA.key][0]  # [500, 12]
    assert style.shape == (500, 12)

    # Frame indices:
    # 0 to 199 (0.0s to 7.96s) -> Block 1
    # 200 to 399 (8.0s to 15.96s) -> Block 2
    # 400 to 499 (16.0s to 19.96s) -> Block 3

    # Check constant values within Block 1
    for f in range(200):
        np.testing.assert_array_equal(style[f], style[0])

    # Check constant values within Block 2
    for f in range(200, 400):
        np.testing.assert_array_equal(style[f], style[200])

    # Check constant values within Block 3
    for f in range(400, 500):
        np.testing.assert_array_equal(style[f], style[400])

    # Check that blocks differ
    assert not np.array_equal(style[0], style[200])
    assert not np.array_equal(style[200], style[400])



def test_multiprocess_export_respects_num_samples(tmp_path):
    """`num_samples` is a total, not a per-worker quota.

    grain's ``mp_prefetch`` replays the whole upstream pipeline inside each
    worker, so a limit placed above it is applied once per worker and the export
    writes ``worker_count * num_samples`` records off the unbounded
    (``num_epochs=None``) source. Every other test in this file runs
    single-process, where that bug is invisible.
    """
    from audiotree.sources import TreeDataSource

    audio_dir = _write_wavs(tmp_path, [2.5, 1.0, 2.0])
    num_samples = 6
    out = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=FakeCodec(),
        style_model=MockMusicCoCa(),
        num_samples=num_samples,
        duration=1.0,
        batch_size=4,          # not a divisor of 6: also covers the partial-batch trim
        worker_count=2,
    )

    assert len(TreeDataSource(out)) == num_samples
