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

"""Tests for the AudioTree-flavored SFT transforms (audiotree-style).

Checkpoint-free: a fake codec and fake token configs exercise the keystone
(EncodeWithCodec: codes-if-present-else-encode), target/source prep, the
augment_batch boundary, and the AudioTree container behaviors (metadata +
batch_fn) the unified pipeline relies on. Audio is channel-major [B, C, T].
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audiotree import AudioTree
from magenta_rt.sft import transforms as T
from magenta_rt.sft.data import prepare_target_tokens


class _FakeCodec:
    """Records its input; returns deterministic [B, T_frames, K] codes."""

    def __init__(self, num_codebooks=4):
        self.num_codebooks = num_codebooks
        self.calls = 0
        self.last_waveform = None

    def waveform_to_codes(self, waveform):
        self.calls += 1
        self.last_waveform = waveform
        B, nsamp = waveform.shape[0], waveform.shape[-1]  # [B, C, T]
        T = max(1, nsamp // 16)
        return np.tile(np.arange(self.num_codebooks, dtype=np.int32), (B, T, 1))


def _audio_wav(B=2, nsamp=64, ch=2):
    return AudioTree(np.zeros((B, ch, nsamp), np.float32), 48000)


def _tgt_cfg(codebook_size=32, num_extra_tokens=6):
    return SimpleNamespace(codebook_size=codebook_size, num_extra_tokens=num_extra_tokens)


# --- EncodeWithCodec (the keystone) -----------------------------------------

def test_encode_skips_when_codes_present():
    codec = _FakeCodec()
    audio = AudioTree(np.zeros((2, 2, 64), np.float32), 48000,
                    codes=np.full((2, 4, 4), 7, np.int32))
    out = T.EncodeWithCodec(codec).map(audio)
    assert codec.calls == 0  # no-op: codes already present
    assert np.array_equal(out.codes, audio.codes)


def test_encode_runs_when_codes_absent():
    codec = _FakeCodec(num_codebooks=4)
    audio = _audio_wav(B=2, nsamp=64)
    out = T.EncodeWithCodec(codec).map(audio)
    assert codec.calls == 1
    assert codec.last_waveform.shape == (2, 2, 64)  # got the [B, C, T] audio
    assert out.codes.shape == (2, 64 // 16, 4)


def test_encode_waveform_fn_adapts_layout():
    codec = _FakeCodec()
    audio = _audio_wav(B=1, nsamp=32, ch=2)
    # e.g. downmix to mono [B, 1, nsamp] before the codec
    out = T.EncodeWithCodec(
        codec, waveform_fn=lambda w: w.mean(axis=1, keepdims=True)
    ).map(audio)
    assert codec.last_waveform.shape == (1, 1, 32)
    assert out.codes is not None


# --- PrepareTarget ----------------------------------------------------------

def test_prepare_target_offsets():
    cfg = _tgt_cfg(codebook_size=32, num_extra_tokens=6)
    codes = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4) % 32
    audio = AudioTree(np.zeros((2, 2, 16), np.float32), 48000, codes=codes)
    out = T.PrepareTarget(cfg).map(audio)
    target = out.extras["target"]
    expected = codes + np.arange(4) * 32 + 6
    assert np.array_equal(target, expected)


def test_prepare_target_requires_codes():
    with pytest.raises(ValueError, match="requires codes"):
        T.PrepareTarget(_tgt_cfg()).map(_audio_wav())


def test_prepare_target_tokens_is_backend_neutral():
    # The offset math must also run on a device array (the on-the-fly GPU path).
    import jax.numpy as jnp

    cfg = _tgt_cfg(codebook_size=32, num_extra_tokens=6)
    codes = np.arange(6, dtype=np.int32).reshape(1, 2, 3) % 32
    host = prepare_target_tokens(codes, cfg)
    dev = prepare_target_tokens(jnp.asarray(codes), cfg)
    assert dev.__class__.__module__.startswith(("jax", "jaxlib"))
    assert np.array_equal(np.asarray(dev), host)


# --- PrepareSource ----------------------------------------------------------

def test_prepare_source_concatenates_and_offsets():
    cfgs = (
        SimpleNamespace(key="a", rvq_truncation_level=2, dropout_prob=None, num_extra_tokens=1),
        SimpleNamespace(key="b", rvq_truncation_level=3, dropout_prob=None, num_extra_tokens=2),
    )
    meta = {
        "a": np.ones((1, 5, 2), np.int32),
        "b": np.full((1, 5, 3), 4, np.int32),
    }
    audio = AudioTree(np.zeros((1, 2, 80), np.float32), 48000, extras=meta)
    out = T.PrepareSource(cfgs).random_map(audio, np.random.default_rng(0))
    source = out.extras["source"]
    assert source.shape == (1, 5, 5)  # 2 + 3 channels concatenated
    # offsets: a -> +2 (num_extra=1 + 1), b -> +3 (num_extra=2 + 1)
    assert np.array_equal(source[..., :2], np.ones((1, 5, 2), np.int32) + 2)
    assert np.array_equal(source[..., 2:], np.full((1, 5, 3), 4, np.int32) + 3)


# --- augment_batch (the boundary) -------------------------------------------

def test_augment_batch_encodes_then_prepares_target():
    codec = _FakeCodec(num_codebooks=4)
    audio = _audio_wav(B=2, nsamp=64)
    out = T.augment_batch(None, audio, [T.EncodeWithCodec(codec), T.PrepareTarget(_tgt_cfg())])
    assert codec.calls == 1
    assert out.codes is not None and "target" in out.extras
    assert out.extras["target"].shape == out.codes.shape


def test_augment_batch_uses_present_codes():
    codec = _FakeCodec()
    audio = AudioTree(np.zeros((2, 2, 64), np.float32), 48000,
                    codes=(np.arange(2 * 4 * 4).reshape(2, 4, 4) % 32).astype(np.int32))
    out = T.augment_batch(None, audio, [T.EncodeWithCodec(codec), T.PrepareTarget(_tgt_cfg())])
    assert codec.calls == 0  # used existing codes, no GPU encode
    assert "target" in out.extras


def test_augment_batch_random_map_gets_rng():
    cfgs = (SimpleNamespace(key="a", rvq_truncation_level=2, dropout_prob=None, num_extra_tokens=1),)
    audio = AudioTree(np.zeros((1, 2, 80), np.float32), 48000,
                    extras={"a": np.ones((1, 5, 2), np.int32)})
    out = T.augment_batch(np.random.default_rng(0), audio, [T.PrepareSource(cfgs)])
    assert "source" in out.extras


def test_augment_batch_rejects_unknown_transform():
    with pytest.raises(TypeError, match="Unsupported transform"):
        T.augment_batch(None, _audio_wav(), [object()])


# --- AudioTree container behaviors (metadata + batch_fn) ---------------------

def test_batch_fn_concatenates_fields_and_metadata():
    items = [
        AudioTree(np.full((1, 2, 8), i, np.float32), 48000,
                  codes=np.full((1, 2, 4), i, np.int32),
                  extras={"cond": np.full((1, 2, 3), i, np.int32)})
        for i in range(3)
    ]
    b = AudioTree.batch(items)
    assert b.waveform.shape == (3, 2, 8)
    assert b.codes.shape == (3, 2, 4)
    assert b.extras["cond"].shape == (3, 2, 3)
    assert [int(b.extras["cond"][i, 0, 0]) for i in range(3)] == [0, 1, 2]


def test_batch_fn_keeps_none_codes():
    items = [AudioTree(np.zeros((1, 2, 8), np.float32), 48000) for _ in range(2)]
    b = AudioTree.batch(items)
    assert b.codes is None and b.waveform.shape == (2, 2, 8)


def test_indexing_carries_metadata():
    b = AudioTree(np.zeros((3, 2, 8), np.float32), 48000,
                  extras={"cond": np.arange(3 * 2 * 3).reshape(3, 2, 3).astype(np.int32)})
    item = b[1]
    assert item.extras["cond"].shape == (1, 2, 3)
    assert np.array_equal(item.extras["cond"][0], b.extras["cond"][1])


# --- AudioTreeRandomCrop / AudioTreeMusicCoCaSticky (grain-pipeline transforms) -

def test_crop_token_mode_is_aligned():
    n, cf = 10, 4
    codes = np.tile(np.arange(n)[None, :, None], (1, 1, 4)).astype(np.int32)  # frame i = i
    cond = np.tile(np.arange(n)[None, :, None], (1, 1, 3)).astype(np.int32)
    audio = AudioTree(None, 48000, codes=codes, extras={"cond": cond})
    out = T.AudioTreeRandomCrop(cf).random_map(audio, np.random.default_rng(0))
    assert out.codes.shape == (1, cf, 4) and out.extras["cond"].shape == (1, cf, 3)
    start = int(out.codes[0, 0, 0])
    # contiguous crop, and metadata cropped at the SAME start (aligned).
    assert np.array_equal(out.codes[0, :, 0], np.arange(start, start + cf))
    assert np.array_equal(out.extras["cond"][0, :, 0], np.arange(start, start + cf))


def test_crop_audio_mode_aligns_samples_to_frames():
    n, cf, spf = 10, 4, 16
    # Audio sample t = t, on both channels; channel-major [1, 2, n*spf].
    waveform = np.tile(np.arange(n * spf)[None, None, :], (1, 2, 1)).astype(np.float32)
    cond = np.tile(np.arange(n)[None, :, None], (1, 1, 3)).astype(np.int32)
    audio = AudioTree(waveform, 48000, extras={"cond": cond})  # codes=None (audio mode)
    out = T.AudioTreeRandomCrop(cf).random_map(audio, np.random.default_rng(1))
    assert out.waveform.shape == (1, 2, cf * spf) and out.extras["cond"].shape == (1, cf, 3)
    fstart = int(out.extras["cond"][0, 0, 0])
    assert int(out.waveform[0, 0, 0]) == fstart * spf  # audio crop aligned to frame crop


def test_crop_pads_short_examples():
    n, cf = 3, 5
    audio = AudioTree(None, 48000, codes=np.ones((1, n, 4), np.int32),
                    extras={"cond": np.ones((1, n, 3), np.int32)})
    out = T.AudioTreeRandomCrop(cf).random_map(audio, np.random.default_rng(0))
    assert out.codes.shape == (1, cf, 4) and out.extras["cond"].shape == (1, cf, 3)
    assert np.array_equal(out.codes[0, n:], np.zeros((cf - n, 4), np.int32))  # zero-padded


def test_sticky_repeats_frames():
    from magenta_rt.config import MUSICCOCA

    mulan = np.arange(5 * 2).reshape(1, 5, 2).astype(np.int32)  # distinct frames
    audio = AudioTree(None, 48000, extras={MUSICCOCA.key: mulan})
    out = T.AudioTreeMusicCoCaSticky(1.0).random_map(audio, np.random.default_rng(0))
    sticky = out.extras[MUSICCOCA.key]
    # fully sticky (prob=1.0) -> every frame collapses to frame 0.
    assert np.array_equal(sticky[0], np.tile(mulan[0, 0], (5, 1)))


def test_sticky_noop_without_musiccoca_channel():
    audio = AudioTree(None, 48000, codes=np.zeros((1, 5, 4), np.int32))
    out = T.AudioTreeMusicCoCaSticky(0.5).random_map(audio, np.random.default_rng(0))
    assert out.extras == {}


# --- end-to-end: the AudioTree pipeline produces trainable source/target -----

def test_waveform_pipeline_source_target(tmp_path):
    """create_audiotree_dataset + the trainer boundary yield correctly-shaped,
    valid source/target (the contract the legacy dict pipeline used to provide;
    byte-for-byte equivalence was proven at migration in commit 7804d18)."""
    from magenta_rt.sft.data import create_audiotree_dataset
    from .test_utils import write_fake_tree_dataset
    from magenta_rt.sft.configs import TinyPOCSpec

    spec = TinyPOCSpec()
    write_fake_tree_dataset(
        str(tmp_path), num_files=4, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    ds = create_audiotree_dataset(
        str(tmp_path), batch_size=2, crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config, seed=0,
    )
    # Trainer boundary: build the target (token mode -> no codec needed).
    source, target = T.to_source_target(next(iter(ds)), spec.target_tokens_config)
    crop_frames = 50
    assert source.shape == (2, crop_frames, spec.input_num_channels)
    assert target.shape == (2, crop_frames, spec.target_tokens_config.rvq_truncation_level)
    assert source.dtype == np.int16
    assert target.dtype == np.int16
    assert source.min() >= 0
    assert target.min() >= spec.target_tokens_config.num_extra_tokens - 1


def test_audio_pipeline_encodes_to_target_on_the_fly(tmp_path):
    """Audio examples (no soundstream_tokens) flow through create_audiotree_dataset
    as channel-major waveform; the trainer boundary encodes them to codes via
    the codec and builds the target — the on-the-fly GPU tokenization path."""
    from magenta_rt.sft.data import create_audiotree_dataset
    from .test_utils import write_fake_tree_dataset
    from magenta_rt.sft.configs import TinyPOCSpec

    spec = TinyPOCSpec()
    rvq = spec.target_tokens_config.rvq_truncation_level
    spf = 32  # small samples-per-frame -> fast test
    write_fake_tree_dataset(str(tmp_path), num_files=4, frames_per_file=75,
                       with_audio=True, samples_per_frame=spf, audio_channels=2)
    ds = create_audiotree_dataset(
        str(tmp_path), batch_size=2, crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config, seed=0,
    )
    batch = next(iter(ds))
    crop_frames = 50
    assert batch.codes is None and batch.waveform is not None  # audio mode
    assert batch.waveform.shape == (2, 2, crop_frames * spf)  # cropped + frame-aligned

    class _SpectroStub:
        def waveform_to_codes(self, waveform):
            B, nsamp = waveform.shape[0], waveform.shape[-1]  # [B, C, T]
            return np.zeros((B, nsamp // spf, rvq), np.int32)

    source, target = T.to_source_target(batch, spec.target_tokens_config, codec=_SpectroStub())
    assert source.shape == (2, crop_frames, spec.input_num_channels)
    assert target.shape == (2, crop_frames, rvq)  # audio -> codec -> target
