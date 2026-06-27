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

"""Tests for ``magenta_rt.audio`` — the module-level helpers operating on
``audiotree.AudioTree`` (imported straight from ``audiotree``; the container
is intentionally not re-exported here)."""

from __future__ import annotations

import numpy as np

from audiotree import AudioTree

from magenta_rt import audio


def _tree(batch: int = 3, channels: int = 2, samples: int = 16) -> AudioTree:
    rng = np.random.default_rng(0)
    return AudioTree(
        rng.standard_normal((batch, channels, samples)).astype(np.float32),
        48_000,
        codes=rng.integers(0, 8, size=(batch, 2, 4), dtype=np.int32),
        metadata={"style": rng.integers(0, 5, size=(batch, 3), dtype=np.int32)},
    )


def test_audio_module_does_not_reexport_the_container():
    assert not hasattr(audio, "AudioTree")
    assert not hasattr(audio, "Waveform")


def test_iteration_terminates_and_keeps_batch_axis():
    tree = _tree(3)
    items = list(tree)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert item.waveform.shape == (1, 2, 16)
        np.testing.assert_array_equal(item.waveform[0], tree.waveform[i])
        np.testing.assert_array_equal(item.codes[0], tree.codes[i])


def test_apply_gain_and_peak_normalize():
    tree = _tree(1)
    doubled = audio.apply_gain(tree, 2.0)
    np.testing.assert_allclose(doubled.waveform, tree.waveform * 2.0, rtol=1e-6)

    normed = audio.peak_normalize(tree, max_value=0.5)
    assert abs(audio.peak_amplitude(normed) - 0.5) < 1e-6


def test_concatenate_time_axis():
    a, b = _tree(2), _tree(2)
    cat = audio.concatenate([a, b])
    assert cat.waveform.shape == (2, 2, 32)  # time-last concat
    assert cat.codes.shape == (2, 4, 4)      # codes frame axis (axis 1)
    np.testing.assert_array_equal(cat.waveform[..., :16], a.waveform)
    np.testing.assert_array_equal(cat.waveform[..., 16:], b.waveform)


def test_compute_rms_uses_channel_major_layout():
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    tree = AudioTree.create(sine, sr)  # [T] -> [1, 1, T]
    _, rms = audio.compute_rms(tree)
    # RMS of a 0.5-amplitude sine is ~0.354.
    assert abs(float(np.median(rms)) - 0.354) < 0.02
