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

"""TreeDataSource-format SFT pipeline tests.

The audiotree ``TreeWriter`` export (manifest.json + one memmap per leaf) is
the on-disk format for the SFT example schema; ``create_audiotree_dataset``
reads it natively. Skipped when the ``audiotree`` package (dev branch) is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("audiotree")
from audiotree.sources import TreeDataSource

from magenta_rt.sft import create_audiotree_dataset, to_source_target
from magenta_rt.sft.configs import TinyPOCSpec


from .test_utils import write_fake_tree_dataset


@pytest.mark.parametrize("with_audio", [False, True])
def test_tree_source_reconstructs_audiotrees(tmp_path, with_audio):
    """TreeWriter records reconstruct as batch-1 AudioTrees with the SFT
    schema: target codes in ``codes`` (or audio in channel-major
    ``waveform``) and conditioning channels in ``metadata``."""
    frames = 75
    write_fake_tree_dataset(
        str(tmp_path), num_files=3, frames_per_file=frames, seed=0,
        with_audio=with_audio, samples_per_frame=32,
    )
    src = TreeDataSource(str(tmp_path))
    assert len(src) == 3

    for i in range(len(src)):
        record = src[i]
        for value in record.extras.values():
            assert value.shape[0] == 1  # batch-1 leading axis
        if with_audio:
            assert record.codes is None
            assert record.waveform.shape == (1, 2, frames * 32)
        else:
            assert record.waveform is None
            assert record.codes.shape[:2] == (1, frames)
            assert record.codes.dtype == np.int32


def test_tree_pipeline_shapes_and_dtypes(tmp_path):
    """create_audiotree_dataset reads the manifest and yields the expected
    batch shapes/dtypes (mirrors test_poc's pipeline test)."""
    spec = TinyPOCSpec()
    write_fake_tree_dataset(
        str(tmp_path),
        num_files=4,
        frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    ds = create_audiotree_dataset(
        str(tmp_path),
        batch_size=2,
        crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config,
        seed=0,
    )
    batch = next(iter(ds))
    source, target = to_source_target(batch, spec.target_tokens_config)
    crop_frames = int(2 * 25)
    assert source.shape == (2, crop_frames, spec.input_num_channels)
    assert target.shape == (
        2, crop_frames, spec.target_tokens_config.rvq_truncation_level,
    )
    assert source.dtype == np.int16
    assert target.dtype == np.int16
    assert source.min() >= 0
    assert target.min() >= spec.target_tokens_config.num_extra_tokens - 1
