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

"""Unit tests for the generate() conditioning broadcast rule.

Pure-NumPy and checkpoint-free: exercises ``broadcast_rows`` / ``broadcast_scalar``
directly so the shared-vs-per-element shaping (the bit ``generate`` relies on for
batched, per-style conditioning) is covered without standing up a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from magenta_rt import conditioning


# --- broadcast_rows ---------------------------------------------------------

def test_rank1_is_shared_across_batch():
    out = conditioning.broadcast_rows([1, 2, 3], expected_len=3, batch_size=4, name="x")
    assert out.shape == (4, 3)
    assert out.dtype == np.int32
    # Every row identical (broadcast), matching the legacy shared behavior.
    assert np.array_equal(out, np.tile([1, 2, 3], (4, 1)))


def test_rank2_single_row_is_shared():
    out = conditioning.broadcast_rows([[5, 6]], expected_len=2, batch_size=3, name="x")
    assert out.shape == (3, 2)
    assert np.array_equal(out, np.tile([5, 6], (3, 1)))


def test_rank2_full_batch_is_per_element():
    rows = [[1, 1], [2, 2], [3, 3]]
    out = conditioning.broadcast_rows(rows, expected_len=2, batch_size=3, name="x")
    assert out.shape == (3, 2)
    # Per-element values preserved exactly.
    assert np.array_equal(out, np.asarray(rows, dtype=np.int32))


def test_result_is_contiguous_int32_even_from_broadcast():
    out = conditioning.broadcast_rows([1, 2], expected_len=2, batch_size=8, name="x")
    # np.broadcast_to returns a non-writable 0-stride view; we must materialize.
    assert out.flags["C_CONTIGUOUS"]
    assert out.dtype == np.int32
    out[0, 0] = 9  # would raise if still a broadcast view
    assert out[1, 0] == 1  # rows are independent, not aliased


def test_rank1_wrong_length_raises():
    with pytest.raises(ValueError, match="expected 3 values, got 2"):
        conditioning.broadcast_rows([1, 2], expected_len=3, batch_size=2, name="notes")


def test_rank2_wrong_last_axis_raises():
    with pytest.raises(ValueError, match="last axis of 3"):
        conditioning.broadcast_rows([[1, 2], [3, 4]], expected_len=3, batch_size=2, name="notes")


def test_rank2_batch_mismatch_raises():
    with pytest.raises(ValueError, match="does not match the number of styles"):
        conditioning.broadcast_rows([[1], [2], [3]], expected_len=1, batch_size=2, name="drums")


def test_rank3_raises():
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        conditioning.broadcast_rows(np.zeros((2, 2, 2)), expected_len=2, batch_size=2, name="x")


# --- broadcast_scalar -------------------------------------------------------

def test_scalar_is_broadcast():
    out = conditioning.broadcast_scalar(1.3, batch_size=3, name="temperature", dtype=np.float32)
    assert out.shape == (3,) and out.dtype == np.float32
    assert np.allclose(out, 1.3)


def test_length1_vector_is_broadcast():
    out = conditioning.broadcast_scalar([7], batch_size=4, name="top_k", dtype=np.int32)
    assert out.shape == (4,) and np.array_equal(out, [7, 7, 7, 7])


def test_full_length_vector_is_per_element():
    out = conditioning.broadcast_scalar([1.0, 2.0, 3.0], batch_size=3, name="temperature", dtype=np.float32)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_scalar_batch_mismatch_raises():
    with pytest.raises(ValueError, match="does not match the number of styles"):
        conditioning.broadcast_scalar([1.0, 2.0], batch_size=3, name="temperature", dtype=np.float32)


def test_scalar_rank2_raises():
    with pytest.raises(ValueError, match="scalar or rank-1"):
        conditioning.broadcast_scalar([[1.0], [2.0]], batch_size=2, name="temperature", dtype=np.float32)
