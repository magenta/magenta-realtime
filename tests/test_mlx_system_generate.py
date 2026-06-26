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

"""Checkpoint-gated tests for the mlx MagentaRT2System.generate surface.

Covers the two behaviors that previously had no test coverage: (1) generate()
populating AudioTree.codes with the RVQ codes it sampled, and (2) the
conditioning broadcast rule (notes/drums/cfgs/temperature/top_k shared across
the batch, or batched [N, ...] per-element). The pure-NumPy broadcast logic has
its own checkpoint-free tests in tests/test_conditioning.py; here we assert the
real system wires it together correctly (region layout, offsets, tokens emit).

Skipped automatically when the mrt2_small checkpoint is unavailable (e.g. CI).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from magenta_rt import paths


@pytest.fixture(scope="module")
def mrt():
    ckpt = paths.resolve_checkpoint("mrt2_small.safetensors")
    if not ckpt.exists():
        pytest.skip(f"checkpoint not found: {ckpt}")
    from magenta_rt.mlx.system import MagentaRT2System

    return MagentaRT2System(size="mrt2_small")


@pytest.fixture(scope="module")
def dims(mrt):
    return (mrt._num_musiccoca_tokens, mrt._num_notes,
            mrt._drum_tokens, mrt._cfg_tokens)


# --- (A) tokens population --------------------------------------------------

def test_generate_populates_tokens(mrt, dims):
    ns, nn, nd, _ = dims
    emb = mrt.embed_styles(["disco funk"])
    mx.random.seed(0)
    wav, _ = mrt.generate(style=emb, notes=[-1] * nn, drums=[-1] * nd,
                          cfgs=[20, 20, 4], frames=4)
    assert wav.codes is not None
    assert wav.codes.shape == (1, 4, 12)
    assert np.issubdtype(wav.codes.dtype, np.integer)
    assert wav.codes.min() >= 0 and wav.codes.max() < mrt._codebook_size
    assert len(np.unique(wav.codes)) > 1  # not degenerate


def test_generate_token_batch_axis_and_indexing(mrt, dims):
    ns, nn, nd, _ = dims
    emb = mrt.embed_styles(["disco funk", "smooth jazz"])
    mx.random.seed(0)
    wav, _ = mrt.generate(style=emb, notes=[-1] * nn, drums=[-1] * nd,
                          cfgs=[20, 20, 4], frames=4)
    assert wav.codes.shape == (2, 4, 12)
    assert wav.waveform.shape[0] == wav.codes.shape[0] == 2
    # __getitem__ keeps the leading batch axis on tokens (rank-aligned w/ waveform)
    item = wav[0]
    assert item.codes.shape == (1, 4, 12)
    assert item.waveform.shape[0] == 1


# --- (B) conditioning broadcast rule ----------------------------------------

def _rows(block):
    return np.array(block.values)[:, 0, :]  # [N, C]


def test_conditioning_shared_gives_identical_rows(mrt, dims):
    ns, nn, nd, _ = dims
    block, _ = mrt._build_conditioning([[0] * ns, [0] * ns],
                                       notes=[-1] * nn, drums=[-1] * nd, cfgs=[20, 20, 4])
    v = _rows(block)
    assert np.array_equal(v[0], v[1])  # backwards-compatible shared behavior


def test_conditioning_per_element_notes(mrt, dims):
    ns, nn, nd, nc = dims
    note_sl = slice(ns, ns + nn)
    cfg_sl = slice(ns + nn + nd, ns + nn + nd + nc)
    notes = np.full((2, nn), -1, dtype=np.int32)
    notes[1, :4] = [2, 2, 0, 1]
    block, _ = mrt._build_conditioning([[0] * ns, [0] * ns],
                                       notes=notes, drums=[-1] * nd, cfgs=[20, 20, 4])
    v = _rows(block)
    assert not np.array_equal(v[0, note_sl], v[1, note_sl])  # notes vary per-element
    assert np.array_equal(v[0, cfg_sl], v[1, cfg_sl])        # cfgs still shared


def test_conditioning_per_element_cfgs_and_scalars(mrt, dims):
    ns, nn, nd, nc = dims
    cfg_sl = slice(ns + nn + nd, ns + nn + nd + nc)
    block, constants = mrt._build_conditioning(
        [[0] * ns, [0] * ns], notes=[-1] * nn, drums=[-1] * nd,
        cfgs=np.array([[20, 20, 4], [0, 0, 0]], dtype=np.int32),
        temperature=[0.5, 1.5], top_k=[10, 40])
    v = _rows(block)
    assert not np.array_equal(v[0, cfg_sl], v[1, cfg_sl])
    assert np.allclose(np.array(constants["temperature"]), [0.5, 1.5])
    assert np.array_equal(np.array(constants["top_k"]), [10, 40])


def test_conditioning_batch_mismatch_raises(mrt, dims):
    ns, nn, _, _ = dims
    with pytest.raises(ValueError, match="does not match the number of styles"):
        mrt._build_conditioning([[0] * ns], notes=np.zeros((2, nn), np.int32))


def test_generate_per_element_cfgs_runs(mrt, dims):
    ns, nn, nd, _ = dims
    emb = mrt.embed_styles(["disco funk", "smooth jazz"])
    mx.random.seed(0)
    wav, _ = mrt.generate(style=emb, notes=[-1] * nn, drums=[-1] * nd,
                          cfgs=[[20, 20, 4], [10, 10, 2]], frames=4)
    assert wav.codes.shape == (2, 4, 12)
