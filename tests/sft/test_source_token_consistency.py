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

"""SFT source tokens must match the generation conditioning convention.

The mrt2 checkpoints are CFG-distilled: conditioning is fed as trained tokens,
offset into a shared encoder embedding. Every generation path — ``jax`` and
``mlx`` ``_build_conditioning`` and the shared
``magenta_rt.conditioning.build_conditioning_rows`` used by ``nnx`` /
``mlx_pure`` — offsets the whole conditioning row by ``NUM_RESERVED_TOKENS + 1``
(the ``+1`` reserves the learned-unconditional *dropout* slot for **every**
channel, including the no-dropout CFG-strength channels).

SFT training builds the same conditioning a different way
(``sft.data.prepare_source_tokens`` over per-channel arrays). If the two
disagree on a single channel's offset, the model is fine-tuned against
conditioning rows it is never given at inference — silently, since inference
parity never exercises the training token-prep. This test pins them together,
channel for channel, with **non-default CFG** so the CFG columns are checked.
It is checkpoint-free (pure token arithmetic) so it runs in CI; it is the guard
that would have caught the CFG off-by-one (``prepare_source_tokens`` used ``+0``
for the no-dropout CFG channels while every generation path used ``+1``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from magenta_rt import config as _cfg
from magenta_rt import conditioning

pytest.importorskip("audiotree")  # prepare_source_tokens lives in sft.data

from magenta_rt.sft.data import prepare_source_tokens  # noqa: E402

# The full mrt2 source spec, in the order the encoder concatenates channels.
SOURCE_CONFIGS = (
    _cfg.MUSICCOCA,                       # 12 style cols
    _cfg.PIANOROLL_WITH_ONSETS,           # 128 note cols
    _cfg.DRUM_PIANOROLL,                  # 1 drum col
    _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES,  # 2 cfg cols (musiccoca, notes)
    _cfg.CFG_CONDITIONING_DRUMS,          # 1 cfg col (drums)
)


class _KeepRng:
    """``random()`` always 1.0 so prepare_source_tokens never drops a channel."""

    def random(self):
        return 1.0


def test_prepare_source_tokens_matches_build_conditioning_rows():
    """Per-frame SFT source tokens == the inference conditioning row, for all
    channels including non-default CFG (the convention every backend uses)."""
    rng = np.random.RandomState(0)
    num_style = _cfg.MUSICCOCA.rvq_truncation_level     # 12
    num_notes = _cfg.PIANOROLL_WITH_ONSETS.rvq_truncation_level  # 128
    num_drums = _cfg.DRUM_PIANOROLL.rvq_truncation_level         # 1
    cfg_mc = _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.rvq_truncation_level  # 2
    cfg_dr = _cfg.CFG_CONDITIONING_DRUMS.rvq_truncation_level            # 1

    # Distinct, non-trivial raw values per channel (well inside each codebook).
    style = rng.randint(0, _cfg.MUSICCOCA.codebook_size, size=num_style)
    notes = rng.randint(0, _cfg.PIANOROLL_WITH_ONSETS.codebook_size, size=num_notes)
    drums = rng.randint(0, _cfg.DRUM_PIANOROLL.codebook_size, size=num_drums)
    # Non-default CFG strengths -> raw token indices (the columns we care about).
    cfg_musiccoca, cfg_notes, cfg_drums = 23, 12, 5  # arbitrary, in range
    cfgs = [cfg_musiccoca, cfg_notes, cfg_drums]      # order: mc, notes, drums

    # --- inference convention (jax/mlx/build_conditioning_rows) ---
    inference_row = conditioning.build_conditioning_rows(
        batch_style=[list(style)],
        notes=list(notes),
        drums=list(drums),
        cfgs=cfgs,
        num_musiccoca=num_style,
        num_notes=num_notes,
        drum_tokens=num_drums,
        cfg_tokens=cfg_mc + cfg_dr,
        offset=_cfg.NUM_RESERVED_TOKENS + 1,
    )[0, 0]  # [C]

    # --- SFT training token-prep over the same raw values ---
    T = 3
    example = {
        _cfg.MUSICCOCA.key: np.tile(style, (T, 1)).astype(np.int32),
        _cfg.PIANOROLL_WITH_ONSETS.key: np.tile(notes, (T, 1)).astype(np.int32),
        _cfg.DRUM_PIANOROLL.key: np.tile(drums, (T, 1)).astype(np.int32),
        _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key:
            np.tile([cfg_musiccoca, cfg_notes], (T, 1)).astype(np.int32),
        _cfg.CFG_CONDITIONING_DRUMS.key:
            np.full((T, cfg_dr), cfg_drums, np.int32),
    }
    # Disable the random per-channel dropout so we compare the offset arithmetic.
    no_dropout = tuple(
        dataclasses.replace(c, dropout_prob=0.0) if c.dropout_prob is not None
        else c
        for c in SOURCE_CONFIGS
    )
    source = prepare_source_tokens(example, no_dropout, _KeepRng())  # [T, C]

    assert source.shape == (T, inference_row.shape[0]), (
        f"shape mismatch: {source.shape} vs row {inference_row.shape}"
    )
    # Every frame equals the (time-constant) inference conditioning row.
    for t in range(T):
        np.testing.assert_array_equal(
            source[t], inference_row,
            err_msg=(
                "SFT source tokens diverge from the inference conditioning row "
                "(check per-channel offsets in prepare_source_tokens vs "
                "build_conditioning_rows)"
            ),
        )

    # Pin the CFG columns explicitly: they are the last cfg_mc + cfg_dr columns,
    # and the regression was exactly here.
    n_cfg = cfg_mc + cfg_dr
    offset = _cfg.NUM_RESERVED_TOKENS + 1
    expected_cfg = np.array(cfgs, np.int32) + offset
    np.testing.assert_array_equal(source[0, -n_cfg:], expected_cfg)


def test_dropped_channels_map_to_the_dropout_token():
    """A dropout-bearing channel that drops (-1) lands on its dropout token at
    index ``num_extra_tokens`` — distinct from every real token (which now start
    at ``num_extra_tokens + 1``)."""
    T = 2
    example = {
        c.key: np.zeros((T, c.rvq_truncation_level), np.int32)
        for c in SOURCE_CONFIGS
    }
    # Force every dropout-bearing channel to drop.
    forced = tuple(
        dataclasses.replace(c, dropout_prob=1.0) if c.dropout_prob is not None
        else c
        for c in SOURCE_CONFIGS
    )
    source = prepare_source_tokens(example, forced, _KeepRng_drop())
    col = 0
    for c in SOURCE_CONFIGS:
        w = c.rvq_truncation_level
        block = source[:, col:col + w]
        if c.dropout_prob is not None:
            assert np.all(block == c.num_extra_tokens), c.key
        else:
            # No dropout slot used; real token 0 -> num_extra_tokens + 1.
            assert np.all(block == c.num_extra_tokens + 1), c.key
        col += w


class _KeepRng_drop:
    """``random()`` always 0.0 so a ``dropout_prob=1.0`` channel always drops."""

    def random(self):
        return 0.0
