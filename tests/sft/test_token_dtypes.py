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

"""``prepare_source_tokens`` / ``prepare_target_tokens`` produce **int16**.

The SFT loader keeps token ids at their natural compact width (int16) instead
of upcasting to int32 — lossless, matches the compact storage dtypes, and the
model's embedding lookups are dtype-agnostic. int16 is only safe while every
model's max token id stays below 32,767, so ``test_token_ids_fit_int16`` pins
that headroom: a future codebook/depth/vocab bump that would overflow int16
fails HERE (loudly) instead of silently wrapping a token id into garbage.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("audiotree")  # prepare_* live in sft.data

from magenta_rt.sft.data import prepare_source_tokens, prepare_target_tokens  # noqa: E402

_INT16_MAX = int(np.iinfo(np.int16).max)


def _registry():
    from magenta_rt.nnx.model import MODEL_REGISTRY
    return MODEL_REGISTRY


class _NoDropRng:
    """``random()`` always 1.0 → prepare_source_tokens never drops a channel."""
    def random(self):
        return 1.0


class _AlwaysDropRng:
    """``random()`` always 0.0 → every dropout-eligible channel is masked."""
    def random(self):
        return 0.0


def test_token_ids_fit_int16_for_all_registered_specs():
    """Every registered model's max source AND target token id must fit int16.

    This is the guard that makes the no-upcast loader safe: if a new model has a
    larger codebook / more RVQ depth / a bigger conditioning vocab such that an
    id reaches 32,767, this fails — at which point that path must widen back to
    int32 rather than wrap.
    """
    for name, cls in _registry().items():
        spec = cls()
        tc = spec.target_tokens_config
        target_max = (
            (tc.codebook_size - 1)
            + (tc.rvq_truncation_level - 1) * tc.codebook_size
            + tc.num_extra_tokens
        )
        assert target_max < _INT16_MAX, (
            f"{name}: max target id {target_max} ≥ int16 max {_INT16_MAX} — "
            f"prepare_target_tokens must widen back to int32 for this model"
        )
        for c in spec.input_configs:
            src_max = (c.per_rvq_vocab_size - 1) + c.num_extra_tokens + 1
            assert src_max < _INT16_MAX, (
                f"{name}/{c.key}: max source id {src_max} ≥ int16 max "
                f"{_INT16_MAX} — prepare_source_tokens must widen to int32"
            )


def test_prepare_target_tokens_is_int16_and_correct():
    spec = _registry()["mrt2_small"]()
    tc = spec.target_tokens_config
    rng = np.random.default_rng(0)
    # codes arrive as uint16 from the compact export.
    codes = rng.integers(
        0, tc.codebook_size, size=(8, tc.rvq_truncation_level)
    ).astype(np.uint16)

    out = np.asarray(prepare_target_tokens(codes, tc))
    assert out.dtype == np.int16, out.dtype

    # Independent int64 reference — int16 output must equal it bit for bit
    # (i.e. no overflow/wrap on the way down to int16).
    ref = (
        codes.astype(np.int64)
        + np.arange(tc.rvq_truncation_level) * tc.codebook_size
        + tc.num_extra_tokens
    )
    np.testing.assert_array_equal(out.astype(np.int64), ref)


def test_prepare_source_tokens_is_int16_across_mixed_input_dtypes():
    """Mixed compact storage dtypes (int8 pianoroll, int16 mulan, int32 CFG)
    must yield a single int16 source row with correct values, and the -1
    dropout/mask sentinel must land on the dropout token (``num_extra_tokens``),
    never a wrapped 255/65535."""
    from magenta_rt import config as _cfg

    spec = _registry()["mrt2_small"]()
    cfgs = spec.input_configs
    T = 6
    rng = np.random.default_rng(1)
    example = {}
    for c in cfgs:
        w = c.rvq_truncation_level
        vals = rng.integers(0, c.per_rvq_vocab_size, size=(T, w))
        # Storage dtypes that the compact export writes / synthesizes.
        if c.key == _cfg.PIANOROLL_WITH_ONSETS.key:
            example[c.key] = vals.astype(np.int8)
        elif c.key == _cfg.MUSICCOCA.key:
            example[c.key] = vals.astype(np.int16)
        else:
            example[c.key] = vals.astype(np.int32)  # CFG synthesized as int32

    # No dropout: every channel offset by num_extra+1; output int16, values right.
    out = prepare_source_tokens(example, cfgs, _NoDropRng())
    assert out.dtype == np.int16, out.dtype
    col = 0
    for c in cfgs:
        w = c.rvq_truncation_level
        block = out[:, col:col + w].astype(np.int64)
        ref = example[c.key].astype(np.int64) + (c.num_extra_tokens + 1)
        np.testing.assert_array_equal(block, ref)
        col += w

    # Force dropout on the eligible channels: the -1 sentinel + offset must be
    # exactly the dropout token ``num_extra_tokens`` (signed-safe, no wrap).
    dropped = prepare_source_tokens(example, cfgs, _AlwaysDropRng())
    assert dropped.dtype == np.int16
    col = 0
    for c in cfgs:
        w = c.rvq_truncation_level
        block = dropped[:, col:col + w]
        if c.dropout_prob is not None and c.dropout_prob > 0:
            assert np.all(block == c.num_extra_tokens), (
                f"{c.key}: dropout token wrong (got {np.unique(block)}, "
                f"want {c.num_extra_tokens}) — a -1 likely wrapped"
            )
        col += w


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
