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

"""NNX counterpart to ``tests/sft/test_cfg_token_layout.py`` (which is MLX-only).

Ties the SFT conditioning-token offsets to the **real mrt2_small** encoder
embedding on the JAX/NNX backend, so the check runs on CUDA/CPU boxes (the MLX
one imports ``train_mlx`` and only runs on Apple Silicon).

It pins three things against the loaded checkpoint:

* the regular (non-style) encoder embedding table is sized exactly as the sum of
  per-channel ``per_rvq_vocab_size`` blocks (rounded to a multiple of 128) — so
  the CFG blocks really are ``codebook + num_extra_tokens`` wide (no extra
  dropout row), as shipped;
* the MusicCoCa dequantizer holds ``rvq_levels × per_rvq_vocab_size`` rows;
* SFT training (``prepare_source_tokens``) and inference
  (``conditioning.build_conditioning_rows``, the convention every backend uses)
  emit **identical** conditioning tokens for the same inputs, and every emitted
  index lands inside the loaded embedding table — i.e. fine-tuning conditions
  the model exactly as generation does.

Gated by ``checkpoint`` + ``slow``; skips without the weights.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from magenta_rt import config as _cfg
from magenta_rt import conditioning, paths

pytestmark = [pytest.mark.checkpoint, pytest.mark.slow]


class _KeepRng:
    def random(self):
        return 1.0  # never drop a channel


@pytest.fixture(scope="module")
def loaded():
    ckpt = paths.resolve_checkpoint("mrt2_small.safetensors")
    if not ckpt.exists():
        pytest.skip(f"checkpoint not found: {ckpt}")
    from flax import nnx

    from magenta_rt.nnx import model as nnx_model

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0)
    )
    mrt.load_checkpoint(ckpt)
    from magenta_rt.nnx.model import get_model_class

    spec = get_model_class("mrt2_small")()
    return mrt, spec


def test_regular_embedding_table_size(loaded):
    """The non-style embedding table == round128(Σ per_rvq_vocab + reserved)."""
    mrt, spec = loaded
    regular = mrt.depthformer.encoder.embedding.regular_embedder
    per_channel = []
    for cfg in spec.input_configs[1:]:  # skip MusicCoCa (separate mulan branch)
        per_channel += [cfg.per_rvq_vocab_size] * cfg.rvq_truncation_level
    total = regular.num_reserved_embeddings + sum(per_channel)
    total = (total + 127) // 128 * 128
    assert int(np.asarray(regular.embedding[...]).shape[0]) == total


def test_mulan_dequantizer_size(loaded):
    """The MusicCoCa dequantizer holds rvq_levels × per_rvq_vocab rows."""
    mrt, spec = loaded
    musiccoca_cfg = spec.input_configs[0]
    assert musiccoca_cfg.key == _cfg.MUSICCOCA.key
    mulan = mrt.depthformer.encoder.embedding.mulan_embedder
    rows = int(np.asarray(mulan.mulan_dequantizer.embedding[...]).shape[0])
    assert rows == musiccoca_cfg.rvq_levels * musiccoca_cfg.per_rvq_vocab_size


def test_train_and_inference_tokens_agree_and_are_in_range(loaded):
    """prepare_source_tokens (train) == build_conditioning_rows (inference) for
    the real spec, including CFG, and every index is inside the loaded table."""
    mrt, spec = loaded
    cfgs_spec = spec.input_configs
    rng = np.random.RandomState(0)

    # raw per-channel values (well inside each codebook), non-default CFG.
    raw = {}
    for c in cfgs_spec:
        if c.key == _cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key:
            raw[c.key] = np.array([23, 12], np.int32)
        elif c.key == _cfg.CFG_CONDITIONING_DRUMS.key:
            raw[c.key] = np.array([5], np.int32)
        else:
            raw[c.key] = rng.randint(
                0, c.codebook_size, size=c.rvq_truncation_level
            ).astype(np.int32)

    # inference row (the convention all four backends share).
    inference_row = conditioning.build_conditioning_rows(
        batch_style=[list(raw[_cfg.MUSICCOCA.key])],
        notes=list(raw[_cfg.PIANOROLL_WITH_ONSETS.key]),
        drums=list(raw[_cfg.DRUM_PIANOROLL.key]),
        cfgs=list(raw[_cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.key])
        + list(raw[_cfg.CFG_CONDITIONING_DRUMS.key]),
        num_musiccoca=_cfg.MUSICCOCA.rvq_truncation_level,
        num_notes=_cfg.PIANOROLL_WITH_ONSETS.rvq_truncation_level,
        drum_tokens=_cfg.DRUM_PIANOROLL.rvq_truncation_level,
        cfg_tokens=(_cfg.CFG_CONDITIONING_MUSICCOCA_NOTES.rvq_truncation_level
                    + _cfg.CFG_CONDITIONING_DRUMS.rvq_truncation_level),
        offset=_cfg.NUM_RESERVED_TOKENS + 1,
    )[0, 0]

    # training row.
    T = 2
    example = {k: np.tile(v, (T, 1)) for k, v in raw.items()}
    no_dropout = tuple(
        dataclasses.replace(c, dropout_prob=0.0) if c.dropout_prob is not None
        else c
        for c in cfgs_spec
    )
    from magenta_rt.sft.data import prepare_source_tokens

    train_row = prepare_source_tokens(example, no_dropout, _KeepRng())[0]

    np.testing.assert_array_equal(
        train_row, inference_row,
        err_msg="SFT source tokens diverge from the inference conditioning row",
    )

    # Every emitted index addresses a real row of the loaded encoder embedding.
    mulan = mrt.depthformer.encoder.embedding.mulan_embedder
    regular = mrt.depthformer.encoder.embedding.regular_embedder
    n_style = _cfg.MUSICCOCA.rvq_truncation_level
    style_ids = train_row[:n_style]
    rest_ids = train_row[n_style:]
    # style ids index per-level dequantizer blocks of per_rvq_vocab_size.
    assert int(style_ids.max()) < _cfg.MUSICCOCA.per_rvq_vocab_size
    # regular ids + the module's per-channel offsets must stay in the table.
    offsets = np.asarray(regular._offsets, np.int64)
    table_rows = int(np.asarray(regular.embedding[...]).shape[0])
    assert np.all(rest_ids.astype(np.int64) + offsets < table_rows)
    assert np.all(rest_ids >= 0)
