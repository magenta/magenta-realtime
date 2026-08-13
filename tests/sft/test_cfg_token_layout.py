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

"""Gated guard: the SFT conditioning-token offsets match the pretrained
checkpoint's encoder embedding layout — including the reserved *dropout token*.

Every conditioning stream reserves a dropout token at index ``num_extra_tokens``
— ``prepare_source_tokens`` (and every generation path) offsets real tokens by a
uniform ``num_extra_tokens + 1``; that dropout row is the learned *unconditional*
representation the model falls back to under classifier-free guidance (CFG). The
no-dropout CFG channels are offset by ``+1`` too (the slot is reserved
table-wide), even though they never emit it. This is the only training-recipe detail in the
conditioning path that inference parity can NOT catch: parity tests only ever
feed *conditional* tokens, so a wrong dropout-token index — or a missing
reserved slot — would pass every parity test yet silently train and condition
the wrong embedding row.

This ties ``prepare_source_tokens``' offset arithmetic to the actual row counts
of the loaded mrt2_small checkpoint, for MusicCoCa, pianoroll, and drum-pianoroll
alike. See the "CFG conditioning-dropout" caveat in PR-NNX-MLX-SFT.md. (It does
NOT — and can not from the checkpoint alone — verify the *policy* around the
dropout token: independent-vs-joint drop, granularity, or rate. Those need the
original internal training recipe.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from magenta_rt import paths

pytest.importorskip("audiotree")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "sft"))

pytestmark = [pytest.mark.checkpoint, pytest.mark.slow]

CHECKPOINT = paths.resolve_checkpoint("mrt2_small.safetensors")


class _ConstRng:
    """Stub Generator whose ``random()`` is constant — forces every
    ``prepare_source_tokens`` dropout coin to drop (0.0) or keep (1.0)."""

    def __init__(self, value: float):
        self._value = value

    def random(self):
        return self._value


def test_pretrained_embedding_layout_matches_prepare_source_tokens():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT}")

    # ``train_mlx`` imports ``mlx.core`` at module scope, so this test only runs
    # where MLX is installed (Apple Silicon), not on a Linux/CUDA host.
    pytest.importorskip("mlx.core")

    import train_mlx as sft_mlx  # type: ignore  # noqa: E402
    from magenta_rt.mlx_pure.configs import get_model_class
    from magenta_rt.sft.data import prepare_source_tokens

    spec = get_model_class("mrt2_small")()
    model = sft_mlx.build_model(
        spec, checkpoint_path=str(CHECKPOINT), model_name="mrt2_small",
    )
    input_configs = spec.input_configs
    musiccoca_cfg = input_configs[0]
    assert musiccoca_cfg.key == "mulan_tokens_25hz"

    # --- (1) Per-modality vocab and the uniform dropout-slot offset ------------
    # ``prepare_source_tokens`` offsets real tokens by ``num_extra_tokens + 1``
    # for EVERY channel — matching the generation convention shared by all four
    # backends (jax/mlx ``_build_conditioning`` and
    # ``conditioning.build_conditioning_rows`` both use ``NUM_RESERVED_TOKENS+1``),
    # which reserves the learned-unconditional dropout slot at ``num_extra_tokens``.
    for cfg in input_configs:
        offset = cfg.num_extra_tokens + 1
        if cfg.dropout_prob is not None:
            # The block holds [reserved..][dropout][real codebook): exact fit.
            assert cfg.per_rvq_vocab_size == cfg.codebook_size + offset, cfg.key
            dropout_idx = -1 + offset  # what a dropped (-1) token maps to
            assert dropout_idx == cfg.num_extra_tokens
            assert 0 <= dropout_idx < offset <= cfg.per_rvq_vocab_size, cfg.key
        else:
            # The CFG channels carry no dropout token, so the shipped embedding
            # block is one row short of ``codebook + offset`` — yet inference
            # still offsets them by ``+1`` (the slot is reserved table-wide), so
            # the single max CFG value spills into the next channel's block. This
            # boundary quirk is inherited from jax; CFG strengths used in practice
            # sit well inside the block.
            assert cfg.per_rvq_vocab_size == cfg.codebook_size + offset - 1, cfg.key

    # --- (2) Loaded MusicCoCa dequantizer: rvq_levels blocks of per_rvq_vocab ---
    mulan = model.encoder.embedding.mulan_embedder
    assert mulan.mulan_dequantizer.weight.shape[0] == (
        musiccoca_cfg.rvq_levels * musiccoca_cfg.per_rvq_vocab_size
    )

    # --- (3) Loaded regular (pianoroll / drum / cfg) flat table size -----------
    regular = model.encoder.embedding.regular_embedder
    per_channel = []
    for cfg in input_configs[1:]:
        per_channel += [cfg.per_rvq_vocab_size] * cfg.rvq_truncation_level
    total = regular.num_reserved_embeddings + sum(per_channel)
    total = (total + 127) // 128 * 128  # round-128 padding (module default)
    assert regular.embedding.shape[0] == total

    # --- (4) Functional: prepared indices land inside each per-channel block ---
    # Use MAX real tokens so the dropout token (index num_extra_tokens) is
    # unambiguously distinct from any real token (offset .. offset+codebook).
    example = {
        cfg.key: np.full((4, cfg.rvq_truncation_level), cfg.codebook_size - 1, np.int32)
        for cfg in input_configs
    }

    dropped = prepare_source_tokens(example, input_configs, _ConstRng(0.0))
    kept = prepare_source_tokens(example, input_configs, _ConstRng(1.0))
    col = 0
    for cfg in input_configs:
        w = cfg.rvq_truncation_level
        d_block = dropped[:, col:col + w]
        k_block = kept[:, col:col + w]
        offset = cfg.num_extra_tokens + 1  # uniform (matches generation paths)
        max_real = (cfg.codebook_size - 1) + offset
        assert np.all(k_block == max_real)
        if cfg.dropout_prob is not None:
            # Block fits exactly: max real token is the last in-block row.
            assert max_real == cfg.per_rvq_vocab_size - 1
            # Dropped -> the dropout token, distinct from every real token.
            assert np.all(d_block == cfg.num_extra_tokens)
            assert np.all(d_block < cfg.per_rvq_vocab_size)
        else:
            # CFG: no dropout row, so the top real token sits AT per_rvq_vocab
            # (one past the block) — the inherited boundary quirk. No drop coin.
            assert max_real == cfg.per_rvq_vocab_size
            assert np.all(d_block == max_real)
        col += w
    assert col == sum(c.rvq_truncation_level for c in input_configs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
