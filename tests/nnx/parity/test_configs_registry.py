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

"""Tests for ``magenta_rt.nnx.configs``: registry round-trips and a
construction smoke for the smallest registered spec.
"""

from __future__ import annotations

import pytest
from flax import nnx

from magenta_rt.nnx.depthformer import EncoderDecoder
from magenta_rt.nnx.configs import (
    MagentaRT2ModelBase,
    MagentaRT2ModelSmall,
    MODEL_REGISTRY,
    get_model_class,
)


def test_registry_has_mrt2():
    assert "mrt2_base" in MODEL_REGISTRY
    assert "mrt2_small" in MODEL_REGISTRY
    assert get_model_class("mrt2_base") is MagentaRT2ModelBase
    assert get_model_class("mrt2_small") is MagentaRT2ModelSmall


def test_get_model_class_unknown_raises():
    with pytest.raises(ValueError):
        get_model_class("not-a-real-spec")


def test_mrt2_target_tokens_config():
    spec = get_model_class("mrt2_small")()
    cfg = spec.target_tokens_config
    assert cfg.rvq_truncation_level == 12
    assert cfg.codebook_size == 1024


def test_mrt2_input_channels():
    # MusicCoCa(12) + onsets(128) + drums(1) + cfg_musiccoca_notes(2) + cfg_drums(1).
    assert get_model_class("mrt2_small")().input_num_channels == 144


@pytest.mark.slow
def test_mrt2_small_from_config_smoke():
    """Construction smoke for the mrt2_small spec (branched MusicCoCa
    embedder + temporal/depth transformers)."""
    spec = get_model_class("mrt2_small")()
    enc_dec = EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    assert enc_dec.encoder is not None
    assert enc_dec.decoder is not None
    assert enc_dec.decoder.num_codebooks == 12
