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

"""Tests for ``magenta_rt.mlx_pure.system.MagentaRT2System`` (tiny, random weights).

Exercises the jax/mlx-shaped system API — ``generate -> (AudioTree, state)``
with a functionally-threaded ``SamplerState`` — on a hand-built tiny model
(the parity e2e builder uses a plain ``nn.Embedding`` encoder; the system
derives the channel count from ``encoder.embedding.num_channels``, so this
builder uses a 1-channel ``MultiChannelEmbedding`` instead).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from magenta_rt.mlx_pure.depthformer import DepthformerDecoder, EncoderDecoder
from magenta_rt.mlx_pure.model import MagentaRT2Sampler
from magenta_rt.mlx_pure.system import MagentaRT2System
from magenta_rt.mlx_pure.transformer import (
    Encoder, MultiChannelEmbedding, Transformer,
)
from tests.mlx_pure.parity.test_system_e2e import _build_tiny_spectrostream


class _ScaledEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = float(math.sqrt(dim))

    def __call__(self, ids: mx.array) -> mx.array:
        return self.embedding(ids) * self.scale


def _build_tiny_model() -> MagentaRT2Sampler:
    dtype = mx.float32
    model_dim = 32
    num_codebooks = 3
    codebook_size = 8
    num_reserved = 4
    vocab_size = num_reserved + num_codebooks * codebook_size

    embedder = _ScaledEmbedding(vocab_size, model_dim)
    temporal = Transformer(
        num_layers=1, model_dim=model_dim, num_heads=4, units_per_head=8,
        ffn_dim=64, max_past_horizon=3, num_sinks=1,
        use_cross_attention=True, cross_attn_source_features=model_dim,
        cross_attn_max_past_horizon=3,
        compute_dtype=dtype, param_dtype=dtype,
    )
    depth = Transformer(
        num_layers=1, model_dim=model_dim, num_heads=4, units_per_head=8,
        ffn_dim=64, max_past_horizon=num_codebooks, num_sinks=0,
        use_cross_attention=False,
        compute_dtype=dtype, param_dtype=dtype,
    )
    decoder = DepthformerDecoder(
        num_codebooks=num_codebooks, codebook_size=codebook_size,
        num_reserved_tokens=num_reserved, vocab_size=vocab_size,
        sos_id=0, num_active_codebooks=None,
        model_dim=model_dim, depth_dim=model_dim,
        temporal=temporal, depth=depth, depth_input_adapter=None,
        embedder=embedder, compute_dtype=dtype, param_dtype=dtype,
    )
    enc_embedding = MultiChannelEmbedding(
        dimension=model_dim,
        num_embeddings_per_channel=[64],
        num_channels=1,
        reduction_fn=mx.mean,
        compute_dtype=dtype, param_dtype=dtype,
        round_num_embeddings_to_multiple_of_128=False,
    )
    encoder = Encoder(
        embedding=enc_embedding, embedding_dimension=model_dim,
        body=None, compute_dtype=dtype, param_dtype=dtype,
    )
    enc_dec = EncoderDecoder(encoder=encoder, decoder=decoder)
    ss = _build_tiny_spectrostream(num_codebooks, codebook_size, model_dim)
    return MagentaRT2Sampler(
        depthformer_model=enc_dec,
        spectrostream=ss,
        num_reserved_tokens=num_reserved,
        codebook_size=codebook_size,
        int16_outputs=False,
    )


def _make_system(seed: int = 0) -> MagentaRT2System:
    return MagentaRT2System(
        size="tiny",
        model=_build_tiny_model(),
        restore=False,
        temperature=1.0,
        top_k=4,
        seed=seed,
    )


def test_generate_shapes_and_tokens():
    sys = _make_system()
    frames = 3
    audio, state = sys.generate(frames=frames)

    # Channel-major [N, C, T] audio.
    assert audio.waveform.shape[0] == 1
    assert audio.waveform.ndim == 3
    # The decoder lookahead may drop leading output, so the chunk length is
    # not necessarily frames-divisible — just require non-empty audio.
    assert audio.waveform.shape[-1] > 0
    assert audio.waveform.dtype == np.float32
    assert np.all(np.isfinite(audio.waveform))
    assert np.abs(audio.waveform).max() <= 1.0

    # Codes are the per-codebook RVQ indices that produced the audio.
    assert audio.codes.shape == (1, frames, 3)
    assert audio.codes.min() >= 0
    assert audio.codes.max() < 8  # tiny codebook_size

    assert state is not None  # SamplerState for continuation


def test_state_continuation_matches_one_shot():
    """generate(2) + generate(2, state) == generate(4) — the threaded
    SamplerState (plus codec module buffers) continues the stream exactly."""
    sys = _make_system()

    wav_full, _ = sys.generate(frames=4)

    wav_a, state = sys.generate(frames=2)  # state=None -> fresh stream
    wav_b, _ = sys.generate(frames=2, state=state)

    np.testing.assert_allclose(
        np.concatenate([wav_a.waveform, wav_b.waveform], axis=-1),
        wav_full.waveform,
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.concatenate([wav_a.codes, wav_b.codes], axis=1),
        wav_full.codes,
    )


def test_per_element_temperature_rejected():
    sys = _make_system()
    with pytest.raises(ValueError, match="per-element"):
        sys.generate(frames=1, temperature=[1.0, 2.0])
