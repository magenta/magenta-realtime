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

"""Smoke tests for the pure depthformer.

NOTE: A full-sequence parity test against
``magenta_rt.mlx.depthformer.MultivariateDecoder.layer_with_emits`` is
not practical: sl's training-time path calls ``self.depth_body(...)``
which raises ``TypeError: 'Serial' object is not callable``, and uses
``einops.rearrange`` on ``mx.array`` (no MLX einops backend).
Production exercises only the streaming ``step_with_emits`` path.

These tests verify that the pure :class:`DepthformerDecoder` can be
constructed, runs end-to-end without errors on representative shapes,
and produces sane outputs (right shape, finite values).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from magenta_rt.mlx_pure.depthformer import DepthformerDecoder, EncoderDecoder
from magenta_rt.mlx_pure.transformer import Encoder, Transformer


class _ScaledEmbedding(nn.Module):
    """Embedding + sqrt(d) scale. Mirrors sl decoder embedder layout."""

    def __init__(self, vocab_size, dim, *, dtype=mx.float32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = float(math.sqrt(dim))

    def __call__(self, ids):
        return self.embedding(ids) * self.scale


def _build_decoder(*, B=1, num_codebooks=3, codebook_size=8, num_reserved=4,
                   model_dim=32, depth_dim=32, num_heads=4, units_per_head=8,
                   ffn_dim=64, max_past=3, cross_max_past=3, num_sinks=1,
                   dtype=mx.float32):
    vocab_size = num_reserved + num_codebooks * codebook_size
    embedder = _ScaledEmbedding(vocab_size, model_dim, dtype=dtype)
    temporal = Transformer(
        num_layers=1, model_dim=model_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=max_past, num_sinks=num_sinks,
        use_cross_attention=True, cross_attn_source_features=model_dim,
        cross_attn_max_past_horizon=cross_max_past,
        compute_dtype=dtype, param_dtype=dtype,
    )
    depth = Transformer(
        num_layers=1, model_dim=depth_dim, num_heads=num_heads,
        units_per_head=units_per_head, ffn_dim=ffn_dim,
        max_past_horizon=num_codebooks, num_sinks=0,
        use_cross_attention=False,
        compute_dtype=dtype, param_dtype=dtype,
    )
    return DepthformerDecoder(
        num_codebooks=num_codebooks, codebook_size=codebook_size,
        num_reserved_tokens=num_reserved, vocab_size=vocab_size,
        sos_id=0, num_active_codebooks=None,
        model_dim=model_dim, depth_dim=depth_dim,
        temporal=temporal, depth=depth, depth_input_adapter=None,
        embedder=embedder, compute_dtype=dtype, param_dtype=dtype,
    )


def test_depthformer_call_runs_and_shapes(rng_key):
    pure = _build_decoder()
    B, T = 1, 4
    Q = pure.num_codebooks
    V = pure.vocab_size
    tokens = mx.random.randint(pure.num_reserved_tokens, V, (B, T, Q), key=rng_key)
    source = mx.random.normal((B, T, pure.model_dim), key=mx.random.split(rng_key)[0]) * 0.1

    logits = pure(tokens, encoded_source=source)
    assert logits.shape == (B, T, Q, V), logits.shape
    assert mx.all(mx.isfinite(logits)).item()


def test_depthformer_step_runs_and_shapes(rng_key):
    pure = _build_decoder()
    B = 1
    state = pure.make_initial_state(B, seed=0)
    source_frame = mx.random.normal((B, 1, pure.model_dim), key=rng_key) * 0.1

    sampled, new_state = pure.step(
        state,
        encoded_source=source_frame,
        temperature=1.0,
        top_k=10,
    )
    assert sampled.shape == (B, 1, pure.num_codebooks)
    # Sampled tokens must be in the valid range for each codebook.
    sampled_np = np.array(sampled)
    for q in range(pure.num_codebooks):
        lo = pure.num_reserved_tokens + q * pure.codebook_size
        hi = lo + pure.codebook_size
        assert ((sampled_np[..., q] >= lo) & (sampled_np[..., q] < hi)).all(), (
            f"codebook {q} samples out of range {lo}..{hi}: {sampled_np[..., q]}"
        )
    assert new_state.step.item() == 1


def test_depthformer_step_multi_frame(rng_key):
    """Multi-frame streaming. Ensures the temporal cache wraps without errors."""
    pure = _build_decoder(max_past=3, cross_max_past=3)
    B = 1
    state = pure.make_initial_state(B, seed=1)
    rng_keys = mx.random.split(rng_key, 8)
    for t in range(8):
        source_frame = mx.random.normal((B, 1, pure.model_dim), key=rng_keys[t]) * 0.1
        sampled, state = pure.step(
            state, encoded_source=source_frame, temperature=1.0, top_k=10
        )
        assert sampled.shape == (B, 1, pure.num_codebooks)


def test_encoder_decoder_smoke(rng_key):
    """End-to-end EncoderDecoder smoke test."""
    pure_dec = _build_decoder()
    # Encoder: an mlx_pure.Encoder wrapping a simple Embedding + identity body.
    src_embed = nn.Embedding(pure_dec.vocab_size, pure_dec.model_dim)
    encoder = Encoder(
        embedding=src_embed,
        embedding_dimension=pure_dec.model_dim,
        body=None,
        param_dtype=mx.float32,
    )
    ed = EncoderDecoder(encoder=encoder, decoder=pure_dec)

    state = ed.make_initial_state(batch_size=1, seed=2)
    sub = mx.random.split(rng_key, 4)
    for t in range(3):
        source_tokens = mx.random.randint(0, pure_dec.vocab_size, (1, 1), key=sub[t])
        source_frame = ed.encode(source_tokens)
        sampled, state = ed.step(state, source_frame=source_frame, temperature=1.0)
        assert sampled.shape == (1, 1, pure_dec.num_codebooks)
