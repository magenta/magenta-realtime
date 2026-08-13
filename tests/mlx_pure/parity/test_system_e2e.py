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

"""End-to-end smoke test: run the pure-MLX MagentaRT2Sampler system with a
tiny depthformer + SpectroStream, all using ``mlx_pure``. No
``sequence_layers`` involved at runtime.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from magenta_rt.mlx_pure.depthformer import (
    DepthformerDecoder, EncoderDecoder,
)
from magenta_rt.mlx_pure.spectrostream import (
    ResidualVectorQuantizer, SpectroStream,
)
from magenta_rt.mlx_pure.model import MagentaRT2Sampler
from magenta_rt.mlx_pure.transformer import Encoder, Transformer


class _ScaledEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, *, dtype=mx.float32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.scale = float(math.sqrt(dim))

    def __call__(self, ids: mx.array) -> mx.array:
        return self.embedding(ids) * self.scale


def _build_tiny_depthformer():
    dtype = mx.float32
    model_dim = 32
    num_codebooks = 3
    codebook_size = 8
    num_reserved = 4
    vocab_size = num_reserved + num_codebooks * codebook_size

    embedder = _ScaledEmbedding(vocab_size, model_dim, dtype=dtype)
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
    src_embed = nn.Embedding(vocab_size, model_dim)
    encoder = Encoder(
        embedding=src_embed, embedding_dimension=model_dim,
        body=None, param_dtype=dtype, compute_dtype=dtype,
    )
    return EncoderDecoder(encoder=encoder, decoder=decoder), num_reserved, codebook_size, num_codebooks


def _build_tiny_spectrostream(num_codebooks: int, codebook_size: int, embedding_dim: int):
    quantizer = ResidualVectorQuantizer(
        num_quantizers=num_codebooks, num_embeddings=codebook_size,
        embedding_dim=embedding_dim,
    )
    return SpectroStream(
        stft_frame_length=64, stft_frame_step=32, stft_fft_length=64,
        ratios=((1, 2), (2, 1)), mults=(2, 1),
        is_resnet=True, activation_fn=nn.elu,
        num_bins=32, num_channels=2, num_features=embedding_dim,
        causal=True,
        encoder_base_conv_depth=8, encoder_base_conv_size=3,
        decoder_base_conv_depth=8, decoder_base_conv_size=3,
        quantizer=quantizer,
    )


def test_system_pure_e2e(rng_key):
    """Drive MagentaRT2Sampler with everything pure-mlx."""
    enc_dec, num_reserved, codebook_size, num_codebooks = _build_tiny_depthformer()
    embedding_dim = 32  # match ResidualVectorQuantizer.embedding_dim
    # Note: in production num_features (SpectroStream input feature dim) and
    # num_codebooks both feed into the RVQ embedding layout. For this smoke
    # test we just need shapes to chain.
    ss = _build_tiny_spectrostream(num_codebooks, codebook_size, embedding_dim)

    system = MagentaRT2Sampler(
        depthformer_model=enc_dec,
        spectrostream=ss,
        num_reserved_tokens=num_reserved,
        codebook_size=codebook_size,
        int16_outputs=True,
    )

    state = system.make_initial_state(batch_size=1, seed=0)
    sub = mx.random.split(rng_key, 4)
    for t in range(3):
        # Source token sequence (one frame at a time) into the encoder.
        source_tokens = mx.random.randint(0, num_reserved + num_codebooks * codebook_size, (1, 1), key=sub[t])
        waveform, state = system.step(
            state,
            source_tokens=source_tokens,
            temperature=1.0,
            top_k=10,
        )
        assert waveform.dtype == mx.int16
        # Audio should come out as [B, T_audio] (or [B, T_audio, C]).
        assert waveform.shape[0] == 1
        assert waveform.ndim in (2, 3)
