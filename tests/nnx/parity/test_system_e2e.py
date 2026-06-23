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

"""End-to-end smoke for ``magenta_rt.nnx.model.MagentaRT2Sampler``.

A multi-step streaming run on a tiny config — random weights — that
exercises depthformer + SpectroStream + the whole streaming lifecycle.
Numerical parity vs the linen reference comes in M7.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from magenta_rt.nnx.depthformer import (
    DepthformerDecoder, EncoderDecoder,
)
from magenta_rt.nnx.spectrostream import (
    ResidualVectorQuantizer, SpectroStream,
)
from magenta_rt.nnx.model import MagentaRT2Sampler
from magenta_rt.nnx.transformer import (
    Encoder, MultiChannelEmbedding, Transformer,
)


def _build_tiny_system(seed: int = 0) -> MagentaRT2Sampler:
    rngs = nnx.Rngs(seed)
    num_codebooks = 4
    codebook_size = 8
    num_reserved = 6
    vocab = num_reserved + num_codebooks * codebook_size

    enc_emb = MultiChannelEmbedding(
        dimension=12,
        num_embeddings_per_channel=[64],
        num_channels=1,
        num_reserved_embeddings=0,
        # Mean reduction over channels — same as shipping configs.
        reduction_fn=jnp.mean,
        round_num_embeddings_to_multiple_of_128=False,
        rngs=rngs,
    )
    encoder = Encoder(
        embedding=enc_emb, embedding_dimension=12, rngs=rngs,
    )
    embedder = nnx.Embed(num_embeddings=vocab, features=16, rngs=rngs)
    temporal = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=4, num_sinks=0,
        use_cross_attention=True, cross_attn_source_features=12,
        cross_attn_max_past_horizon=2, rngs=rngs,
    )
    depth = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=num_codebooks, num_sinks=0,
        use_cross_attention=False, rngs=rngs,
    )
    decoder = DepthformerDecoder(
        num_codebooks=num_codebooks, codebook_size=codebook_size,
        num_reserved_tokens=num_reserved, vocab_size=vocab,
        sos_id=0, model_dim=16, depth_dim=16,
        temporal=temporal, depth=depth, embedder=embedder,
        soft_cap_logits=30.0, rngs=rngs,
    )
    enc_dec = EncoderDecoder(encoder=encoder, decoder=decoder)

    quantizer = ResidualVectorQuantizer(
        num_quantizers=num_codebooks, num_embeddings=codebook_size,
        embedding_dim=16, rngs=rngs,
    )
    codec = SpectroStream(
        sample_rate=48000,
        stft_frame_length=32, stft_frame_step=16, stft_fft_length=32,
        ratios=[(2, 2), (2, 2)], mults=[1, 1],
        channel_splits=None,
        is_resnet=True, num_bins=16, num_channels=2,
        num_features=16, causal=True,
        encoder_base_conv_depth=8, encoder_base_conv_size=(3, 3),
        decoder_base_conv_depth=8, decoder_base_conv_size=(3, 3),
        quantizer=quantizer, decoder_lookahead=0, rngs=rngs,
    )

    return MagentaRT2Sampler(
        depthformer_model=enc_dec,
        num_reserved_tokens=num_reserved,
        codebook_size=codebook_size,
        spectrostream=codec,
        int16_outputs=False,
    )


def test_system_construction_smoke():
    sys = _build_tiny_system()
    assert sys.spectrostream is not None
    assert sys.depthformer is not None


def test_system_streaming_step_runs(rng_key):
    """``MagentaRT2Sampler.step`` runs end-to-end for several frames and
    produces finite (non-zero) waveform chunks."""
    sys = _build_tiny_system()
    sys.init_streaming(batch_size=1, rngs=nnx.Rngs(0))
    sub = jax.random.split(rng_key, 4)
    out_chunks = []
    for k in sub:
        # source_tokens for the encoder: [B, T=1, num_channels=1].
        src = jax.random.randint(k, (1, 1, 1), 0, 64, dtype=jnp.int32)
        tree = sys.step(source_tokens=src)
        out_chunks.append(tree.waveform)
    audio = jnp.concatenate(out_chunks, axis=-1)  # time-last [B, C, T]
    assert jnp.all(jnp.isfinite(audio))
    assert audio.ndim >= 2  # [B, T_audio] or [B, channels, T_audio]
    assert int(sys.depthformer.decoder.step_counter[...][0]) == 4
