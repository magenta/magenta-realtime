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

"""Tests for ``magenta_rt.nnx.depthformer`` and friends.

Construction smoke + a multi-step streaming run on a tiny config.
Numerical parity against the JAX/Linen implementation lives in the
jax<->nnx parity tests (``test_jax_logit_parity.py`` for a single step,
``test_e2e_jax_nnx_code_parity.py`` end to end).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from magenta_rt.nnx.depthformer import (
    DepthformerDecoder, EncoderDecoder,
)
from magenta_rt.nnx.transformer import (
    Encoder, MultiChannelEmbedding, Transformer,
)


def _build_tiny_decoder(seed: int = 0) -> DepthformerDecoder:
    rngs = nnx.Rngs(seed)
    num_codebooks = 4
    codebook_size = 8
    num_reserved = 6
    vocab = num_reserved + num_codebooks * codebook_size

    # Decoder embedder is a plain Embedding over the full vocab; the
    # depthformer broadcasts it across the codebook axis at lookup
    # time. (Real shipping configs use a ScaledEmbedding instead.)
    embedder = nnx.Embed(
        num_embeddings=vocab, features=16, rngs=rngs,
    )
    temporal = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=4, num_sinks=0,
        use_cross_attention=True,
        cross_attn_source_features=12,
        cross_attn_max_past_horizon=2,
        rngs=rngs,
    )
    depth = Transformer(
        num_layers=2, model_dim=16, num_heads=2, units_per_head=8,
        ffn_dim=32, max_past_horizon=num_codebooks, num_sinks=0,
        use_cross_attention=False,
        rngs=rngs,
    )
    return DepthformerDecoder(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        num_reserved_tokens=num_reserved,
        vocab_size=vocab,
        sos_id=0,
        model_dim=16,
        depth_dim=16,
        temporal=temporal,
        depth=depth,
        embedder=embedder,
        soft_cap_logits=30.0,
        rngs=rngs,
    )


def test_depthformer_decoder_construction():
    dec = _build_tiny_decoder()
    assert dec.num_codebooks == 4
    assert dec.num_active_codebooks == 4
    # Initial state shape sanity.
    dec.init_streaming(batch_size=1, rngs=nnx.Rngs(0))
    assert dec.previous_frame[...].shape == (1, 1, 4)
    assert isinstance(dec.rng_state.get_value(), nnx.Rngs)


def test_depthformer_decoder_full_seq_forward(rng_key):
    """Non-streaming forward returns ``[B, T, num_codebooks, vocab]`` logits."""
    dec = _build_tiny_decoder()
    sub = jax.random.split(rng_key, 2)
    tokens = jax.random.randint(sub[0], (1, 3, dec.num_codebooks), 0, dec.vocab_size)
    src = jax.random.normal(sub[1], (1, 3, 12))
    logits = dec(tokens, encoded_source=src)
    assert logits.shape == (1, 3, dec.num_codebooks, dec.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_depthformer_streaming_step_shapes(rng_key):
    """Three streaming steps produce ``[B, 1, num_codebooks]`` int tokens
    and monotonically advance ``step_counter``.
    """
    dec = _build_tiny_decoder()
    dec.set_attributes(streaming=True, raise_if_not_found=False)
    dec.init_cache(batch=1, dtype=jnp.float32)
    dec.init_streaming(batch_size=1, rngs=nnx.Rngs(0))

    sub = jax.random.split(rng_key, 3)
    for i, src_key in enumerate(sub):
        src = jax.random.normal(src_key, (1, 1, 12))
        sampled = dec.step(encoded_source=src, temperature=1.0)
        assert sampled.shape == (1, 1, dec.num_codebooks)
        assert sampled.dtype == jnp.int32
        # Tokens land in the configured valid range per codebook.
        for q in range(dec.num_codebooks):
            tok = int(sampled[0, 0, q])
            min_v = dec.num_reserved_tokens + q * dec.codebook_size
            max_v = min_v + dec.codebook_size
            assert min_v <= tok < max_v
        assert int(dec.step_counter[...][0]) == i + 1


def test_depthformer_forced_tokens_short_circuit(rng_key):
    """``forced_tokens`` skips the depth sampling loop and returns the
    forced values directly (after still consuming a temporal step).
    """
    dec = _build_tiny_decoder()
    dec.set_attributes(streaming=True, raise_if_not_found=False)
    dec.init_cache(batch=1, dtype=jnp.float32)
    dec.init_streaming(batch_size=1, rngs=nnx.Rngs(0))

    src = jax.random.normal(rng_key, (1, 1, 12))
    forced = jnp.array([[[6, 14, 22, 30]]], dtype=jnp.int32)
    sampled = dec.step(encoded_source=src, forced_tokens=forced)
    np.testing.assert_array_equal(np.asarray(sampled), np.asarray(forced))
    assert int(dec.step_counter[...][0]) == 1
    np.testing.assert_array_equal(
        np.asarray(dec.previous_frame[...]), np.asarray(forced),
    )


def test_encoder_decoder_step(rng_key):
    """``EncoderDecoder.step`` runs one frame end-to-end with a
    user-supplied encoded source frame."""
    dec = _build_tiny_decoder()
    encoder = Encoder(
        embedding=MultiChannelEmbedding(
            dimension=12,
            num_embeddings_per_channel=[64],
            num_channels=1,
            num_reserved_embeddings=0,
            reduction_fn=jnp.mean,
            round_num_embeddings_to_multiple_of_128=False,
            rngs=nnx.Rngs(0),
        ),
        embedding_dimension=12,
        rngs=nnx.Rngs(0),
    )
    enc_dec = EncoderDecoder(encoder=encoder, decoder=dec)
    enc_dec.set_attributes(streaming=True, raise_if_not_found=False)
    enc_dec.init_cache(batch=1, dtype=jnp.float32)

    enc_dec.init_streaming(batch_size=1, rngs=nnx.Rngs(0))
    src_tokens = jax.random.randint(rng_key, (1, 1, 1), 0, 64, dtype=jnp.int32)
    sampled = enc_dec.step(source_tokens=src_tokens)
    assert sampled.shape == (1, 1, dec.num_codebooks)
    assert int(dec.step_counter[...][0]) == 1
