# Copyright 2025 The MT3 Authors.
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

"""MT3 encoder-decoder Transformer in pure MLX.

Port of ``magenta_rt.nnx.mt3.model`` — a T5.1.1 model with fixed
sinusoidal positions and a continuous-input projection in place of
encoder token embeddings. Inference-focused (dropout layers exist but
are inert in eval mode; call ``model.eval()`` as ``load_model`` does).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from magenta_rt.mt3.config import MT3Config
from magenta_rt.mt3.spectrograms import input_depth

from .layers import FixedEmbed
from .t5_layers import (
    T5MLP,
    T5Attention,
    T5DenseGeneral,
    T5Embed,
    T5LayerNorm,
    make_attention_mask,
    make_decoder_mask,
)

_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def _resolve_dtype(dtype) -> mx.Dtype:
    if isinstance(dtype, mx.Dtype):
        return dtype
    return _DTYPES[str(dtype)]


class MT3EncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.pre_attention_layer_norm = T5LayerNorm(config.emb_dim)
        self.attention = T5Attention(config.emb_dim, config.num_heads, config.head_dim)
        self.post_attention_dropout = nn.Dropout(config.dropout_rate)
        self.pre_mlp_layer_norm = T5LayerNorm(config.emb_dim)
        self.mlp = T5MLP(
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=tuple(config.mlp_activations),
            intermediate_dropout_rate=config.dropout_rate,
        )
        self.post_mlp_dropout = nn.Dropout(config.dropout_rate)

    def __call__(self, inputs: mx.array, encoder_mask: Optional[mx.array] = None) -> mx.array:
        x = self.pre_attention_layer_norm(inputs)
        x = self.attention(x, x, encoder_mask)
        x = self.post_attention_dropout(x)
        x = x + inputs

        y = self.pre_mlp_layer_norm(x)
        y = self.mlp(y)
        y = self.post_mlp_dropout(y)
        return y + x


class MT3DecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.pre_self_attention_layer_norm = T5LayerNorm(config.emb_dim)
        self.self_attention = T5Attention(config.emb_dim, config.num_heads, config.head_dim)
        self.post_self_attention_dropout = nn.Dropout(config.dropout_rate)
        self.pre_cross_attention_layer_norm = T5LayerNorm(config.emb_dim)
        self.encoder_decoder_attention = T5Attention(
            config.emb_dim, config.num_heads, config.head_dim
        )
        self.post_cross_attention_dropout = nn.Dropout(config.dropout_rate)
        self.pre_mlp_layer_norm = T5LayerNorm(config.emb_dim)
        self.mlp = T5MLP(
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=tuple(config.mlp_activations),
            intermediate_dropout_rate=config.dropout_rate,
        )
        self.post_mlp_dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self,
        inputs: mx.array,
        encoded: mx.array,
        decoder_mask: Optional[mx.array] = None,
        encoder_decoder_mask: Optional[mx.array] = None,
        decode: bool = False,
    ) -> mx.array:
        x = self.pre_self_attention_layer_norm(inputs)
        x = self.self_attention(x, x, decoder_mask, decode=decode)
        x = self.post_self_attention_dropout(x)
        x = x + inputs

        y = self.pre_cross_attention_layer_norm(x)
        y = self.encoder_decoder_attention(y, encoded, encoder_decoder_mask)
        y = self.post_cross_attention_dropout(y)
        y = y + x

        z = self.pre_mlp_layer_norm(y)
        z = self.mlp(z)
        z = self.post_mlp_dropout(z)
        return z + y


class MT3Encoder(nn.Module):
    """A stack of encoder layers operating on continuous inputs."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config
        self.continuous_inputs_projection = T5DenseGeneral(
            in_features=input_depth(config.spectrogram_config),
            features=config.emb_dim,
        )
        self.fixed_embed = FixedEmbed(config.emb_dim, max_length=config.max_positions)
        self.input_dropout = nn.Dropout(config.dropout_rate)
        self.layers = [MT3EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        self.encoder_norm = T5LayerNorm(config.emb_dim)
        self.output_dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self, encoder_input_tokens: mx.array, encoder_mask: Optional[mx.array] = None
    ) -> mx.array:
        assert encoder_input_tokens.ndim == 3  # [batch, length, depth]
        seq_length = encoder_input_tokens.shape[-2]
        inputs_positions = mx.arange(seq_length)[None, :]

        x = self.continuous_inputs_projection(encoder_input_tokens)
        x = x + self.fixed_embed(inputs_positions)
        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_mask)

        return self.output_dropout(self.encoder_norm(x))


class MT3Decoder(nn.Module):
    """A stack of decoder layers as part of an encoder-decoder architecture."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config
        self.token_embedder = T5Embed(config.vocab_size, config.emb_dim)
        self.fixed_embed = FixedEmbed(config.emb_dim, max_length=config.max_positions)
        self.input_dropout = nn.Dropout(config.dropout_rate)
        self.layers = [MT3DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        self.decoder_norm = T5LayerNorm(config.emb_dim)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        if config.logits_via_embedding:
            self.logits_dense = None
        else:
            self.logits_dense = T5DenseGeneral(
                in_features=config.emb_dim, features=config.vocab_size
            )

    def __call__(
        self,
        encoded: mx.array,
        decoder_input_tokens: mx.array,
        decoder_mask: Optional[mx.array] = None,
        encoder_decoder_mask: Optional[mx.array] = None,
        decode: bool = False,
    ) -> mx.array:
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]
        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = mx.arange(seq_length)[None, :]

        y = self.token_embedder(decoder_input_tokens.astype(mx.int32))
        y = y + self.fixed_embed(decoder_positions, decode=decode)
        y = self.input_dropout(y)

        for layer in self.layers:
            y = layer(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
                decode=decode,
            )

        y = self.output_dropout(self.decoder_norm(y))

        if cfg.logits_via_embedding:
            logits = self.token_embedder.attend(y)
            logits = logits / mx.sqrt(mx.array(float(y.shape[-1])))
        else:
            logits = self.logits_dense(y)
        return logits


class MT3(nn.Module):
    """MT3 encoder-decoder Transformer for music transcription (MLX)."""

    def __init__(self, config: MT3Config):
        super().__init__()
        # Shared config stores dtype framework-neutrally; the MLX port
        # computes in float32 (validate, then drop the resolved value —
        # layers are fp32 throughout, matching the pretrained weights).
        _resolve_dtype(config.dtype)
        self.config = config
        self.encoder = MT3Encoder(config)
        self.decoder = MT3Decoder(config)

    def encode(self, encoder_input_tokens: mx.array, deterministic: bool = True) -> mx.array:
        """Encode spectrogram frames ``[batch, length, depth]``."""
        del deterministic  # signature parity with the nnx port
        assert encoder_input_tokens.ndim == 3
        # As in the original model, no encoder positions are masked out.
        return self.encoder(encoder_input_tokens, None)

    def decode(
        self,
        encoded: mx.array,
        decoder_input_tokens: mx.array,
        decoder_target_tokens: Optional[mx.array] = None,
        deterministic: bool = True,
        decode: bool = False,
    ) -> mx.array:
        """Decode target tokens against encoder output; see the nnx port."""
        del deterministic
        encoder_ones = mx.ones(encoded.shape[:-1])

        if decode:
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                mx.ones(decoder_input_tokens.shape), encoder_ones
            )
        else:
            if decoder_target_tokens is None:
                raise ValueError("decoder_target_tokens is required when decode=False.")
            decoder_mask = make_decoder_mask(decoder_target_tokens)
            encoder_decoder_mask = make_attention_mask(
                (decoder_target_tokens > 0).astype(mx.float32), encoder_ones
            )

        return self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            decode=decode,
        )

    def init_cache(self, batch_size: int, max_decode_length: Optional[int] = None):
        """Initialize the autoregressive decoding cache."""
        cfg = self.config
        if max_decode_length is None:
            max_decode_length = cfg.targets_length
        self.decoder.fixed_embed.init_cache()
        for layer in self.decoder.layers:
            layer.self_attention.init_cache(batch_size, max_decode_length)

    def __call__(
        self,
        encoder_input_tokens: mx.array,
        decoder_input_tokens: mx.array,
        decoder_target_tokens: mx.array,
    ) -> mx.array:
        """Full encoder-decoder forward on a training-style batch."""
        encoded = self.encode(encoder_input_tokens)
        return self.decode(encoded, decoder_input_tokens, decoder_target_tokens)
