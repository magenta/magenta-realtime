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

"""MT3 encoder-decoder Transformer in Flax NNX.

Ported from https://github.com/magenta/mt3 (network.py), a T5.1.1 model with
two differences from standard T5:

- Fixed sinusoidal absolute position embeddings instead of relative position
  biases.
- The encoder takes continuous inputs (log mel spectrogram frames) through a
  linear projection instead of a token embedding.
"""

from typing import Optional

from flax import nnx
from jax import numpy as jnp

from .t5_layers import (
    T5Attention,
    T5DenseGeneral,
    T5Embed,
    T5LayerNorm,
    T5MLP,
    make_attention_mask,
    make_decoder_mask,
)

from magenta_rt.mt3.config import MT3Config
from .layers import FixedEmbed
from .spectrograms import input_depth


class MT3EncoderLayer(nnx.Module):
    """Transformer encoder layer."""

    def __init__(self, config: MT3Config, rngs: nnx.Rngs):
        self.deterministic = False

        self.pre_attention_layer_norm = T5LayerNorm(
            features=config.emb_dim, dtype=config.dtype, rngs=rngs
        )
        self.attention = T5Attention(
            in_features=config.emb_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        self.post_attention_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.pre_mlp_layer_norm = T5LayerNorm(
            features=config.emb_dim, dtype=config.dtype, rngs=rngs
        )
        self.mlp = T5MLP(
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=tuple(config.mlp_activations),
            intermediate_dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.post_mlp_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        encoder_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )

        # Attention block.
        x = self.pre_attention_layer_norm(inputs)
        x = self.attention(x, x, encoder_mask, deterministic=deterministic, rngs=rngs)
        x = self.post_attention_dropout(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = self.pre_mlp_layer_norm(x)
        y = self.mlp(y, deterministic=deterministic)
        y = self.post_mlp_dropout(y, deterministic=deterministic)
        y = y + x

        return y


class MT3DecoderLayer(nnx.Module):
    """Transformer decoder layer that attends to the encoder."""

    def __init__(self, config: MT3Config, rngs: nnx.Rngs):
        self.deterministic = False

        self.pre_self_attention_layer_norm = T5LayerNorm(
            features=config.emb_dim, dtype=config.dtype, rngs=rngs
        )
        self.self_attention = T5Attention(
            in_features=config.emb_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        self.post_self_attention_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.pre_cross_attention_layer_norm = T5LayerNorm(
            features=config.emb_dim, dtype=config.dtype, rngs=rngs
        )
        self.encoder_decoder_attention = T5Attention(
            in_features=config.emb_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        self.post_cross_attention_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.pre_mlp_layer_norm = T5LayerNorm(
            features=config.emb_dim, dtype=config.dtype, rngs=rngs
        )
        self.mlp = T5MLP(
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=tuple(config.mlp_activations),
            intermediate_dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.post_mlp_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        encoded: jnp.ndarray,
        decoder_mask: Optional[jnp.ndarray] = None,
        encoder_decoder_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
        decode: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )

        # Self-attention block.
        x = self.pre_self_attention_layer_norm(inputs)
        x = self.self_attention(
            x, x, decoder_mask, deterministic=deterministic, decode=decode, rngs=rngs
        )
        x = self.post_self_attention_dropout(x, deterministic=deterministic)
        x = x + inputs

        # Encoder-decoder block.
        y = self.pre_cross_attention_layer_norm(x)
        y = self.encoder_decoder_attention(
            y, encoded, encoder_decoder_mask, deterministic=deterministic, rngs=rngs
        )
        y = self.post_cross_attention_dropout(y, deterministic=deterministic)
        y = y + x

        # MLP block.
        z = self.pre_mlp_layer_norm(y)
        z = self.mlp(z, deterministic=deterministic)
        z = self.post_mlp_dropout(z, deterministic=deterministic)
        z = z + y

        return z


class MT3Encoder(nnx.Module):
    """A stack of encoder layers operating on continuous inputs."""

    def __init__(self, config: MT3Config, rngs: nnx.Rngs):
        self.config = config
        self.deterministic = False

        self.continuous_inputs_projection = T5DenseGeneral(
            in_features=input_depth(config.spectrogram_config),
            features=config.emb_dim,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.fixed_embed = FixedEmbed(features=config.emb_dim, max_length=config.max_positions)
        self.input_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.layers = nnx.List(
            [MT3EncoderLayer(config, rngs=rngs) for _ in range(config.num_encoder_layers)]
        )

        self.encoder_norm = T5LayerNorm(features=config.emb_dim, dtype=config.dtype, rngs=rngs)
        self.output_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        encoder_input_tokens: jnp.ndarray,
        encoder_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )
        cfg = self.config
        assert encoder_input_tokens.ndim == 3  # [batch, length, depth]

        seq_length = encoder_input_tokens.shape[-2]
        inputs_positions = jnp.arange(seq_length)[None, :]

        # [batch, length, depth] -> [batch, length, emb_dim]
        x = self.continuous_inputs_projection(encoder_input_tokens)
        x = x + self.fixed_embed(inputs_positions)
        x = self.input_dropout(x, deterministic=deterministic)
        x = x.astype(cfg.dtype)

        for layer in self.layers:
            x = layer(x, encoder_mask, deterministic=deterministic, rngs=rngs)

        x = self.encoder_norm(x)
        return self.output_dropout(x, deterministic=deterministic)


class MT3Decoder(nnx.Module):
    """A stack of decoder layers as part of an encoder-decoder architecture."""

    def __init__(self, config: MT3Config, rngs: nnx.Rngs):
        self.config = config
        self.deterministic = False

        self.token_embedder = T5Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            dtype=config.dtype,
            embedding_init=nnx.initializers.normal(stddev=1.0),
            one_hot=True,
            rngs=rngs,
        )
        self.fixed_embed = FixedEmbed(features=config.emb_dim, max_length=config.max_positions)
        self.input_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.layers = nnx.List(
            [MT3DecoderLayer(config, rngs=rngs) for _ in range(config.num_decoder_layers)]
        )

        self.decoder_norm = T5LayerNorm(features=config.emb_dim, dtype=config.dtype, rngs=rngs)
        self.output_dropout = nnx.Dropout(
            rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        if config.logits_via_embedding:
            self.logits_dense = None
        else:
            self.logits_dense = T5DenseGeneral(
                in_features=config.emb_dim,
                features=config.vocab_size,
                dtype=jnp.float32,  # Use float32 for stability.
                rngs=rngs,
            )

    def __call__(
        self,
        encoded: jnp.ndarray,
        decoder_input_tokens: jnp.ndarray,
        decoder_mask: Optional[jnp.ndarray] = None,
        encoder_decoder_mask: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
        decode: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]

        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = jnp.arange(seq_length)[None, :]

        # [batch, length] -> [batch, length, emb_dim]
        y = self.token_embedder(decoder_input_tokens.astype("int32"))
        y = y + self.fixed_embed(decoder_positions, decode=decode)
        y = self.input_dropout(y, deterministic=deterministic)
        y = y.astype(cfg.dtype)

        for layer in self.layers:
            y = layer(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
                deterministic=deterministic,
                decode=decode,
                rngs=rngs,
            )

        y = self.decoder_norm(y)
        y = self.output_dropout(y, deterministic=deterministic)

        # [batch, length, emb_dim] -> [batch, length, vocab_size]
        if cfg.logits_via_embedding:
            # Use the transpose of the embedding matrix for the logit transform,
            # normalizing pre-softmax logits for this shared case.
            logits = self.token_embedder.attend(y.astype(jnp.float32))
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = self.logits_dense(y)
        return logits


class MT3(nnx.Module):
    """MT3 encoder-decoder Transformer for music transcription.

    Args:
        config: MT3 configuration.
        rngs: RNG state for parameter initialization and dropout.
    """

    def __init__(self, config: MT3Config, rngs: nnx.Rngs = None):
        # The shared config stores dtype framework-neutrally (e.g. the
        # string "float32"); resolve it to a jnp dtype for the layers.
        if isinstance(config.dtype, str):
            config = config.replace(dtype=jnp.dtype(config.dtype))
        self.config = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.encoder = MT3Encoder(config, rngs=rngs)
        self.decoder = MT3Decoder(config, rngs=rngs)
        self.rngs = nnx.data(None) if rngs is None else rngs.fork()

    def encode(
        self,
        encoder_input_tokens: jnp.ndarray,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        """Applies the Transformer encoder on continuous inputs.

        Args:
            encoder_input_tokens: Spectrogram frames of shape
                [batch, length, depth].
            deterministic: Whether to run in deterministic mode (no dropout).
                If None, uses self.deterministic from train()/eval().
            rngs: Optional RNG state for dropout.

        Returns:
            Encoded representation of shape [batch, length, emb_dim].
        """
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)
        # As in the original model, no input positions are masked out; the model
        # may attend to the zero vectors used as padding. An all-ones attention
        # mask is equivalent to no mask at all.
        return self.encoder(encoder_input_tokens, None, deterministic=deterministic, rngs=rngs)

    def decode(
        self,
        encoded: jnp.ndarray,
        decoder_input_tokens: jnp.ndarray,
        decoder_target_tokens: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
        decode: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        """Applies the Transformer decoder on encoded inputs and target tokens.

        Args:
            encoded: Encoder output of shape [batch, encoder_length, emb_dim].
            decoder_input_tokens: Decoder input tokens [batch, length]; during
                autoregressive decoding this is the single previous token
                [batch, 1].
            decoder_target_tokens: Decoder target tokens [batch, length], used
                to construct attention masks when not decoding autoregressively.
            deterministic: Whether to run in deterministic mode (no dropout).
            decode: Whether to use the autoregressive cache (see init_cache).
            rngs: Optional RNG state for dropout.

        Returns:
            Logits of shape [batch, length, vocab_size].
        """
        cfg = self.config
        encoder_ones = jnp.ones(encoded.shape[:-1])

        if decode:
            # At decoding time the causal constraint is enforced by the cache;
            # the decoder may attend to all encoder positions.
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                jnp.ones_like(decoder_input_tokens), encoder_ones, dtype=cfg.dtype
            )
        else:
            if decoder_target_tokens is None:
                raise ValueError("decoder_target_tokens is required when decode=False.")
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens, dtype=cfg.dtype
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0, encoder_ones, dtype=cfg.dtype
            )

        logits = self.decoder(
            encoded,
            decoder_input_tokens=decoder_input_tokens,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=deterministic,
            decode=decode,
            rngs=rngs,
        )
        return logits.astype(cfg.dtype)

    def materialize_constants(self) -> None:
        """Rebuild the derived constants an abstract build cannot restore.

        Part of the contract for loading into a model built with
        ``nnx.eval_shape`` (see :mod:`magenta_rt.nnx.checkpoint_utils`); here
        that is the encoder and decoder sinusoidal position tables.
        """
        self.encoder.fixed_embed.materialize_constants()
        self.decoder.fixed_embed.materialize_constants()

    def init_cache(self, batch_size: int, max_decode_length: Optional[int] = None):
        """Initialize the autoregressive decoding cache.

        Args:
            batch_size: Batch size of the sequences to decode.
            max_decode_length: Maximum number of decoding steps. Defaults to
                ``config.targets_length``.
        """
        cfg = self.config
        if max_decode_length is None:
            max_decode_length = cfg.targets_length
        self.decoder.fixed_embed.init_cache()
        for layer in self.decoder.layers:
            layer.self_attention.init_cache(
                (batch_size, max_decode_length, cfg.emb_dim), dtype=cfg.dtype
            )

    def __call__(
        self,
        encoder_input_tokens: jnp.ndarray,
        decoder_input_tokens: jnp.ndarray,
        decoder_target_tokens: jnp.ndarray,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jnp.ndarray:
        """Applies the full encoder-decoder on a training-style batch.

        Args:
            encoder_input_tokens: Spectrogram frames [batch, length, depth].
            decoder_input_tokens: Decoder input tokens [batch, targets_length],
                a shifted version of ``decoder_target_tokens``.
            decoder_target_tokens: Decoder target tokens [batch, targets_length].
            deterministic: Whether to run in deterministic mode (no dropout).
            rngs: Optional RNG state for dropout.

        Returns:
            Logits of shape [batch, targets_length, vocab_size].
        """
        deterministic = nnx.module.first_from(
            deterministic,
            self.encoder.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )
        if not deterministic:
            rngs = nnx.module.first_from(
                rngs,
                self.rngs,
                error_msg="`deterministic` is False, but rngs was not provided.",
            )

        encoded = self.encode(encoder_input_tokens, deterministic=deterministic, rngs=rngs)
        return self.decode(
            encoded,
            decoder_input_tokens,
            decoder_target_tokens,
            deterministic=deterministic,
            rngs=rngs,
        )
