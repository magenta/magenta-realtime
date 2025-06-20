# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for DepthFormer models."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
from jax import random
import jax.numpy as jnp
import seqio
from t5x import adafactor
from t5x import decoding
from t5x import models as t5x_models

from flaxformer.architectures.t5 import t5_architecture
from flaxformer.architectures.t5 import t5_architecture_test_utils as t5_test_utils
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from . import modules as depthformer


FINAL_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)
BIAS_INIT = nn.initializers.normal(stddev=1e-6)
make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
make_layer_norm = layer_norm.T5LayerNorm
DTYPE = jnp.float32

MODEL_DIMS = 8
DEPTH_DIMS = 4
VOCAB_SIZE = 16386
NUM_ATTN_HEADS = 2
MAX_DECODE_LENGTH = 16


def _make_output_logits():
  return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
      VOCAB_SIZE,
      dtype=DTYPE,
      kernel_init=FINAL_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
  )


def _make_depth_dims_converter(depth_dims):
  return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
      depth_dims,
      dtype=DTYPE,
      kernel_init=FINAL_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
  )


def _make_decoder_layer(shared_relative_position_bias):
  # assert shared_relative_position_bias is not None
  return t5_architecture.DecoderLayer(
      self_attention=t5_test_utils.make_attention1(NUM_ATTN_HEADS, DTYPE),
      encoder_decoder_attention=t5_test_utils.make_attention1(
          NUM_ATTN_HEADS, DTYPE
      ),
      mlp=t5_test_utils.make_mlp1(DTYPE),
      dropout_factory=make_dropout,
      layer_norm_factory=make_layer_norm,
      scanned=False,
      sow_intermediates=False,
      shared_relative_position_bias=shared_relative_position_bias,
  )


def _make_depth_layer(shared_relative_position_bias):
  # assert shared_relative_position_bias is not None
  return t5_architecture.DecoderLayer(
      self_attention=t5_test_utils.make_attention1(NUM_ATTN_HEADS, DTYPE),
      encoder_decoder_attention=None,
      mlp=t5_test_utils.make_mlp1(DTYPE),
      dropout_factory=make_dropout,
      layer_norm_factory=make_layer_norm,
      scanned=False,
      sow_intermediates=False,
      shared_relative_position_bias=shared_relative_position_bias,
  )


def _make_depthformer_decoder(shared_token_embedder):
  del shared_token_embedder  # unused
  return depthformer.DepthformerDecoder(
      num_levels=2,
      num_encoder_levels=2,
      num_layers=1,
      num_depth_layers=1,
      layer_factory=_make_decoder_layer,
      depth_layer_factory=_make_depth_layer,
      dropout_factory=make_dropout,
      layer_norm_factory=make_layer_norm,
      output_logits_factory=lambda: _make_output_logits(),  # pylint: disable=unnecessary-lambda
      token_embedder_factory=lambda: t5_test_utils.make_token_emb1(  # pylint: disable=g-long-lambda
          VOCAB_SIZE, DTYPE, features=MODEL_DIMS
      ),
      shared_relative_position_depth_bias_factory=lambda: t5_test_utils._make_relative_position_bias(  # pylint: disable=g-long-lambda
          NUM_ATTN_HEADS, DTYPE
      ),
      depth_dims_converter_factory=lambda: _make_depth_dims_converter(  # pylint: disable=g-long-lambda
          DEPTH_DIMS
      ),
  )


def _make_encoder_layer(shared_relative_position_bias):
  assert shared_relative_position_bias is None
  return t5_architecture.EncoderLayer(
      attention=t5_test_utils.make_attention1(NUM_ATTN_HEADS, DTYPE),
      mlp=t5_test_utils.make_mlp1(DTYPE),
      dropout_factory=make_dropout,
      layer_norm_factory=make_layer_norm,
      relative_position_bias_factory=(
          lambda: t5_test_utils._make_relative_position_bias(
              NUM_ATTN_HEADS, DTYPE
          )
      ),
      scanned=False,
      sow_intermediates=False,
  )


def _make_depthformer_encoder(shared_token_embedder):
  assert shared_token_embedder is None
  return depthformer.DepthformerEncoder(
      num_levels=2,
      num_layers=3,
      token_embedder_factory=(
          lambda: t5_test_utils.make_token_emb1(2_000, DTYPE)
      ),
      layer_factory=_make_encoder_layer,
      input_dropout_factory=make_dropout,
      output_dropout_factory=make_dropout,
      layer_norm_factory=make_layer_norm,
      dtype=DTYPE,
      scan_layers=False,
      layer_remat='legacy',
      sow_intermediates=False,
  )


class EncoderDecoderTest(parameterized.TestCase):

  # Instantiates the test data.
  def _get_data(self):
    rng_key = random.PRNGKey(42)
    batch_size = 2
    encoder_length = MAX_DECODE_LENGTH
    decoder_length = MAX_DECODE_LENGTH
    input_batch = {
        'encoder_input_tokens': random.randint(  # codec input
            rng_key,
            shape=(batch_size, encoder_length),
            dtype=jnp.int32,
            minval=2,
            maxval=8,
        ),
        'decoder_input_tokens': random.randint(  # codec input
            rng_key,
            shape=(batch_size, decoder_length),
            dtype=jnp.int32,
            minval=8,
            maxval=VOCAB_SIZE,
        ),
        'decoder_target_tokens': random.randint(  # codec target
            rng_key,
            shape=(batch_size, decoder_length),
            dtype=jnp.int32,
            minval=8,
            maxval=VOCAB_SIZE,
        ),
        'decoder_loss_weights': random.randint(  # loss mask
            rng_key,
            shape=(batch_size, decoder_length),
            dtype=jnp.int32,
            minval=0,
            maxval=1,
        ),
    }
    # Gets the initial variables.
    input_shapes = {k: v.shape for k, v in input_batch.items()}
    # input_dtypes = {k: v.dtype for k, v in input_batch.items()}
    return input_batch, input_shapes

  def setUp(self):
    super().setUp()

    transformer = t5_test_utils.make_config1()
    transformer.decoder_factory = _make_depthformer_decoder
    transformer.encoder_factory = _make_depthformer_encoder
    self.model = t5x_models.EncoderDecoderModel(
        module=transformer,
        input_vocabulary=seqio.PassThroughVocabulary(size=1024),
        output_vocabulary=seqio.PassThroughVocabulary(size=2048),
        optimizer_def=adafactor.Adafactor,
        decode_fn=decoding.temperature_sample,
    )
    self.input_batch, input_shapes = self._get_data()
    rng_key = {'params': random.PRNGKey(42), 'dropout': random.PRNGKey(42)}
    self.initial_vars = self.model.get_initial_variables(
        rng=rng_key, input_shapes=input_shapes
    )

    self.init_cache, _ = self.model._compute_kv_cache(
        self.initial_vars['params'],
        self.input_batch['encoder_input_tokens'],
        self.input_batch['decoder_input_tokens'],
        self.input_batch['decoder_input_tokens'],
    )

    self.encoded_inputs = self.model.module.apply(
        {'params': self.initial_vars['params']},
        self.input_batch['encoder_input_tokens'],
        enable_dropout=False,
        method=self.model.module.encode,
    )

  def test_cache_reset(self):
    # Do one decode step with pre-init decoder cache
    _, first_vars = self.model.module.apply(
        {'params': self.initial_vars['params'], 'cache': self.init_cache},
        self.encoded_inputs,
        self.input_batch[
            'encoder_input_tokens'
        ],  # only needed for encoder padding mask
        self.input_batch['decoder_input_tokens'][:, :1],
        self.input_batch['decoder_target_tokens'][:, :1],
        enable_dropout=False,
        decode=True,
        max_decode_length=MAX_DECODE_LENGTH,
        mutable=['cache'],
        method=self.model.module.decode,
    )
    cache = first_vars['cache']
    # Run decode num_rvq times with the same input to see if depth decoder cache
    # resets to init value.
    for _ in range(2):
      _, new_vars = self.model.module.apply(
          {'params': self.initial_vars['params'], 'cache': cache},
          self.encoded_inputs,
          self.input_batch[
              'encoder_input_tokens'
          ],  # only needed for encoder padding mask.
          self.input_batch['decoder_input_tokens'][:, :1],
          self.input_batch['decoder_target_tokens'][:, :1],
          enable_dropout=False,
          decode=True,
          max_decode_length=MAX_DECODE_LENGTH,
          mutable=['cache'],
          method=self.model.module.decode,
      )
      cache = new_vars['cache']

    initial_depth_cache = first_vars['cache']['decoder']['decoder'][
        'depth_decoder'
    ]['depth_layers_0']
    final_depth_cache = cache['decoder']['decoder']['depth_decoder'][
        'depth_layers_0'
    ]
    chex.assert_trees_all_close(initial_depth_cache, final_depth_cache)

  def test_predict_batch_with_aux(self):
    decode_batch = {
        'encoder_input_tokens': self.input_batch['encoder_input_tokens'],
        'decoder_input_tokens': self.input_batch['decoder_input_tokens'],
    }
    decodes, _ = self.model.predict_batch_with_aux(
        self.initial_vars['params'], decode_batch
    )

    chex.assert_equal_shape([decodes, self.input_batch['decoder_input_tokens']])

  def test_cached_mean_pooling(self):
    _, first_vars = self.model.module.apply(
        {'params': self.initial_vars['params'], 'cache': self.init_cache},
        self.encoded_inputs,
        self.input_batch[
            'encoder_input_tokens'
        ],  # only needed for encoder padding mask
        self.input_batch['decoder_input_tokens'][:, :1],
        self.input_batch['decoder_target_tokens'][:, :1],
        enable_dropout=False,
        decode=True,
        max_decode_length=MAX_DECODE_LENGTH,
        mutable=['cache'],
        method=self.model.module.decode,
    )

    cache = first_vars['cache']
    for idx in range(1, 3):
      _, new_vars = self.model.module.apply(
          {'params': self.initial_vars['params'], 'cache': cache},
          self.encoded_inputs,
          self.input_batch[
              'encoder_input_tokens'
          ],  # only needed for encoder padding mask
          jnp.expand_dims(self.input_batch['decoder_input_tokens'][:, idx], -1),
          jnp.expand_dims(
              self.input_batch['decoder_target_tokens'][:, idx], -1
          ),
          enable_dropout=False,
          decode=True,
          max_decode_length=MAX_DECODE_LENGTH,
          mutable=['cache'],
          method=self.model.module.decode,
      )
      cache = new_vars['cache']

    # embed input tokens and test that mean is equivalent
    # to mean of cached_inputs
    token_embedder = embedding.Embed(
        num_embeddings=VOCAB_SIZE,
        cast_input_dtype=jnp.int32,
        attend_dtype=jnp.float32,
        features=8,
        one_hot=False,
    )
    embedding_params = self.initial_vars['params']['decoder']['token_embedder'][
        'embedding'
    ]
    embedded_inputs = token_embedder.apply(
        {'params': {'embedding': embedding_params}},
        self.input_batch['decoder_input_tokens'][:, 1:3],
    )

    mean_pooled_embedded_inputs = jnp.expand_dims(
        jnp.mean(embedded_inputs, axis=-2, keepdims=True), -2
    )
    mean_pooled_cached_inputs = jnp.expand_dims(
        jnp.mean(
            cache['decoder']['decoder']['temporal_decoder']['cached_input'],
            axis=0,
            keepdims=False,
        ),
        -2,
    )
    chex.assert_trees_all_equal(
        mean_pooled_embedded_inputs, mean_pooled_cached_inputs
    )

  def test_relative_position_bias(self):
    # Test whether shared relative position bias in the depth decoder is
    # correctly instantiated.
    depth_bias_exists = (
        'relpos_bias_depth'
        in self.initial_vars['params']['decoder']['decoder']['depth_decoder']
    )
    chex.assert_equal(depth_bias_exists, True)

  def test_decode(self):
    # Test that decoding one step at a time produces the same logits as a
    # forward pass on the same data.
    logits = self.model._compute_logits(
        self.initial_vars['params'], self.input_batch, dropout_rng=None
    )

    idx = 0
    flat_logits, new_vars = self.model.module.apply(
        {'params': self.initial_vars['params'], 'cache': self.init_cache},
        self.encoded_inputs,
        self.input_batch[
            'encoder_input_tokens'
        ],  # only needed for encoder padding mask
        jnp.expand_dims(self.input_batch['decoder_input_tokens'][:, idx], -1),
        jnp.expand_dims(self.input_batch['decoder_input_tokens'][:, idx], -1),
        enable_dropout=False,
        decode=True,
        mutable=['cache'],
        method=self.model.module.decode,
    )
    new_cache = new_vars['cache']

    chex.assert_trees_all_close(
        flat_logits, jnp.expand_dims(logits[:, 0, :], axis=1), atol=1e-2
    )

    cache = new_cache
    for _ in range(MAX_DECODE_LENGTH - 1):
      idx += 1
      flat_logits, new_vars = self.model.module.apply(
          {'params': self.initial_vars['params'], 'cache': cache},
          self.encoded_inputs,
          self.input_batch[
              'encoder_input_tokens'
          ],  # only needed for encoder padding mask
          jnp.expand_dims(self.input_batch['decoder_input_tokens'][:, idx], -1),
          jnp.expand_dims(self.input_batch['decoder_input_tokens'][:, idx], -1),
          enable_dropout=False,
          decode=True,
          mutable=['cache'],
          method=self.model.module.decode,
      )
      cache = new_vars['cache']
      chex.assert_trees_all_close(
          flat_logits, jnp.expand_dims(logits[:, idx, :], axis=1), atol=1e-1
      )


if __name__ == '__main__':
  absltest.main()
