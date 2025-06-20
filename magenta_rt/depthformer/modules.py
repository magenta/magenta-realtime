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

"""Implementation of depthformer models.

The idea behind depthformer models is to decompse autoregressive
modelling of token sequences into a temporal model and a separate depth model.
In the example of modelling SpectroStream token sequences, the temporal model
operates at the sampling frequency and the depth model predicts all RVQ levels
in parallel. This accelerates both training and inference compared to modelling
the flattened token sequence while removing the need for multiple stages as in
AudioLM.

The key step is to override the layer construction of the Decoder class combine
a temporal sequence prediction and a depth-wise token prediction.
"""

from typing import Any, Callable, Mapping, Optional

import chex
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from flaxformer import transformer_common as common
from flaxformer import types
from flaxformer.architectures.t5 import t5_architecture

Array = types.Array

PyTree = Any


def _copy_to_scope(scope, collection, collection_name):
  """Helper to copy collections to the current Module scope.

  Required for dealing with cache variables manually.

  Args:
    scope: The scope in which the variable is stored.
    collection: The collection of the variable (e.g., "params").
    collection_name: The name of the variable (e.g., "cache").
  """
  for k, v in collection.items():
    if isinstance(v, Mapping):
      subscope = scope.push(k, reuse=True)
      _copy_to_scope(subscope, v, collection_name)
    else:
      scope.put_variable(collection_name, k, v)


def _to_temporal_embedded_inputs(embedded_inputs: jnp.ndarray, num_levels: int):
  """Construct input for temporal decoder stack at train-time.

  This function assumes the first item is BOS. It pads this so the first Q
  items are BOS meaning new shape is T+1. We therefore return :-1 in the T
  axis.

  Args:
    embedded_inputs: Embedded tokens [B, T*Q, D].
    num_levels: RVQ depth, Q.

  Returns:
    temporal_embedded_inputs: [B, T, Q, D]
  """
  # embedded_inputs: [B, T' Q, D]
  padded_inputs = jnp.pad(
      embedded_inputs,
      pad_width=((0, 0), (num_levels - 1, 1), (0, 0)),
      mode='edge',
  )
  temporal_embedded_inputs = jnp.reshape(
      padded_inputs,
      [padded_inputs.shape[0], -1, num_levels, padded_inputs.shape[-1]],
  )
  # We pad the sequence by (Q-1) since the first token is an additional BOS.
  # We return all but the last element in the T axis since the last target is
  # frame T-1.
  temporal_embedded_inputs = temporal_embedded_inputs[:, :-1, :, :]
  return temporal_embedded_inputs


def _to_temporal_decoder_mask(decoder_mask: jnp.ndarray, num_levels: int):
  # Slice Decoder mask [B, 1, T*Q, T*Q] -> [B, 1, T, T] by taking
  # every Qth element.
  temporal_decoder_mask = decoder_mask[:, :, ::num_levels, ::num_levels]
  return temporal_decoder_mask


def _to_depth_decoder_mask(decoder_mask: jnp.ndarray, num_levels: int):
  """Convert decoder_mask to depth_decoder_mask.

  We slice decoder_mask [B, 1, T*Q, T*Q] -> [B*T, 1, Q, Q] by decomposing into
  [B, 1, T, Q, T, Q] and concatenating the [QxQ] components for each element
  of T to obtain [B*T, 1, Q, Q].

  Args:
    decoder_mask: Decoder self-attention mask.
    num_levels: Number of RVQ levels.

  Returns:
    depth_decoder_mask: Correctly sliced and shaped decoder self-attention mask.
  """
  batch_size, *_, full_seq_len = decoder_mask.shape
  seq_len = full_seq_len // num_levels
  mask = jnp.reshape(
      decoder_mask, (batch_size, 1, seq_len, num_levels, seq_len, num_levels)
  )
  mask_collection = [
      jnp.expand_dims(mask[:, :, j, :, j, :], 1) for j in range(seq_len)
  ]
  return jnp.reshape(
      jnp.concatenate(mask_collection, axis=1), (-1, 1, num_levels, num_levels)
  )


def _to_depth_logit_mask(logit_mask: jnp.ndarray, num_levels: int):
  padded_mask = jnp.pad(
      logit_mask, pad_width=((0, 0), (num_levels - 1, 1), (0, 0)), mode='edge'
  )
  padded_mask = jnp.reshape(
      padded_mask, (padded_mask.shape[0], -1, num_levels, 1)
  )
  depth_logit_mask = jnp.reshape(padded_mask[:, 1:, :, :], (-1, num_levels, 1))
  return depth_logit_mask


def _to_depth_embedded_inputs(
    temporal_context: jnp.ndarray, embedded_inputs: jnp.ndarray, num_levels: int
):
  """Construct input for depth decoder stack at train-time.

  Args:
    temporal_context: Mean-pooled and reshaped embeddings. [B, T, D]
    embedded_inputs: Input Embeddings [B, T*Q, D]
    num_levels: Number of RVQ levels.

  Returns:
    depth_embedded_inputs: [B*T, Q, D]
  """

  # temporal_context: [B, T', D]
  # embedded_inputs: [B, T' Q, D]
  padded_inputs = jnp.pad(
      embedded_inputs,
      pad_width=((0, 0), (num_levels - 1, 1), (0, 0)),
      mode='edge',
  )
  depth_embedded_inputs = jnp.reshape(
      padded_inputs,
      [padded_inputs.shape[0], -1, num_levels, padded_inputs.shape[-1]],
  )

  temporal_context = jnp.expand_dims(temporal_context, axis=-2)
  depth_embedded_inputs = jnp.concatenate(
      [temporal_context, depth_embedded_inputs[:, 1:, :-1, :]], axis=-2
  )
  return jnp.reshape(
      depth_embedded_inputs,
      [-1, depth_embedded_inputs.shape[2], depth_embedded_inputs.shape[3]],
  )


class TemporalDecoderStack(nn.Module):
  """Stack of temporal decoder layers for DepthFormer.

  Attributes:
    temporal_layer_factory: A MakeDecoderLayerFn
    gather_mode: How the temporal module pools the Q rvq levels.
    num_temporal_layers: Number of temporal decoder layers.
    num_levels: Number of RVQ levels.
    relpos_bias: An instance of a shared relative position bias module, usually
      owned by the Decoder.
    layer_remat: What remat to use, only 'none' is supported now
    scan_layers: Whether to use scan to construct layer stack. Only False is
      supported.
  """

  temporal_layer_factory: t5_architecture.MakeDecoderLayerFn
  gather_mode: str = 'mean'
  num_temporal_layers: Optional[int] = None
  num_levels: Optional[int] = None

  relpos_bias: Optional[nn.Module] = None

  layer_remat: str = 'none'
  scan_layers: bool = False

  def setup(self):
    if self.layer_remat != 'none':
      raise ValueError('Remat not supported.')
    if self.scan_layers:
      raise ValueError('Scan layers not supported.')
    self.temporal_decoder = self._setup_layer_sequence()

  def _setup_layer_sequence(self):
    """Follows setup_layer_sequence of the base Decoder class."""
    lyrf = lambda: self.temporal_layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias
    )
    self.layers = [lyrf() for _ in range(self.num_temporal_layers)]
    return common.TransparentLayerSequence(self.layers)

  def __call__(
      self,
      embedded_inputs,
      encoded=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
      *,
      logit_mask=None,
      enable_dropout: bool = False,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      init_cache: bool = False,
      **kwargs,
  ):
    """Applies the temporal decoder stack.

    Args:
      embedded_inputs: Embedded input tokens.
      encoded: The outputs from the encoder. If None, do not attend to encoder
        outputs, resulting in a decoder only model (i.e. language model).
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      logit_mask: a mask to be applied to the attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters. cache, lengths are inferred from the mask if not
        provided.
      init_cache: Sets whether we are initialising the decoding cache.
      **kwargs: Optional keyword arguments to pass to
        decode_from_continuous_inputs.

    Returns:
      Temporal context for the depth decoder stack.

    There are two main modes of behaviour: train (default) and decode.
    init_cache is a special case that proceeds like train but initialises the
    cache for decoding.
    """
    train_or_init_cache = init_cache or not decode
    if train_or_init_cache:
      # Reshape temporal decoder inputs from [B, T*Q, D] -> [B, T, Q, D] then
      # Pool temporal decoder inputs along Q axis: [B, T, Q, D] -> [B, T, D]
      # We prefer to have reshaping and pooling here rather than a level higher
      # so we only have to check that we are training or initing once.
      # During decoding, reshaping and pooling is more complex, it gets handled
      # by the wrapping class.
      temporal_embedded_inputs = _to_temporal_embedded_inputs(
          embedded_inputs, self.num_levels
      )
      if self.gather_mode == 'mean':
        temporal_embedded_inputs = jnp.mean(temporal_embedded_inputs, axis=-2)
      else:
        raise ValueError(f'Gather mode {self.gather_mode} not supported.')

      # If we are doing EncoderDecoder we must adjust the EncDec attention mask.
      if encoder_decoder_mask is not None:
        encoder_decoder_mask = encoder_decoder_mask[
            :, :, self.num_levels - 1 :: self.num_levels, :
        ]
    else:
      # inputs are coming from the decoder cache. They will arrive mean-pooled.
      temporal_embedded_inputs = embedded_inputs

    if decoder_mask is not None:
      decoder_mask = _to_temporal_decoder_mask(decoder_mask, self.num_levels)
    if logit_mask is not None:
      # _to_temporal_ method returns [B, T, Q, D]. The logit_mask for the
      # temporal decoder needs to be [B, T, D_logit_mask] (D_logit_mask=1.)
      # We pick the index 1 in case the given frame isn't full.
      logit_mask = _to_temporal_embedded_inputs(logit_mask, self.num_levels)[
          :, :, 1, :
      ]

    # Run temporal decoder stack
    temporal_context = self.temporal_decoder(
        temporal_embedded_inputs,
        encoded=encoded,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
    )
    return temporal_context


class DepthDecoderStack(nn.Module):
  """Stack of depth transformer decoder layers."""

  depth_layer_factory: t5_architecture.MakeDecoderLayerFn

  num_depth_layers: Optional[int] = None
  num_levels: Optional[int] = None

  relpos_bias_depth: Optional[nn.Module] = None

  layer_remat: str = 'none'
  scan_layers: bool = False

  depth_dims_converter_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    self.depth_dims_converter = (
        self.depth_dims_converter_factory()  # pylint: disable=not-callable
        if self.depth_dims_converter_factory
        else None
    )
    self.depth_decoder = self._setup_depth_layer_sequence()

  def _setup_depth_layer_sequence(self):
    """Creates depth-wise transformer layers.

    Mirrors `t5_architecture.Decoder._setup_layer_sequence`.

    Returns:
      TransparentLayerSequence of decoder layers.
    """
    if self.layer_remat != 'none':
      raise ValueError('Remat not supported.')
    if self.scan_layers:
      raise ValueError('Scan layers not supported.')

    lyrf_depth = lambda: self.depth_layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias_depth
    )
    self.depth_layers = [lyrf_depth() for _ in range(self.num_depth_layers)]
    return common.TransparentLayerSequence(self.depth_layers)

  def __call__(
      self,
      temporal_context,
      embedded_inputs,
      decoder_mask=None,
      *,
      logit_mask=None,
      enable_dropout=False,
      decode=False,
      max_decode_length=None,
      init_cache: bool = False,
  ):
    batch_size, seq_len, _ = temporal_context.shape

    train_or_init_cache = init_cache or not decode
    if train_or_init_cache:
      if init_cache:
        # We need to make sure the shapes are correct for the AR cache. Input to
        # the depth stack should not push sequence length into batch dims
        # since during decode, T=1. We simply reshape to [B, T, Q, D] and
        # average (arbitrary reduce operation) over T.
        depth_embedded_inputs = jnp.reshape(
            embedded_inputs, (batch_size, seq_len, self.num_levels, -1)
        )
        depth_embedded_inputs = jnp.mean(depth_embedded_inputs, axis=1)
        logit_mask = None
      else:
        # Default train behaviour
        # Combine and reshape temporal context and first (Q-1) embeddings into
        # depth-input [B, T, D] -> [B*T, 1, D]
        depth_embedded_inputs = _to_depth_embedded_inputs(
            temporal_context, embedded_inputs, self.num_levels
        )
    else:
      # During decode attention is cached and we don't compute logits.
      decoder_mask = None
      logit_mask = None
      # current temporal_context has been cached.
      # the inputs are coming conditionally from the calling wrapper
      # depending on the internal count state.
      depth_embedded_inputs = temporal_context

    if decoder_mask is not None:
      # slice and reshape attention mask to depth [B*T, 1, Q, Q]
      decoder_mask = _to_depth_decoder_mask(decoder_mask, self.num_levels)

    if logit_mask is not None:
      logit_mask = _to_depth_logit_mask(logit_mask, self.num_levels)

    if self.depth_dims_converter is not None:
      # Convert between temporal model dims and depth model dims.
      depth_embedded_inputs = self.depth_dims_converter(depth_embedded_inputs)

    # Run depth decoder stack
    pre_logits = self.depth_decoder(
        depth_embedded_inputs,
        encoded=None,
        decoder_mask=decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
    )

    # Reshape logits to flattened T*Q output if not decoding.
    if train_or_init_cache:
      if init_cache:
        # We previously reduced over T to get the correct dims for input.
        # Now we must repeat the logits T times to get the expected shape.
        pre_logits = jnp.repeat(pre_logits, seq_len, axis=1)
      pre_logits = jnp.reshape(
          pre_logits,
          (
              batch_size,
              seq_len * self.num_levels,
              -1,
          ),
      )
    return pre_logits


class DepthformerDecoderStack(nn.Module):
  """Plumbs together temporal and depth decoder layers."""

  temporal_layer_factory: t5_architecture.MakeDecoderLayerFn
  depth_layer_factory: t5_architecture.MakeDecoderLayerFn

  gather_mode: str = 'mean'
  num_levels: Optional[int] = None
  num_encoder_levels: int = 1
  num_temporal_layers: Optional[int] = None
  num_depth_layers: Optional[int] = None
  relpos_bias: Optional[nn.Module] = None
  relpos_bias_depth: Optional[nn.Module] = None
  depth_dims_converter_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    self.temporal_decoder = (
        PeriodicCallableTemporalDecoder(  # TemporalDecoderStack
            num_levels=self.num_levels,
            num_temporal_layers=self.num_temporal_layers,
            temporal_layer_factory=self.temporal_layer_factory,
            relpos_bias=self.relpos_bias,
        )
    )
    self.depth_decoder = PeriodicResetDepthDecoder(  # DepthDecoderStack
        num_levels=self.num_levels,
        num_depth_layers=self.num_depth_layers,
        depth_layer_factory=self.depth_layer_factory,
        relpos_bias_depth=self.relpos_bias_depth,
        depth_dims_converter_factory=self.depth_dims_converter_factory,
    )

  @nn.compact
  def __call__(
      self,
      embedded_inputs,
      encoder_outputs=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
      *,
      logit_mask=None,
      enable_dropout: bool = False,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      **kwargs,
  ):
    # Reshape temporal decoder inputs from [B, T*Q, D] -> [B, T, Q, D]
    input_dims = chex.Dimensions()
    input_dims['BLD'] = embedded_inputs.shape
    if not decode:
      # At training time check the sequence length is divisible by Q.
      chex.assert_is_divisible(input_dims.L, self.num_levels)

    if max_decode_length is not None:
      temporal_max_decode_length = max_decode_length // self.num_levels
      depth_max_decode_length = self.num_levels
    else:
      temporal_max_decode_length = None
      depth_max_decode_length = None

    if encoder_decoder_mask is not None:
      encoder_decoder_mask = encoder_decoder_mask[
          :, :, :, :: self.num_encoder_levels
      ]

    temporal_context = self.temporal_decoder(
        embedded_inputs,
        encoder_outputs=encoder_outputs,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=temporal_max_decode_length,
    )

    pre_logits = self.depth_decoder(
        temporal_context,
        embedded_inputs,
        decoder_mask=decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=depth_max_decode_length,
    )

    return pre_logits


class DepthformerDecoder(t5_architecture.Decoder):
  """Depthformer decoder component.

  Given a flattened sequence of tokens [B, T*Q], DepthFormer is a modified
  Decoder which operates on the structured input [B, T, Q]. At each element of
  the temporal sequence, T, it predicts Q "depth" tokens sequentially.
  Overrides `t5_architecture.Decoder._setup_layer_sequence`.

  num_layers in the base class provides the number of layers in the temporal
  sequence model.

  Attributes:
    num_levels: Number of RVQ levels (depth).
    num_encoder_levels: Number of RVQ levels of the encoder (depth).
    num_depth_layers: Number of layers in the depth transformer.
    gather_mode: Strategy for pooling embeddings over Q.
    shared_relative_position_depth_bias_factory: An instance of a shared
      relative position bias module, usually owned by the Decoder.
    depth_layer_factory: A callable that returns a DecoderLayer for the depth
      decoder.
    depth_dims_converter_factory: A callable that instantiates a Linear or MLP
      nn module to convert between encoder/temporal model dimensions and depth
      model dimensions. If this is active we require output_logits_factory to
      convert between the smaller model dimensions and the output logits (the
      token embedder will be in the larger temporal model dimensional space).
  """

  num_levels: Optional[int] = None
  num_depth_layers: Optional[int] = None
  num_encoder_levels: int = 1
  gather_mode: str = 'mean'
  shared_relative_position_bias_factory: Optional[Callable[[], nn.Module]] = (
      None
  )
  shared_relative_position_depth_bias_factory: Optional[
      Callable[[], nn.Module]
  ] = None
  depth_layer_factory: Optional[t5_architecture.MakeDecoderLayerFn] = None
  depth_dims_converter_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    self.relpos_bias_depth = (
        self.shared_relative_position_depth_bias_factory()  # pylint: disable=not-callable
        if self.shared_relative_position_depth_bias_factory is not None
        else None
    )
    if self.depth_dims_converter_factory is not None:
      if self.output_logits_factory is None:
        raise ValueError(
            'When using depth_dims_converter_factory '
            'we must use an output_logits_factory '
            'with the correct embed_dims.'
        )
    super().setup()

  def _setup_layer_sequence(self):
    return DepthformerDecoderStack(
        temporal_layer_factory=self.layer_factory,
        depth_layer_factory=self.depth_layer_factory,
        num_levels=self.num_levels,
        num_encoder_levels=self.num_encoder_levels,
        num_temporal_layers=self.num_layers,
        num_depth_layers=self.num_depth_layers,
        gather_mode=self.gather_mode,
        relpos_bias=self.relpos_bias,  # comes from super().setup()
        relpos_bias_depth=self.relpos_bias_depth,
        depth_dims_converter_factory=self.depth_dims_converter_factory,
    )


class DepthformerEncoder(t5_architecture.Encoder):
  """Depthformer encoder component.

  Given a flattened sequence of tokens [B, T*Q], DepthFormer is a modified
  Encoder which operates on the structured input [B, T, Q]. At each element of
  the temporal sequence, T, the Q "depth" tokens are embedded and pooled
  together.

  Attributes:
    num_levels: Number of RVQ levels (depth).
  """

  num_levels: Optional[int] = None

  def __call__(
      self,
      inputs,
      inputs_positions=None,
      encoder_mask=None,
      *,
      segment_ids: Optional[Array] = None,
      enable_dropout: bool = True,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: encoder self-attention mask.
      segment_ids: Input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output of a transformer encoder.
    """
    if self.sow_intermediates:
      self.sow('intermediates', 'input_tokens_ids', inputs)
    embedded_inputs = self.embed_and_combine_inputs(
        inputs,
        inputs_positions=inputs_positions,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout,
    )
    # Depthformer special, reshape and mean-pool
    b, t_times_q, d = embedded_inputs.shape
    t = t_times_q // self.num_levels
    embedded_inputs = embedded_inputs.reshape((b, t, self.num_levels, d)).mean(
        axis=2
    )
    logit_mask = (inputs > 0).reshape((b, t, self.num_levels)).any(axis=2)
    logit_mask = jnp.expand_dims(
        jnp.array(logit_mask, dtype=embedded_inputs.dtype), axis=-1
    )
    if encoder_mask is not None:
      bool_mask = (encoder_mask == 1).reshape(
          (b, 1, t, self.num_levels, t, self.num_levels)
      )
      shrunk_bool_mask = bool_mask.any(axis=3).any(axis=-1)
      encoder_mask = jnp.array(shrunk_bool_mask, dtype=encoder_mask.dtype)
    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_inputs,
        encoder_mask=encoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
    )
    if self.sow_intermediates:
      self.sow('intermediates', 'final_encoder_outputs', encoder_outputs)
    return encoder_outputs


#################################
###### PERIODIC FUNCTIONS #######
#################################
#
# The following are nn.Modules with periodic behaviour which wrap child
# nn.Modules.
# They support 3 distinct modes of operation: training, init_cache and decode.
#
# During training, we manually .init() to obtain 'params' and
# 'params_axes' collections for the child modules. We manually .apply() the
# underlying modules. The periodic modules work transparently during training
# otherwise.
#
# During init_cache we manually initialize the
# decoding cache (attention, positional biases) of the child module.
# We add additional variables into 'cache' to maintain counters to allow for
# periodic behaviour conditioned on the counter value. We also initialize
# dummy shapes and cached variable values into 'cache' when required or allow
# periodic resetting of the entire cache of the child module. This added
# complexity is necessary for proper and efficient decoding behavior.
#
# During decoding we conditionally execute specific behaviour of module
# depending on the counter value.


class PeriodicCallableTemporalDecoder(nn.Module):
  """Wrapper for TemporalDecoderStack which is called periodically.

  This module wraps a stack of decoder layers. During training it behaves
  transparently. During decoding we maintain a counter which allows the module
  to either be applied every num_levels iterations or to otherwise return cached
  values.

  Attributes:
    temporal_layer_factory: A MakeDecoderLayerFn
    num_temporal_layers: Number of temporal decoder layers.
    num_levels: Number of RVQ levels.
    relpos_bias: An instance of a shared relative position bias module, usually
      owned by the Decoder.
  """

  temporal_layer_factory: t5_architecture.MakeDecoderLayerFn
  num_temporal_layers: Optional[int] = None
  num_levels: Optional[int] = None
  relpos_bias: Optional[nn.Module] = None
  layer_remat: str = 'none'
  mean_pool_input: bool = True

  def setup(self):
    # The underlying nn.Module being wrapped
    self._module = TemporalDecoderStack(
        num_levels=self.num_levels,
        num_temporal_layers=self.num_temporal_layers,
        temporal_layer_factory=self.temporal_layer_factory,
        relpos_bias=self.relpos_bias,
    )

  @nn.compact
  def __call__(
      self,
      embedded_inputs,
      encoder_outputs=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
      *,
      logit_mask=None,
      enable_dropout: bool = False,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      **kwargs,
  ):
    """Runs the temporal decoder stack periodically.

    Args:
      embedded_inputs: Embedded input tokens.
      encoder_outputs: The outputs from the encoder. If None, do not attend to
        encoder outputs, resulting in a decoder only model.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      logit_mask: a mask to be applied to the attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      **kwargs: Optional keyword arguments to pass to
        decode_from_continuous_inputs.

    Returns:
        Temporal context for the depth decoder stack.
    """

    if self.is_initializing():
      # True if model.init() is called
      init_rng = {'params': self.make_rng('params')}
      if self.has_rng('dropout'):
        init_rng['dropout'] = self.make_rng('dropout')

      module_params = self._module.init(
          init_rng,
          embedded_inputs,
          encoded=encoder_outputs,
          decoder_mask=decoder_mask,
          encoder_decoder_mask=encoder_decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=False,
          decode=decode,
          max_decode_length=max_decode_length,
      )  # this is a dict which should have {'params', 'params_axes', ...}
      params = module_params['params']
      params_axes = module_params['params_axes']
      _copy_to_scope(self.scope, params, 'params')
      _copy_to_scope(self.scope, params_axes, 'params_axes')
    else:
      params = self.variables.get('params', {})
      params_axes = self.variables.get('params_axes', {})
    if enable_dropout:
      rngs = {'dropout': self.make_rng('dropout')}
    else:
      rngs = {}

    if not decode:
      # This is the default train-time behaviour.
      # Run the module with empty cache
      return self._module.apply(
          {'params': params, 'params_axes': params_axes, 'cache': {}},
          embedded_inputs,
          encoded=encoder_outputs,
          decoder_mask=decoder_mask,
          encoder_decoder_mask=encoder_decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout,
          decode=False,
          max_decode_length=max_decode_length,
          rngs=rngs,
      )

    # Check if we have already initialized the counter in cache.
    is_initialized = self.has_variable('cache', 'call_counter')

    # Initializes or retrieves the call_counter.
    # If call_counter exists self.variable doesn't throw an error because
    # 'cache' is a mutable collection.
    call_counter = self.variable(
        'cache', 'call_counter', jnp.zeros, (), jnp.int32
    )
    module_scope = self.scope

    if not is_initialized:
      # This is the first time we call with decode=True so we pass
      # init_cache=True and 'cache'={}. We must make 'cache' mutable.
      y, module_variables = self._module.apply(
          {'params': params, 'params_axes': params_axes, 'cache': {}},
          embedded_inputs,
          encoded=encoder_outputs,
          decoder_mask=decoder_mask,
          encoder_decoder_mask=encoder_decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          init_cache=True,
          rngs=rngs,
          mutable=['cache'],
      )
      module_variables = module_variables['cache']
      # We must create dummy shapes which are like embedded_inputs with T=1.
      dummy_batch, *_, dummy_dims = embedded_inputs.shape
      dummy_shapes = [dummy_batch, 1, dummy_dims]

      # Need to be careful with cached input shapes
      if self.mean_pool_input:
        cached_input = self.variable(
            'cache',
            'cached_input',
            jnp.zeros,
            [self.num_levels] + dummy_shapes,  # tuple(embedded_inputs.shape),
            embedded_inputs.dtype,
        )
      cached_output = self.variable(
          'cache', 'cached_output', jnp.zeros, dummy_shapes, y.dtype
      )

    else:
      if self.mean_pool_input:
        cached_input = self.variable('cache', 'cached_input')
        # The first input will have sequence length 1 and therefore correspond
        # to RVQ depth 0. However, the temporal model expects a mean embedding
        # as input. Therefore we duplicate this num_levels times and pass it
        # to the mean-pooling operator.
        updated_cached_input0 = jnp.concatenate(
            [embedded_inputs[jnp.newaxis]] * self.num_levels, axis=0
        )
        # Otherwise we append the latest input embedding to the cache 'queue'
        # (dropping the first, stale embedding) and pass this to the mean-pool
        # operator (note it won't get mean-pooled until this operation has
        # happened num_levels times indicating we have a full input frame).
        updated_cached_input1 = jnp.concatenate(
            [cached_input.value[1:], embedded_inputs[jnp.newaxis]], axis=0
        )
        updated_cached_input = jax.lax.cond(
            call_counter.value == 0,
            lambda: updated_cached_input0,
            lambda: updated_cached_input1,
        )
        module_inputs = jnp.mean(updated_cached_input, axis=0, keepdims=False)
      else:
        module_inputs = embedded_inputs
      cached_output = self.variable('cache', 'cached_output')
      # We must pop the newly created variables from the cache before performing
      # an apply step: the cache passed to the child modules must contain only
      # expected variables.
      module_cache, _ = flax.core.pop(
          module_scope.variables()['cache'], 'call_counter'
      )
      module_cache, _ = flax.core.pop(module_cache, 'cached_output')

      def _return_cached_output(cached_output, module_cache, inputs):
        del inputs  # not needed, we will return the cache.
        return cached_output, module_cache

      def _return_updated_output(cached_output, module_cache, inputs):
        del cached_output  # not needed, we will return new values.
        y, module_variables = self._module.apply(
            {
                'params': params,
                'params_axes': params_axes,
                'cache': module_cache,
            },
            inputs,
            encoded=encoder_outputs,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            logit_mask=None,
            enable_dropout=enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
            rngs=rngs,
            mutable=['cache'],
        )
        module_variables = module_variables['cache']

        return y, module_variables

      is_call = call_counter.value % self.num_levels == 0
      y, module_variables = jax.lax.cond(
          is_call,
          _return_updated_output,
          _return_cached_output,
          cached_output.value,
          module_cache,
          module_inputs,
      )

    # Update the cache.
    _copy_to_scope(module_scope, module_variables, 'cache')

    # It seems important to do this after _copy_to_scope instead of the
    # opposite, otherwise the following 2 values get wiped out by the
    # _copy_to_scope call.
    if is_initialized:
      # The cache initialization time is considered as a "dummy" step.
      # The counter is only increased when doing real steps.
      call_counter.value = call_counter.value + 1
      cached_output.value = y
      if self.mean_pool_input:
        # pylint: disable-next=undefined-variable
        cached_input.value = updated_cached_input

    return y


class PeriodicResetDepthDecoder(nn.Module):
  """Wrapper for DepthDecoderStack which allows periodic resetting.

  This module wraps a stack of decoder layers. During training it behaves
  transparently. During decoding we maintain a counter and manually reset the
  cache every num_levels calls, ensuring the effective sequence length in the
  depth decoder is <Q.

  Attributes:
    depth_layer_factory: A MakeDecoderLayerFn
    num_depth_layers: Number of depth decoder layers.
    num_levels: Number of RVQ levels.
    relpos_bias_depth: An instance of a shared relative position bias module,
      usually owned by the Decoder.
    depth_dims_converter_factory: An optional nn.Module which can convert
      between larger model dimensions and smaller dimensions in the depth
      decoder for more efficient operation.
  """

  depth_layer_factory: t5_architecture.MakeDecoderLayerFn
  num_depth_layers: Optional[int] = None
  num_levels: Optional[int] = None

  relpos_bias_depth: Optional[nn.Module] = None
  depth_dims_converter_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    self._module = DepthDecoderStack(
        num_levels=self.num_levels,
        num_depth_layers=self.num_depth_layers,
        depth_layer_factory=self.depth_layer_factory,
        relpos_bias_depth=self.relpos_bias_depth,
        depth_dims_converter_factory=self.depth_dims_converter_factory,
    )
    # Now we expect resetting to happen once every Q steps.

  @nn.compact
  def __call__(
      self,
      temporal_context,
      embedded_inputs,
      decoder_mask,
      *,
      logit_mask=None,
      enable_dropout=False,
      decode=False,
      max_decode_length=None,
  ) -> Array:
    if self.is_initializing():
      init_rng = {'params': self.make_rng('params')}
      if self.has_rng('dropout'):
        init_rng['dropout'] = self.make_rng('dropout')
      module_params = self._module.init(
          init_rng,
          temporal_context,
          embedded_inputs,
          decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=False,
          decode=decode,
          max_decode_length=max_decode_length,
      )  # this is a dict which should have {'params', 'params_axes', ...}
      params = module_params['params']
      params_axes = module_params['params_axes']
      _copy_to_scope(self.scope, params, 'params')
      _copy_to_scope(self.scope, params_axes, 'params_axes')
    else:
      params = self.variables.get('params', {})
      params_axes = self.variables.get('params_axes', {})
    if enable_dropout:
      rngs = {'dropout': self.make_rng('dropout')}
    else:
      rngs = {}

    if not decode:
      return self._module.apply(
          {'params': params, 'params_axes': params_axes, 'cache': {}},
          temporal_context,
          embedded_inputs,
          decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          rngs=rngs,
      )

    is_initialized = self.has_variable('cache', 'reset_counter')
    reset_counter = self.variable(
        'cache', 'reset_counter', jnp.zeros, (), jnp.int32
    )
    # We create dummy variables to initialize the depth decoder cache.
    dummy_temporal_shape = self.variable(
        'cache',
        'dummy_temporal_shape',
        jnp.zeros,
        len(temporal_context.shape),
        jnp.int32,
    )
    dummy_embedded_shape = self.variable(
        'cache',
        'dummy_embedded_shape',
        jnp.zeros,
        len(embedded_inputs.shape),
        jnp.int32,
    )

    if not is_initialized:
      # The cache is reset every num_levels decodes so we must remember the
      # original shapes of temporal_context and embedded_inputs at init
      # (during decode they will have T=1).
      dummy_temporal_shape.value = jnp.ones(temporal_context.shape)
      dummy_embedded_shape.value = jnp.ones(embedded_inputs.shape)

    module_scope = self.scope

    # If we are decoding or initing the cache, max_decode_len is always
    # num_levels
    max_decode_length = self.num_levels

    # Note that this is cheap, since we don't use the output values but only
    # the freshly initialized cache. So jitting should get rid of part of the
    # tree that computes the output (since not used). Here we use the dummy
    # versions of temporal_context and embedded_inputs to init the depth cache.
    dummy_y, initial_module_variables = self._module.apply(
        {'params': params, 'params_axes': params_axes, 'cache': {}},
        temporal_context=dummy_temporal_shape.value,
        embedded_inputs=dummy_embedded_shape.value,
        decoder_mask=decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        init_cache=True,
        rngs=rngs,
        mutable=['cache'],
    )
    initial_module_variables = initial_module_variables['cache']

    if not is_initialized:  # len(module_cache) == 0:
      y = dummy_y
      module_variables = initial_module_variables

    else:
      module_cache, _ = flax.core.pop(
          module_scope.variables()['cache'], 'reset_counter'
      )
      module_cache, _ = flax.core.pop(module_cache, 'dummy_temporal_shape')
      module_cache, _ = flax.core.pop(module_cache, 'dummy_embedded_shape')

      def _first_step(
          initial_module_variables,
          module_cache,
          temporal_context,
          embedded_inputs,
      ):
        del module_cache
        # For one real step, we do the cache initialization step (does not count
        # as a decode step) and a real decode step.
        y1, module_variables1 = self._module.apply(
            {
                'params': params,
                'params_axes': params_axes,
                'cache': initial_module_variables,
            },
            temporal_context=temporal_context,
            embedded_inputs=embedded_inputs,
            enable_dropout=enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
            rngs=rngs,
            mutable=['cache'],
        )
        module_variables1 = module_variables1['cache']
        return y1, module_variables1

      def _standard_step(
          initial_module_variables,
          module_cache,
          temporal_context,
          embedded_inputs,
      ):
        # We don't need the temporal context, the previously decoded state is
        # passed in through embedded_inputs
        del initial_module_variables, temporal_context
        # Standard step where we re-use the previous state.
        y2, module_variables2 = self._module.apply(
            {
                'params': params,
                'params_axes': params_axes,
                'cache': module_cache,
            },
            temporal_context=embedded_inputs,  # Ignored.
            embedded_inputs=embedded_inputs,  # Previously decoded embedding.
            enable_dropout=enable_dropout,
            decode=decode,
            max_decode_length=max_decode_length,
            rngs=rngs,
            mutable=['cache'],
        )
        module_variables2 = module_variables2['cache']
        return y2, module_variables2

      # Select the state based on whether we should reset or not the underlying
      # module.
      is_reset = reset_counter.value % self.num_levels == 0
      y, module_variables = jax.lax.cond(
          is_reset,
          _first_step,
          _standard_step,
          initial_module_variables,
          module_cache,
          temporal_context,
          embedded_inputs,
      )

    # Update the cache.
    _copy_to_scope(module_scope, module_variables, 'cache')

    # It seems important to do that after _copy_to_scope instead of the
    # opposite.
    if is_initialized:
      # The cache initialization time is considered as a "dummy" step.
      # The counter is only increased when doing real steps.
      reset_counter.value = reset_counter.value + 1

    return y
