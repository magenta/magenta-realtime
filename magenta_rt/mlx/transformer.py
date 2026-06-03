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

"""Transformer layers."""

import dataclasses
from typing import Callable, Literal, Protocol, Sequence

from mlx import nn
import flax.linen as flax_nn
import mlx.core as mx
import numpy as np
import sequence_layers.mlx as sl


NormType = Literal['rms_normalization', 'layer_normalization']
NormPolicy = Literal['primer_hybrid']

stacked_attention_input_projection_init = flax_nn.initializers.variance_scaling(
    1.0,
    'fan_in',
    'truncated_normal',
    in_axis=-4,
    out_axis=(-2, -1),
    batch_axis=(-3,),
)

attention_input_projection_init = flax_nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal', in_axis=-3, out_axis=(-2, -1)
)

attention_output_projection_kernel_init = flax_nn.initializers.variance_scaling(
    1.0,
    'fan_in',
    'truncated_normal',
    in_axis=(-2, -1),
    out_axis=-3,
)

dense_kernel_init = flax_nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)

LEVEL_AXIS = -2


class ReductionFn(Protocol):
  """Callable that reduces an array along an axis= kwarg."""

  def __call__(self, a, axis: int):
    pass


class MultiChannelEmbedding(sl.Stateless):
  """Retrieves learnable embeddings of multi-channel integer input codes.

  The layer expects multi-channel/multi-level int-valued tokens as inputs.  Two
  examples are the residual-vector-quantized (RVQ) tokens of a speech encoder or
  product-quantized (PQ) tokens.

  The parameter num_channels controls the number of token levels that are
  expected. For each token level/channel, separate embeddings are learned.

  After the level/channel-wise embedding lookup, the embeddings can optionally
  be reduced/aggregated (e.g., via mean or sum over the channel axis (RVQ), or
  by unstacking channels and concatenating on the depth axis (PQ)).

  NOTE: Unlike sl.SequenceEmbedding, this layer is stateless/time-invariant and
  expects a stack of tokens (i.e., multiple channels/levels) for every time
  step.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Config for MultiChannelEmbedding.

    Attributes:
      dimension: The common dimensionality the per-channel embeddings.
      num_embeddings_per_channel: The number of tokens/embeddings per channel.
        If a single int is provided, the same number of embeddings is used for
        each channel.
      num_channels: The number of channels (e.g. RVQ levels). The input sequence
        is expected to have a channel shape of (num_channels,), i.e., its values
        should be of shape (batch, time, num_channels).
      num_reserved_embeddings: If non-zero, the number of embeddings to treat as
        "reserved". If they occur in any channel, they are not offset by that
        channel's offset.
      reduction_fn: Optional reduction function to aggregate the channel-wise
        embeddings over the channel axis. The reduction should accept an axis=
        keyword argument, the index of the axis over which to reduce. Standard
        examples for reduction_fn are mx.mean and mx.sum. If None, no
        reduction is applied.
      compute_dtype: Dtype to which the outputs are promoted. If None, the
        outputs are of param_dtype.
      param_dtype: The dtype of the learned embedding parameters.
      embedding_init: The initializer of the embedding.
      round_num_embeddings_to_multiple_of_128: Whether to round the total number
        of embeddings across all channels (including num_reserved_embeddings) to
        a multiple of 128. This is useful to improve behavior on TPU.
      name: The name of the layer.
    """

    dimension: int
    num_embeddings_per_channel: Sequence[int]
    num_channels: int
    num_reserved_embeddings: int = 0
    reduction_fn: ReductionFn | None = None
    compute_dtype: sl.DType | None = None
    param_dtype: sl.DType = mx.float32
    # By default, initialize embeddings to have a norm of 1.
    embedding_init: flax_nn.initializers.Initializer = flax_nn.linear.default_embed_init
    round_num_embeddings_to_multiple_of_128: bool = True
    name: str | None = None

    def make(self) -> 'MultiChannelEmbedding':
      return MultiChannelEmbedding(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    num_embeddings = config.num_reserved_embeddings + sum(
        config.num_embeddings_per_channel
    )
    if config.round_num_embeddings_to_multiple_of_128:
      num_embeddings = (num_embeddings + 128 - 1) // 128 * 128

    self.embedding = mx.zeros(
        (
            num_embeddings,
            config.dimension,
        ),
        dtype=config.param_dtype,
    )


  def _validate_dtype(self, dtype: sl.DType):
    if not mx.issubdtype(dtype, mx.integer):
      raise ValueError(
          'Input to Embedding must be an integer or unsigned integer, got:'
          f' {dtype}'
      )


  def _validate_input(self, x: sl.Sequence):
    self._validate_dtype(x.dtype)
    if x.channel_shape != (self.config.num_channels,):
      raise ValueError(
          f'Expected a channel shape of ({self.config.num_channels=},) but got'
          f' input sequence with {x.channel_shape=}.'
      )


  def get_output_shape(
      self,
      input_shape: sl.ShapeLike,
      *,
      constants: sl.Constants | None = None,
  ) -> sl.Shape:
    embeddings_shape = tuple(input_shape) + (self.config.dimension,)
    if self.config.reduction_fn is None:
      return embeddings_shape
    else:
      # Compute the reduced shape via numpy to avoid needing jax.eval_shape.
      dummy = np.empty(embeddings_shape)
      return np.mean(dummy, axis=LEVEL_AXIS).shape


  def get_output_dtype(
      self, input_dtype: sl.DType, *, constants: sl.Constants | None = None
  ) -> sl.DType:
    self._validate_dtype(input_dtype)
    if self.config.compute_dtype is None:
      return self.config.param_dtype
    return self.config.compute_dtype

  @sl.check_layer
  def layer(
      self,
      x: sl.Sequence,
      *,
      training: bool = False,
      constants: sl.Constants | None = None,
  ) -> sl.Sequence:
    del training
    del constants

    self._validate_input(x)
    if self.config.compute_dtype is not None:
      embedding = self.embedding.astype(self.config.compute_dtype)
    else:
      embedding = self.embedding

    offsets = mx.cumsum(
        mx.array(
            [0] + list(self.config.num_embeddings_per_channel)[:-1], mx.int32
        )
    )

    # Do not offset reserved embeddings.
    if self.config.num_reserved_embeddings:
      offsets = mx.where(
          x.values < self.config.num_reserved_embeddings,
          0,
          offsets[mx.newaxis, mx.newaxis, :],
      )

    # Embed offset indices in x, preserving the channel shape.
    y = x.apply_values(lambda v: mx.take(embedding, v + offsets, axis=0))

    # Aggregate/reduce embeddings over the group / RVQ axis.
    if self.config.reduction_fn is not None:
      y = y.apply_values_masked(self.config.reduction_fn, axis=LEVEL_AXIS)
    return y


def get_pre_and_post_norm(
    norm_type: NormType,
    norm_policy: NormPolicy,
    param_dtype: sl.DType | None = None,
    reductions_in_at_least_fp32: bool = False,
    adaptive_norm_condition_name: str | None = None,
) -> tuple[sl.SequenceLayerConfig, sl.SequenceLayerConfig]:
  """Returns pre / post norm determined by norm_policy and norm_type."""

  def get_norm(name: str) -> sl.SequenceLayerConfig:
    use_scale = adaptive_norm_condition_name is None

    match norm_type:
      case 'rms_normalization':
        config = sl.RMSNormalization.Config(
            name=name,
            use_scale=use_scale,
        )
      case _:
        raise NotImplementedError(f'Unsupported norm type: {norm_type}')

    norm = config.copy(param_dtype=param_dtype) if param_dtype else config

    if adaptive_norm_condition_name is not None:
      norm = sl.Serial.Config([
          norm,
          sl.Conditioning.Config(
              conditioning_name=adaptive_norm_condition_name,
              projection=sl.Conditioning.Projection.LINEAR_AFFINE,
              combination=sl.Conditioning.Combination.AFFINE,
              streaming=True,
              param_dtype=param_dtype,
          ),
      ])

    return norm

  match norm_policy:
    case 'primer_hybrid':
      pre_norm = get_norm('pre_norm')
      post_norm = get_norm('post_norm')
    case _:
      raise NotImplementedError(f'Unsupported norm policy: {norm_policy}')
  return pre_norm, post_norm


def _get_query_key_value_networks(
    use_rope: bool,
    *,
    rope_only_advance_position_for_valid_timesteps: bool,
    rope_positions_in_at_least_fp32: bool,
    query_positions_name: str | None = None,
    key_positions_name: str | None = None,
) -> tuple[
    sl.SequenceLayerConfig, sl.SequenceLayerConfig, sl.SequenceLayerConfig
]:
  """Returns query, key, and value networks for self and cross attention."""

  # Apply RoPE to queries and keys if enabled.
  if use_rope:

    if rope_only_advance_position_for_valid_timesteps and (
        query_positions_name or key_positions_name
    ):
      raise ValueError(
          'rope_only_advance_position_for_valid_timesteps is incompatible'
          ' with externally fed positions'
          f' ({query_positions_name=} {key_positions_name=})'
      )

    # Use RoPE for the queries and keys.
    maybe_query_rope = sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=10000,
        only_advance_position_for_valid_timesteps=rope_only_advance_position_for_valid_timesteps,
        positions_in_at_least_fp32=rope_positions_in_at_least_fp32,
        positions_name=query_positions_name,
        name='rope',
    )
    maybe_key_rope = sl.ApplyRotaryPositionalEncoding.Config(
        max_wavelength=10000,
        only_advance_position_for_valid_timesteps=rope_only_advance_position_for_valid_timesteps,
        positions_in_at_least_fp32=rope_positions_in_at_least_fp32,
        positions_name=key_positions_name,
        name='rope',
    )
  else:
    maybe_query_rope = maybe_key_rope = sl.Identity.Config()

  # Shard Q, K, and V projection outputs with activation_btnh.
  query_network = sl.Serial.Config(
      [
          sl.CheckpointName.Config('query_proj'),
          maybe_query_rope,
      ],
      name='query_network',
  )
  key_network = sl.Serial.Config(
      [
          sl.CheckpointName.Config('key_proj'),
          maybe_key_rope,
      ],
      name='key_network',
  )
  value_network = sl.Serial.Config(
      [
          sl.CheckpointName.Config('value_proj'),
      ],
      name='value_network',
  )

  return query_network, key_network, value_network


@dataclasses.dataclass(frozen=True)
class SLSelfAttention(sl.SequenceLayerConfig):
  """A residual self-attention layer."""

  # The model dimension. Input and output sequences from this layer are shaped
  # [b, t, model_dimension].
  model_dimension: int
  # The number of units per attention head.
  units_per_head: int
  # The number of attention heads.
  num_heads: int
  # The maximum number of timesteps that each timestep can look into the past
  # (not counting itself). -1 means unmasked (infinite past). Must be
  # non-negative if use_local_attention is True.
  max_past_horizon: int
  # The maximum number of timesteps that each timestep can look into the future
  # (not counting itself). -1 means unmasked (infinite future). Must be
  # non-negative if use_local_attention is True.
  max_future_horizon: int
  # If positive, a soft cap applied to attention logits to prevent blowup.
  logits_soft_cap: float | None
  # Whether to learn a [units_per_head] query scale factor across all query
  # heads. If false, queries are scaled by 1/sqrt(units_per_head).
  per_dim_scale: bool
  # Outputs all-zeros context vectors for queries which have nothing to attend
  # to (i.e. all possible keys are masked).
  zero_fully_masked: bool
  # Whether to use Rotary Positional Encodings (RoPE) for queries and keys.
  use_rope: bool
  # If true, uses biases for the query, key and value projections.
  use_bias: bool
  # The norm type to use (e.g. RMSNorm, LayerNorm, etc.).
  norm_type: NormType
  # The norm "policy" to use, e.g. pre-norm, post-norm, or both.
  norm_policy: NormPolicy
  # If true, uses "local" dot product self attention. Functionally equivalent to
  # regular dot product self attention when max_past_horizon >= 0 and
  # max_future_horizon >= 0, but uses a more efficient implementation.
  use_local_attention: bool
  # If positive, the dropout rate to use.
  dropout_rate: float
  # Whether to broadcast dropout across time.
  broadcast_dropout_across_time: bool

  # The dtype of the layer's computations.
  compute_dtype: sl.DType | None
  # The dtype of the layer's parameters.
  param_dtype: sl.DType
  rope_only_advance_position_for_valid_timesteps: bool = True
  rope_positions_in_at_least_fp32: bool = True
  # Whether to perform reductions (for LayerNorm/RMSNorm) in at least fp32.
  reductions_in_at_least_fp32: bool = False
  # Whether to use separate or combined query, key and value projections for
  # self attention. This has no impact on the algorithm or number of
  # parameters, but in practice separate QKV matrices can lead to improved
  # performance on TPU.
  use_separate_qkv: bool = False
  # Whether to use an experimental ring buffer implementation for the KV cache
  # updates. Limitations:
  # * Incompatible with attention sinks.
  # * Incompatible with relative_position_embedding.
  # * Requires streaming step sizes of 1.
  use_kv_cache_ringbuffer: bool = False
  # The number of attentions sinks.
  num_sink_embeddings: int = 0
  # If True, use learnable attention sink scalars (one per head).
  use_sink_scalars: bool = False
  # If specified, the name of a [batch_size, time] constant which indicates
  # query/key positions for the relative position embedding. If unspecified, the
  # position along the query's time dimension will be used.
  # NOTE: Currently only supported for RoPE.
  positions_name: str | None = None
  # If not None, use adaptive normalization with the given condition name.
  adaptive_norm_condition_name: str | None = None
  # If defined, the dropout rate to use for self-attention probabilities.
  # Otherwise, dropout_rate is used.
  attention_dropout_rate: float | None = None
  # An optional name for the layer.
  name: str | None = 'self_attention'

  def make(self) -> sl.SequenceLayer:
    pre_norm, post_norm = get_pre_and_post_norm(
        self.norm_type,
        self.norm_policy,
        self.param_dtype,
        self.reductions_in_at_least_fp32,
        self.adaptive_norm_condition_name,
    )
    dropout_broadcast_dims = (1,) if self.broadcast_dropout_across_time else ()

    query_network, key_network, value_network = _get_query_key_value_networks(
        self.use_rope,
        rope_only_advance_position_for_valid_timesteps=self.rope_only_advance_position_for_valid_timesteps,
        rope_positions_in_at_least_fp32=self.rope_positions_in_at_least_fp32,
        query_positions_name=self.positions_name,
        key_positions_name=self.positions_name,
    )

    if self.use_separate_qkv:
      input_projection = sl.SeparateQueryKeyValueProjection(
          q_kernel_init=attention_input_projection_init,
          k_kernel_init=attention_input_projection_init,
          v_kernel_init=attention_input_projection_init,
      )
    else:
      input_projection = sl.CombinedQueryKeyValueProjection(
          qkv_kernel_init=stacked_attention_input_projection_init,
      )

    attention_dropout_rate = (
        self.attention_dropout_rate
        if self.attention_dropout_rate is not None
        else self.dropout_rate
    )

    if self.use_local_attention:
      block_size = max(1, self.max_past_horizon, self.max_future_horizon)
      self_attention = sl.LocalDotProductSelfAttention.Config(
          units_per_head=self.units_per_head,
          num_heads=self.num_heads,
          block_size=block_size,
          max_past_horizon=self.max_past_horizon,
          max_future_horizon=self.max_future_horizon,
          use_bias=self.use_bias,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
          attention_logits_soft_cap=self.logits_soft_cap,
          per_dim_scale=self.per_dim_scale,
          attention_probabilities_dropout_rate=attention_dropout_rate,
          broadcast_dropout_across_queries=self.broadcast_dropout_across_time,
          input_projection=input_projection,
          zero_fully_masked=self.zero_fully_masked,
          compute_dtype=self.compute_dtype,
          param_dtype=self.param_dtype,
          num_sink_embeddings=self.num_sink_embeddings,
          use_sink_scalars=self.use_sink_scalars,
          use_kv_cache_ringbuffer=self.use_kv_cache_ringbuffer,
          name='attention',
      )
    else:
      self_attention = sl.DotProductSelfAttention.Config(
          units_per_head=self.units_per_head,
          num_heads=self.num_heads,
          max_past_horizon=self.max_past_horizon,
          max_future_horizon=self.max_future_horizon,
          use_bias=self.use_bias,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
          attention_logits_soft_cap=self.logits_soft_cap,
          per_dim_scale=self.per_dim_scale,
          attention_probabilities_dropout_rate=attention_dropout_rate,
          broadcast_dropout_across_queries=self.broadcast_dropout_across_time,
          input_projection=input_projection,
          zero_fully_masked=self.zero_fully_masked,
          compute_dtype=self.compute_dtype,
          param_dtype=self.param_dtype,
          num_sink_embeddings=self.num_sink_embeddings,
          use_sink_scalars=self.use_sink_scalars,
          use_kv_cache_ringbuffer=self.use_kv_cache_ringbuffer,
          name='attention',
      )

    return sl.Residual.Config(
        [
            pre_norm,
            self_attention,
            sl.EinsumDense.Config(
                '...nh,dnh->...d',
                [self.model_dimension],
                bias_axes='d' if self.use_bias else None,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name='output_projection',
            ),
            sl.CheckpointName.Config('output_projection'),
            post_norm,
            sl.Dropout.Config(
                self.dropout_rate,
                broadcast_dims=dropout_broadcast_dims,
                name='output_dropout',
            ),
        ],
        # Add a step-only delay to match the input latency introduced by when
        # max_future_horizon > 0.
        shortcut_layers=[
            sl.Delay.Config(
                self.max_future_horizon if self.max_future_horizon > 0 else 0,
                delay_layer_output=False,
            )
        ],
        name=self.name,
    ).make(backend="mlx")


@dataclasses.dataclass(frozen=True)
class SLStreamingCrossAttention(sl.SequenceLayerConfig):
  """A residual streaming cross-attention layer."""

  source_name: str
  # The model dimension. Input and output sequences from this layer are shaped
  # [b, t, model_dimension].
  model_dimension: int
  # The number of units per attention head.
  units_per_head: int
  # The number of attention heads.
  num_heads: int
  # The maximum number of timesteps that each timestep can look into the past
  # (not counting itself). Must be non-negative.
  max_past_horizon: int
  # The maximum number of timesteps that each timestep can look into the future
  # (not counting itself). Must be non-negative.
  max_future_horizon: int
  # If true, achieves lookahead by max_future_horizon by internally computing a
  # query delay buffer.
  use_query_delay_buffer: bool
  # If positive, a soft cap applied to attention logits to prevent blowup.
  logits_soft_cap: float | None
  # Whether to learn a [units_per_head] query scale factor across all query
  # heads. If false, queries are scaled by 1/sqrt(units_per_head).
  per_dim_scale: bool
  # Outputs all-zeros context vectors for queries which have nothing to attend
  # to (i.e. all possible keys are masked).
  zero_fully_masked: bool
  # Whether to use Rotary Positional Encodings (RoPE) for queries and keys.
  use_rope: bool
  # If true, uses biases for the query, key and value projections.
  use_bias: bool
  # The norm type to use (e.g. RMSNorm, LayerNorm, etc.).
  norm_type: NormType
  # The norm "policy" to use, e.g. pre-norm, post-norm, or both.
  norm_policy: NormPolicy
  # If positive, the dropout rate to use.
  dropout_rate: float
  # Whether to broadcast dropout across time.
  broadcast_dropout_across_time: bool

  # The dtype of the layer's computations.
  compute_dtype: sl.DType | None
  # The dtype of the layer's parameters.
  param_dtype: sl.DType
  # If specified, the name of a [batch_size, time] constant which indicates
  # query positions for the relative position embedding. If unspecified, the
  # position along the query's time dimension will be used.
  # NOTE: Currently only supported for RoPE.
  query_positions_name: str | None = None
  # If specified, the name of a [batch_size, time] constant which indicates key
  # positions for the relative position embedding. If unspecified, the position
  # along the key's time dimension will be used.
  # NOTE: Currently only supported for RoPE.
  key_positions_name: str | None = None
  rope_only_advance_position_for_valid_timesteps: bool = True
  rope_positions_in_at_least_fp32: bool = True
  reductions_in_at_least_fp32: bool = False
  # Whether to use separate or combined key / value projections. This has no
  # impact on the algorithm or number of parameters, but in practice separate KV
  # matrices can lead to improved performance on TPU.
  use_separate_kv: bool = False
  # Whether to use an experimental ring buffer implementation for the KV cache
  # updates. This implementation is more compute and memory efficient than the
  # default implementation on TPU.
  #
  # Limitations:
  # * Incompatible with attention sinks.
  # * Incompatible with relative_position_embedding.
  # * Requires streaming step sizes of 1.
  use_kv_cache_ringbuffer: bool = False

  # Number of attention sinks.
  num_sink_embeddings: int = 0
  # If True, use learnable attention sink scalars (one per head).
  use_sink_scalars: bool = False
  # If not None, use adaptive normalization with the given condition name.
  adaptive_norm_condition_name: str | None = None
  # An optional name for the layer.
  name: str | None = 'cross_attention'

  def make(self) -> sl.SequenceLayer:
    pre_norm, post_norm = get_pre_and_post_norm(
        self.norm_type,
        self.norm_policy,
        self.param_dtype,
        self.reductions_in_at_least_fp32,
        self.adaptive_norm_condition_name,
    )
    dropout_broadcast_dims = (1,) if self.broadcast_dropout_across_time else ()
    if (
        not self.use_query_delay_buffer
        and self.use_rope
        and not self.rope_only_advance_position_for_valid_timesteps
    ):
      raise ValueError(
          'use_query_delay_buffer=False requires'
          ' rope_only_advance_position_for_valid_timesteps if rope is enabled.'
      )

    query_network, key_network, value_network = _get_query_key_value_networks(
        self.use_rope,
        rope_only_advance_position_for_valid_timesteps=self.rope_only_advance_position_for_valid_timesteps,
        rope_positions_in_at_least_fp32=self.rope_positions_in_at_least_fp32,
        query_positions_name=self.query_positions_name,
        key_positions_name=self.key_positions_name,
    )

    if self.use_separate_kv:
      input_projection = sl.SeparateQueryKeyValueProjection(
          q_kernel_init=attention_input_projection_init,
          k_kernel_init=attention_input_projection_init,
          v_kernel_init=attention_input_projection_init,
      )
    else:
      input_projection = sl.QueryAndKeyValueProjection(
          q_kernel_init=attention_input_projection_init,
          kv_kernel_init=stacked_attention_input_projection_init,
      )

    return sl.Residual.Config(  # pylint: disable=g-long-ternary
        [
            pre_norm,
            sl.StreamingDotProductAttention.Config(
                source_name=self.source_name,
                units_per_head=self.units_per_head,
                num_heads=self.num_heads,
                block_size=max(
                    1, self.max_past_horizon, self.max_future_horizon
                ),
                max_past_horizon=self.max_past_horizon,
                max_future_horizon=self.max_future_horizon,
                use_query_delay_buffer=self.use_query_delay_buffer,
                use_bias=self.use_bias,
                query_network=query_network,
                key_network=key_network,
                value_network=value_network,
                attention_logits_soft_cap=self.logits_soft_cap,
                per_dim_scale=self.per_dim_scale,
                attention_probabilities_dropout_rate=self.dropout_rate,
                broadcast_dropout_across_queries=self.broadcast_dropout_across_time,
                input_projection=input_projection,
                zero_fully_masked=self.zero_fully_masked,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                num_sink_embeddings=self.num_sink_embeddings,
                use_sink_scalars=self.use_sink_scalars,
                use_kv_cache_ringbuffer=self.use_kv_cache_ringbuffer,
                name='attention',
            ),
            sl.EinsumDense.Config(
                '...nh,dnh->...d',
                [self.model_dimension],
                bias_axes='d' if self.use_bias else None,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name='output_projection',
            ),
            sl.CheckpointName.Config('output_projection'),
            post_norm,
            sl.Dropout.Config(
                self.dropout_rate,
                broadcast_dims=dropout_broadcast_dims,
                name='output_dropout',
            ),
        ],
        # Add a step-only delay to match the input latency introduced by
        # StreamingLocalDotProductAttention when max_future_horizon > 0.
        shortcut_layers=[
            sl.Delay.Config(
                self.max_future_horizon if self.use_query_delay_buffer else 0,
                delay_layer_output=False,
            )
        ],
        name=self.name,
    ).make(backend="mlx")


@dataclasses.dataclass(frozen=True)
class SLTransformerFFN(sl.SequenceLayerConfig):
  """A residual feed-forward network (FFN) layer."""

  # The model dimension. Input and output sequences from this layer are shaped
  # [b, t, model_dimension].
  model_dimension: int
  # The dimension of the hidden Dense layer. If gated is True, this is doubled
  # to predict parameters for the gate and activation.
  hidden_dimension: int
  # The activation to use for the hidden Dense layer.
  activation: Callable[[mx.array], mx.array]
  # The norm type to use (e.g. RMSNorm, LayerNorm, etc.).
  norm_type: NormType
  # The norm "policy" to use, e.g. pre-norm, post-norm, or both.
  norm_policy: NormPolicy
  # If true, a gated activation is applied as in
  # https://arxiv.org/abs/2002.05202
  gated: bool
  # If positive, the dropout rate to use.
  dropout_rate: float
  # A list of axes to broadcast dropout across.
  dropout_broadcast_dims: tuple[int, ...]
  # Whether to use biases for the dense layers in this FFN.
  use_bias: bool

  # The dtype of the layer's computations.
  compute_dtype: sl.DType | None
  # The dtype of the layer's parameters.
  param_dtype: sl.DType
  # Whether to perform reductions (for LayerNorm/RMSNorm) in at least fp32.
  reductions_in_at_least_fp32: bool = False
  # If not None, use adaptive normalization with the given condition name.
  adaptive_norm_condition_name: str | None = None

  # An optional name for the layer.
  name: str | None = 'ffn'

  def make(self) -> sl.SequenceLayer:
    if self.gated:
      ffn_layer1 = [
          sl.Dense.Config(
              self.hidden_dimension * 2,
              use_bias=self.use_bias,
              compute_dtype=self.compute_dtype,
              param_dtype=self.param_dtype,
              name='ffn_layer1',
          ),
          sl.GatedUnit.Config(
              feature_activation=None, gate_activation=self.activation
          ),
      ]
    else:
      ffn_layer1 = [
          sl.Dense.Config(
              self.hidden_dimension,
              use_bias=self.use_bias,
              activation=self.activation,
              compute_dtype=self.compute_dtype,
              param_dtype=self.param_dtype,
              name='ffn_layer1',
          )
      ]

    pre_norm, post_norm = get_pre_and_post_norm(
        self.norm_type,
        self.norm_policy,
        self.param_dtype,
        self.reductions_in_at_least_fp32,
        self.adaptive_norm_condition_name,
    )
    return sl.Residual.Config(
        [
            pre_norm,
            *ffn_layer1,
            sl.Dropout.Config(
                self.dropout_rate,
                broadcast_dims=self.dropout_broadcast_dims,
                name='hidden_dropout',
            ),
            sl.Dense.Config(
                self.model_dimension,
                use_bias=self.use_bias,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name='ffn_layer2',
            ),
            sl.CheckpointName.Config('ffn2'),
            post_norm,
            sl.Dropout.Config(
                self.dropout_rate,
                broadcast_dims=self.dropout_broadcast_dims,
                name='output_dropout',
            ),
        ],
        name=self.name,
    ).make(backend="mlx")


class SLTransformer(sl.SerialCombinatorMixin, sl.Emitting):
  """Transformer in sequence layer fashion."""

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class Config(sl.SequenceLayerConfig):
    """Config for a praxis equivalent stacked transformer in sequence layer."""

    num_layers: int
    max_past_horizon: int
    max_future_horizon: int
    attention_logits_soft_cap: float | None = None
    attention_zero_fully_masked: bool
    # Whether to broadcast dropout across time (as in T5).
    broadcast_dropout_across_time: bool = False
    use_cross_attention: bool
    cross_attention_source_name: str | None
    use_local_attention: bool = True
    # The dtype of the layer's computations.
    compute_dtype: sl.DType | None = None
    # The dtype of the layer's parameters.
    param_dtype: sl.DType = mx.float32

    # Whether to use separate or combined query, key and value projections for
    # self attention. This has no impact on the algorithm or number of
    # parameters, but in practice separate QKV matrices can lead to improved
    # performance on TPU.
    self_attention_use_separate_qkv: bool = False
    # Whether to use separate or combined key and value projections for cross
    # attention. This has no impact on the algorithm or number of parameters,
    # but in practice separate KV matrices can lead to improved performance on
    # TPU.
    cross_attention_use_separate_kv: bool = False

    # If true, use a streaming cross attention implementation instead of global
    # cross attention.
    use_streaming_cross_attention: bool = False
    # If use_streaming_cross_attention is True, the maximum past horizon for
    # streaming cross attention. If None, defaults to max_past_horizon.
    streaming_cross_attention_max_past_horizon: int | None = None
    # If use_streaming_cross_attention is True, the maximum future horizon for
    # streaming cross attention. If None, defaults to max_future_horizon.
    streaming_cross_attention_max_future_horizon: int | None = None
    # Whether to use a query delay buffer in streaming cross attention to
    # support cross attention lookahead.
    streaming_cross_attention_use_query_delay_buffer: bool = True

    # Some defaults that don't usually need to be updated.
    # The model dimension. Input and output sequences from this layer are shaped
    # [b, t, model_dimension].
    model_dimension: int = 1024
    use_rope: bool = True
    rope_only_advance_position_for_valid_timesteps: bool = True
    rope_positions_in_at_least_fp32: bool = True
    # Whether to perform reductions (for LayerNorm/RMSNorm) in at least fp32.
    reductions_in_at_least_fp32: bool = False
    num_heads: int = 16
    units_per_head: int = 64
    # If positive, the dropout rate to use.
    dropout_rate: float = 0.0
    # If defined, the dropout rate to use for self-attention probabilities.
    # Otherwise, dropout_rate is used.
    self_attention_dropout_rate: float | None = None
    # Attention related.
    attention_use_bias: bool = False
    attention_per_dim_scale: bool = True
    # The norm type to use (e.g. RMSNorm, LayerNorm, etc.).
    norm_type: NormType = 'rms_normalization'
    # The norm "policy" to use, e.g. pre-norm, post-norm, or both.
    norm_policy: NormPolicy = 'primer_hybrid'
    # FFN related.
    ffn_activation: Callable[[mx.array], mx.array] = nn.gelu_approx
    ffn_dim: int = 4096
    ffn_use_bias: bool = True
    ffn_gated: bool = True
    # Defaults for repeated layer usage.
    use_repeated: bool = False
    num_repeats: int = 1
    # The number of attention sinks to use.
    num_attention_sink_embeddings: int = 0
    # If True, use learnable attention sink scalars (one per head).
    use_attention_sink_scalars: bool = False
    # Whether to use an experimental ring buffer implementation for the KV cache
    # updates for either self or streaming cross attention. Limitations:
    # * Incompatible with attention sinks.
    # * Incompatible with GQA.
    # * Incompatible with relative_position_embedding.
    # * Requires streaming step sizes of 1.
    self_attention_use_kv_cache_ringbuffer: bool = False
    streaming_cross_attention_use_kv_cache_ringbuffer: bool = False

    # An optional name for the layer.
    name: str = 'transformer'

    def make(self) -> 'SLTransformer':
      return SLTransformer(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    dropout_broadcast_dims = (
        (1,) if config.broadcast_dropout_across_time else ()
    )

    def transformer_block(name: str) -> sl.SequenceLayerConfig:

      if config.use_streaming_cross_attention:
        cross_attention = SLStreamingCrossAttention(
            source_name=config.cross_attention_source_name,
            model_dimension=config.model_dimension,
            units_per_head=config.units_per_head,
            num_heads=config.num_heads,
            max_past_horizon=config.streaming_cross_attention_max_past_horizon
            if config.streaming_cross_attention_max_past_horizon
            is not None
            else config.max_past_horizon,
            max_future_horizon=config.streaming_cross_attention_max_future_horizon
            if config.streaming_cross_attention_max_future_horizon
            is not None
            else config.max_future_horizon,
            use_query_delay_buffer=config.streaming_cross_attention_use_query_delay_buffer,
            logits_soft_cap=config.attention_logits_soft_cap,
            per_dim_scale=config.attention_per_dim_scale,
            zero_fully_masked=config.attention_zero_fully_masked,
            use_rope=config.use_rope,
            rope_only_advance_position_for_valid_timesteps=config.rope_only_advance_position_for_valid_timesteps,
            rope_positions_in_at_least_fp32=config.rope_positions_in_at_least_fp32,
            reductions_in_at_least_fp32=config.reductions_in_at_least_fp32,
            use_bias=config.attention_use_bias,
            norm_type=config.norm_type,
            norm_policy=config.norm_policy,
            dropout_rate=config.dropout_rate,
            broadcast_dropout_across_time=config.broadcast_dropout_across_time,
            compute_dtype=config.compute_dtype,
            param_dtype=config.param_dtype,
            num_sink_embeddings=config.num_attention_sink_embeddings,
            use_sink_scalars=config.use_attention_sink_scalars,
            use_separate_kv=config.cross_attention_use_separate_kv,
            use_kv_cache_ringbuffer=config.streaming_cross_attention_use_kv_cache_ringbuffer,
        )
      else:
        cross_attention = sl.Identity.Config()

      return sl.Serial.Config(
          [
              SLSelfAttention(
                  model_dimension=config.model_dimension,
                  units_per_head=config.units_per_head,
                  num_heads=config.num_heads,
                  max_past_horizon=config.max_past_horizon,
                  max_future_horizon=config.max_future_horizon,
                  logits_soft_cap=config.attention_logits_soft_cap,
                  per_dim_scale=config.attention_per_dim_scale,
                  zero_fully_masked=config.attention_zero_fully_masked,
                  use_rope=config.use_rope,
                  rope_only_advance_position_for_valid_timesteps=config.rope_only_advance_position_for_valid_timesteps,
                  rope_positions_in_at_least_fp32=config.rope_positions_in_at_least_fp32,
                  reductions_in_at_least_fp32=config.reductions_in_at_least_fp32,
                  use_bias=config.attention_use_bias,
                  norm_type=config.norm_type,
                  norm_policy=config.norm_policy,
                  use_local_attention=config.use_local_attention,
                  dropout_rate=config.dropout_rate,
                  attention_dropout_rate=config.self_attention_dropout_rate,
                  broadcast_dropout_across_time=config.broadcast_dropout_across_time,
                  compute_dtype=config.compute_dtype,
                  param_dtype=config.param_dtype,
                  num_sink_embeddings=config.num_attention_sink_embeddings,
                  use_sink_scalars=config.use_attention_sink_scalars,
                  use_separate_qkv=config.self_attention_use_separate_qkv,
                  use_kv_cache_ringbuffer=config.self_attention_use_kv_cache_ringbuffer,
              ),
              cross_attention,
              SLTransformerFFN(
                  model_dimension=config.model_dimension,
                  hidden_dimension=config.ffn_dim,
                  activation=config.ffn_activation,
                  norm_type=config.norm_type,
                  norm_policy=config.norm_policy,
                  gated=config.ffn_gated,
                  dropout_rate=config.dropout_rate,
                  dropout_broadcast_dims=dropout_broadcast_dims,
                  reductions_in_at_least_fp32=config.reductions_in_at_least_fp32,
                  use_bias=config.ffn_use_bias,
                  compute_dtype=config.compute_dtype,
                  param_dtype=config.param_dtype,
              ),
          ],
          name=name,
      )

    layer_configs = [
        transformer_block(f'x_layers_{i}')
        for i in range(config.num_layers)
    ]

    layer_configs = [
        *layer_configs,
    ]


    self.layers = [cfg.make(backend="mlx") for cfg in layer_configs]


# Register custom SequenceLayerConfig types as MLX backend factories.
# This allows them to be instantiated by parent configs (e.g. Serial.Config)
# when make(backend='mlx') is called.
_SLC = sl.SequenceLayerConfig
_SLC.register_backend_factory('mlx', MultiChannelEmbedding.Config, lambda c: c.make())
_SLC.register_backend_factory('mlx', SLSelfAttention, lambda c: c.make())
_SLC.register_backend_factory('mlx', SLStreamingCrossAttention, lambda c: c.make())
_SLC.register_backend_factory('mlx', SLTransformerFFN, lambda c: c.make())
_SLC.register_backend_factory('mlx', SLTransformer.Config, lambda c: c.make())


# Monkey-patch get_accumulated_input_latency onto MLX SequenceLayer base class.
# This method is defined in JAX's Steppable but is missing from the MLX backend.
# It's needed by SerialCombinatorMixin.input_latency when iterating serial layers.
import math as _math
if not hasattr(sl.SequenceLayer, 'get_accumulated_input_latency'):
  def _get_accumulated_input_latency(self, input_latency: int) -> int:
    """Returns the accumulated input latency of this layer."""
    return _math.ceil(input_latency / self.output_ratio) + self.input_latency
  sl.SequenceLayer.get_accumulated_input_latency = _get_accumulated_input_latency
