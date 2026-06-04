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

"""Depthformer model for sequence generation.

This file contains the implementation of a Depthformer model, including a
`MultivariateDecoder` and an `EncoderDecoder`. The model is designed for
generating multivariate sequences, potentially using vector quantization.
"""

import dataclasses
import fractions
import logging
import math
from typing import Any

import einops
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import jaxtyping
import optax
import sequence_layers.jax as sl
from sequence_layers.jax import utils

# pylint: disable=g-multiple-import, g-importing-member
from sequence_layers.jax.typing import AnyPyTree, ArrayT, Float, Int, ScalarFloat, ScalarInt, typed


# Constants for sampling.
TEMPERATURE_CONSTANT = 'temperature'
TOP_K_CONSTANT = 'top_k'
TOP_P_CONSTANT = 'top_p'
CLASSIFIER_FREE_GUIDANCE_SCALE_CONSTANT = 'classifier_free_guidance_scale'
CLASSIFIER_FREE_GUIDANCE_NEGATIVE_CONSTANT = 'classifier_free_guidance_negative'
PRNG_KEY_CONSTANT = 'prng_key'
MAX_LENGTH_CONSTANT = 'max_length'

# TODO(kehanghan): upstream this to sequence_layers.jax.typing.
PRNGKeyBatch = (
    jaxtyping.Key[jax.Array, '*B']
    | jaxtyping.UInt32[jax.Array, '*B 2']
    | jaxtyping.UInt32[jax.Array, '*B 4']
)


def get_prefixed_constants(constants: sl.Constants, prefix: str):
  """Extracts a dict of constants with the given prefix + arbitrary suffixes."""
  result = {}
  for key, value in constants.items():
    if key.startswith(prefix) and value is not None:
      result[key[len(prefix) + 1 :]] = value
  return result


def split_batch_dim(tree: AnyPyTree, split_size: int) -> AnyPyTree:
  def split_array(v):
    if v.shape[0] % split_size != 0:
      raise ValueError(
          f'Array {v.shape=} / {v.dtype} cannot be split with {split_size=}.'
      )
    return v.reshape((v.shape[0] // split_size, split_size) + v.shape[1:])

  return jax.tree.map(split_array, tree)


def merge_batch_dim(tree: AnyPyTree) -> AnyPyTree:
  def merge_array(v):
    batch_size, split_size = v.shape[:2]
    return v.reshape((batch_size * split_size,) + v.shape[2:])

  return jax.tree.map(merge_array, tree)


def interleave_sequences(
    a: sl.Sequence, b: sl.Sequence, *c: sl.Sequence
) -> sl.Sequence:
  """Interleaves at least two sequences on the batch dimension."""
  if a is b and all(a is seq for seq in c):
    # Special case reference equality as a tile.
    result = type(a)(
        jnp.tile(
            a.values[:, jnp.newaxis], (1, 2 + len(c)) + (1,) * (a.ndim - 1)
        ),
        jnp.tile(a.mask[:, jnp.newaxis], (1, 2 + len(c), 1)),
    )
  else:
    if (
        isinstance(a, sl.MaskedSequence)
        and isinstance(b, sl.MaskedSequence)
        and all(isinstance(seq, sl.MaskedSequence) for seq in c)
    ):
      result_type = sl.MaskedSequence
    else:
      result_type = sl.Sequence

    values = jnp.stack([a.values, b.values, *[seq.values for seq in c]], axis=1)
    mask = jnp.stack([a.mask, b.mask, *[seq.mask for seq in c]], axis=1)
    result = result_type(values, mask)

  return merge_batch_dim(result)


def get_large_negative_number(dtype: jnp.dtype) -> jnp.ndarray:
  """Returns a large negative value for the given dtype."""
  # JAX may canonicalize 64-bit types to smaller types if x64 mode isn't
  # enabled. This code avoids an integer overflow in that case.
  dtype = jnp.asarray(0, dtype=dtype).dtype
  # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError('Unsupported dtype for inputs.')
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _flatten_batch_time(x: sl.Sequence) -> sl.Sequence:
  """Flattens the batch and time dimensions of a Sequence object.

  Args:
    x: The input Sequence object.

  Returns:
    A new Sequence object with the batch and time dimensions flattened.
  """
  return sl.Sequence.from_values(
      einops.rearrange(x.values, 'B T ... -> (B T) ...')
  )


def _unflatten_batch_time(x: sl.Sequence, batch_size: int) -> sl.Sequence:
  """Unflattens the batch and time dimensions of a Sequence object.

  Args:
    x: The input Sequence object with flattened batch and time dimensions.
    batch_size: The original batch size.

  Returns:
    A new Sequence object with the batch and time dimensions restored.
  """
  return sl.Sequence.from_values(
      einops.rearrange(x.values, '(B T) ... -> B T ...', B=batch_size)
  )


@typed
def _sample_categorical_with_temperature(
    logits: sl.Sequence[Float[ArrayT, 'B T V'], sl.MaskT],
    temperature: float | ScalarFloat | Float[ArrayT, 'B'],
    top_k: int | ScalarInt | Int[ArrayT, 'B'] | None,
    top_p: float | ScalarFloat | Float[ArrayT, 'B'] | None,
    rng_key: PRNGKeyBatch,
    classifier_free_guidance_scale: (
        float | ScalarFloat | Float[ArrayT, 'B'] | None
    ),
    classifier_free_guidance_arity: int,
    valid_range: tuple[int, int] | None,
) -> sl.Sequence[Int[ArrayT, 'B T'], sl.MaskT]:
  """Samples from the provided logits with the given seed and temperature."""
  original_dtype = logits.dtype

  temperature = jnp.array(temperature, logits.dtype)
  assert (
      top_k is None or top_p is None
  ), 'Only one of top_k and top_p can be set.'
  top_k = jnp.array(top_k, jnp.int32) if top_k is not None else None
  top_p = jnp.array(top_p, jnp.float32) if top_p is not None else None

  if classifier_free_guidance_scale is not None:
    # 'arity' is the number of partial conditional inputs plus the one fully
    # conditional input.
    arity = classifier_free_guidance_arity + 1
    classifier_free_guidance_scale = jnp.array(
        classifier_free_guidance_scale, logits.dtype
    )
    # The batch is expected to be structured like [fully_cond, partial_cond1,
    # partial_cond2, partial_cond3, ...]. So, the batch size must be divisible
    # by the arity.
    batch_size = logits.shape[0]
    if batch_size % arity != 0:
      raise ValueError(
          f'Batch size {batch_size=} must be divisible by arity'
          f' {arity=} when using CFG.'
      )

    # Since we sample once per group, we only need RNG keys for the original
    # conditional inputs.
    rng_key = rng_key[::arity]

    # Similarly, adjust other per-batch parameters to match the pre-CFG batch
    # size.
    if temperature.ndim == 1:
      temperature = temperature[::arity]
    if top_k is not None and top_k.ndim == 1:
      top_k = top_k[::arity]
    if top_p is not None and top_p.ndim == 1:
      top_p = top_p[::arity]

    # Reshape the CFG scale to allow broadcasting with the logits tensor.
    if classifier_free_guidance_scale.ndim == 0:
      classifier_free_guidance_scale = jnp.full(
          [batch_size, 1, 1],
          jnp.array(classifier_free_guidance_scale, dtype=logits.dtype),
      )
    elif classifier_free_guidance_scale.ndim == 1:
      classifier_free_guidance_scale = classifier_free_guidance_scale[
          :, jnp.newaxis, jnp.newaxis
      ]

    # The core CFG logic:
    # new_logits = full_conditional + scale * (full_conditional - negative)
    # This steers the distribution towards the conditional logits.
    orig_logits = logits[::arity]
    cfg_logits = orig_logits
    for i in range(1, arity):
      scale = classifier_free_guidance_scale[i::arity]
      negative = logits.values[i::arity]
      cfg_logits = cfg_logits.apply_values(
          lambda v, s=scale, neg=negative: v + s * (orig_logits.values - neg)
      )
    logits = cfg_logits

  # Reshape temperature, top_k, top_p so they can be applied to the (B, T, V)
  # logits tensor.
  if temperature.ndim == 1:
    temperature = temperature[:, jnp.newaxis, jnp.newaxis]
  if top_k is not None:
    if top_k.ndim == 1:
      top_k = top_k[:, jnp.newaxis, jnp.newaxis]
    else:
      assert top_k.ndim == 0
      top_k = top_k[jnp.newaxis, jnp.newaxis, jnp.newaxis]
  if top_p is not None:
    if top_p.ndim == 1:
      top_p = top_p[:, jnp.newaxis, jnp.newaxis]
    else:
      assert top_p.ndim == 0
      top_p = top_p[jnp.newaxis, jnp.newaxis, jnp.newaxis]

  # Generate Gumbel noise. The Gumbel-Max trick (argmax(logits + gumbel_noise))
  # is a way to sample from a categorical distribution.
  if rng_key.shape[0] == 1:
    gumbel_noise = jax.random.gumbel(
        rng_key[0], logits.shape, dtype=logits.dtype
    )
  else:

    def gumbel(seed):
      return jax.random.gumbel(seed, logits.shape[1:], logits.dtype)

    gumbel_noise = jax.vmap(gumbel)(rng_key)

  def sample_logits_fn(logits: jax.Array) -> jax.Array:
    # If a valid range is given, mask out all logits outside this range by
    # setting them to a large negative number.
    if valid_range is not None:
      # Mask out logits outside the valid range.
      indices = jnp.arange(logits.shape[-1])[jnp.newaxis, jnp.newaxis]
      logits = jnp.where(
          (indices >= valid_range[0]) & (indices < valid_range[1]),
          logits,
          get_large_negative_number(logits.dtype),
      )
    # If top_k is set, find the k-th largest logit and mask out all smaller
    # logits.
    if top_k is not None:
      # Apply top-k filtering.
      # TODO(kehanghan): use jax.lax.top_k(logits, topk) instead of sorting?
      # eg: third_party/deepmind/lyria_live/internal/odml/pipeline/python/lm.py
      k = jnp.clip(top_k, 1, logits.shape[-1])
      sorted_logits = jax.lax.sort(logits, dimension=-1, is_stable=False)
      kth_logit = jnp.take_along_axis(sorted_logits, -k, axis=-1)
      logits = jnp.where(
          logits >= kth_logit,
          logits,
          get_large_negative_number(logits.dtype),
      )

    # If top_p is set, sort logits, find the smallest set of tokens whose
    # cumulative probability is >= p, and mask out all other logits.
    if top_p is not None:
      sorted_logits = jax.lax.sort(logits, dimension=-1, is_stable=False)[
          ..., ::-1
      ]
      cum_p = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
      cutoff_indices = jnp.sum(
          (cum_p < top_p).astype(jnp.int32), axis=-1, keepdims=True
      )
      # Clip the indices to prevent out-of-bounds access if top_p is >= 1.0
      cutoff_indices = jnp.minimum(cutoff_indices, logits.shape[-1] - 1)
      logits_threshold = jnp.take_along_axis(
          sorted_logits, cutoff_indices, axis=-1
      )
      logits = jnp.where(
          logits >= logits_threshold,
          logits,
          get_large_negative_number(logits.dtype),
      )
    # Apply temperature to the noise and add it to the logits.
    # Taking argmax is equivalent to sampling from the distribution.
    logits += gumbel_noise * temperature
    return jnp.argmax(logits, axis=-1)

  assert logits.dtype == original_dtype
  assert gumbel_noise.dtype == logits.dtype
  assert temperature.dtype == logits.dtype

  sample = logits.apply_values(sample_logits_fn).mask_invalid()

  # If CFG was used, the resulting sample batch is smaller.
  # Duplicate the samples to match the original batch size.
  if classifier_free_guidance_scale is not None:
    # Use the sample for all trajectories if CFG is enabled.
    arity = classifier_free_guidance_arity + 1
    sample = interleave_sequences(*([sample] * arity))

  return sample


class Encoder(nn.Module):
  """Embeds an integer sequence and processes it with a SequenceLayer.

  This doesn't really need to be a Module, it could be a pure SequenceLayer.
  Hard-coding processing like embedding w/ vocab size as a param may make
  interpreting the hparams easier for readers.
  """

  @dataclasses.dataclass(frozen=True)
  class Config:
    """Config for Encoder."""

    vocab_size: int
    embedding_dimension: int
    body: sl.SequenceLayerConfig
    use_basic_conditioning: bool = True
    # Whether to perform reductions (for LayerNorm/RMSNorm) in at least fp32.
    reductions_in_at_least_fp32: bool = True
    # If provided, use instead of the default embedding layer.
    embedding: sl.SequenceLayerConfig | None = None
    param_dtype: sl.DType = jnp.float32
    compute_dtype: sl.DType | None = None
    name: str | None = None

    def make(self) -> 'Encoder':
      return Encoder(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.global_conditioning_encoder = None

    if self.config.embedding is not None:
      embedding_layers = [self.config.embedding]
    else:
      embedding_layers = [
          sl.Embedding.Config(
              num_embeddings=self.config.vocab_size,
              dimension=self.config.embedding_dimension,
              param_dtype=self.config.param_dtype,
              compute_dtype=self.config.compute_dtype,
              name='encoder_embedding',
          ),
          # Implement scale_sqrt_depth for compatibility with PAX.
          sl.Scale.Config(math.sqrt(self.config.embedding_dimension)),
      ]
    conditioning = sl.Identity.Config()
    self.body = sl.Serial.Config([
        sl.Logging.Config("encoder_input"),
        *embedding_layers,
        # Optional: Encoder input adapter. Default no-op.
        # Optional: Add position embedding. Default no-op.
        conditioning,
        self.config.body,
        sl.LayerNormalization.Config(
            epsilon=1e-6,
            use_scale=True,
            use_bias=True,
            reductions_in_at_least_fp32=self.config.reductions_in_at_least_fp32,
            param_dtype=self.config.param_dtype,
            name='encoder_ln',
        ),
    ]).make()


class MultivariateDecoder(sl.Emitting):
  """A decoder model that generates multivariate sequences.

  This decoder uses an embedder, a temporal body, and a depth body to process
  an input sequence and generate an output sequence. It handles reserved tokens
  and uses vector quantization for the output space.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Configuration for the MultivariateDecoder.

    Attributes:
      embedder: Configuration for the embedder layer.
      temporal_body: Configuration for the temporal body layer.
      depth_body: Configuration for the depth body layer.
      sos_id: The ID of the start-of-sequence (SOS) token.
      num_reserved_tokens: The number of reserved tokens (e.g., SOS, padding).
      codebook_size: The size of the codebook for the output tokens.
      num_codebooks: The number of vector quantization (RVQ) layers used.
      name: Optional name for the layer.
    """

    embedder: sl.SequenceLayerConfig
    temporal_body: sl.SequenceLayerConfig
    depth_body: sl.SequenceLayerConfig

    sos_id: int
    num_reserved_tokens: int
    codebook_size: int
    num_codebooks: int

    soft_cap_logits: float | None = None
    name: str | None = None

    def make(self):
      """Creates an instance of MultivariateDecoder based on the config."""
      return MultivariateDecoder(self, name=self.name)

  config: Config

  def setup(self):
    """Sets up the sub-layers of the decoder."""
    self.embedder = self.config.embedder.make()
    self.temporal_body = self.config.temporal_body.make()
    self.depth_body = self.config.depth_body.make()

  @nn.nowrap
  def get_sos(self, batch_size: int) -> sl.MaskedSequence:
    return sl.Sequence.from_values(
        jnp.full(
            (batch_size, 1, self.config.num_codebooks),
            self.config.sos_id,
        )
    )

  def layer_with_emits(
      self,
      x: sl.Sequence[Any, Any],
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> tuple[sl.Sequence[Any, Any], sl.Emits]:
    """Processes the input sequence and generates the output sequence and targets.

    Args:
      x: The input Sequence object.
      training: Whether the model is in training mode.
      constants: Optional constants passed to the layers.

    Returns:
      A tuple containing:
        - The generated output sequence (depth_outputs).
        - A dictionary containing the target sequence.
    """
    batch_size = x.shape[0]
    # X has shape (B, T, Q) -> (B, T+1, Q)
    x = x.pad_time(
        pad_left=1,
        pad_right=0,
        valid=True,
        pad_value=self.config.sos_id,
    )
    # embedded has shape (B, T+1, Q, D)
    embedded: sl.Sequence = self.embedder(x, training=training)

    # temporal_inputs has shape (B, T, D)
    temporal_inputs = embedded.apply_values(jnp.mean, axis=-2)[:, :-1]

    # temporal_outputs has shape (B, T, D)
    temporal_outputs = self.temporal_body.layer(
        temporal_inputs,
        training=training,
        constants=constants,
    )

    # depth_inputs has shape (B, T, Q, D)
    depth_inputs = sl.Sequence.from_values(
        jnp.concatenate(
            [
                temporal_outputs.values[..., None, :],
                embedded.values[:, 1:, :-1],
            ],
            axis=-2,
        )
    )

    # depth_outputs has shape (B, T, Q, D)
    # _flatten_batch_time(depth_inputs) has shape (B*T, Q, D)
    # depth_body has shape (B*T, Q, D) -> (B*T, Q, D)
    # depth_outputs via _unflatten_batch_time has shape (B, T, Q, D)
    depth_outputs = _unflatten_batch_time(
        self.depth_body(_flatten_batch_time(depth_inputs), training),
        batch_size=batch_size,
    )

    if self.config.soft_cap_logits is not None:
      depth_outputs = depth_outputs.apply_values(
          lambda x: jnp.tanh(x / self.config.soft_cap_logits)
          * self.config.soft_cap_logits,
      )

    # targets has shape (B, T, Q)
    targets = x[:, 1:]

    return depth_outputs, dict(targets=targets)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: sl.ChannelSpec | None,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> sl.State:
    """Gets the initial state for the decoder.

    Args:
      batch_size: The batch size for the state.
      input_spec: The channel specification of the input to the first step.
      training: Whether the model is in training mode.
      constants: Optional constants passed to the layers.

    Returns:
      A tuple containing:
        - A PRNG key.
        - The initial frame containing the SOS token.
        - The initial state of the temporal body.
    """
    sos_frame = self.get_sos(batch_size)
    embedded = self.embedder.layer(sos_frame, training=False)
    embedded = embedded.apply_values(jnp.mean, axis=-2)

    temporal_state = self.temporal_body.get_initial_state(
        batch_size,
        input_spec=embedded.channel_spec,
        training=training,
        constants=constants,
    )
    rng = random.split(self.make_rng('random'), batch_size)
    step = jnp.zeros([batch_size], jnp.int32)
    return (rng, sos_frame, temporal_state, step)

  def get_output_shape(
      self, input_shape: sl.ShapeLike, *, constants: sl.Constants | None = None
  ) -> sl.Shape:
    """Gets the shape of the output sequence.

    Args:
      input_shape: The shape of the input sequence.
      constants: Optional constants passed to the layers.

    Returns:
      The shape of the output sequence.
    """
    return sl.Shape([
        *input_shape,
        self.config.num_codebooks * self.config.codebook_size
        + self.config.num_reserved_tokens,
    ])

  def get_output_dtype(self, input_dtype: sl.DType[Any]) -> sl.DType[Any]:
    """Gets the data type of the output sequence.

    Args:
      input_dtype: The data type of the input sequence.

    Returns:
      The data type of the output sequence.
    """
    return jnp.float32

  def step_with_emits(
      self,
      x: sl.Sequence[Any, Any],
      state: sl.State,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> tuple[sl.Sequence[Any, Any], sl.State, sl.Emits]:
    """Performs one step of the decoding process.

    Args:
      x: The input sequence (unused in this implementation).
      state: The current state of the decoder.
      training: Whether the model is in training mode.
      constants: Optional constants passed to the layers.

    Returns:
      A tuple containing:
        - The generated output sequence for this step.
        - The updated state of the decoder.
        - Any emitted values for this step (None in this implementation).
    """
    rng, previous_frame, temporal_state, step = state
    constants = constants or dict()
    # embedded_frame has shape (B, 1, Q, D)
    batch_size, _, _ = previous_frame.shape
    embedded_frame: sl.Sequence = self.embedder(
        previous_frame, training=training
    )
    # temporal_inputs has shape (B, 1, D)
    temporal_inputs = embedded_frame.apply_values(jnp.mean, axis=-2)
    self.sow("intermediates", "temporal_inputs", temporal_inputs.values)

    # temporal_outputs has shape (B, 1, D)
    temporal_outputs, temporal_state = self.temporal_body.step(
        temporal_inputs,
        temporal_state,
        training=training,
        constants=constants,
    )
    self.sow("intermediates", "temporal_outputs", temporal_outputs.values)

    depth_state = self.depth_body.get_initial_state(
        batch_size=temporal_outputs.shape[0],
        input_spec=temporal_outputs.channel_spec,
        training=training,
    )

    depth_samples = []
    depth_inputs = temporal_outputs

    # rngs = iter(random.split(rng[0], self.config.num_codebooks + 1))
    classifier_free_guidance_scales = get_prefixed_constants(
        constants or {},
        prefix=CLASSIFIER_FREE_GUIDANCE_SCALE_CONSTANT,
    )

    # Construct a batch of CFG scales for each CFG key, then rearrange these
    # into a single batch ordered by (batch index, CFG key).
    classifier_free_guidance_arity = len(classifier_free_guidance_scales)
    if classifier_free_guidance_arity == 0:
      # No CFG scale constants provided; skip classifier-free guidance.
      classifier_free_guidance_scale = None
    else:
      original_batch_size = batch_size // (classifier_free_guidance_arity + 1)
      cfg_scale_batches = [jnp.zeros(original_batch_size)] + [
          v[:: classifier_free_guidance_arity + 1]
          if jnp.array(v).size == batch_size
          else v * jnp.ones(original_batch_size)
          for _, v in sorted(classifier_free_guidance_scales.items())
      ]
      classifier_free_guidance_scale = jnp.stack(
          cfg_scale_batches, axis=1
      ).flatten()

    def _sample_from_logits(
        logits: jax.Array,
        rng: jax.Array,
        rvq_index: int,
        temperature: float,
        top_k: int,
        cfg_scale: jax.Array | None,
    ) -> jax.Array:
      """Samples a token from logits, masking invalid tokens for the current RVQ layer.

      Args:
        logits: The logits from the depth body step.
        rng: The RNG key for sampling.
        rvq_index: The index of the current RVQ layer.
        temperature: The temperature for sampling.
        top_k: The top-k for sampling.
        cfg_scale: The scale for classifier-free guidance.

      Returns:
        The sampled token index.
      """

      if cfg_scale is not None:
        assert cfg_scale.shape[0] == 1, 'only 1 cfg scale supported'
        pos_logits = logits[::2]
        neg_logits = logits[1::2]
        logits = neg_logits + cfg_scale * (pos_logits - neg_logits)

      num_embeddings = logits.shape[-1]
      min_valid_value = (
          self.config.num_reserved_tokens
          + rvq_index * self.config.codebook_size
      )
      max_valid_value = min_valid_value + self.config.codebook_size

      indices = jnp.arange(num_embeddings)
      mask = jnp.logical_and(
          indices >= min_valid_value, indices < max_valid_value
      )
      mask = jnp.where(mask, 0.0, -float('inf'))

      logits = logits.astype(jnp.float32)
      logits = logits + mask
      logits = logits / temperature

      if top_k is not None:
        sorted_logits = jax.lax.sort(logits, is_stable=True)
        min_logits_val = sorted_logits[..., -top_k][..., None]
        logits = jnp.where(logits < min_logits_val, -float('inf'), logits)

      logits = jax.nn.log_softmax(logits, axis=-1)
      logits = logits + random.gumbel(rng, logits.shape)
      sample = jnp.argmax(logits, axis=-1)

      if cfg_scale is not None:
        assert cfg_scale.shape[0] == 1, 'only 1 cfg scale supported'
        sample = sample.repeat(repeats=2, axis=0)

      return sample

    for rvq_index in range(self.config.num_codebooks):
      logits, depth_state = self.depth_body.step(
          depth_inputs,
          depth_state,
          training=training,
      )
      self.sow("intermediates", "depth_logits", logits.values)

      if self.config.soft_cap_logits is not None:
        logits = logits.apply_values(
            lambda x: jnp.tanh(x / self.config.soft_cap_logits)
            * self.config.soft_cap_logits,
        )

      # Perform sampling in float32 for improved stability.
      logits = logits.apply_values_masked(lambda v: v.astype(jnp.float32))
      # depth_sample = logits.apply_values(
      #     _sample_from_logits,
      #     rng=next(rngs),
      #     rvq_index=rvq_index,
      #     temperature=constants.get('temperature', [1.0])[0],
      #     top_k=constants.get('top_k', [None])[0],
      #     cfg_scale=constants.get('cfg', None),
      # )
      min_valid_value = (
          self.config.num_reserved_tokens
          + rvq_index * self.config.codebook_size
      )
      max_valid_value = min_valid_value + self.config.codebook_size
      depth_sample = _sample_categorical_with_temperature(
          logits,
          temperature=constants.get('temperature'),
          top_k=constants.get('top_k'),
          top_p=constants.get('top_p', None),
          rng_key=rng,
          classifier_free_guidance_scale=classifier_free_guidance_scale,
          classifier_free_guidance_arity=classifier_free_guidance_arity,
          valid_range=(min_valid_value, max_valid_value),
      )
      rng = jax.vmap(random.fold_in)(rng, step)
      self.sow("intermediates", "depth_samples", depth_sample.values)
      depth_samples.append(depth_sample.values)
      depth_inputs = self.embedder(depth_sample, training=training)

    depth_samples = jnp.stack(depth_samples, axis=-1)
    depth_samples = sl.Sequence.from_values(depth_samples)

    state = (rng, depth_samples, temporal_state, step + 1)
    return depth_samples, state, None


class EncoderDecoder(nn.Module):
  """An Encoder-Decoder model for sequence-to-sequence tasks.

  This module combines an encoder and a `MultivariateDecoder` to process a
  source sequence and generate a target sequence. It supports static and
  streaming conditioning.
  """

  @dataclasses.dataclass(frozen=True)
  class Config:
    encoder: Encoder.Config
    decoder: MultivariateDecoder.Config
    conditioning_name: str | None = None
    streaming_encoder: bool = False
    name: str | None = None

    def make(self) -> 'EncoderDecoder':
      return EncoderDecoder(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.encoder = self.config.encoder.make()
    self.decoder = self.config.decoder.make()
    self.sampler = StreamingEncoderDecoderSampler(
        encoder=self.encoder,
        decoder=self.decoder,
        streaming_encoder=self.config.streaming_encoder,
        encoder_lookahead=0,
        conditioning_name=self.config.conditioning_name,
    )

  def get_sampler_sequence_layer(self) -> sl.SequenceLayer:
    """Returns a SequenceLayer that samples from this model."""
    return self.sampler

  def encode(
      self,
      source: sl.Sequence,
      training: bool,
      conditioning: sl.Sequence | None = None,
  ) -> sl.Sequence:
    del conditioning
    return self.encoder.layer(source, training=training)

  def __call__(
      self,
      source: sl.Sequence,
      target: sl.Sequence,
      training: bool,
      conditioning: sl.Sequence | None = None,
  ):
    del conditioning
    encoded = self.encode(source, training=training)
    constants = dict()

    if self.config.conditioning_name is not None:
      constants[self.config.conditioning_name] = encoded

    logits, emits = self.decoder.layer_with_emits(
        target,
        training=training,
        constants=constants,
    )
    logits = logits.apply_values(lambda v: v.astype(jnp.float32))

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.values, labels=emits['targets'].values
    )
    loss = sl.Sequence.from_values(jnp.mean(loss, -1) / loss.shape[1])

    return {
        'logits': logits,
        'negative_log_likelihood': loss,
    }


class StreamingEncoderDecoderSampler(sl.SequenceLayer):
  """A streaming encoder and autoregressive sampler as a SequenceLayer."""

  encoder: Encoder
  decoder: MultivariateDecoder
  streaming_encoder: bool
  encoder_lookahead: int
  conditioning_name: str | None = None

  @property
  def block_size(self) -> int:
    if self.streaming_encoder:
      return self.encoder.body.block_size
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    if self.streaming_encoder:
      return self.encoder.body.output_ratio
    else:
      return fractions.Fraction(1)

  def setup(self) -> None:
    assert self.decoder.temporal_body.input_latency == 0
    assert self.decoder.temporal_body.output_latency == 0

  @property
  def input_latency(self) -> int:
    return self.encoder.body.input_latency

  @property
  def output_latency(self) -> int:
    return self.encoder.body.output_latency

  @nn.nowrap
  def get_output_shape(
      self, input_shape: sl.ShapeLike, *, constants: sl.Constants | None = None
  ) -> sl.Shape:
    return (self.decoder.config.num_codebooks,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: sl.DType,
      *,
      constants: sl.Constants | None = None,
  ) -> sl.DType:
    return jnp.int32

  @sl.check_layer
  def layer(
      self,
      x: sl.Sequence,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> sl.Sequence:
    encoded = self.encoder.body.layer(x, training=training, constants=constants)

    if constants is None:
      constants = {}

    temperature = (
        constants[TEMPERATURE_CONSTANT]
        if TEMPERATURE_CONSTANT in constants
        else 1.0
    )
    top_k = (
        constants[TOP_K_CONSTANT]
        if TOP_K_CONSTANT in constants
        else None
    )
    top_p = (
        constants[TOP_P_CONSTANT]
        if TOP_P_CONSTANT in constants
        else None
    )
    prng_key = (constants or {}).get(PRNG_KEY_CONSTANT)
    max_length = (
        constants[MAX_LENGTH_CONSTANT]
        if MAX_LENGTH_CONSTANT in constants
        else x.shape[1]
    )

    samples = self.decoder.sample(
        batch_size=x.shape[0],
        conditioning=encoded,
        training=training,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        prng_key=prng_key,
    )
    return samples

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: sl.ChannelSpec,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> sl.State:
    constants = dict(constants) or {}
    classifier_free_guidance_scales = get_prefixed_constants(
        constants,
        prefix=CLASSIFIER_FREE_GUIDANCE_SCALE_CONSTANT,
    )
    if classifier_free_guidance_scales:
      # Multiply the effective batch size to account for the negative guidance
      # states.
      batch_size *= len(classifier_free_guidance_scales) + 1

    encoder_state = self.encoder.body.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    encoder_spec = self.encoder.body.get_output_spec(
        input_spec, constants=constants
    )

    zero_encoded = sl.Sequence(
        jnp.zeros(
            (batch_size, self.encoder.body.block_size) + encoder_spec.shape,
            dtype=encoder_spec.dtype,
        ),
        jnp.zeros((batch_size, self.encoder.body.block_size), jnp.bool_),
    )

    if self.streaming_encoder:
      constants[self.conditioning_name] = zero_encoded

    sampler_sos = self.decoder.get_sos(batch_size)
    sampler_state = self.decoder.get_initial_state(
        batch_size,
        sampler_sos.channel_spec,
        training=training,
        constants=constants,
    )

    # Delay the SOS input until this many timesteps have passed.
    sampler_delay_countdown = jnp.full(
        [batch_size], self.encoder_lookahead, jnp.int32
    )

    state = (
        encoder_state,
        sampler_sos,
        sampler_state,
        sampler_delay_countdown,
    )
    if classifier_free_guidance_scales:
      # Pack the ordinary and negative guidance states into a single batch
      # element.
      state = split_batch_dim(
          state, len(classifier_free_guidance_scales) + 1
      )

    return state

  @sl.check_step
  def step(
      self,
      x: sl.Sequence,
      state: sl.State,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> tuple[sl.Sequence, sl.State]:
    assert x.shape[1] % self.encoder.body.block_size == 0
    if self.block_size > 1:
      raise NotImplementedError(
          'Block sizes greater than 1 are not implemented.'
      )

    if not self.streaming_encoder:
      raise NotImplementedError(
          'Stepwise processing requires a streaming encoder.'
      )

    if x.shape[1] > 1:
      output, state, unused_emits = utils.step_by_step_static(
          self,
          x,
          training=training,
          initial_state=state,
          constants=constants,
          with_emits=False,
          stream_constants=self.streaming_encoder,
      )
      return output, state

    if constants is None:
      constants = {}

    classifier_free_guidance_scales = get_prefixed_constants(
        constants, prefix=CLASSIFIER_FREE_GUIDANCE_SCALE_CONSTANT
    )
    classifier_free_guidance_negatives = get_prefixed_constants(
        constants,
        prefix=CLASSIFIER_FREE_GUIDANCE_NEGATIVE_CONSTANT,
    )

    if classifier_free_guidance_scales:
      # Due to SLEngine requirements, elements of a batch can be mixed up
      # arbitrarily and so must be handled individually.  This forces us to
      # store the negative guidance state(s) in the same batch element as the
      # ordinary state, and take the negative inputs as auxiliary constants.
      state = merge_batch_dim(state)

      if not classifier_free_guidance_negatives:
        logging.warning(
            'Classifier free guidance scale(s) specified, but negative(s) not'
            ' specified; using zeros as negatives.'
        )
        x_neg = [sl.Sequence(values=jnp.zeros_like(x), mask=x.mask)] * len(
            classifier_free_guidance_scales
        )
      else:
        if set(classifier_free_guidance_scales.keys()) != set(
            classifier_free_guidance_negatives.keys()
        ):
          raise ValueError(
              'Classifier free guidance scale keys and negative keys do not'
              ' match.'
          )
        # Order negatives canonically by key to make sure they line up with the
        # scales.
        x_neg = [
            v for _, v in sorted(classifier_free_guidance_negatives.items())
        ]

      arity = len(x_neg) + 1
      x = interleave_sequences(x, *x_neg)

      # We also need to repeat the batched constants to account for the
      # multiplied batch size.
      is_repeatable = lambda v: (
          v is not None
          and not isinstance(v, sl.Sequence)
          and jnp.array(v).ndim >= 1
      )
      constants = {
          k: (
              jax.tree.map(lambda a: jnp.repeat(a, arity, axis=0), v)
              if is_repeatable(v)
              else v
          )
          for k, v in constants.items()
          if v is not None
      }

    (
        encoder_state,
        sampler_previous_output,
        sampler_state,
        sampler_delay_countdown,
    ) = state

    encoded, encoder_state = self.encoder.body.step(
        x, encoder_state, training=training, constants=constants
    )

    sampler_constants = constants | {self.conditioning_name: encoded}

    delay_active = sampler_delay_countdown > 0
    sampler_delay_countdown = jnp.maximum(0, sampler_delay_countdown - 1)

    def broadcast_delay_active(ndim) -> jax.Array:
      return delay_active.reshape(delay_active.shape + (1,) * (ndim - 1))

    delay_active_values = broadcast_delay_active(sampler_previous_output.ndim)
    delay_active_mask = broadcast_delay_active(2)

    # Use an invalid step input until the sampler_delay_countdown countdown
    # timer has elapsed.
    step_input = sl.MaskedSequence(
        jnp.where(
            delay_active_values,
            jnp.zeros_like(sampler_previous_output.values),
            sampler_previous_output.values,
        ),
        jnp.where(
            delay_active_mask,
            jnp.zeros_like(sampler_previous_output.mask),
            sampler_previous_output.mask,
        ),
    )

    sampler_output, sampler_state = self.decoder.step(
        step_input,
        sampler_state,
        training=training,
        constants=sampler_constants,
    )
    assert sampler_output.shape[1] == 1, sampler_output.shape

    # Only update sampler_previous_output if the sampler_delay_countdown
    # countdown timer has elapsed.
    sampler_previous_output = sl.MaskedSequence(
        jnp.where(
            delay_active_values,
            sampler_previous_output.values,
            sampler_output.values,
        ),
        jnp.where(
            delay_active_mask,
            sampler_previous_output.mask,
            sampler_output.mask,
        ),
    )

    state = (
        encoder_state,
        sampler_previous_output,
        sampler_state,
        sampler_delay_countdown,
    )
    if classifier_free_guidance_scales:
      # Pack the negative guidance state into the same batch element as the
      # ordinary state.
      arity = len(classifier_free_guidance_scales) + 1
      sampler_output = sampler_output[::arity]
      state = split_batch_dim(state, arity)

    return sampler_output, state
