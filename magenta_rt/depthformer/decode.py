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

"""Decoding functions for Depthformer."""

from typing import Mapping

import gin
import jax
import jax.numpy as jnp
import t5x.decoding
import t5x.models
import typing_extensions


class DecodeFnWithCallbacks(typing_extensions.Protocol):
  """Call signature of T5X decoding function with callbacks."""

  def __call__(
      self,
      *,
      inputs: jnp.ndarray,
      cache: Mapping[str, jnp.ndarray],
      tokens_to_logits: t5x.models.TokensIdsToLogitsCallable,
      eos_id: int,
      num_decodes: int,
      decode_rng: jax.Array | None,
      cache_offset: int,
      logit_callback_fn: t5x.decoding.LogitCallbackFn,
      state_callback_fn: t5x.decoding.StateCallbackFn,
      **kwargs,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    ...


@gin.configurable
def decode_with_classifier_free_guidance(
    *args,
    decode_fn: DecodeFnWithCallbacks = t5x.decoding.temperature_sample,
    guidance_weight: float = 0.0,
    initial_index: jnp.ndarray | None = None,
    logit_callback_fn: t5x.decoding.LogitCallbackFn | None = None,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Applies classifier-free guidance to compatible decode function.

  Assumes that even-indexed batch elements are conditioned inputs and
  consecutive odd-indexed elements are matching, but unconditioned.

  At each decoding step, the conditioned and unconditioned logits are combined
  based on `guidance_weight` before sampling. After sampling, the sampled value
  is copied over to both the conditioned and unconditioned sequences so that
  they match.

  Args:
    *args: Positional args to pass to `decode_fn`.
    decode_fn: The decoding function to use. Must include `logit_callback_fn`
      and `state_callback_fn` args (e.g., t5x.decoding.temperature_sample).
    guidance_weight: A hyperparameter controlling the mixture of the conditioned
      and unconditioned logits.
    initial_index: Initial index following prefill.
    logit_callback_fn: A logit callback function to apply to logits after CFG.
    **kwargs: Keyword arguments to pass to `decode_fn`.

  Returns:
    A tuple (decodes, log_prob) where decodes is sampled sequences with
    shape [batch_size, num_decodes, max_decode_len] sorted by log_prob, which is
    log probability of each of the sampled sequences.
  """

  def _cfg_logit_callback_fn(
      logits: jnp.ndarray,
      state: t5x.decoding.SamplingLoopState,
  ) -> jnp.ndarray:
    logits = (1 + guidance_weight) * logits[::2] - guidance_weight * logits[
        1::2
    ]
    if logit_callback_fn is not None:
      logits = logit_callback_fn(logits, state)
    return jnp.repeat(logits, 2, axis=0)

  def _cfg_state_callback_fn(
      state: t5x.decoding.SamplingLoopState,
  ) -> t5x.decoding.SamplingLoopState:
    def _override_samples(state):
      override = lambda x: jnp.repeat(x[::2], 2, axis=0)
      return state.replace(
          sequences=override(state.sequences),
          cur_token=override(state.cur_token),
          ended=override(state.ended),
      )

    return jax.lax.cond(
        state.step > 0, lambda: _override_samples(state), lambda: state
    )

  return jax.lax.cond(
      guidance_weight != 0.0,
      lambda: decode_fn(  # pylint:disable=g-long-lambda
          *args,
          logit_callback_fn=_cfg_logit_callback_fn,
          state_callback_fn=_cfg_state_callback_fn,
          initial_index=initial_index,
          **kwargs,
      ),
      lambda: decode_fn(
          *args,
          logit_callback_fn=logit_callback_fn,
          initial_index=initial_index,
          **kwargs,
      ),
  )


@gin.configurable
def constrained_logit_callback_fn(
    logits: jnp.ndarray,
    state: t5x.decoding.SamplingLoopState,
    split_point: int = 0,
    reserved_tokens: int = 2,
    acoustic_depth: int = 4,
    style_depth: int = 0,
    tokens_per_level: int = 1024,
) -> jnp.ndarray:
  """Masks logits to constrain decoding to only valid token ranges at each step.

  Assumes the targets contain tokens in the following order:
    1. (Optional) Style
    2. Semantic
    3. Acoustic

  Args:
    logits: Logits for single step of decoding.
    state: The current decoder state.
    split_point: Index where acoustic tokens begin.
    reserved_tokens: The number of reserved tokens in the vocabulary.
    acoustic_depth: The number of levels of acoustic tokens in the vocabulary.
    style_depth: The number of style tokens in the vocabulary.
    tokens_per_level: The number of tokens per acousic, style, and semantic
      levels.

  Returns:
    The masked logits.
  """
  idx = state.cur_index[0]
  level = jax.lax.switch(
      jnp.searchsorted(jnp.array([style_depth, split_point]), idx + 1),
      [
          # style token range begins after semantic and acoustic
          lambda: idx + acoustic_depth + 1,
          # semantic token range begins after acoustic
          lambda: acoustic_depth,
          # acoustic range comes first
          lambda: (idx - split_point) % acoustic_depth,
      ],
  )
  range_start = reserved_tokens + tokens_per_level * level
  indices = jnp.arange(jnp.shape(logits)[-1])
  return jnp.where(
      jnp.logical_and(
          indices >= range_start, indices < range_start + tokens_per_level
      ),
      logits,
      -jnp.inf,
  )
