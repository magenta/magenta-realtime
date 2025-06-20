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

"""Custom utilties for t5x Models and decoding."""

import importlib
import sys
from typing import Any, Callable, Mapping, MutableMapping, Optional

import gin
import jax
import jax.numpy as jnp
import numpy as np
import t5x
import t5x.interactive_model
import t5x.models
import t5x.partitioning

from .. import path


class RTSongInferenceEncoderDecoderModel(t5x.models.EncoderDecoderModel):
  """Wrapper around EncoderDecoderModel for inference on Magenta RT models."""

  def predict_batch_with_aux(
      self,
      *args,
      decoder_params: MutableMapping[str, Any] | None = None,
      **kwargs,
  ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Override predict_batch_with_aux in EncoderDecoderModel to handle seed."""
    if decoder_params is None:
      decoder_params = {}
    for key in [
        'seed',
        'guidance_weight',
        'topk',
        'temperature',
    ]:
      if key in decoder_params and decoder_params[key].ndim == 1:
        decoder_params[key] = decoder_params[key][0]
    if 'seed' in decoder_params:
      if 'decode_rng' in decoder_params:
        raise ValueError('cannot provide both `seed` and `decode_rng`')
      decoder_params['decode_rng'] = jax.random.PRNGKey(decoder_params['seed'])
      del decoder_params['seed']
    return super().predict_batch_with_aux(
        *args, decoder_params=decoder_params, **kwargs
    )


_GIN_CONFIGS_DIR = path.MODULE_DIR / 'depthformer' / 'configs'


def _parse_global_gin_config(
    size: str = 'base', overrides: str = ''
) -> tuple[Any, dict[str, int]]:
  """Parses the global gin config and returns the model and task lengths."""
  if size not in ['base', 'large']:
    raise ValueError(f'Unsupported size: {size}')
  gin.enter_interactive_mode()
  gin.clear_config()
  # TODO(chrisdonahue): Fix gin relative includes to improve modularity.
  # gin.add_config_file_search_path(str(_GIN_CONFIGS_DIR))
  # for config_file in ['size_base.gin', 'depthformer.gin', 'magenta_rt.gin']:
  # gin.parse_config_file(str(_GIN_CONFIGS_DIR / config_file))
  gin.parse_config_file(str(_GIN_CONFIGS_DIR / f'mrt_merged_{size}.gin'))
  gin.parse_config(overrides)
  return (
      gin.get_configurable('MODEL/macro')(),
      gin.get_configurable('TASK_FEATURE_LENGTHS/macro')(),
  )


def load_pretrained_model(
    checkpoint_dir: str,
    size: str = 'base',
    default_guidance_weight: float = 4.0,
    default_temperature: float = 1.1,
    default_topk: int = 40,
    batch_size: int = 1,
    num_partitions: Optional[int] = 1,
    model_parallel_submesh: Optional[tuple[int, int, int, int]] = None,
    gin_overrides: Optional[str] = '',
) -> Callable:  # pylint: disable=g-bare-generic
  """Loads a pretrained Magenta RT t5x.InteractiveModel."""
  # Check that one of `model_parallel_submesh` or `num_partitions` specified.
  if (model_parallel_submesh, num_partitions).count(None) != 1:
    raise ValueError(
        'Exactly one of `model_parallel_submesh` or `num_partitions` must'
        ' be specified.'
    )

  # Parse the global gin config.
  model, task_feature_lengths = _parse_global_gin_config(size, gin_overrides)

  # TODO(chrisdonahue): Relax this assertion to support other sizes in future.
  # 1006 = 10 seconds x 25 Hz x 4 levels + 6 style tokens
  assert task_feature_lengths['inputs'] == 1006
  # 800 = 2 seconds x 25 Hz x 16 levels
  assert task_feature_lengths['targets'] == 800

  # Create the TPU partitioner.
  partitioner = t5x.partitioning.PjitPartitioner(
      num_partitions=num_partitions,
      model_parallel_submesh=model_parallel_submesh,
  )

  # Check if running on Colab and apply workaround for checkpoint loading.
  if 'google.colab' in sys.modules:
    try:
      nest_asyncio = importlib.import_module('nest_asyncio')
      nest_asyncio.apply()
    except ImportError:
      pass

  # Create the interactive model.
  interactive_model = t5x.interactive_model.InteractiveModel(
      batch_size=batch_size,
      task_feature_lengths=task_feature_lengths,
      output_dir='/tmp',
      partitioner=partitioner,
      model=model,
      dtype=None,
      restore_mode='specific',
      checkpoint_path=checkpoint_dir,
      input_shapes={
          'encoder_input_tokens': (
              batch_size,
              task_feature_lengths['inputs'],
          ),
          'decoder_input_tokens': (
              batch_size,
              task_feature_lengths['targets'],
          ),
      },
  )

  # Create the inference function.
  def _infer_fn(params, batch, decoder_params, seed):
    assert isinstance(interactive_model.model, t5x.models.EncoderDecoderModel)
    return interactive_model.model.predict_batch_with_aux(
        params,
        batch,
        rng=seed,
        prompt_with_targets=True,
        decoder_params=decoder_params,
    )

  # Compile the inference function for TPU (if needed).
  train_state = interactive_model.train_state
  partitioned_infer_fn = partitioner.partition(
      _infer_fn,
      in_axis_resources=(
          interactive_model.train_state_axes.params,
          partitioner.data_partition_spec,
          None,
          None,
      ),
      out_axis_resources=(
          partitioner.data_partition_spec,
          partitioner.data_partition_spec,
      ),
  )
  partitioned_infer_fn = partitioner.compile(
      partitioned_infer_fn,
      train_state.params,
      {
          'encoder_input_tokens': jnp.zeros(
              (batch_size, task_feature_lengths['inputs']),
              np.int32,
          ),
          'decoder_input_tokens': jnp.zeros(
              (batch_size, task_feature_lengths['targets']),
              np.int32,
          ),
      },
      {
          'max_decode_steps': task_feature_lengths['targets'],
          'guidance_weight': default_guidance_weight,
          'temperature': default_temperature,
          'topk': default_topk,
      },
      jax.random.PRNGKey(0),  # placeholder
  )

  def _final_wrapper(batch, decoder_params, seed):
    return partitioned_infer_fn(
        train_state.params,
        batch,
        decoder_params,
        seed,
    )

  return _final_wrapper
