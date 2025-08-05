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

"""Training tasks."""

from collections.abc import Mapping, MutableMapping, Sequence
import functools

import seqio
import tensorflow as tf

from .. import musiccoca
from .. import system


@seqio.map_over_dataset(num_seeds=1)
def mask_inputs(
    ex: MutableMapping[str, tf.Tensor],
    seed=None,
    ranges_and_probs: Sequence[tuple[Sequence[tuple[int, int]], float]] = (),
    mask_value: int = 1,
    axis: int = 0,
) -> Mapping[str, tf.Tensor]:
  """Preprocessor to stochastically mask input tokens.

  Args:
    ex: The unmasked input example.
    seed: Seed to use for stochastic op.
    ranges_and_probs: A sequence of pairs of ranges along with the probability
      that the ranges should be masked. For example, `[(((1, 3)), 0.2), (((0,
      0)), 0.8)]` would signify that positions 1 and 2 should be masked with
      probability 0.2 and nothing should be masked with probability 0.8.
      Supports negative indices as well as `None` to represent the final
      position.
    mask_value: The value to replace the masked tokens with.
    axis: The axis on which ranges are specified.

  Returns:
    Masked example.
  """
  if not ranges_and_probs:
    return ex

  ranges, probs = zip(*ranges_and_probs)
  probs = tf.constant(probs, tf.float32)

  inputs = ex['inputs']
  inputs_shape = tf.shape(inputs)

  # Handle negative and `None` indices.
  def _map_index(i):
    if i is None:
      return inputs_shape[axis]
    if i < 0:
      return i + inputs_shape[axis]
    return i

  def _apply_mask(ranges_to_mask):
    if not ranges_to_mask:
      return inputs
    ranges_to_mask = [[_map_index(i) for i in rng] for rng in ranges_to_mask]
    arange = tf.broadcast_to(tf.range(inputs_shape[axis]), inputs_shape)
    is_masked = tf.reduce_any(
        tf.stack([
            tf.logical_and(r[0] <= arange, arange < r[1])
            for r in ranges_to_mask
        ]),
        axis=0,
    )
    return tf.where(is_masked, tf.cast(mask_value, inputs.dtype), inputs)

  rnd_i = tf.random.stateless_categorical(
      tf.math.log([probs + 1e-8]), 1, seed, dtype=tf.int32
  )[0, 0]

  ex['inputs'] = tf.switch_case(
      rnd_i, [(lambda r=rng: _apply_mask(r)) for rng in ranges]
  )

  return ex


def _load_style_tokens(
    raw_tensor: tf.Tensor,
    magenta_rt_config: system.MagentaRTConfiguration,
    keep_frame_dim: bool = False,
) -> tf.Tensor:
  """Parses and loads style tokens from serialized input tensor."""
  toks = tf.io.parse_tensor(raw_tensor, tf.int32)
  toks.set_shape([None, 12])
  toks = toks[tf.newaxis][:, :, : magenta_rt_config.encoder_style_rvq_depth]
  tokens = (
      toks
      + tf.range(magenta_rt_config.encoder_style_rvq_depth)
      * magenta_rt_config.style_rvq_codebook_size
  )
  tokens = tokens + magenta_rt_config.vocab_style_offset
  tokens = (
      tf.reshape(tokens, (tokens.shape[0], -1))
      if not keep_frame_dim
      else tokens
  )[0]
  return tokens


def _load_acoustic_tokens(
    raw_tensor: tf.Tensor,
    magenta_rt_config: system.MagentaRTConfiguration,
    num_chunks: int,
) -> tf.Tensor:
  """Parses and lays out acoustic tokens from serialized input tensor."""
  toks = tf.io.parse_tensor(raw_tensor, tf.int32)
  toks.set_shape([None, 64])
  tokens = toks[:, : magenta_rt_config.decoder_codec_rvq_depth]
  tokens = (
      tokens
      + tf.range(magenta_rt_config.decoder_codec_rvq_depth)
      * magenta_rt_config.codec_rvq_codebook_size
  )
  tokens = tokens + magenta_rt_config.vocab_codec_offset
  tokens = tf.reshape(tokens, (-1,))
  return tf.reshape(tokens, (num_chunks, -1))


@seqio.map_over_dataset(num_seeds=3)
def preprocess_example(
    inp: Mapping[str, tf.Tensor],
    magenta_rt_config: system.MagentaRTConfiguration,
    style_config: musiccoca.MusicCoCaConfiguration = musiccoca.MusicCoCaConfiguration(),
    seeds=None,
    acoustic_key: str = 'acoustic_tokens',
    style_key: str = 'style_tokens',
    encoder_codec_rvq_depth: int = 4,
    decoder_codec_rvq_depth: int = 16,
    max_prompt_secs: int = 10,
    num_gen_secs: int = 2,
) -> Mapping[str, tf.Tensor]:
  """Preprocess tf.Example."""
  ex = {}

  codec_frame_rate = magenta_rt_config.codec_frame_rate

  acoustic_tokens = _load_acoustic_tokens(
      inp[acoustic_key],
      num_chunks=1,
      magenta_rt_config=magenta_rt_config,
  )[0]

  num_aco_frames = tf.shape(acoustic_tokens)[0] // decoder_codec_rvq_depth
  clip_length_secs = num_aco_frames // int(codec_frame_rate)

  # 75% of the time, use the maximum prompt length. The remaining 25% of the
  # time, choose uniformly from the range [0, max_prompt_secs).
  num_prompt_secs = tf.clip_by_value(
      tf.random.stateless_uniform(
          (),
          seed=seeds[0],
          minval=0,
          maxval=4 * max_prompt_secs,
          dtype=tf.int32,
      ),
      0,
      tf.math.minimum(max_prompt_secs, clip_length_secs - num_gen_secs),
  )
  start_sec = tf.random.stateless_uniform(
      (),
      seed=seeds[1],
      minval=0,
      maxval=clip_length_secs - (num_prompt_secs + num_gen_secs) + 1,
      dtype=tf.int32,
  )
  start_frame = start_sec * int(codec_frame_rate)
  end_frame = (start_sec + num_prompt_secs) * int(codec_frame_rate)
  start_i = start_frame * decoder_codec_rvq_depth
  end_i = end_frame * decoder_codec_rvq_depth

  style_tokens = _load_style_tokens(
      inp[style_key],
      keep_frame_dim=True,
      magenta_rt_config=magenta_rt_config,
  )

  # use the MusicCoCa frame closest to the target position
  avg_target_sec = (
      tf.cast(start_sec, tf.float32)
      + tf.cast(num_prompt_secs, tf.float32)
      + 0.5 * num_gen_secs
  )

  style_idx = int(avg_target_sec / style_config.clip_length)
  style_tokens = style_tokens[style_idx, :]

  # tokenized inputs
  acoustic_inputs = tf.reshape(
      acoustic_tokens[start_i:end_i], [-1, decoder_codec_rvq_depth]
  )

  concat_inputs = [
      tf.ones(
          (max_prompt_secs - num_prompt_secs)
          * int(codec_frame_rate)
          * encoder_codec_rvq_depth,
          tf.int32,
      ),
      tf.reshape(acoustic_inputs[:, :encoder_codec_rvq_depth], [-1]),
      style_tokens,
  ]

  ex['inputs'] = tf.concat(concat_inputs, -1)
  ex['targets'] = acoustic_tokens[
      end_i : end_i
      + num_gen_secs * int(codec_frame_rate) * decoder_codec_rvq_depth
  ]

  return ex


def register_task(
    name: str,
    split_to_filepattern: Mapping[str, str],
    reader_cls: seqio.DatasetReaderType = tf.data.TFRecordDataset,
    acoustic_key='acoustic_tokens',
    style_key='style_tokens',
    encoder_codec_rvq_depth=4,
    decoder_codec_rvq_depth=16,
    encoder_style_rvq_depth=6,
    max_prompt_secs=10,
    num_gen_secs=2,
    num_eval_chunks=5,
):
  """Registers task with the given configuration and input files.

  Args:
    name: Name for the task.
    split_to_filepattern: Mapping from split to input filepattern.
    reader_cls: The tf.data reader class to use.
    acoustic_key: Acoustic tokens feature key.
    style_key: MusicCoCa tokens feature key.
    encoder_codec_rvq_depth: Acoustic token inputs depth.
    decoder_codec_rvq_depth: Acoustic token targets depth.
    encoder_style_rvq_depth: Style token depth.
    max_prompt_secs: Maximum prompt length in seconds.
    num_gen_secs: Number of seconds to generate.
    num_eval_chunks: Number of chunks to generate for evaluation.
  """
  magenta_rt_config = system.MagentaRTConfiguration(
      context_length=10.0,
      encoder_codec_rvq_depth=encoder_codec_rvq_depth,
      encoder_style_rvq_depth=encoder_style_rvq_depth,
      decoder_codec_rvq_depth=decoder_codec_rvq_depth,
  )

  seqio_vocab = seqio.PassThroughVocabulary(
      magenta_rt_config.vocab_size_pretrained
  )

  features = [acoustic_key, style_key]

  feature_description = {
      feat: tf.io.FixedLenFeature([], tf.string) for feat in features
  }

  mask_ranges_and_probs = (
      ([(0, 0)], 0.9),  # No mask: 90%
      ([(-encoder_style_rvq_depth, None)], 0.1),  # Mask style tokens: 10%
  )
  mask_value = 1
  range_axis = 0

  seqio.TaskRegistry.add(
      name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern=split_to_filepattern,
          feature_description=feature_description,
          reader_cls=reader_cls,
      ),
      preprocessors=[
          # pylint: disable=protected-access
          functools.partial(
              preprocess_example,
              magenta_rt_config=magenta_rt_config,
              acoustic_key=acoustic_key,
              style_key=style_key,
              encoder_codec_rvq_depth=encoder_codec_rvq_depth,
              decoder_codec_rvq_depth=decoder_codec_rvq_depth,
              max_prompt_secs=max_prompt_secs,
              num_gen_secs=num_gen_secs,
          ),
          # pylint: enable=protected-access
          seqio.CacheDatasetPlaceholder(),
          functools.partial(
              mask_inputs,
              ranges_and_probs=mask_ranges_and_probs,
              mask_value=mask_value,
              axis=range_axis,
          ),
      ],
      output_features={
          'inputs': seqio.Feature(seqio_vocab, add_eos=False),
          'targets': seqio.Feature(seqio_vocab, add_eos=False),
      },
  )

  seqio.TaskRegistry.add(
      name + '_eval',
      source=seqio.TFExampleDataSource(
          split_to_filepattern=split_to_filepattern,
          feature_description=feature_description,
          reader_cls=reader_cls,
      ),
      preprocessors=[
          # pylint: disable=protected-access
          functools.partial(
              preprocess_example,
              acoustic_key=acoustic_key,
              style_key=style_key,
              magenta_rt_config=magenta_rt_config,
              encoder_codec_rvq_depth=encoder_codec_rvq_depth,
              decoder_codec_rvq_depth=decoder_codec_rvq_depth,
              max_prompt_secs=max_prompt_secs,
              num_gen_secs=num_gen_secs * num_eval_chunks,
          ),
          # Add targets feature that won't be trimmed by the feature converter
          seqio.map_over_dataset(
              lambda ex: {**ex, 'full_targets': ex['targets']}
          ),
          # pylint: enable=protected-access
          seqio.CacheDatasetPlaceholder(),
      ],
      output_features={
          'inputs': seqio.Feature(seqio_vocab, add_eos=False),
          'targets': seqio.Feature(seqio_vocab, add_eos=False),
      },
  )
