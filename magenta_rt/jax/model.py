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

"""Magenta RealTime 2 models."""

import abc
from collections.abc import Sequence
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import sequence_layers.jax as sl

from . import depthformer
from . import transformer

from magenta_rt.config import (  # noqa: F401
    CFG_CONDITIONING_DRUMS,
    CFG_CONDITIONING_MUSICCOCA_NOTES,
    DRUM_PIANOROLL,
    L_SHALLOW_TPU_OPTIMIZED,
    L_SHALLOW_TPU_OPTIMIZED_6,
    L_TPU_OPTIMIZED,
    M_SHALLOW_TPU_OPTIMIZED,
    MUSICCOCA,
    ModelSpec,
    NUM_RESERVED_TOKENS,
    PIANOROLL_WITH_ONSETS,
    S,
    SPECTROSTREAM,
    TOKEN_DROPOUT_PROB,
    TokensConfig,
    XXL_SHALLOW,
)


def branch_config(
    split_sizes: Sequence[int],
    *layer_configs: sl.SequenceLayerConfig,
    name: str | None = None,
) -> sl.Parallel.Config:
  """Creates a branched config with the given layer configs and split sizes.

  This allows to use different embedders for different input features
  concatenated along the last dimension.

  For example, if the encoder input is np.concatenate([mulan_tokens,
  pianoroll_tokens], axis=-1), with num_mulan_tokens = 12 and
  num_pianoroll_tokens = 128, then the split_sizes should be [12, 128] and the
  layer_configs should be [mulan embedder config, pianoroll embedder config].

  Args:
    split_sizes: The number of tokens in each branch.
    *layer_configs: The layer configs to apply to each branch.
    name: The name of the branched config.

  Returns:
    A branched config with the given layer configs and split sizes.
  """
  assert len(layer_configs) == len(
      split_sizes
  ), f'{len(layer_configs)}!={len(split_sizes)}'

  bounds = [0] + [int(e) for e in np.cumsum(split_sizes)]

  def _crop_apply(config, start, stop):
    crop_name = getattr(config, 'name', None)
    if crop_name is not None:
      crop_name = f'branched_{crop_name}'
    crop_config = sl.Lambda.Config(
        lambda s: s[..., start:stop],
        mask_required=False,
        expected_input_spec=sl.ShapeDType(
            shape=bounds[-1:],
            dtype=jnp.int32,
        ),
        name=crop_name,
    )
    return sl.Serial.Config([crop_config, config], name=crop_name)

  return sl.Parallel.Config(
      layers=[
          _crop_apply(config, bounds[i], bounds[i + 1])
          for i, config in enumerate(layer_configs)
      ],
      combination=sl.utils.CombinationMode.MEAN,
      name=name,
  )


class MagentaRT2ModelBase(metaclass=abc.ABCMeta):
  """Magenta RealTime 2 base model."""

  encoder_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED
  decoder_temporal_size: ModelSpec = XXL_SHALLOW
  decoder_depth_size: ModelSpec = L_SHALLOW_TPU_OPTIMIZED_6

  self_attention_use_separate_qkv: bool = True
  cross_attention_use_separate_kv: bool = True
  temporal_transformer_self_attention_use_kv_cache_ringbuffer: bool = False
  temporal_transformer_cross_attention_use_kv_cache_ringbuffer: bool = False

  param_dtype: sl.DType = jnp.float32
  compute_dtype: sl.DType = jnp.bfloat16

  # Number of attention sink embeddings to use in the encoder and temporal
  # decoder.
  num_attention_sink_embeddings: int = 1

  # If True, use attention sink scalars in the encoder and temporal decoder.
  # Incompatible with num_attention_sink_embeddings > 0.
  use_attention_sink_scalars: bool = False

  # Whether to use RoPE positional embeddings in all transformer modules
  # (encoder, temporal decoder, depth decoder). If False, no positional
  # embeddings are used, i.e., "NoPE".
  use_rope: bool = False  # NoPE

  # Whether to randomly mask MusicCoCa tokens during training. If True, a random
  # suffix of MusicCoCa tokens is replaced with masks, and no truncation is applied
  # during training or eval. If False, MusicCoCa tokens are always truncated to
  # MUSICCOCA_TOKENS.rvq_truncation_level.
  mask_musiccoca: bool = False

  # Number of MusicCoCa tokens to mask. If set, the specified number of MusicCoCa tokens
  # are masked during training and eval. Needs mask_musiccoca to be true to be able
  # to access all 12 MusicCoCa tokens.
  num_musiccoca_tokens_to_mask: int | None = None

  # Optionally specify the number of RVQ levels stored in the input data for
  # all SpectroStream features. If not specified, this will be inferred from the
  # spectrostream config.
  data_rvq_levels: int | None = None

  # Delay input tokens by N frames. The model learns to predict outputs without
  # access to the most recent N input frames, matching the inference setting
  # where inputs are delayed due to network and model latency.
  io_offset_frames: int | None = None

  # Delay output tokens by N frames. This is useful for creating a
  # "pass-through" model that parrots inputs back with a fixed delay.
  output_delay_frames: int | None = None

  # Trim this many frames from the beginning and end of each feature. This can
  # be used to avoid training the model on SpectroStream tokens that contain
  # "edge artifacts".
  trim_edge_frames: int | None = None

  # Make the first N frames of MusicCoCa silent for eval.
  silent_musiccoca_frames_eval: int | None = None

  musiccoca_shift_frames: int = 0

  # Probability that a MusicCoCa frame will be copied from the previous frame.
  musiccoca_sticky_prob: float = 0.995  # changes every 8 seconds on average

  # Use pretrained MusicCoCa embedder instead of training from scratch.
  use_pretrained_musiccoca_embedder: bool = True

  # 20s * 25Hz / 20 layers = 25 frames per layer.
  encoder_max_past_horizon: int = 25
  decoder_temporal_self_attention_max_past_horizon: int = 25
  decoder_temporal_cross_attention_max_past_horizon: int = 25

  dropout_prob: float = 0.1
  # Optional dropout probability for decoder temporal self-attention. If not
  # specified, the default `dropout_prob` will be used. A high value "blurs"
  # the model's view of its own past output. This can be useful to encourage
  # Audio-to-Audio models to rely more on the encoder-side input signal.
  temporal_self_attention_dropout_prob: float | None = None

  spectrostream: TokensConfig = SPECTROSTREAM

  @property
  def target_tokens_config(self) -> TokensConfig:
    return dataclasses.replace(
        self.spectrostream,
        key='ss_target_tokens',
        frame_rate=SPECTROSTREAM.frame_rate,
    )

  @property
  def input_configs(self) -> Sequence[TokensConfig]:
    return (
        MUSICCOCA,
        PIANOROLL_WITH_ONSETS,
        DRUM_PIANOROLL,
        CFG_CONDITIONING_MUSICCOCA_NOTES,
        CFG_CONDITIONING_DRUMS,
    )

  @property
  def input_num_channels(self) -> int:
    return sum(cfg.rvq_truncation_level for cfg in self.input_configs)

  @property
  def input_frame_rate(self) -> float:
    frame_rates = {cfg.frame_rate for cfg in self.input_configs}
    if len(frame_rates) != 1:
      raise ValueError(
          f'Input configs have different frame rates: {frame_rates}'
      )
    return frame_rates.pop()

  def depthformer_config(self):
    """Returns the configuration for the Depthformer EncoderDecoder model."""
    encoder_spec = dataclasses.replace(
        self.encoder_size, dropout_prob=self.dropout_prob
    )
    decoder_temporal_spec = dataclasses.replace(
        self.decoder_temporal_size, dropout_prob=self.dropout_prob
    )
    decoder_depth_spec = dataclasses.replace(
        self.decoder_depth_size, dropout_prob=0.0
    )
    # Convert from decoder_temporal_spec to decoder_depth_spec if different.
    if decoder_temporal_spec.model_dims != decoder_depth_spec.model_dims:
      depth_input_adapter = sl.Dense.Config(
          decoder_depth_spec.model_dims,
          use_bias=False,
          param_dtype=self.param_dtype,
          compute_dtype=self.compute_dtype,
          name='depth_input_adapter',
      )
    else:
      depth_input_adapter = sl.Identity.Config(name='depth_input_adapter')

    if self.use_pretrained_musiccoca_embedder:
      musiccoca_cfg = self.input_configs[0]
      assert musiccoca_cfg.key == 'mulan_tokens_25hz', (
          f'Expected first input config to be MusicCoCa, got {musiccoca_cfg.key}'
      )

      offset = (
          np.arange(musiccoca_cfg.rvq_truncation_level)
          * musiccoca_cfg.per_rvq_vocab_size
      )

      musiccoca_embedder = sl.Serial.Config(
          [
              sl.Lambda.Config(
                  lambda x: x + offset,
                  mask_required=False,
                  expected_input_spec=sl.ShapeDType(
                      shape=[musiccoca_cfg.rvq_truncation_level],
                      dtype=jnp.int32,
                  ),
              ),
              sl.Embedding.Config(
                  dimension=musiccoca_cfg.embedding_size,
                  num_embeddings=musiccoca_cfg.rvq_levels
                  * musiccoca_cfg.per_rvq_vocab_size,
                  compute_dtype=self.compute_dtype,
                  param_dtype=self.param_dtype,
                  embedding_init=jax.nn.initializers.zeros,
                  name='mulan_dequantizer',
              ),
              sl.Lambda.Config(
                  lambda x: jnp.sum(x, axis=-2),
                  mask_required=False,
                  expected_input_spec=sl.ShapeDType(
                      shape=[
                          musiccoca_cfg.rvq_truncation_level,
                          musiccoca_cfg.embedding_size,
                      ],
                      dtype=self.compute_dtype,
                  ),
              ),
              sl.Dense.Config(
                  encoder_spec.model_dims,
                  use_bias=False,
                  param_dtype=self.param_dtype,
                  compute_dtype=self.compute_dtype,
                  name='depth_input_adapter',
              ),
          ],
          name='mulan_embedder',
      )

      # Other condition embedder (pianoroll, etc.)
      num_embeddings_per_channel = []
      for cfg in self.input_configs[1:]:
        num_embeddings_per_channel += [
            cfg.per_rvq_vocab_size
        ] * cfg.rvq_truncation_level

      num_regular_channels = (
          self.input_num_channels - musiccoca_cfg.rvq_truncation_level
      )
      regular_embedder = transformer.MultiChannelEmbedding.Config(
          num_embeddings_per_channel=num_embeddings_per_channel,
          dimension=encoder_spec.model_dims,
          num_channels=num_regular_channels,
          reduction_fn=jnp.mean,
          param_dtype=self.param_dtype,
          compute_dtype=self.compute_dtype,
          name='regular_embedder',
      )

      encoder_embedding = branch_config(
          [
              musiccoca_cfg.rvq_truncation_level,
              num_regular_channels,
          ],
          musiccoca_embedder,
          regular_embedder,
      )
    else:
      # MultiChannelEmbedding now supports different vocab sizes per channel
      num_embeddings_per_channel = []
      for cfg in self.input_configs:
        num_embeddings_per_channel += [
            cfg.per_rvq_vocab_size
        ] * cfg.rvq_truncation_level

      encoder_embedding = transformer.MultiChannelEmbedding.Config(
          num_embeddings_per_channel=num_embeddings_per_channel,
          dimension=encoder_spec.model_dims,
          num_channels=self.input_num_channels,
          reduction_fn=jnp.mean,
          param_dtype=self.param_dtype,
          compute_dtype=self.compute_dtype,
          name='encoder_embedding',
      )

    return depthformer.EncoderDecoder.Config(
        name='depthformer',
        conditioning_name='source',
        streaming_encoder=True,
        encoder=depthformer.Encoder.Config(
            vocab_size=0,  # unused
            embedding_dimension=encoder_spec.model_dims,
            body=sl.Identity.Config(),
            embedding=encoder_embedding,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
        ),
        decoder=depthformer.MultivariateDecoder.Config(
            num_codebooks=self.target_tokens_config.rvq_truncation_level,
            sos_id=0,
            soft_cap_logits=30.0,
            # Shared embedding between time-wise and depth-wise model.
            embedder=sl.Serial.Config(
                [
                    # Embed
                    # [self.num_classes, self.input_dims],
                    sl.Embedding.Config(
                        num_embeddings=self.target_tokens_config.vocab_size,
                        dimension=decoder_temporal_spec.model_dims,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name='embedding',
                    ),
                    # Manually implement Embedding.scale_sqrt_depth=True
                    sl.Scale.Config(np.sqrt(decoder_temporal_spec.model_dims)),
                ],
                name='decoder_embedding',
            ),
            temporal_body=sl.Serial.Config(
                [
                    # Input: [b, t, target_depth, model_dims]
                    transformer.SLTransformer.Config(
                        model_dimension=decoder_temporal_spec.model_dims,
                        num_layers=1
                        if decoder_temporal_spec.use_repeat_layers
                        else decoder_temporal_spec.num_layers,
                        use_repeated=decoder_temporal_spec.use_repeat_layers,
                        num_repeats=decoder_temporal_spec.num_layers
                        if decoder_temporal_spec.use_repeat_layers
                        else 1,
                        ffn_dim=decoder_temporal_spec.hidden_dims,
                        num_heads=decoder_temporal_spec.num_heads,
                        units_per_head=decoder_temporal_spec.dim_per_head,
                        dropout_rate=decoder_temporal_spec.dropout_prob,
                        self_attention_dropout_rate=self.temporal_self_attention_dropout_prob,
                        max_past_horizon=self.decoder_temporal_self_attention_max_past_horizon,
                        max_future_horizon=0,
                        use_rope=self.use_rope,
                        ffn_gated=decoder_temporal_spec.ffn_use_gated_activation,
                        attention_zero_fully_masked=self.num_attention_sink_embeddings
                        + self.use_attention_sink_scalars
                        == 0,
                        use_cross_attention=True,
                        cross_attention_source_name='source',
                        use_streaming_cross_attention=True,
                        streaming_cross_attention_max_past_horizon=self.decoder_temporal_cross_attention_max_past_horizon,
                        streaming_cross_attention_max_future_horizon=0,
                        streaming_cross_attention_use_query_delay_buffer=False,
                        num_attention_sink_embeddings=self.num_attention_sink_embeddings,
                        use_attention_sink_scalars=self.use_attention_sink_scalars,
                        self_attention_use_separate_qkv=self.self_attention_use_separate_qkv,
                        cross_attention_use_separate_kv=self.cross_attention_use_separate_kv,
                        self_attention_use_kv_cache_ringbuffer=self.temporal_transformer_self_attention_use_kv_cache_ringbuffer,
                        streaming_cross_attention_use_kv_cache_ringbuffer=self.temporal_transformer_cross_attention_use_kv_cache_ringbuffer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                    ),
                    # Final layernorm disabled.
                ],
                name='temporal_body',
            ),
            depth_body=sl.Serial.Config(
                [
                    depth_input_adapter,
                    transformer.SLTransformer.Config(
                        model_dimension=decoder_depth_spec.model_dims,
                        num_layers=1
                        if decoder_depth_spec.use_repeat_layers
                        else decoder_depth_spec.num_layers,
                        use_repeated=decoder_depth_spec.use_repeat_layers,
                        num_repeats=decoder_depth_spec.num_layers
                        if decoder_depth_spec.use_repeat_layers
                        else 1,
                        ffn_dim=decoder_depth_spec.hidden_dims,
                        num_heads=decoder_depth_spec.num_heads,
                        units_per_head=decoder_depth_spec.dim_per_head,
                        dropout_rate=decoder_depth_spec.dropout_prob,
                        max_past_horizon=self.target_tokens_config.rvq_truncation_level,
                        max_future_horizon=0,
                        use_rope=self.use_rope,
                        rope_positions_in_at_least_fp32=False,
                        reductions_in_at_least_fp32=True,
                        ffn_gated=decoder_depth_spec.ffn_use_gated_activation,
                        attention_zero_fully_masked=True,
                        use_cross_attention=False,
                        cross_attention_source_name=None,
                        streaming_cross_attention_use_query_delay_buffer=False,
                        self_attention_use_separate_qkv=self.self_attention_use_separate_qkv,
                        cross_attention_use_separate_kv=self.cross_attention_use_separate_kv,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                    ),
                    sl.LayerNormalization.Config(
                        epsilon=1e-6,
                        use_bias=True,
                        use_scale=True,
                        param_dtype=self.param_dtype,
                        name='final_ln',
                    ),
                    sl.Dense.Config(
                        self.target_tokens_config.vocab_size,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name='to_logits',
                    ),
                ],
                name='depth_body',
            ),
            num_reserved_tokens=self.target_tokens_config.num_extra_tokens,
            codebook_size=self.target_tokens_config.codebook_size,
        ),
    )



class MagentaRT2ModelSmall(MagentaRT2ModelBase):
  encoder_size: ModelSpec = S
  decoder_temporal_size: ModelSpec = L_TPU_OPTIMIZED
  decoder_depth_size: ModelSpec = M_SHALLOW_TPU_OPTIMIZED

  # 20s * 25Hz / 12 layers = 41 frames per layer.
  encoder_max_past_horizon: int = 41
  decoder_temporal_self_attention_max_past_horizon: int = 41
  decoder_temporal_cross_attention_max_past_horizon: int = 41


# ---------------------------------------------------------------------------
# Model registry – maps short CLI-friendly names to model classes.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[MagentaRT2ModelBase]] = {
    'mrt2_base': MagentaRT2ModelBase,
    'mrt2_small': MagentaRT2ModelSmall,
}



def get_model_class(name: str) -> type[MagentaRT2ModelBase]:
  """Look up a model class by its short registry name."""
  if name not in MODEL_REGISTRY:
    available = ', '.join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(
        f"Unknown model '{name}'. Available models: {available}"
    )
  return MODEL_REGISTRY[name]
