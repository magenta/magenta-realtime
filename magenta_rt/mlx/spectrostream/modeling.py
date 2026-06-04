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

"""SpectroStream implemented in SequenceLayers for MLX."""

from collections.abc import Sequence
import dataclasses
import fractions
import math
from typing import Literal, Optional, Union

import mlx.core as mx
from mlx import nn
import numpy as np
import sequence_layers.mlx as sl
from sequence_layers.mlx import signal
from sequence_layers.mlx import utils


def conv2d(
    filters: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    dilation: tuple[int, int],
    groups: int,
    weight_norm: bool,
    wrap: bool,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
    name: str | None = None,
) -> sl.SequenceLayerConfig:
  """Creates a Conv2D SequenceLayer with custom spatial padding."""
  pad_freq = max((kernel_size[1] - 1) * dilation[1] + 1 - strides[1], 0)
  spatial_padding = (pad_freq // 2, pad_freq - pad_freq // 2)

  if weight_norm:
    raise NotImplementedError('weight_norm is not implemented.')

  conv = sl.Conv2D.Config(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      dilation_rate=dilation,
      time_padding='semicausal' if padding == 'causal' else padding,
      spatial_padding=spatial_padding,
      groups=groups,
      param_dtype=param_dtype,
      compute_dtype=compute_dtype,
      name='conv' if wrap else name,
  )

  if wrap:
    return sl.Serial.Config([conv], name=name)
  else:
    return conv


def pre_activated_conv2d(
    activation: sl.SequenceLayerConfig,
    filters: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    dilation: tuple[int, int],
    transpose: bool,
    weight_norm: bool,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
    name: str | None = None,
):
  """Creates a Conv2D or transpose Conv2D with activated inputs."""
  layers = [activation]

  if transpose and (strides[0] != 1 or strides[1] != 1):
    if padding not in ('same', 'causal'):
      raise NotImplementedError(
          'Transpose padding other than same and causal is not implemented.'
      )
    layers.append(
        sl.Conv2DTranspose.Config(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            time_padding=padding,
            spatial_padding='same',
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name='conv',
        )
    )
  else:
    layers.append(
        conv2d(
            filters,
            kernel_size,
            strides=strides,
            padding='semicausal' if padding == 'causal' else padding,
            dilation=dilation,
            groups=1,
            weight_norm=weight_norm,
            wrap=False,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name='conv',
        )
    )
  return sl.Serial.Config(layers, name=name)


def conv2d_residual_unit(
    input_channels: int,
    output_channels: int,
    strides: tuple[int, int],
    dilation: tuple[int, int],
    transposed: bool,
    activation: sl.SequenceLayerConfig,
    padding: str,
    weight_norm: bool,
    use_shortcut: bool,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
    name: str | None = None,
):
  """Creates a residual act -> conv2d (transpose) -> act conv2d block."""
  layers = []
  resample_kernel_size = (max(3, 2 * strides[0]), max(3, 2 * strides[1]))

  if transposed:
    filters = output_channels
    if strides == (1, 1):
      layers.append(
          pre_activated_conv2d(
              activation,
              filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding=padding,
              dilation=(1, 1),
              transpose=False,
              weight_norm=weight_norm,
              param_dtype=param_dtype,
              compute_dtype=compute_dtype,
              name='conv2d_3x3_a',
          )
      )
    else:
      layers.append(
          pre_activated_conv2d(
              activation,
              filters,
              kernel_size=resample_kernel_size,
              strides=strides,
              padding=padding,
              dilation=(1, 1),
              transpose=True,
              weight_norm=weight_norm,
              param_dtype=param_dtype,
              compute_dtype=compute_dtype,
              name='conv2dtranspose_%dx%d' % resample_kernel_size,
          )
      )
  else:
    filters = input_channels

  layers.append(
      pre_activated_conv2d(
          activation,
          filters,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding=padding,
          dilation=dilation,
          transpose=False,
          weight_norm=weight_norm,
          param_dtype=param_dtype,
          compute_dtype=compute_dtype,
          name='conv2d_3x3',
      )
  )

  if not transposed:
    layers.append(
        pre_activated_conv2d(
            activation,
            output_channels,
            kernel_size=resample_kernel_size,
            strides=strides,
            padding=padding,
            dilation=(1, 1),
            transpose=False,
            weight_norm=weight_norm,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name='conv2d_%dx%d_a' % resample_kernel_size,
        )
    )

  if use_shortcut:
    shortcut_layers = []

    if strides != (1, 1) and not transposed:
      shortcut_layers.append(
          sl.AveragePooling2D.Config(
              pool_size=strides,
              strides=strides,
              time_padding='semicausal' if padding == 'causal' else padding,
              spatial_padding='valid',
              name='avgpool2d',
          )
      )

    if input_channels != output_channels:
      shortcut_layers.append(
          conv2d(
              output_channels,
              kernel_size=(1, 1),
              strides=(1, 1),
              padding='causal',
              dilation=(1, 1),
              groups=1,
              weight_norm=weight_norm,
              wrap=True,
              param_dtype=param_dtype,
              compute_dtype=compute_dtype,
              name='conv1x1',
          ),
      )

    if strides != (1, 1) and transposed:
      shortcut_layers.append(
          sl.Upsample2D.Config(rate=strides, name='upsample2d')
      )

    return sl.Residual.Config(
        layers, shortcut_layers=shortcut_layers, name=name
    )
  else:
    return sl.Serial.Config(layers, name=name)


# ---------------------------------------------------------------------------
# STFT / Inverse STFT config functions
# ---------------------------------------------------------------------------

def spectrostream_stft_config(
    frame_length: int,
    frame_step: int,
    fft_length: int,
    time_padding: Literal['semicausal', 'causal', 'reverse_causal'],
    keep_dc: bool,
    num_channels: int,
    compute_dtype: sl.DType = mx.float32,
    name: str | None = None,
) -> sl.SequenceLayerConfig:
  """Returns an STFT SequenceLayerConfig compatible with SpectroStream."""

  assert num_channels % 2 == 0
  num_audio_channels = num_channels // 2

  def slice_and_bitcast(v: mx.array) -> mx.array:
    """Post-processes the complex STFT (tiles channels, bitcasts, slices)."""
    assert v.ndim in (3, 4)
    if v.ndim == 3:
      v = v[..., np.newaxis]

    if v.shape[3] == 1 and num_audio_channels > 1:
      v = mx.tile(v, [1, 1, 1, num_audio_channels])

    # Bitcast complex64 -> 2x float32
    float_dtype = mx.float32
    v = v.view(float_dtype)

    assert v.shape[2] == fft_length // 2 + 1, v.shape
    v = v[:, :, :-1] if keep_dc else v[:, :, 1:]
    return v

  return sl.Serial.Config(
      [
          sl.STFT.Config(
              frame_length=frame_length,
              frame_step=frame_step,
              fft_length=fft_length,
              window_fn=signal.hann_window,
              time_padding=time_padding,
              fft_padding='right',
          ),
          sl.Lambda.Config(
              slice_and_bitcast,
              mask_required=False,
          ),
          sl.Cast.Config(compute_dtype),
      ],
      name=name,
  )


def spectrostream_inverse_stft_config(
    frame_length: int,
    frame_step: int,
    fft_length: int,
    causal: bool,
    keep_dc: bool,
    num_bins: int,
    num_channels: int,
    compute_dtype: sl.DType = mx.float32,
    name: str | None = None,
):
  """Returns an InverseSTFT SequenceLayerConfig compatible with SpectroStream."""

  def cast_pad_and_bitcast(v: mx.array) -> mx.array:
    if compute_dtype not in (mx.float32, mx.float64):
      v = v.astype(mx.float32)

    assert v.shape[2:] == (num_bins, num_channels), v.shape

    channel_padding = [0, 1] if keep_dc else [1, 0]
    paddings = [[0, 0], [0, 0], channel_padding, [0, 0]]
    v = mx.pad(v, paddings)
    complex_dtype = mx.complex64

    assert v.shape[2] == fft_length // 2 + 1
    v = v.view(complex_dtype)
    assert v.shape[3] == num_channels // 2

    if v.shape[-1] == 1:
      v = v.squeeze(-1)

    return v

  return sl.Serial.Config(
      [
          sl.Lambda.Config(
              cast_pad_and_bitcast,
              mask_required=False,
          ),
          sl.InverseSTFT.Config(
              frame_length=frame_length,
              frame_step=frame_step,
              fft_length=fft_length,
              time_padding='causal' if causal else 'same',
              window_fn=signal.inverse_stft_window_fn(
                  frame_step,
                  signal.hann_window,
              ),
              fft_padding='right',
          ),
      ],
      name=name,
  )


# ---------------------------------------------------------------------------
# Encoder / Decoder config functions
# ---------------------------------------------------------------------------

def spectrostream_encoder_config(
    base_conv_depth: int,
    base_conv_size: Union[int, tuple[int, int]],
    ratios: Sequence[tuple[int, int]],
    mults: Sequence[Union[int, float]],
    dilations: Optional[
        Union[Sequence[tuple[int, int]], tuple[int, int]]
    ] = None,
    channel_splits: int | None = None,
    channel_recombo_block: int = -1,
    is_resnet: bool = True,
    activation: sl.SequenceLayerConfig = sl.Elu.Config(),
    num_input_bins: int = 160,
    num_output_features: int = 64,
    causal: bool = True,
    lookahead: int = 0,
    global_weight_norm: bool = False,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
    input_latency: int = 0,
    embedding_normalizer: sl.SequenceLayerConfig | None = None,
) -> sl.SequenceLayerConfig:
  """Creates a SpectroStream encoder as a SequenceLayer."""

  num_blocks = len(ratios) + 1

  if isinstance(base_conv_size, int):
    base_conv_size = (base_conv_size,) * 2
  if dilations is None:
    dilations = (1, 1)
  if isinstance(dilations[0], int):
    dilations = (dilations,) * len(ratios)

  padding = 'causal' if causal else 'same'

  if lookahead:
    raise NotImplementedError('Lookahead not supported in encoder.')

  if channel_splits:
    if not -num_blocks <= channel_recombo_block < num_blocks:
      raise ValueError(
          f'channel_recombo_block {channel_recombo_block} out of range for only'
          f' {num_blocks} blocks (excluding base conv).'
      )
    channel_recombo_block %= num_blocks

  layers = [
      conv2d(
          base_conv_depth,
          kernel_size=base_conv_size,
          strides=(1, 1),
          padding=padding,
          dilation=(1, 1),
          groups=1,
          weight_norm=global_weight_norm,
          wrap=True,
          param_dtype=param_dtype,
          compute_dtype=compute_dtype,
          name='base_conv_first',
      )
  ]
  share_scope = [False]

  input_channels = base_conv_depth
  output_channels = base_conv_depth
  curr_num_bins = num_input_bins
  for level_index, (strides, dilation, mult) in enumerate(
      zip(ratios, dilations, mults)
  ):
    if channel_splits and channel_recombo_block == level_index:
      layers = [
          sl.ParallelChannels.Config(
              sl.Serial.Config(layers),
              num_groups=channel_splits,
              combination=utils.CombinationMode.CONCAT,
          )
      ]
      share_scope = [True]
      input_channels *= channel_splits

    output_channels = int(np.round(output_channels * mult))
    curr_num_bins //= strides[1]

    time_stride = strides[0]

    conv_i = conv2d_residual_unit(
        input_channels=input_channels,
        output_channels=output_channels,
        strides=strides,
        dilation=dilation,
        transposed=False,
        activation=activation,
        padding=padding,
        weight_norm=global_weight_norm,
        use_shortcut=is_resnet,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        name=f'encoder_{level_index}',
    )
    conv_i_output_latency = utils.get_output_latency(conv_i)
    conv_i_output_ratio = fractions.Fraction(1, time_stride)

    delay_amount = utils.get_required_stepwise_delay(
        conv_i_output_ratio, input_latency
    )
    if delay_amount:
      layers.append(sl.Delay.Config(delay_amount, delay_layer_output=False))
      share_scope.append(False)
      input_latency += delay_amount

    input_latency = (
        int(input_latency * conv_i_output_ratio) + conv_i_output_latency
    )

    layers.append(conv_i)
    share_scope.append(False)
    input_channels = output_channels

  if channel_splits and channel_recombo_block == num_blocks - 1:
    layers = [
        sl.ParallelChannels.Config(
            sl.Serial.Config(layers),
            num_groups=channel_splits,
            combination=utils.CombinationMode.CONCAT,
        )
    ]
    share_scope = [True]
    input_channels *= channel_splits

  layers.append(
      conv2d_residual_unit(
          input_channels=input_channels,
          output_channels=output_channels,
          strides=(1, 1),
          dilation=(1, 1),
          transposed=False,
          activation=activation,
          padding=padding,
          weight_norm=global_weight_norm,
          use_shortcut=is_resnet,
          param_dtype=param_dtype,
          compute_dtype=compute_dtype,
          name='bottleneck',
      )
  )
  share_scope.append(False)

  layers.extend([
      sl.Flatten.Config(),
      sl.ExpandDims.Config(axis=0),
      sl.Residual.Config(
          [
              conv2d(
                  num_output_features,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_last',
              ),
          ],
          shortcut_layers=[
              activation,
              conv2d(
                  curr_num_bins * output_channels,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_b1',
              ),
              activation,
              conv2d(
                  num_output_features,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_b2',
              ),
          ],
          name='output_convs',
      ),
      sl.Flatten.Config(),
  ])
  share_scope.extend([False, False, False, False])

  if embedding_normalizer is not None:
    layers.append(embedding_normalizer)
    share_scope.append(False)

  return sl.Serial.Config(layers, name='encoder')


def spectrostream_decoder_config(
    base_conv_depth: int,
    base_conv_size: Union[int, tuple[int, int]],
    ratios: Sequence[tuple[int, int]],
    mults: Sequence[Union[int, float]],
    dilations: Optional[
        Union[Sequence[tuple[int, int]], tuple[int, int]]
    ] = None,
    channel_splits: int | None = None,
    channel_recombo_block: int = -1,
    is_resnet: bool = True,
    activation: sl.SequenceLayerConfig = sl.Elu.Config(),
    num_output_bins: int = 160,
    num_output_channels: int = 2,
    causal: bool = True,
    lookahead: int = 0,
    global_weight_norm: bool = False,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
):
  """Creates a SpectroStream decoder as a SequenceLayer."""

  total_time_stride, total_freq_stride = np.prod(ratios, axis=0)
  input_bins = num_output_bins // total_freq_stride
  if input_bins * total_freq_stride != num_output_bins:
    raise ValueError(
        'Ratios applied to the frequency dimension do not match '
        'the expected output shape.'
    )
  num_blocks = len(ratios) + 1

  output_channels = base_conv_depth * np.prod(mults)
  if isinstance(base_conv_size, int):
    base_conv_size = (base_conv_size,) * 2
  if dilations is None:
    dilations = (1, 1)
  if isinstance(dilations[0], int):
    dilations = (dilations,) * len(ratios)
  padding = 'causal' if causal else 'same'

  assert causal or not lookahead

  proj_filters = input_bins * output_channels

  if channel_splits:
    if not -num_blocks <= channel_recombo_block < num_blocks:
      raise ValueError(
          f'channel_recombo_block {channel_recombo_block} out of range for only'
          f' {num_blocks} blocks (excluding base conv).'
      )
    channel_recombo_block %= num_blocks

  layers = [
      sl.ExpandDims.Config(axis=0),
      sl.Residual.Config(
          [
              conv2d(
                  proj_filters,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_first',
              ),
          ],
          shortcut_layers=[
              conv2d(
                  proj_filters,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_b1',
              ),
              activation,
              conv2d(
                  proj_filters,
                  kernel_size=(1, 1),
                  strides=(1, 1),
                  padding='causal',
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=global_weight_norm,
                  wrap=True,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='conv1x1_b2',
              ),
          ],
          name='input_layer',
      ),
      sl.Reshape.Config([input_bins, output_channels]),
  ]

  curr_freq_dim = input_bins

  if channel_splits and channel_recombo_block == num_blocks - 1:
    output_channels *= channel_splits

  layers.append(
      conv2d_residual_unit(
          input_channels=output_channels,
          output_channels=output_channels,
          strides=(1, 1),
          dilation=(1, 1),
          transposed=True,
          activation=activation,
          padding=padding,
          weight_norm=global_weight_norm,
          use_shortcut=is_resnet,
          param_dtype=param_dtype,
          compute_dtype=compute_dtype,
          name='input_layers_residual_unit',
      )
  )
  input_channels = output_channels

  ungrouped_layers = []

  if channel_splits and channel_recombo_block == num_blocks - 1:
    ungrouped_layers = layers
    layers = []
    output_channels //= channel_splits

  for level_index, (strides, dilation, mult) in enumerate(
      zip(ratios[::-1], dilations[::-1], mults[::-1])
  ):
    output_channels = int(np.round(output_channels / mult))

    if channel_splits and channel_recombo_block == num_blocks - 2 - level_index:
      output_channels *= channel_splits

    layers.append(
        conv2d_residual_unit(
            input_channels=input_channels,
            output_channels=output_channels,
            strides=strides,
            dilation=dilation,
            transposed=True,
            activation=activation,
            padding=padding,
            weight_norm=global_weight_norm,
            use_shortcut=is_resnet,
            param_dtype=param_dtype,
            compute_dtype=compute_dtype,
            name=f'decoder_{level_index}',
        )
    )
    input_channels = output_channels
    if channel_splits and channel_recombo_block == num_blocks - 2 - level_index:
      ungrouped_layers = layers
      layers = []
      output_channels //= channel_splits
    curr_freq_dim = curr_freq_dim * strides[1]

  if channel_splits:
    if num_output_channels % channel_splits != 0:
      raise ValueError(
          f'num_output_channels {num_output_channels} is not a multiple of'
          f' channel_splits {channel_splits}'
      )
    num_output_channels //= channel_splits

  layers.append(
      sl.Serial.Config(
          [
              activation,
              conv2d(
                  num_output_channels,
                  kernel_size=base_conv_size,
                  strides=(1, 1),
                  padding=padding,
                  dilation=(1, 1),
                  groups=1,
                  weight_norm=False,
                  param_dtype=param_dtype,
                  compute_dtype=compute_dtype,
                  name='base_conv_last',
                  wrap=True,
              ),
          ],
          name='output_layer',
      )
  )

  if channel_splits:
    layers = ungrouped_layers + [
        sl.ParallelChannels.Config(
            sl.Serial.Config(layers),
            num_groups=channel_splits,
            combination=utils.CombinationMode.CONCAT,
        )
    ]
    share_scope = [False] * len(ungrouped_layers) + [True]
    num_output_channels *= channel_splits
  else:
    assert not ungrouped_layers
    share_scope = [False] * len(layers)

  if lookahead:
    layers.append(
        sl.Lookahead.Config(int(lookahead * total_time_stride), name='lookahead')
    )
    share_scope.append(False)

  return sl.Serial.Config(layers, name='decoder')


# ---------------------------------------------------------------------------
# ResidualVectorQuantizer
# ---------------------------------------------------------------------------

class ResidualVectorQuantizer(nn.Module):
  """An implementation of residual vector quantization. Only supports inference."""

  @dataclasses.dataclass(frozen=True)
  class Config:
    """Config for ResidualVectorQuantizer."""

    num_quantizers: int
    num_embeddings: int
    embedding_dim: int
    use_unique_codes: bool

    # Training configuration. Unsupported for inference.
    beta: float
    dynamic_masking: bool
    target_num_quantizers: Sequence[int]
    full_quantizer_dropout_rate: float
    full_quantizer_commitment: float = False

    use_quantizer_stopgradient: bool = True
    dtype: sl.DType | None = None
    param_dtype: sl.DType = mx.float32

    truncation_level: int | None = None
    encoded_truncation_level: int | None = None
    name: str | None = None

    @property
    def num_expected_output_codes(self) -> int:
      if self.encoded_truncation_level is not None:
        return self.encoded_truncation_level
      else:
        return self.num_quantizers

    @property
    def num_expected_input_codes(self) -> int:
      if self.truncation_level is not None:
        return self.truncation_level
      else:
        return self.num_quantizers

    def make(self) -> 'ResidualVectorQuantizer':
      return ResidualVectorQuantizer(self)

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    embedding_shape = (
        config.num_quantizers,
        config.num_embeddings,
        config.embedding_dim,
    )
    self.embedding = mx.zeros(embedding_shape, dtype=config.param_dtype)

    if config.num_expected_output_codes > config.num_quantizers:
      raise ValueError('num_expected_output_codes must be <= num_quantizers.')

    self.embeddings_to_codes_layer = sl.Lambda.Config(
        self.embeddings_to_codes,
        mask_required=False,
        sequence_input=True,
    ).make()

    if config.num_expected_input_codes > config.num_quantizers:
      raise ValueError('num_expected_input_codes must be <= num_quantizers.')

    self.codes_to_embeddings_layer = sl.Lambda.Config(
        self.codes_to_embeddings,
        mask_required=False,
        sequence_input=True,
        expected_output_spec=sl.ShapeDType(
            (config.embedding_dim,), mx.float32
        ),
    ).make()

  def embeddings_to_codes(
      self, inputs: sl.Sequence, num_quantizers: int | None = None
  ) -> sl.Sequence:
    """Encodes embeddings to codes."""
    num_quantizers = num_quantizers or self.config.num_expected_output_codes

    residual = inputs.values
    codes = []
    for i in range(num_quantizers):
      # Find nearest neighbor
      distances = (
          mx.sum(residual ** 2, axis=-1, keepdims=True)
          - 2 * residual @ self.embedding[i].T
          + mx.sum(self.embedding[i] ** 2, axis=-1)
      )
      code_i = mx.argmin(distances, axis=-1)
      quantized_i = mx.take(self.embedding[i], code_i, axis=0)
      residual = residual - quantized_i
      codes.append(code_i)

    codes = mx.stack(codes, axis=-1)
    if self.config.use_unique_codes:
      offsets = mx.arange(num_quantizers) * self.config.num_embeddings
      codes = codes + offsets
    return sl.Sequence(codes, inputs.mask)

  def codes_to_embeddings(
      self, codes: sl.Sequence, use_gather: bool = True
  ) -> sl.Sequence:
    """A [b, t, num_quantizers] sequence of codes."""
    if self.config.use_unique_codes:
      codes = codes.apply_values_masked(
          lambda v: mx.mod(v, self.config.num_embeddings)  # was jnp.mod
      )

    num_input_quantizers = codes.shape[2]
    if num_input_quantizers > self.config.num_expected_input_codes:
      raise ValueError(
          f'Input to codes_to_embeddings has {num_input_quantizers=}, which is'
          ' greater than the number of expected input codebooks'
          f' {self.config.num_expected_input_codes=}.'
      )

    if codes.ndim != 3 or codes.dtype not in (mx.int32, mx.uint32):
      raise ValueError(
          'ResidualVectorQuantizer expects 3D int32/uint32 input. Got:'
          f' {codes.shape=} {codes.dtype=}'
      )
    codes = codes.astype(mx.int32)

    if use_gather:
      quantized = None
      for i in range(num_input_quantizers):
        quantized_i = mx.take(self.embedding[i], codes.values[:, :, i], axis=0)
        quantized = (
            quantized_i if quantized is None else quantized + quantized_i
        )
      if quantized is None:
        quantized = mx.zeros(
            codes.shape[:2] + (self.config.embedding_dim,), self.embedding.dtype
        )
    else:
      indices = mx.one_hot(
          codes.values, self.config.num_embeddings
      ).astype(mx.float32)
      if self.config.num_quantizers != num_input_quantizers:
        pad_amount = self.config.num_quantizers - num_input_quantizers
        indices = mx.pad(indices, [[0, 0], [0, 0], [0, pad_amount], [0, 0]])
      quantized = mx.einsum('btqn,qnd->btd', indices, self.embedding)

    return sl.Sequence(quantized, codes.mask).mask_invalid()


# ---------------------------------------------------------------------------
# SpectroStream
# ---------------------------------------------------------------------------

class SpectroStream(nn.Module):
  """SpectroStream implemented with SequenceLayers."""

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Config for SpectroStream."""

    audio_sample_rate: float

    stft_frame_length: int
    stft_frame_step: int
    stft_fft_length: int

    ratios: Sequence[tuple[int, int]]
    mults: Sequence[int | float]
    dilations: Sequence[tuple[int, int]] | tuple[int, int] | None
    is_resnet: bool
    activation: sl.SequenceLayerConfig

    num_bins: int
    num_channels: int
    num_features: int

    causal: bool

    encoder_base_conv_depth: int
    encoder_base_conv_size: int | tuple[int, int]
    encoder_lookahead: int

    quantizer: ResidualVectorQuantizer.Config | None

    decoder_base_conv_depth: int
    decoder_base_conv_size: int | tuple[int, int]
    decoder_lookahead: int

    channel_splits: int | None = None
    channel_recombo_block: int = -1

    param_dtype: sl.DType = mx.float32
    compute_dtype: sl.DType = mx.float32

    embedding_normalizer: sl.SequenceLayerConfig | None = None
    name: str | None = None

    mock_embeddings_to_waveform_layer: sl.SequenceLayerConfig | None = None
    mock_embeddings_to_waveform_layer_ratio: int = 1

    def make(self, backend: str = 'mlx') -> 'SpectroStream':
      return SpectroStream(self)

    @property
    def stft(self) -> sl.SequenceLayerConfig:
      return spectrostream_stft_config(
          self.stft_frame_length,
          self.stft_frame_step,
          fft_length=self.stft_fft_length,
          time_padding='reverse_causal',
          keep_dc=True,
          num_channels=self.num_channels,
          compute_dtype=self.compute_dtype,
      )

    @property
    def inverse_stft(self) -> sl.SequenceLayerConfig:
      return spectrostream_inverse_stft_config(
          self.stft_frame_length,
          self.stft_frame_step,
          fft_length=self.stft_fft_length,
          causal=True,
          keep_dc=True,
          num_channels=self.num_channels,
          num_bins=self.num_bins,
          compute_dtype=self.compute_dtype,
      )

    @property
    def encoder(self) -> sl.SequenceLayerConfig:
      frontend_latency = utils.get_output_latency(self.stft)

      return spectrostream_encoder_config(
          base_conv_depth=self.encoder_base_conv_depth,
          base_conv_size=self.encoder_base_conv_size,
          ratios=self.ratios,
          mults=self.mults,
          dilations=self.dilations,
          channel_splits=self.channel_splits,
          channel_recombo_block=self.channel_recombo_block,
          is_resnet=self.is_resnet,
          activation=self.activation,
          num_input_bins=self.num_bins,
          num_output_features=self.num_features,
          causal=self.causal,
          lookahead=self.encoder_lookahead,
          param_dtype=self.param_dtype,
          compute_dtype=self.compute_dtype,
          input_latency=frontend_latency,
          embedding_normalizer=self.embedding_normalizer,
      )

    @property
    def decoder(self) -> sl.SequenceLayerConfig:
      return spectrostream_decoder_config(
          base_conv_depth=self.decoder_base_conv_depth,
          base_conv_size=self.decoder_base_conv_size,
          ratios=self.ratios,
          mults=self.mults,
          dilations=self.dilations,
          channel_splits=self.channel_splits,
          channel_recombo_block=self.channel_recombo_block,
          is_resnet=self.is_resnet,
          activation=self.activation,
          num_output_bins=self.num_bins,
          num_output_channels=self.num_channels,
          causal=self.causal,
          lookahead=self.decoder_lookahead,
          param_dtype=self.param_dtype,
          compute_dtype=self.compute_dtype,
      )

    @property
    def waveform_to_codes_ratio(self) -> int:
      """Returns the ratio of waveform length to codes length via the config."""
      if self.mock_embeddings_to_waveform_layer:
        return self.mock_embeddings_to_waveform_layer_ratio
      return math.prod(r[0] for r in self.ratios) * self.stft_frame_step

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.stft = config.stft.make(backend='mlx')
    self.inverse_stft = config.inverse_stft.make(backend='mlx')
    self.encoder = config.encoder.make(backend='mlx')
    self.decoder = config.decoder.make(backend='mlx')

    self.waveform_to_embeddings_layer = sl.SerialModules((
        self.stft,
        self.encoder,
    ))

    if config.mock_embeddings_to_waveform_layer:
      self.embeddings_to_waveform_layer = (
          config.mock_embeddings_to_waveform_layer.make(backend='mlx')
      )
    else:
      self.embeddings_to_waveform_layer = sl.SerialModules((
          self.decoder,
          self.inverse_stft,
      ))

    if config.quantizer is not None:
      self.quantizer = config.quantizer.make()
      self.waveform_to_codes_layer = sl.SerialModules((
          self.stft,
          self.encoder,
          self.quantizer.embeddings_to_codes_layer,
      ))
      self.codes_to_waveform_layer = sl.SerialModules((
          self.quantizer.codes_to_embeddings_layer,
          self.decoder,
          self.inverse_stft,
      ))
      if not config.mock_embeddings_to_waveform_layer:
        assert (
            self.codes_to_waveform_layer.output_ratio
            == config.waveform_to_codes_ratio
        )
    else:
      self.quantizer = None


# ---------------------------------------------------------------------------
# Standard config
# ---------------------------------------------------------------------------

def stft_spectrostream_40ms_generic_48khz_stereo_config(
    rvq_truncation_level: int | None = None,
    encoded_truncation_level: int | None = None,
    param_dtype: sl.DType = mx.float32,
    compute_dtype: sl.DType = mx.float32,
    use_unique_codes: bool = False,
) -> SpectroStream.Config:
  """Config for a 40ms 48 kHz stereo STFT SpectroStream model."""
  return SpectroStream.Config(
      audio_sample_rate=48000.0,
      param_dtype=param_dtype,
      compute_dtype=compute_dtype,
      stft_frame_length=960,
      stft_frame_step=480,
      stft_fft_length=960,
      ratios=((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1)),
      mults=(2, 1, 2, 1, 1, 2, 1),
      dilations=None,
      is_resnet=True,
      activation=sl.Elu.Config(),
      causal=True,
      num_bins=480,
      num_channels=4,
      channel_splits=2,
      channel_recombo_block=-2,
      num_features=256,
      encoder_base_conv_depth=32,
      encoder_base_conv_size=7,
      encoder_lookahead=0,
      decoder_base_conv_depth=64,
      decoder_base_conv_size=7,
      decoder_lookahead=1,
      quantizer=ResidualVectorQuantizer.Config(
          num_quantizers=64,
          truncation_level=rvq_truncation_level,
          encoded_truncation_level=encoded_truncation_level,
          num_embeddings=1024,
          embedding_dim=256,
          use_unique_codes=use_unique_codes,
          beta=1.0,
          dynamic_masking=True,
          target_num_quantizers=(
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
              18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64,
          ),
          full_quantizer_dropout_rate=0.5,
          full_quantizer_commitment=False,
      ),
  )


# Keep SpectroStreamConfig as an alias for backward compatibility with system.py
# until it's updated to use SpectroStream.Config.
SpectroStreamConfig = SpectroStream.Config
