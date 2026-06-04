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

"""SpectroStream implemented in SequenceLayers."""

from collections.abc import Sequence
import dataclasses
import fractions
import math
from typing import Literal, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import sequence_layers.jax as sl
from sequence_layers.jax import signal
from sequence_layers.jax import utils

UNCONSTRAINED = jax.sharding.PartitionSpec.UNCONSTRAINED
DimSharding = str | Sequence[str] | None | type(UNCONSTRAINED)
Sharding = Sequence[DimSharding] | None


def conv2d(
    filters: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: sl.PaddingModeString,
    dilation: tuple[int, int],
    groups: int,
    weight_norm: bool,
    wrap: bool,
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
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
    padding: sl.PaddingModeString,
    dilation: tuple[int, int],
    transpose: bool,
    weight_norm: bool,
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
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
    padding: sl.PaddingModeString,
    weight_norm: bool,
    use_shortcut: bool,
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
    name: str | None = None,
):
  """Creates a residual act -> conv2d (transpose) -> act conv2d block."""
  layers = []

  # Make the type Tuple[int, int] explicit, since
  # `tuple(max(3, s * 2) for s in strides)` would produce type Tuple[int, ...]).
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
      # The KWS streaming classes use semi-causal padding, so if kernel_size ==
      # stride, no padding is necessary. This is why valid padding on
      # AveragePooling2D doesn't cause an issue. We need to use causal padding
      # to avoid introducing a time offset with respect to the above
      # convolutions.
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


def spectrostream_stft_config(
    frame_length: int,
    frame_step: int,
    fft_length: int,
    time_padding: Literal['semicausal', 'causal', 'reverse_causal'],
    keep_dc: bool,
    num_channels: int,
    compute_dtype: sl.DType = jnp.float32,
    name: str | None = None,
) -> sl.SequenceLayerConfig:
  """Returns an STFT SequenceLayerConfig compatible with SpectroStream."""

  assert num_channels % 2 == 0
  num_audio_channels = num_channels // 2

  def slice_and_bitcast(v: jax.Array) -> jax.Array:
    """Post-processes the complex STFT (tiles channels, bitcasts, slices)."""
    assert v.ndim in (3, 4)
    # Expand from [b, t // frame_step, fft_length // 2 + 1] to [b, t //
    # frame_step, fft_length // 2 + 1, 1]
    if v.ndim == 3:
      v = v[..., jnp.newaxis]

    # If input is single-channel, tile it to the number of audio channels before
    # bitcasting.
    if v.shape[3] == 1 and num_audio_channels > 1:
      v = jnp.tile(v, [1, 1, 1, num_audio_channels])

    float_dtype = jnp.float32 if v.dtype == jnp.complex64 else jnp.float64

    # Bitcast from [b, t // frame_step, fft_length // 2 + 1, c] to [b, t //
    # frame_step, fft_length // 2 + 1, c * 2].
    v = v.view(float_dtype)

    # keep_dc controls whether we drop the DC bin or the Nyquist bin.
    assert v.shape[2] == fft_length // 2 + 1, v.shape
    v = v[:, :, :-1] if keep_dc else v[:, :, 1:]
    return v

  return sl.Serial.Config(
      [
          # [b, t] or [b, t, c]
          sl.STFT.Config(
              frame_length=frame_length,
              frame_step=frame_step,
              fft_length=fft_length,
              window_fn=signal.hann_window,
              time_padding=time_padding,
              fft_padding='right',
          ),
          # [b, t // frame_step, fft_length // 2 + 1]
          # or
          # [b, t // frame_step, fft_length // 2 + 1, c]
          sl.Lambda.Config(
              slice_and_bitcast,
              expected_input_spec=sl.ShapeDType(
                  (fft_length // 2 + 1,), jnp.complex64
              ),
              mask_required=False,
          ),
          # [b, t // frame_step, fft_length // 2, c * 2]
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
    compute_dtype: sl.DType = jnp.float32,
    name: str | None = None,
):
  """Returns an InverseSTFT SequenceLayerConfig compatible with SpectroStream."""

  def cast_pad_and_bitcast(v: jax.Array) -> jax.Array:
    # Cast to float32 if the input dtype is not one of the supported types for
    # IRFFT.
    if compute_dtype not in (jnp.float32, jnp.float64):
      v = v.astype(jnp.float32)

    assert v.shape[2:] == (num_bins, num_channels), v.shape

    # Input is [b, t, num_bins, num_channels].
    channel_padding = [0, 1] if keep_dc else [1, 0]
    paddings = [[0, 0], [0, 0], channel_padding, [0, 0]]
    v = jnp.pad(v, paddings)
    complex_dtype = jnp.complex64 if v.dtype == jnp.float32 else jnp.complex128

    assert v.shape[2] == fft_length // 2 + 1
    v = v.view(complex_dtype)
    assert v.shape[3] == num_channels // 2

    # [b, t, fft_length // 2 + 1, num_channels // 2]
    # If num_channels is 2, squeeze it out.
    if v.shape[-1] == 1:
      v = v.squeeze(-1)

    return v

  return sl.Serial.Config(
      [
          sl.Lambda.Config(
              cast_pad_and_bitcast,
              expected_input_spec=sl.ShapeDType(
                  [num_bins, num_channels],
                  compute_dtype,
              ),
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
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
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
    # Lookahead currently handled outside of the encoder. Handle
    # lookahead padding here. In training, pad lookahead * total_time_stride
    # timesteps to the end of the sequence.
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

    # b/372530075 - Insert a delay such that the input latency is divisible by
    # this layer's stride. Accumulate output latencies until we hit a strided
    # conv.
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
            sl.Serial.Config(layers, share_scope=share_scope),
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

  # Aggregate over frequency channels.
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

  if lookahead:
    # Lookahead currently handled outside of the encoder. Handle
    # lookahead cropping here. In training, crop lookahead samples from the
    # front of the sequence.
    pass

  return sl.Serial.Config(layers, share_scope=share_scope, name='encoder')


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
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
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

  if lookahead:
    # Lookahead currently handled outside of the decoder. Handle
    # lookahead padding here. In training, pad lookahead samples to the end of
    # the sequence.
    pass

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
                  weight_norm=False,  # No normalization for the final output.
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
        sl.Lookahead.Config(lookahead * total_time_stride, name='lookahead')
    )
    share_scope.append(False)

  return sl.Serial.Config(layers, share_scope=share_scope, name='decoder')


class ResidualVectorQuantizer(nn.Module):
  """An implementation of residual vector quantization.

  Only supports inference.
  """

  @dataclasses.dataclass(frozen=True)
  class Config:
    """Config for ResidualVectorQuantizer."""

    num_quantizers: int
    num_embeddings: int
    embedding_dim: int

    # If true, use unique codes in the range
    # [0, num_quantizers * num_embeddings). Codes for each quantizer are offset
    # by a quantizer_index * num_embeddings offset.
    # If false, codes for all codebooks are in the range [0, num_embeddings).
    use_unique_codes: bool

    # Training configuration. Unsupported.
    beta: float
    dynamic_masking: bool
    target_num_quantizers: Sequence[int]
    full_quantizer_dropout_rate: float
    full_quantizer_commitment: float

    # VQ params
    use_quantizer_stopgradient: bool = True

    dtype: sl.DType | None = None
    param_dtype: sl.DType = jnp.float32
    embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
    embedding_sharding: Sharding | None = None
    # Optional expected truncation level (num. input codes expected) for
    # SpectroStream decoding (de-quantization) into embeddings.
    truncation_level: int | None = None
    # Optional expected truncation level (num. output codes to produce) for
    # SpectroStream tokenization (quantization) to avoid extra computation.
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
      return ResidualVectorQuantizer(self, name=self.name)

  def setup(self) -> None:
    embedding_init = utils.shard_initializer(
        self.config.embedding_init,
        self.config.embedding_sharding,
    )
    embedding_shape = (
        self.config.num_quantizers,
        self.config.num_embeddings,
        self.config.embedding_dim,
    )
    self.embedding = self.param(
        'embedding', embedding_init, embedding_shape, self.config.param_dtype
    )
    if self.config.num_expected_output_codes > self.config.num_quantizers:
      raise ValueError('num_expected_output_codes must be <= num_quantizers.')

    self.embeddings_to_codes_layer = sl.Lambda.Config(
        self.embeddings_to_codes,
        mask_required=False,
        sequence_input=True,
        expected_input_spec=sl.ShapeDType(
            [self.config.embedding_dim], jnp.float32
        ),
    ).make()
    if self.config.num_expected_input_codes > self.config.num_quantizers:
      raise ValueError('num_expected_input_codes must be <= num_quantizers.')

    self.codes_to_embeddings_layer = sl.Lambda.Config(
        self.codes_to_embeddings,
        mask_required=False,
        sequence_input=True,
        expected_input_spec=sl.ShapeDType(
            [self.config.num_expected_input_codes], jnp.int32
        ),
    ).make()

  @nn.nowrap
  def _compute_pairwise_distances(
      self,
      inputs: jax.Array,
      centroids: jax.Array,
      include_input_sq_norms: bool = False,
  ) -> jax.Array:
    """Computes the squared pairwise distances between inputs and centroids.

    Args:
      inputs: Inputs of shape [N, D].
      centroids: Centroids of shape [K, D].
      include_input_sq_norms: Whether to include the squared norm of the inputs
        in the calculated distances. Omitting them is more efficient when the
        results are only used to determine the nearest neighbors, where the
        input norms are constant and hence omittable. In this case, the returned
        result is d(i,j) = -2 X_i · C_j + ||C_j||^2.

    Returns:
      Pairwise distances of shape [N, K].
    """

    distances = (
        -2 * jnp.einsum('nd,kd->nk', inputs, centroids)
        + jnp.einsum('kd,kd->k', centroids, centroids)[jnp.newaxis, :]
    )
    if include_input_sq_norms:
      distances += jnp.einsum('nd,nd->n', inputs, inputs)[:, jnp.newaxis]
    return distances

  @nn.nowrap
  def _get_encodings_and_neighbors(
      self,
      inputs: jax.Array,
      centroids: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_embeddings, embedding_dim = centroids.shape
    assert num_embeddings == self.config.num_embeddings
    assert embedding_dim == inputs.shape[-1]
    if inputs.ndim > 2:
      inputs = jnp.reshape(inputs, (-1, embedding_dim))
    pairwise_distances = self._compute_pairwise_distances(inputs, centroids)
    nearest_neighbors_idx = jnp.argmin(pairwise_distances, axis=-1).astype(
        jnp.int32
    )
    encoding = jax.nn.one_hot(nearest_neighbors_idx, num_embeddings, axis=-1)
    counts = jnp.sum(encoding, axis=0)
    return encoding, counts, nearest_neighbors_idx

  def encode(
      self, inputs: jax.Array, codebook: jax.Array, training: bool = False
  ) -> tuple[jax.Array, jax.Array]:
    encoding, _, nearest_neighbors_idx = self._get_encodings_and_neighbors(
        inputs, codebook
    )
    if training:
      # counts = self._reduce_var(counts, tf.distribute.ReduceOp.SUM)
      # self._assignment_counts.update_state(counts)
      raise NotImplementedError('Training not supported yet.')

    quantized = jnp.take(codebook, nearest_neighbors_idx, axis=0)
    return encoding, quantized

  @nn.nowrap
  def _quantize(
      self, inputs: jax.Array, embedding_table: jax.Array, training: bool
  ) -> tuple[jax.Array, jax.Array]:
    embedding_dim = embedding_table.shape[-1]
    batch_shape = inputs.shape[:-1]
    assert inputs.shape[-1] == embedding_dim, inputs.shape
    inputs = jnp.reshape(inputs, (-1, embedding_dim))
    encoding, quantized = self.encode(inputs, embedding_table, training)

    def _reshape_to(tensor, last_dim):
      """Reshapes to [*dims, last_dim]."""
      return jnp.reshape(tensor, batch_shape + (last_dim,))

    # Inputs, quantized and encoding are respectively [-1, C], [-1,C] and
    # [-1, K], so we reshape it before returning for the next layer.
    inputs = _reshape_to(inputs, embedding_dim)
    quantized = _reshape_to(quantized, embedding_dim)
    encoding = _reshape_to(encoding, self.config.num_embeddings)
    if self.config.use_quantizer_stopgradient:
      quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    return quantized, encoding

  def __call__(
      self, embeddings: sl.Sequence, training: bool
  ) -> tuple[sl.Sequence, sl.Sequence]:
    if embeddings.ndim != 3:
      raise ValueError('ResidualVectorQuantizer expects 3D input.')

    embedding_table = self.embedding

    masked_quantized = 0.0
    encodings = []
    residual = embeddings.values

    # Use a scan?
    for quantizer_i in range(self.config.num_quantizers):
      quantizer_embeddings_i = embedding_table[quantizer_i]
      # Compute the unmasked quantized value because quantizer.call() adds an
      # embedding distance loss. Using the masked residual would treat all finer
      # RVQs as if their embedding space was from the first, coarser, masked
      # RVQ, whose distribution may have a different shape and scale.
      current_quantized, current_encoding = self._quantize(
          residual, quantizer_embeddings_i, training
      )
      residual -= current_quantized
      masked_quantized += current_quantized
      encodings.append(current_encoding)

    masked_quantized = sl.Sequence(
        masked_quantized, embeddings.mask
    ).mask_invalid()
    encodings = sl.Sequence(
        jnp.stack(encodings, axis=2), embeddings.mask
    ).mask_invalid()
    return masked_quantized, encodings

  def embeddings_to_codes(
      self, inputs: sl.Sequence, num_quantizers: int | None = None
  ) -> sl.Sequence:
    if num_quantizers is None:
      num_quantizers = self.config.num_expected_output_codes
    codes = []
    residual = inputs.values
    embedding_table = self.embedding
    for quantizer_i in range(num_quantizers):
      quantizer_embeddings_i = embedding_table[quantizer_i]
      current_quantized, current_encoding = self._quantize(
          residual, quantizer_embeddings_i, training=False
      )
      residual -= current_quantized
      codes.append(jnp.argmax(current_encoding, axis=-1).astype(jnp.int32))
    codes = jnp.stack(codes, axis=2)
    if self.config.use_unique_codes:
      # Broadcast-add a per-quantizer offset to the [batch, time,
      # num_quantizers] codes.
      codes += (
          np.arange(self.config.num_quantizers, dtype=np.int32)
          * self.config.num_embeddings
      )
    return sl.Sequence(codes, inputs.mask).mask_invalid()

  def codes_to_embeddings(
      self, codes: sl.Sequence, use_gather: bool = True
  ) -> sl.Sequence:
    """A [b, t, num_quantizers] sequence of codes."""

    # If unique codes are in use, divide by num_embeddings to get non-unique
    # codes per quantizer.
    # Error checking / validation of quantizer values.
    if self.config.use_unique_codes:
      codes = codes.apply_values_masked(
          lambda v: jnp.mod(v, self.config.num_embeddings)
      )

    num_input_quantizers = codes.shape[2]
    if num_input_quantizers > self.config.num_expected_input_codes:
      raise ValueError(
          f'Input to codes_to_embeddings has {num_input_quantizers=}, which is'
          ' greater than the number of expected input codebooks'
          f' {self.config.num_expected_input_codes=}.'
      )

    if codes.ndim != 3 or codes.dtype != jnp.int32:
      raise ValueError(
          'ResidualVectorQuantizer expects 3D int32 input. Got:'
          f' {codes.shape=} {codes.dtype=}'
      )

    if use_gather:
      quantized = None
      for i in range(num_input_quantizers):
        quantized_i = jnp.take(self.embedding[i], codes.values[:, :, i], axis=0)
        quantized = (
            quantized_i if quantized is None else quantized + quantized_i
        )
      if quantized is None:
        quantized = jnp.zeros(
            codes.shape[:2] + (self.config.embedding_dim,), self.embedding.dtype
        )
    else:
      indices = jax.nn.one_hot(
          codes.values, self.config.num_embeddings, dtype=jnp.float32, axis=-1
      )
      if self.config.num_quantizers != num_input_quantizers:
        pad_amount = self.config.num_quantizers - num_input_quantizers
        indices = jnp.pad(indices, [[0, 0], [0, 0], [0, pad_amount], [0, 0]])
      quantized = jnp.einsum('btqn,qnd->btd', indices, self.embedding)

    return sl.Sequence(quantized, codes.mask).mask_invalid()

  config: Config


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

    # Number of bins and channels in the input STFT.
    num_bins: int
    num_channels: int
    # Number of continuous features in the bottleneck representation between the
    # encoder and decoder.
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

    param_dtype: sl.DType = jnp.float32
    compute_dtype: sl.DType = jnp.float32

    embedding_normalizer: sl.SequenceLayerConfig | None = None
    name: str | None = None

    # For unit test.
    mock_embeddings_to_waveform_layer: sl.SequenceLayerConfig | None = None
    mock_embeddings_to_waveform_layer_ratio: int = 1

    def make(self) -> 'SpectroStream':
      return SpectroStream(self, name=self.name)

    @property
    def stft(self) -> sl.SequenceLayerConfig:
      return spectrostream_stft_config(
          self.stft_frame_length,
          self.stft_frame_step,
          fft_length=self.stft_fft_length,
          # Even when the encoder is causal, the STFT is not causal.
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
          # The inverse STFT is always causal (since it behaves like a transpose
          # convolution).
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

  config: Config

  def setup(self) -> None:
    self.stft = self.config.stft.make()
    self.inverse_stft = self.config.inverse_stft.make()
    self.encoder = self.config.encoder.make()
    self.decoder = self.config.decoder.make()

    self.waveform_to_embeddings_layer = sl.SerialModules((
        self.stft,
        self.encoder,
    ))

    if self.config.mock_embeddings_to_waveform_layer:
      self.embeddings_to_waveform_layer = (
          self.config.mock_embeddings_to_waveform_layer.make()
      )
    else:
      self.embeddings_to_waveform_layer = sl.SerialModules((
          self.decoder,
          self.inverse_stft,
      ))

    if self.config.quantizer is not None:
      self.quantizer = self.config.quantizer.make()
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
      # We don't use this in the mock test
      if not self.config.mock_embeddings_to_waveform_layer:
        assert (
            self.codes_to_waveform_layer.output_ratio
            == self.config.waveform_to_codes_ratio
        )
    else:
      self.quantizer = None

  def __call__(self, waveforms: sl.Sequence, training: bool) -> sl.Sequence:
    """A __call__ method to create all variables."""
    if training:
      raise NotImplementedError(
          'SpectroStream training support is not implemented.'
      )

    if self.quantizer is None:
      embeddings = self.waveform_to_embeddings(waveforms, training=training)
      return self.embeddings_to_waveform(embeddings, training=training)
    else:
      codes = self.waveform_to_codes(waveforms, training=training)
      codes = codes[:, :, : self.quantizer.config.num_expected_input_codes]
      return self.codes_to_waveform(codes, training=training)

  def get_waveform_to_embeddings_layer(self) -> sl.SequenceLayer:
    return self.waveform_to_embeddings_layer

  def get_waveform_to_codes_layer(self) -> sl.SequenceLayer:
    if self.quantizer is None:
      raise ValueError('Quantizer is not set.')
    return self.waveform_to_codes_layer

  def waveform_to_embeddings(
      self, waveform: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Encodes [b, t] waveforms in [-1, 1] to [b, t, e] embeddings."""
    spectrogram = self.stft.layer(waveform, training=training)
    return self.encoder.layer(spectrogram, training=training)

  def embeddings_to_waveform(
      self, embeddings: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Encodes [b, t, e] embeddings [b, t] waveforms."""
    features = self.decoder.layer(embeddings, training=training)
    return self.inverse_stft.layer(features, training=training)

  def waveform_to_codes(
      self, waveform: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Encodes [b, t] waveforms in [-1, 1] to [b, t, c] codes."""
    if self.quantizer is None:
      raise ValueError('Quantizer is not set.')
    embeddings = self.waveform_to_embeddings(waveform, training=training)
    return self.quantizer.embeddings_to_codes(embeddings)

  def get_embeddings_to_waveform_layer(self) -> sl.SequenceLayer:
    return self.embeddings_to_waveform_layer

  def get_codes_to_waveform_layer(self) -> sl.SequenceLayer:
    if self.quantizer is None:
      raise ValueError('Quantizer is not set.')
    return self.codes_to_waveform_layer

  def codes_to_features(
      self, codes: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Decodes [b, t, c] codes to [b, t, d] embeddings."""
    if self.quantizer is None:
      raise ValueError('Quantizer is not set.')
    embeddings = self.quantizer.codes_to_embeddings(codes, use_gather=False)
    return self.decoder.layer(embeddings, training=training)

  def codes_to_waveform(
      self, codes: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Decodes [b, t, c] codes to [b, t] waveforms in [-1, 1]."""
    if self.quantizer is None:
      raise ValueError('Quantizer is not set.')
    embeddings = self.codes_to_features(codes, training=training)
    return self.inverse_stft.layer(embeddings, training=training)


def stft_spectrostream_40ms_generic_48khz_stereo_config(
    rvq_truncation_level: int | None = None,
    encoded_truncation_level: int | None = None,
    param_dtype: sl.DType = jnp.float32,
    compute_dtype: sl.DType = jnp.float32,
    use_unique_codes: bool = False,
) -> SpectroStream.Config:
  """Config for a 40ms 48 kHz stereo STFT SpectroStream model.

  Args:
    rvq_truncation_level: Optional expected truncation level for SpectroStream
      decoding. Will raise ValueError if more codes that this are provided.
    encoded_truncation_level: Optional number of residual quantization stages to
      apply for SpectroStream tokenization; else all quantizers will be used.
    param_dtype: The dtype to use for model parameters.
    compute_dtype: The dtype to use for model computation.
    use_unique_codes: Whether to use unique codes.

  Returns:
    A SpectroStream.Config for the above model.
  """
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
              1,
              2,
              3,
              4,
              5,
              6,
              7,
              8,
              9,
              10,
              11,
              12,
              13,
              14,
              15,
              16,
              18,
              20,
              22,
              24,
              26,
              28,
              30,
              32,
              36,
              40,
              44,
              48,
              52,
              56,
              60,
              64,
          ),  # Low RVQ biased.
          full_quantizer_dropout_rate=0.5,
          full_quantizer_commitment=False,
      ),
  )
