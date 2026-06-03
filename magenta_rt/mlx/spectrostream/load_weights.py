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

"""Load JAX safetensors weights into an MLX SpectroStream model.

This module is intentionally self-contained: it depends only on the MLX
SpectroStream model (and generic sequence-layers utilities), not on the rest of
the MagentaRT system. Callers that just need a SpectroStream (e.g. the
``mrt mlx export-spectrostream`` command) can load its weights without building
a full sampler / depthformer.
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import safetensors.flax as safetensors_flax
import flax.traverse_util as flaxtu

import sequence_layers.mlx as sl
from sequence_layers.mlx import weight_converter


def _load_jax_params(path):
  """Load JAX checkpoint as a nested dict of numpy arrays."""
  flat_weights = safetensors_flax.load_file(path)
  nested_dict = {tuple(k.split('/')): v for k, v in flat_weights.items()}
  return flaxtu.unflatten_dict(nested_dict)


_loaded_param_count = [0]


def _to_mx(arr):
  """Convert numpy/jax array to mx.array."""
  result = mx.array(np.array(arr))
  _loaded_param_count[0] += result.size
  return result


def _collect_all_params(module):
  """Collect all parameters, including those in unregistered lists."""
  params = {}
  def _recurse(obj, path):
    if isinstance(obj, mx.array):
      params[path] = obj
    elif isinstance(obj, dict):
      for k, v in obj.items():
        _recurse(v, path + (k,))
    elif isinstance(obj, list):
      for i, v in enumerate(obj):
        _recurse(v, path + (str(i),))
    elif isinstance(obj, nn.Module):
      _recurse(obj.parameters(), path)
  _recurse(module, ())
  return params


def _check_unupdated_params(module, params_before):
  """Check if any parameters in the module were NOT updated (same id)."""
  params_after = _collect_all_params(module)
  unupdated = []
  for k in params_before:
    if k in params_after and id(params_before[k]) == id(params_after[k]):
      unupdated.append(k)
  return unupdated


def _get_inner(layer):
  """Unwrap Deferred wrapper to get the actual layer."""
  inner = layer
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner
  return inner


def _load_conv2d(mlx_conv, jax_params):
  """Load Conv2D: kernel [kH, kW, in, out] -> [out, kH, kW, in]."""
  inner = mlx_conv
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner
  kernel = np.array(jax_params['kernel'])
  inner.kernel = _to_mx(np.transpose(kernel, (3, 0, 1, 2)))
  if 'bias' in jax_params:
    inner.bias = _to_mx(jax_params['bias'])


def _load_conv2d_transpose(mlx_conv, jax_params):
  """Load Conv2DTranspose: kernel [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin].

  JAX's conv_transpose flips the kernel (true mathematical transposition),
  but MLX's conv_transpose2d does NOT. We must flip the kernel along both
  spatial dimensions (kH, kW) to compensate.
  """
  inner = mlx_conv
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner
  kernel = np.array(jax_params['kernel'])
  # JAX ConvTranspose kernel: [kH, kW, in_channels, out_channels]
  # Flip along kH and kW to match JAX's implicit kernel flip.
  kernel = kernel[::-1, ::-1, :, :]
  # Reorder: [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin]
  inner.kernel = _to_mx(np.transpose(kernel, (3, 0, 1, 2)))
  if 'bias' in jax_params:
    inner.bias = _to_mx(jax_params['bias'])


def _load_spectrostream_decoder(mlx_decoder, jax_params):
  """Load SpectroStream decoder weights from JAX params into MLX Serial.

  The MLX decoder is a Serial with structure:
    [0] ExpandDims
    [1] Residual — input_layer
        body: Serial([Serial([Conv2D])])                        # conv1x1_first
        shortcut: Serial([Serial([Conv2D]), Elu, Serial([Conv2D])])  # conv1x1_b1, conv1x1_b2
    [2] Reshape
    [3] Residual — input_layers_residual_unit
        body: Serial([Serial([Elu, Conv2D]), Serial([Elu, Conv2D])])  # conv2d_3x3_a, conv2d_3x3
        shortcut: Identity
    [4] Residual — decoder_0
        body: Serial([Serial([Elu, Conv2DTranspose]), Serial([Elu, Conv2D])])
        shortcut: Serial([Serial([Conv2D]), Upsample2D])
    [5] ParallelChannels — decoder_1 through decoder_6 + output_layer
        child: Serial([
          [0] Residual — decoder_1
          [1] Residual — decoder_2
          [2] Residual — decoder_3
          [3] Residual — decoder_4
          [4] Residual — decoder_5
          [5] Residual — decoder_6
          [6] Serial([Elu, Serial([Conv2D])]) — output_layer
        ])
    [6] Lookahead
  """
  def _get_jax_conv(jax_block, conv_name):
    """Get JAX conv params, handling the extra 'conv/' nesting."""
    p = jax_block[conv_name]
    if 'conv' in p:
      p = p['conv']
    return p

  def _load_decoder_block(mlx_residual, jax_block):
    """Load a standard decoder block (ConvTranspose + Conv2D + optional shortcut).

    body: Serial([Serial([Elu, Conv2DTranspose]), Serial([Elu, Conv2D])])
    """
    body = mlx_residual.body

    # Find conv transpose key (conv2dtranspose_*)
    transpose_key = None
    conv_key = None
    shortcut_key = None
    for k in jax_block:
      if 'conv2dtranspose' in k:
        transpose_key = k
      elif 'conv2d' in k:
        conv_key = k
      elif 'shortcut' in k:
        shortcut_key = k

    assert transpose_key is not None, f'No transpose conv in block: {list(jax_block.keys())}'
    assert conv_key is not None, f'No conv2d in block: {list(jax_block.keys())}'

    # body.layers[0] = Serial([Elu, Conv2DTranspose]) or Serial([Elu, Conv2D, ...])
    conv_t = body.layers[0].layers[-1]  # Last layer in first sub-serial
    _load_conv2d_transpose(conv_t, _get_jax_conv(jax_block, transpose_key))

    # body.layers[1] = Serial([Elu, Conv2D])
    conv = body.layers[1].layers[-1]
    _load_conv2d(conv, _get_jax_conv(jax_block, conv_key))

    # Shortcut conv1x1 (if present)
    if shortcut_key is not None:
      sc = mlx_residual.shortcut
      jax_sc = jax_block[shortcut_key]
      # Find the conv1x1 key inside shortcut
      sc_conv_key = None
      for k in jax_sc:
        if 'conv1x1' in k:
          sc_conv_key = k
          break
      assert sc_conv_key is not None, f'No conv1x1 in shortcut: {list(jax_sc.keys())}'
      # shortcut is typically Serial([Serial([Conv2D]), Upsample2D])
      sc_conv = sc.layers[0].layers[0]
      _load_conv2d(sc_conv, _get_jax_conv(jax_sc, sc_conv_key))

  # === Load input_layer ===
  if 'input_layer' in jax_params:
    jax_input = jax_params['input_layer']
    res_input = mlx_decoder.layers[1]  # Residual

    # body: Serial([Serial([Conv2D])])  — conv1x1_first
    body_conv = res_input.body.layers[0].layers[0]
    _load_conv2d(body_conv, _get_jax_conv(jax_input, 'conv1x1_first'))

    # shortcut: Serial([Serial([Conv2D]), Elu, Serial([Conv2D])])
    sc = res_input.shortcut
    # First conv: conv1x1_b1
    sc_conv1 = sc.layers[0].layers[0]
    _load_conv2d(sc_conv1, _get_jax_conv(jax_input['shortcut_layer'], 'conv1x1_b1'))
    # Second conv: conv1x1_b2 (after Elu)
    sc_conv2 = sc.layers[2].layers[0]
    _load_conv2d(sc_conv2, _get_jax_conv(jax_input['shortcut_layer'], 'conv1x1_b2'))

  # === Load input_layers_residual_unit ===
  if 'input_layers_residual_unit' in jax_params:
    jax_ru = jax_params['input_layers_residual_unit']
    res_ru = mlx_decoder.layers[3]  # Residual

    # body: Serial([Serial([Elu, Conv2D]), Serial([Elu, Conv2D])])
    # First conv: conv2d_3x3_a
    conv_a = res_ru.body.layers[0].layers[-1]
    _load_conv2d(conv_a, _get_jax_conv(jax_ru, 'conv2d_3x3_a'))
    # Second conv: conv2d_3x3
    conv_b = res_ru.body.layers[1].layers[-1]
    _load_conv2d(conv_b, _get_jax_conv(jax_ru, 'conv2d_3x3'))

  # === Load decoder_0 ===
  if 'decoder_0' in jax_params:
    _load_decoder_block(mlx_decoder.layers[4], jax_params['decoder_0'])

  # === Load decoder_1 through decoder_6 + output_layer ===
  # These are inside ParallelChannels.child (a Serial with 7 layers)
  pc = mlx_decoder.layers[5]  # ParallelChannels
  pc_child = pc.child  # Serial with child decoder blocks

  for i in range(6):
    jax_key = f'decoder_{i + 1}'
    if jax_key in jax_params:
      _load_decoder_block(pc_child.layers[i], jax_params[jax_key])

  # output_layer: Serial([Elu, Serial([Conv2D])])
  if 'output_layer' in jax_params:
    jax_out = jax_params['output_layer']
    out_serial = pc_child.layers[6]
    # out_serial.layers = [Elu, Serial([Conv2D])]
    out_conv = out_serial.layers[1].layers[0]
    _load_conv2d(out_conv, _get_jax_conv(jax_out, 'base_conv_last'))

  print('    SpectroStream decoder weights loaded.')


def _load_spectrostream_encoder(mlx_encoder, jax_params):
  """Load SpectroStream encoder weights from JAX params into MLX Serial."""
  def _get_jax_conv(jax_block, conv_name):
    """Get JAX conv params, handling the extra 'conv/' nesting."""
    p = jax_block[conv_name]
    if 'conv' in p:
      p = p['conv']
    return p

  def _load_encoder_block(mlx_residual, jax_block, name):
    """Load a standard encoder block (Conv2D + Conv2D + optional shortcut)."""
    body = mlx_residual.body

    conv_3x3_key = None
    conv_a_key = None
    shortcut_key = None
    for k in jax_block:
      if '_a' in k:
        conv_a_key = k
      elif 'conv2d_3x3' in k:
        conv_3x3_key = k
      elif 'shortcut' in k:
        shortcut_key = k

    assert conv_3x3_key is not None, f'No conv2d_3x3 in block: {list(jax_block.keys())}'
    assert conv_a_key is not None, f'No conv_a in block: {list(jax_block.keys())}'

    conv_3x3 = body.layers[0].layers[-1]
    _load_conv2d(conv_3x3, _get_jax_conv(jax_block, conv_3x3_key))

    conv_a = body.layers[1].layers[-1]
    _load_conv2d(conv_a, _get_jax_conv(jax_block, conv_a_key))

    if shortcut_key is not None:
      sc = mlx_residual.shortcut
      jax_sc = jax_block[shortcut_key]
      sc_conv_key = None
      for k in jax_sc:
        if 'conv1x1' in k:
          sc_conv_key = k
          break
      assert sc_conv_key is not None, f'No conv1x1 in shortcut: {list(jax_sc.keys())}'

      def find_conv(module):
        inner = _get_inner(module)
        if type(inner).__name__ == 'Conv2D':
          return inner
        if hasattr(module, 'layers'):
          for l in module.layers:
            c = find_conv(l)
            if c: return c
        return None

      sc_conv = find_conv(sc)
      assert sc_conv is not None, f"Shortcut Conv2D not found in {name}"
      _load_conv2d(sc_conv, _get_jax_conv(jax_sc, sc_conv_key))

  def find_modules(module, type_name):
    found = []
    if type(module).__name__ == type_name:
      found.append(module)
    elif type_name == 'Conv2D' and type(module).__name__ == 'DeferredConv2D':
      found.append(module)

    if hasattr(module, 'layers'):
      for l in module.layers:
        found.extend(find_modules(l, type_name))
    elif hasattr(module, 'child'):
      found.extend(find_modules(module.child, type_name))

    return found

  residuals = find_modules(mlx_encoder, 'Residual')

  for i in range(7):
    jax_key = f'encoder_{i}'
    if jax_key in jax_params:
      if len(residuals) > i:
        _load_encoder_block(residuals[i], jax_params[jax_key], jax_key)

  if 'bottleneck' in jax_params:
    if len(residuals) >= 8:
      _load_encoder_block(residuals[7], jax_params['bottleneck'], 'bottleneck')

  if 'output_convs' in jax_params:
    jax_out = jax_params['output_convs']
    if len(residuals) >= 9:
      out_res = residuals[8]
      convs = find_modules(out_res.body, 'Conv2D')
      if convs:
        _load_conv2d(convs[0], _get_jax_conv(jax_out, 'conv1x1_last'))

      sc_convs = find_modules(out_res.shortcut, 'Conv2D')
      if len(sc_convs) >= 2:
        _load_conv2d(sc_convs[0], _get_jax_conv(jax_out['shortcut_layer'], 'conv1x1_b1'))
        _load_conv2d(sc_convs[1], _get_jax_conv(jax_out['shortcut_layer'], 'conv1x1_b2'))

  all_convs = find_modules(mlx_encoder, 'Conv2D')
  if all_convs:
    _load_conv2d(all_convs[0], _get_jax_conv(jax_params, 'base_conv_first'))

  print('    SpectroStream encoder weights loaded.')


def load_spectrostream_weights(
    spectrostream, checkpoint_path, *, soundstream_params=None,
):
  """Load SpectroStream weights (quantizer + decoder + encoder) into an MLX model.

  Args:
    spectrostream: An MLX ``SpectroStream`` instance.
    checkpoint_path: Path to the JAX safetensors checkpoint. The quantizer and
      decoder weights are read from its top-level ``soundstream`` params; the
      encoder weights are read from a sibling ``encoder.safetensors`` file, if
      present.
    soundstream_params: Optional pre-loaded ``soundstream`` params (the value of
      ``_load_jax_params(checkpoint_path)['params']['soundstream']``). Pass this
      to avoid re-reading a large checkpoint when the caller has already loaded
      it.

  Returns:
    The number of SpectroStream parameters loaded.
  """
  start_count = _loaded_param_count[0]

  print('Loading spectrostream...')
  if soundstream_params is None:
    all_params = _load_jax_params(checkpoint_path)
    # NB: do not rename to spectrostream
    soundstream_params = all_params['params']['soundstream']
  jax_ss = soundstream_params

  # Quantizer embedding
  print('  Loading quantizer embeddings...')
  jax_q = jax_ss['quantizer']
  spectrostream.quantizer.embedding = _to_mx(jax_q['embedding'])

  # SpectroStream decoder (conv layers)
  print('  Loading SpectroStream decoder...')
  jax_ss_dec = weight_converter._unbox_params(jax_ss['decoder'])

  # Materialize all deferred conv layers by running a dummy step.
  mlx_ss_dec = spectrostream.embeddings_to_waveform_layer.layers[0]
  _dummy_x = sl.Sequence(
      mx.zeros((1, 1, 256), dtype=mx.float32),
      mx.ones((1, 1), dtype=mx.bool_),
  )
  _dummy_state = mlx_ss_dec.get_initial_state(
      1, sl.ChannelSpec(shape=[256], dtype=mx.float32)
  )
  mlx_ss_dec.step(_dummy_x, _dummy_state)
  params_before_dec = _collect_all_params(mlx_ss_dec)

  _load_spectrostream_decoder(mlx_ss_dec, jax_ss_dec)
  unupdated_dec = _check_unupdated_params(mlx_ss_dec, params_before_dec)
  if unupdated_dec:
    print(f"Warning: {len(unupdated_dec)} SpectroStream decoder parameters were NOT updated during loading.")
    if len(unupdated_dec) < 10:
      for k in unupdated_dec:
        print(f"  NOT UPDATED: {'.'.join(k)}")

  # SpectroStream encoder (conv layers)
  encoder_path = os.path.join(os.path.dirname(checkpoint_path), 'encoder.safetensors')
  if os.path.exists(encoder_path):
    print('  Loading SpectroStream encoder...')
    # Materialize all deferred conv layers by running a dummy step for encoder.
    num_bins = spectrostream.config.num_bins
    num_channels = spectrostream.config.num_channels
    _dummy_enc_x = sl.Sequence(
        mx.zeros((1, 2, num_bins, num_channels), dtype=mx.float32),
        mx.ones((1, 2), dtype=mx.bool_),
    )
    spectrostream.encoder.layer(_dummy_enc_x)
    params_before_enc = _collect_all_params(spectrostream.encoder)

    enc_params = _load_jax_params(encoder_path)
    jax_enc_params = enc_params['params']['encoder']
    _load_spectrostream_encoder(spectrostream.encoder, jax_enc_params)
    unupdated_enc = _check_unupdated_params(spectrostream.encoder, params_before_enc)
    if unupdated_enc:
      print(f"Warning: {len(unupdated_enc)} SpectroStream encoder parameters were NOT updated during loading.")
      if len(unupdated_enc) < 10:
        for k in unupdated_enc:
          print(f"  NOT UPDATED: {'.'.join(k)}")
  else:
    print(f'  Encoder weights not found at {encoder_path}, skipping.')

  return _loaded_param_count[0] - start_count
