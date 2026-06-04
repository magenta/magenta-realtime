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

"""Load JAX safetensors weights into the MLX MagentaRT2Sampler system."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import safetensors.flax as safetensors_flax
import flax.traverse_util as flaxtu

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export

from .spectrostream.load_weights import load_spectrostream_weights


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


def _load_attention_weights(mlx_attn, jax_attn_params, jax_output_params=None):
  """Load self-attention or cross-attention weights.

  JAX layout:
    attention/query_projection/kernel: [in, heads, uph]
    attention/key_projection/kernel: [in, heads, uph]
    attention/value_projection/kernel: [in, heads, uph]
    attention/per_dim_scale: [uph]

  MLX layout (inner DotProductSelfAttention or StreamingDotProductAttention):
    inner.q_proj: [heads*uph, in] (Dense weight is transposed)
    inner.k_proj: [heads*uph, in]
    inner.v_proj: [heads*uph, in]
    inner.per_dim_scale: [uph]
  """
  # Get the inner attention module (unwrap Deferred)
  inner = _get_inner(mlx_attn)

  # Query/Key/Value projections
  if 'query_key_value_projection' in jax_attn_params:
    # Combined QKV projection
    qkv_kernel = np.array(jax_attn_params['query_key_value_projection']['kernel'])
    # Shape: [in, 3, heads, uph]
    q_kernel = qkv_kernel[:, 0, :, :]  # [in, heads, uph]
    k_kernel = qkv_kernel[:, 1, :, :]
    v_kernel = qkv_kernel[:, 2, :, :]
  elif 'key_value_projection' in jax_attn_params:
    # Streaming attention: separate query, combined key+value
    q_kernel = np.array(jax_attn_params['query_projection']['kernel'])  # [in, heads, uph]
    kv_kernel = np.array(jax_attn_params['key_value_projection']['kernel'])  # [source, 2, heads, uph]
    k_kernel = kv_kernel[:, 0, :, :]
    v_kernel = kv_kernel[:, 1, :, :]
  elif 'shared_key_value_projection' in jax_attn_params:
    # Streaming attention: shared KV
    q_kernel = np.array(jax_attn_params['query_projection']['kernel'])
    shared_kv = np.array(jax_attn_params['shared_key_value_projection']['kernel'])
    k_kernel = shared_kv
    v_kernel = shared_kv
  else:
    # Separate Q/K/V projections
    q_kernel = np.array(jax_attn_params['query_projection']['kernel'])
    k_kernel = np.array(jax_attn_params['key_projection']['kernel'])
    v_kernel = np.array(jax_attn_params['value_projection']['kernel'])

  # Reshape kernel: [in, heads, uph] -> [in, heads*uph]
  # Reshape kernel: [in, heads, uph] -> [in, heads*uph]
  # Note: attention uses mx.matmul(x, proj) so proj should be [in, out]
  def reshape_proj(kernel):
    in_dim = kernel.shape[0]
    out_dim = int(np.prod(kernel.shape[1:]))
    return kernel.reshape(in_dim, out_dim)

  q_flat = reshape_proj(q_kernel)
  k_flat = reshape_proj(k_kernel)
  v_flat = reshape_proj(v_kernel)

  if hasattr(inner, 'qkv_proj'):
    inner.qkv_proj = _to_mx(np.concatenate([q_flat, k_flat, v_flat], axis=-1))
    if hasattr(inner, 'q_proj'): del inner.q_proj
    if hasattr(inner, 'k_proj'): del inner.k_proj
    if hasattr(inner, 'v_proj'): del inner.v_proj
    if hasattr(inner, 'kv_proj'): del inner.kv_proj
  elif hasattr(inner, 'kv_proj'):
    inner.q_proj = _to_mx(q_flat)
    inner.kv_proj = _to_mx(np.concatenate([k_flat, v_flat], axis=-1))
    if hasattr(inner, 'k_proj'): del inner.k_proj
    if hasattr(inner, 'v_proj'): del inner.v_proj
  else:
    inner.q_proj = _to_mx(q_flat)
    inner.k_proj = _to_mx(k_flat)
    inner.v_proj = _to_mx(v_flat)

  # Per-dim scale
  if 'per_dim_scale' in jax_attn_params:
    assert hasattr(inner, "_per_dim_scale")
    inner._per_dim_scale = _to_mx(jax_attn_params['per_dim_scale'])

  # Sink embeddings
  if 'sink_key_embeddings' in jax_attn_params:
    inner.sink_key_embeddings = _to_mx(jax_attn_params['sink_key_embeddings'])
  if 'sink_value_embeddings' in jax_attn_params:
    inner.sink_value_embeddings = _to_mx(jax_attn_params['sink_value_embeddings'])


def _load_rms_norm(mlx_norm, jax_params):
  """Load RMSNormalization weights."""
  scale = _to_mx(jax_params['scale'])
  # Trigger lazy initialization if needed
  if mlx_norm._rms_norm is None and not hasattr(mlx_norm, '_scale'):
    # Initialize with dummy shape
    dummy_shape = (1, 1) + scale.shape
    mlx_norm._ensure_initialized(dummy_shape)

  if mlx_norm._use_builtin and mlx_norm._rms_norm is not None:
    mlx_norm._rms_norm.weight = scale
  elif hasattr(mlx_norm, '_scale') and mlx_norm._scale is not None:
    mlx_norm._scale = scale
  else:
    mlx_norm.weight = scale


def _load_layer_norm(mlx_norm, jax_params):
  """Load LayerNormalization weights."""
  scale = _to_mx(jax_params['scale'])
  bias = _to_mx(jax_params['bias']) if 'bias' in jax_params else None

  # Trigger lazy initialization if needed
  if mlx_norm._layer_norm is None and mlx_norm._manual_scale is None:
    dummy_shape = (1, 1) + scale.shape
    mlx_norm._ensure_initialized(dummy_shape)

  if mlx_norm._use_builtin and mlx_norm._layer_norm is not None:
    mlx_norm._layer_norm.weight = scale
    if bias is not None:
      mlx_norm._layer_norm.bias = bias
  else:
    if mlx_norm._manual_scale is not None:
      mlx_norm._manual_scale = scale
    if bias is not None and mlx_norm._manual_bias is not None:
      mlx_norm._manual_bias = bias


def _load_dense(mlx_dense, jax_params):
  """Load Dense weights: kernel [in, out] -> weight [out, in]."""
  inner = mlx_dense
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner
  kernel = np.array(jax_params['kernel'])
  inner._linear.weight = _to_mx(kernel.T)
  if 'bias' in jax_params:
    inner._linear.bias = _to_mx(jax_params['bias'])


def _get_inner(layer):
  """Unwrap Deferred wrapper to get the actual layer."""
  inner = layer
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner
  return inner


def _load_attn_residual(mlx_residual, jax_params):
  """Load attention residual block.

  MLX body: [RMSNorm(pre), Attention(Deferred), EinsumDense(output), CheckpointName, RMSNorm(post), Dropout]
  JAX: attention/{q,k,v projections, per_dim_scale}, output_projection/{kernel}, pre_norm/{scale}, post_norm/{scale}
  """
  body = mlx_residual.body
  layers = body.layers if hasattr(body, 'layers') else [body]

  rms_count = 0
  for layer in layers:
    inner = _get_inner(layer)
    t = type(inner).__name__

    if 'Attention' in t:
      _load_attention_weights(layer, jax_params['attention'], None)
    elif 'EinsumDense' in t:
      inner.kernel = _to_mx(jax_params['output_projection']['kernel'])
    elif 'RMS' in t:
      norm_key = 'pre_norm' if rms_count == 0 else 'post_norm'
      _load_rms_norm(layer, jax_params[norm_key])
      rms_count += 1
    elif t in ["CheckpointName", "Dropout"]:
      pass
    else:
      raise RuntimeError(f"layer: {t}")


def _load_ffn_residual(mlx_residual, jax_params):
  """Load FFN residual block.

  MLX body: [RMSNorm(pre), Dense(ffn_layer1), Dropout, Dense(ffn_layer2), CheckpointName, RMSNorm(post), Dropout]
  JAX: ffn_layer1/{kernel,bias}, ffn_layer2/{kernel,bias}, pre_norm/{scale}, post_norm/{scale}
  """
  body = mlx_residual.body
  layers = body.layers if hasattr(body, 'layers') else [body]

  rms_count = 0
  dense_count = 0
  for layer in layers:
    inner = _get_inner(layer)
    t = type(inner).__name__

    if 'Dense' in t and 'Norm' not in t and 'Einsum' not in t:
      dense_key = 'ffn_layer1' if dense_count == 0 else 'ffn_layer2'
      if dense_key in jax_params:
        _load_dense(layer, jax_params[dense_key])
      dense_count += 1
    elif 'RMS' in t:
      norm_key = 'pre_norm' if rms_count == 0 else 'post_norm'
      _load_rms_norm(layer, jax_params[norm_key])
      rms_count += 1
    elif t == "Dropout":
      pass
    else:
      pass


def _load_transformer_block(mlx_block, jax_block_params):
  """Load a single transformer block.

  MLX: Serial(Residual(self_attn), Residual(cross_attn), Residual(ffn))
  JAX: self_attention/..., cross_attention/..., ffn/...
  """
  residuals = [l for l in mlx_block.layers if type(l).__name__ == 'Residual']

  if len(residuals) == 3:
    # self-attention, cross-attention, ffn
    _load_attn_residual(residuals[0], jax_block_params['self_attention'])
    if 'cross_attention' in jax_block_params:
      _load_attn_residual(residuals[1], jax_block_params['cross_attention'])
    _load_ffn_residual(residuals[2], jax_block_params['ffn'])
  elif len(residuals) == 2:
    # self-attention, ffn (no cross-attention)
    _load_attn_residual(residuals[0], jax_block_params['self_attention'])
    _load_ffn_residual(residuals[1], jax_block_params['ffn'])
  else:
    raise ValueError(f'unexpected number of residuals: {len(residuals)}')


def _load_transformer(mlx_xformer, jax_xformer_params):
  """Load transformer weights block by block."""
  for i, mlx_block in enumerate(mlx_xformer.layers):
    jax_key = f'x_layers_{i}'
    if jax_key in jax_xformer_params:
      _load_transformer_block(mlx_block, jax_xformer_params[jax_key])
    else:
      raise ValueError(f'{jax_key} not found in JAX params')


def load_weights(
    mrt_sampler, checkpoint_path, num_input_channels, constants=None,
):
  """Load JAX safetensors checkpoint into MLX MagentaRT2Sampler.

  Args:
    mrt_sampler: MLX MagentaRT2Sampler instance.
    checkpoint_path: Path to the safetensors checkpoint.
    num_input_channels: Total encoder input width.
    constants: Constants dict for materializing deferred layers.
  """
  # 1. Materialize deferred layers
  print('Materializing deferred layers...')
  input_width = num_input_channels
  input_spec = bt.ShapeDType((input_width,), mx.int32)
  if constants is None:
    constants = {
      'classifier_free_guidance_scale_musiccoca': mx.array([1.0]),
      'classifier_free_guidance_scale_notes': mx.array([1.0]),
      'temperature': mx.array([1.0]),
      'top_k': mx.array([40]),
    }
  export._materialize_deferred(
      mrt_sampler, batch_size=1, input_spec=input_spec, constants=constants,
  )
  print('Materialized ✓')
  params_before_depth = _collect_all_params(mrt_sampler.depthformer)

  # 2. Load JAX params
  print(f'Loading checkpoint: {checkpoint_path}')
  all_params = _load_jax_params(checkpoint_path)
  jax_params = all_params['params']

  loaded_count = 0
  _loaded_param_count[0] = 0

  # 3. Load depthformer params
  jax_df = jax_params['depthformer']
  mlx_df = mrt_sampler.depthformer

  # 3a. Encoder
  print('Loading encoder...')
  jax_enc = jax_df['encoder']['body']
  mlx_enc_body = mlx_df.encoder.body  # Serial

  # In JAX, the branched Parallel lives at 'layers_1' (because a Logging
  # layer occupies 'layers_0'). The MLX encoder omits Logging, so the
  # Parallel / MultiChannelEmbedding is layers[0].
  if 'layers_1' in jax_enc:
    # ------------------------------------------------------------------
    # Branched encoder path.
    # JAX checkpoint structure:
    #   layers_1/branched_<name>/mulan_embedder/mulan_dequantizer/embedding
    #   layers_1/branched_<name>/mulan_embedder/depth_input_adapter/kernel
    #   layers_1/branched_<name>/regular_embedder/embedding
    # MLX encoder body structure:
    #   layers[0] = Parallel
    #     layers[0] = Serial (branched_mulan_embedder)
    #       layers[0] = Lambda (crop)
    #       layers[1] = Serial (mulan_embedder)
    #         layers[0] = Lambda (offset)
    #         layers[1] = Embedding (mulan_dequantizer)
    #         layers[2] = Lambda (sum)
    #         layers[3] = Dense (depth_input_adapter)
    #     layers[1] = Serial (branched_regular_embedder)
    #       layers[0] = Lambda (crop)
    #       layers[1] = MultiChannelEmbedding (regular_embedder)
    # ------------------------------------------------------------------
    print('  Detected branched encoder embedding.')
    jax_branched = jax_enc['layers_1']
    mlx_parallel = mlx_enc_body.layers[0]  # Parallel

    # --- Branch 0: MuLan embedder ---
    jax_mulan = jax_branched['branched_mulan_embedder']['mulan_embedder']
    # mlx_parallel.layers[0] is Serial(Lambda_crop, Serial(mulan_embedder))
    mlx_mulan_serial = mlx_parallel.layers[0].layers[1]  # inner Serial

    # mulan_dequantizer Embedding
    for layer in mlx_mulan_serial.layers:
      inner = _get_inner(layer)
      if type(inner).__name__ == 'Embedding':
        inner._embedding.weight = _to_mx(
            jax_mulan['mulan_dequantizer']['embedding']
        )
        loaded_count += 1
        break

    # depth_input_adapter Dense
    for layer in mlx_mulan_serial.layers:
      inner = _get_inner(layer)
      if 'Dense' in type(inner).__name__:
        _load_dense(layer, jax_mulan['depth_input_adapter'])
        loaded_count += 1
        break

    # --- Branch 1: Regular embedder ---
    jax_regular = jax_branched['branched_regular_embedder']['regular_embedder']
    # mlx_parallel.layers[1] is Serial(Lambda_crop, MultiChannelEmbedding)
    mlx_regular_serial = mlx_parallel.layers[1]
    for layer in mlx_regular_serial.layers:
      if hasattr(layer, 'embedding'):
        layer.embedding = _to_mx(jax_regular['embedding'])
        loaded_count += 1
        break
    else:
      raise ValueError('Regular embedder Embedding not found in branch 1')

    print('  Branched encoder embeddings loaded.')
  else:
    raise ValueError(
        'Could not find encoder embedding in checkpoint. '
        f'Available keys: {list(jax_enc.keys())}'
    )

  # encoder_ln (LayerNormalization)
  for layer in mlx_enc_body.layers:
    if type(layer).__name__ == 'LayerNormalization':
      _load_layer_norm(layer, jax_enc['encoder_ln'])
      loaded_count += 1
      break
  else:
    raise ValueError("LayerNormalization not found")

  # 3b. Decoder
  print('Loading decoder...')
  jax_dec = jax_df['decoder']
  mlx_dec = mlx_df.decoder

  # Decoder embedding
  jax_emb = jax_dec['decoder_embedding']
  # The embedder is a Serial with Embedding + Scale
  embed_layer, scale_layer = mlx_dec.embedder.layers
  embed_layer._embedding.weight = _to_mx(jax_emb["embedding"]["embedding"])
  loaded_count += 1

  # Temporal body: Serial with SLTransformer
  print('Loading temporal body (transformer)...')
  jax_temporal = jax_dec['temporal_body']
  mlx_temporal_xformer = mlx_dec.temporal_body.layers[0]  # SLTransformer
  _load_transformer(mlx_temporal_xformer, jax_temporal['transformer'])
  loaded_count += 1

  # Depth body: Serial with [depth_input_adapter, transformer, final_ln, to_logits]
  print('Loading depth body...')
  jax_depth = jax_dec['depth_body']
  mlx_depth_body = mlx_dec.depth_body  # Serial

  # depth_input_adapter (Dense that projects temporal_dims -> depth_dims)
  if 'depth_input_adapter' in jax_depth:
    # Find Dense layers in order; the first one is the adapter.
    dense_layers = []
    for layer in mlx_depth_body.layers:
      inner = _get_inner(layer)
      if 'Dense' in type(inner).__name__:
        dense_layers.append(layer)
    if len(dense_layers) >= 2:
      # First Dense = depth_input_adapter, Last Dense = to_logits
      _load_dense(dense_layers[0], jax_depth['depth_input_adapter'])
      loaded_count += 1
      print('  Loaded depth_input_adapter weights.')

  # Transformer
  # Find the SLTransformer in depth_body.layers
  for layer in mlx_depth_body.layers:
    if type(layer).__name__ == 'SLTransformer':
      _load_transformer(layer, jax_depth['transformer'])
      loaded_count += 1
      break
  else:
    raise ValueError("SLTransformer not found")

  # Final LayerNorm
  for layer in mlx_depth_body.layers:
    if type(layer).__name__ == 'LayerNormalization':
      _load_layer_norm(layer, jax_depth['final_ln'])
      loaded_count += 1
      break
  else:
    raise ValueError("LayerNormalization not found")

  # to_logits Dense — find the *last* Dense in depth_body
  dense_layers = []
  for layer in mlx_depth_body.layers:
    inner = _get_inner(layer)
    if 'Dense' in type(inner).__name__:
      dense_layers.append(layer)
  if dense_layers:
    _load_dense(dense_layers[-1], jax_depth['to_logits'])
    loaded_count += 1
  else:
    raise ValueError("Dense (to_logits) not found")

  depthformer_param_count = _loaded_param_count[0]
  unupdated_depth = _check_unupdated_params(mrt_sampler.depthformer, params_before_depth)
  if unupdated_depth:
    print(f"Warning: {len(unupdated_depth)} Depthformer parameters were NOT updated during loading.")
    if len(unupdated_depth) < 10:
      for k in unupdated_depth:
        print(f"  NOT UPDATED: {'.'.join(k)}")

  # 4. Load SpectroStream params (quantizer + decoder + encoder).
  # The actual loading lives in the self-contained spectrostream loader; we
  # hand it the already-loaded 'soundstream' params so it doesn't re-read the
  # (large) checkpoint.
  spectrostream_param_count = load_spectrostream_weights(
      mrt_sampler.spectrostream,
      checkpoint_path,
      soundstream_params=jax_params['soundstream'],  # NB: do not rename to spectrostream
  )
  loaded_count += 1

  # 5. Convert depthformer params to float16 to match hardware capabilities and quantization bins.
  print('Converting depthformer params to bfloat16...')
  convert_to_bf16(mrt_sampler.depthformer)

  total_param_count = depthformer_param_count + spectrostream_param_count
  print(f'Weight loading complete ({loaded_count} groups loaded)')
  print(f'  Depthformer params:  {depthformer_param_count:>15,}')
  print(f'  SpectroStream params:  {spectrostream_param_count:>15,}')
  print(f'  Total loaded params: {total_param_count:>15,}')
  mx.eval(mrt_sampler.parameters())
  return mrt_sampler


def convert_to_bf16(module):
  """Recursively convert all float32 parameters to bfloat16.

  Uses MLX's tree_map + update API to properly traverse all parameters,
  including those in private sub-modules (e.g. _rms_norm, _linear).
  """
  import mlx.utils

  def _cast(x):
    if isinstance(x, mx.array) and x.dtype == mx.float32:
      return x.astype(mx.bfloat16)
    return x

  # Convert parameters visible to this module
  params = dict(mlx.utils.tree_map(_cast, module.parameters()))
  module.update(params)

  # Recurse into child modules (including private ones like _rms_norm)
  for name, child in module.children().items():
    if isinstance(child, nn.Module):
      convert_to_bf16(child)
    elif isinstance(child, list):
      for item in child:
        if isinstance(item, nn.Module):
          convert_to_bf16(item)
