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

"""GPTQ calibration-based int4 weight quantization for MagentaRT.

Applies the GPTQ algorithm (Frantar et al., 2022) to the non-QAT baseline
checkpoint. Uses activation statistics from a short calibration run to find
smarter int4 rounding decisions that minimize output-space error.

Key advantage: **no grid mismatch by construction** — GPTQ adjusts float
weights so that standard nn.quantize() nearest-rounding produces near-optimal
int4 values. The model structure and file format are identical to naive int4.

Algorithm per layer:
  1. Collect input activations X during calibration steps
  2. Compute Hessian: H = X^T @ X / n  (+ damping)
  3. Invert H for error sensitivity
  4. Quantize columns in blocks, compensating remaining columns for each
     rounding error using the Hessian

Usage:
    from magenta_rt.mlx.gptq import gptq_calibrate_and_quantize
    gptq_calibrate_and_quantize(mrt_sampler, calibrate_fn, bits=4, group_size=32)
"""

from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

import sequence_layers.mlx as sl


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

# Import concrete layer types for isinstance checks.
from sequence_layers.mlx.dense import Dense, DenseDeferred, EinsumDense
from sequence_layers.mlx.attention import (
    DotProductSelfAttention,
    DeferredDotProductSelfAttention,
    DotProductAttention,
    DeferredDotProductAttention,
)


def _find_quantizable_layers(module):
  """Walk the model tree and find all layers with quantizable weights.

  Uses MLX's built-in named_modules() to traverse the module tree correctly.
  Returns a list of (dotted_name, module, layer_type, shape_info) tuples.

  Handles three layer types:
    - Dense / DenseDeferred (via inner._linear.weight): shape [out, in]
    - EinsumDense (kernel): shape [d, n, h] for '...nh,dnh->...d'
    - DotProductSelfAttention / DotProductAttention (q_proj, kv_proj)
  """
  results = []
  seen = set()  # avoid duplicates (Deferred and its .inner)

  for name, mod in module.named_modules():
    mod_id = id(mod)
    if mod_id in seen:
      continue

    # DenseDeferred -> check .inner
    if isinstance(mod, DenseDeferred):
      inner = getattr(mod, 'inner', None)
      if inner is not None:
        linear = getattr(inner, '_linear', None)
        if linear is not None and hasattr(linear, 'weight') and linear.weight is not None:
          results.append((name, mod, 'dense', linear.weight.shape))
          seen.add(mod_id)
          seen.add(id(inner))
          continue

    # Dense (direct, not via Deferred)
    elif isinstance(mod, Dense):
      linear = getattr(mod, '_linear', None)
      if linear is not None and hasattr(linear, 'weight') and linear.weight is not None:
        results.append((name, mod, 'dense', linear.weight.shape))
        seen.add(mod_id)
        continue

    # EinsumDense
    elif isinstance(mod, EinsumDense):
      if mod.kernel is not None and hasattr(mod, '_equation'):
        results.append((name, mod, 'einsum', mod.kernel.shape))
        seen.add(mod_id)
        continue

    # DeferredDotProductSelfAttention -> check .inner
    elif isinstance(mod, (DeferredDotProductSelfAttention, DeferredDotProductAttention)):
      inner = getattr(mod, 'inner', None)
      if inner is not None and hasattr(inner, 'q_proj') and inner.q_proj is not None:
        results.append((name, mod, 'attention', {
            'q_proj': inner.q_proj.shape,
            'kv_proj': inner.kv_proj.shape,
        }))
        seen.add(mod_id)
        seen.add(id(inner))
        continue

    # DotProductSelfAttention / DotProductAttention (direct)
    elif isinstance(mod, (DotProductSelfAttention, DotProductAttention)):
      if hasattr(mod, 'q_proj') and mod.q_proj is not None:
        results.append((name, mod, 'attention', {
            'q_proj': mod.q_proj.shape,
            'kv_proj': mod.kv_proj.shape,
        }))
        seen.add(mod_id)
        continue

  return results


# ---------------------------------------------------------------------------
# Activation capture via monkey-patching
# ---------------------------------------------------------------------------

class ActivationCapture:
  """Context manager that instruments Dense/EinsumDense/Attention layers
  to capture input activations during calibration.

  Since sequence_layers modules don't support register_forward_hook(),
  we monkey-patch the layer/step methods to intercept activations.
  """

  def __init__(self, model, max_samples=2048):
    self.model = model
    self.max_samples = max_samples
    self.captured = defaultdict(list)
    self._originals = {}  # (name, method_name) -> original_method
    self._layers = _find_quantizable_layers(model)

  def __enter__(self):
    for name, module, layer_type, _ in self._layers:
      self._instrument_layer(name, module, layer_type)
    return self

  def __exit__(self, *args):
    self._restore_all()

  def _instrument_layer(self, name, module, layer_type):
    """Wrap the layer's forward to capture input activations."""
    inner = getattr(module, 'inner', module)

    if layer_type == 'dense':
      # Wrap Dense.layer() — wrapping nn.Linear.__call__ doesn't work
      # because MLX dispatches via type(obj).__call__, not instance.
      target = inner  # Dense or DenseDeferred.inner (which is Dense)
      self._wrap_layer_method(name, target, self._dense_capture)
    elif layer_type == 'einsum':
      self._wrap_layer_method(name, inner, self._einsum_capture)
    elif layer_type == 'attention':
      attn = getattr(module, 'inner', module)
      self._wrap_forward(name, attn, '_project_qkv', self._attention_capture)

  def _wrap_forward(self, name, target, method_name, capture_fn):
    """Wrap a method to capture inputs before calling the original."""
    original = getattr(target, method_name)
    self._originals[(name, id(target), method_name)] = (target, original)
    captured = self.captured
    max_samples = self.max_samples

    def wrapped(*args, **kwargs):
      capture_fn(name, captured, max_samples, args, kwargs)
      return original(*args, **kwargs)

    setattr(target, method_name, wrapped)

  def _wrap_layer_method(self, name, target, capture_fn):
    """Wrap a SequenceLayer's layer() method."""
    original_layer = target.layer
    self._originals[(name, id(target), 'layer')] = (target, original_layer)
    captured = self.captured
    max_samples = self.max_samples

    def wrapped_layer(x, *, constants=None, **kwargs):
      capture_fn(name, captured, max_samples, x)
      return original_layer(x, constants=constants, **kwargs)

    target.layer = wrapped_layer

  @staticmethod
  def _dense_capture(name, captured, max_samples, x):
    """Capture Dense input from Sequence object."""
    if len(captured[name]) * 128 >= max_samples:
      return
    if isinstance(x, sl.Sequence):
      v = x.values
    elif isinstance(x, mx.array):
      v = x
    else:
      return
    v = v.reshape(-1, v.shape[-1])
    captured[name].append(v)

  @staticmethod
  def _einsum_capture(name, captured, max_samples, x):
    """Capture EinsumDense input from Sequence object."""
    if len(captured[name]) * 128 >= max_samples:
      return
    if isinstance(x, sl.Sequence):
      v = x.values
    elif isinstance(x, mx.array):
      v = x
    else:
      return
    # For '...nh,dnh->...d', v has shape [..., n, h] -> flatten to [..., n*h]
    if v.ndim > 2:
      last_two = v.shape[-2] * v.shape[-1]
      v = v.reshape(-1, last_two)
    else:
      v = v.reshape(-1, v.shape[-1])
    captured[name].append(v)

  @staticmethod
  def _attention_capture(name, captured, max_samples, args, kwargs):
    """Capture attention input: first arg (self) then x (Sequence)."""
    if len(captured[name]) * 128 >= max_samples:
      return
    x = args[1] if len(args) > 1 else args[0]
    if isinstance(x, sl.Sequence):
      v = x.values.reshape(-1, x.values.shape[-1])
    elif isinstance(x, mx.array):
      v = x.reshape(-1, x.shape[-1])
    else:
      return
    captured[name].append(v)

  def _restore_all(self):
    for (name, obj_id, method_name), (target, original) in self._originals.items():
      setattr(target, method_name, original)
    self._originals.clear()

  def get_hessians(self, debug_identity_hessian=False):
    """Compute Hessian H = X^T @ X / n for each captured layer.

    If debug_identity_hessian=True, returns identity matrices instead.
    Calibration still runs so activation stats are visible.
    """
    hessians = {}
    for name, chunks in self.captured.items():
      if not chunks:
        continue
      X = mx.concatenate(chunks, axis=0).astype(mx.float32)
      n_samples = X.shape[0]
      dim = X.shape[1]

      if debug_identity_hessian:
        H = mx.eye(dim)
        mx.eval(H)
        hessians[name] = H
        print(f"  [DEBUG] Identity Hessian for {name}: dim={dim}, "
              f"n_samples={n_samples}")
      else:
        H = (X.T @ X) / n_samples
        diag_mean = mx.mean(mx.diag(H))
        damp = 0.01 * diag_mean
        H = H + damp * mx.eye(dim)
        mx.eval(H)
        hessians[name] = H
        print(f"  Hessian for {name}: shape={H.shape}, "
              f"n_samples={n_samples}, damp={damp.item():.4e}")
    return hessians


# ---------------------------------------------------------------------------
# Core GPTQ algorithm
# ---------------------------------------------------------------------------

def _pack_int4(Q, bits=4):
  """Pack int4 values into uint32 (8 nibbles per uint32, LSB first).

  Matches MLX's mx.quantize() packing convention.
  Q: [rows, cols] int values in [0, 2^bits - 1]
  Returns: [rows, cols // 8] uint32 array
  """
  elems_per_int = 32 // bits
  Q = Q.astype(mx.uint32)
  rows, cols = Q.shape
  packed = mx.zeros((rows, cols // elems_per_int), dtype=mx.uint32)
  for k in range(elems_per_int):
    packed = packed | (Q[:, k::elems_per_int] << (k * bits))
  return packed


def gptq_quantize_weight(W, H, bits=4, group_size=32, block_size=128):
  """GPTQ error compensation for a single weight matrix.

  Uses mx.quantize()'s own scale/bias convention to ensure alignment.
  Returns (packed, scales, biases) in the same format as mx.quantize(),
  ready to set directly on QuantizedLinear/EinsumDense/Attention layers.

  With H=I (identity), produces bit-identical output to mx.quantize(W).
  """
  orig_dtype = W.dtype
  W = W.astype(mx.float32)
  rows, cols = W.shape
  assert cols % group_size == 0

  # mx.linalg.inv is CPU-only in current MLX.
  H = H.astype(mx.float32)
  cpu = mx.cpu
  try:
    H_inv = mx.linalg.inv(H, stream=cpu)
    mx.eval(H_inv)
  except Exception:
    diag_mean = mx.mean(mx.diag(H))
    H = H + 0.1 * diag_mean * mx.eye(H.shape[0])
    H_inv = mx.linalg.inv(H, stream=cpu)
    mx.eval(H_inv)

  W_work = mx.array(W)  # working copy (modified by error compensation)
  max_val = 2**bits - 1
  num_groups = cols // group_size
  elems_per_int = 32 // bits

  # Store int4 values for direct packing
  Q_all = mx.zeros((rows, cols), dtype=mx.int32)

  # Per-group scale/bias — from mx.quantize()
  out_scales = mx.zeros((rows, num_groups), dtype=orig_dtype)
  out_biases = mx.zeros((rows, num_groups), dtype=orig_dtype)

  for block_start in range(0, cols, block_size):
    block_end = min(block_start + block_size, cols)
    H_inv_block = H_inv[block_start:block_end, block_start:block_end]
    err_block = mx.zeros((rows, block_end - block_start), dtype=mx.float32)

    cur_scale_f32 = None
    cur_bias_f32 = None
    group_q = None  # [rows, group_size] int4 values from mx.quantize()
    group_modified = None  # tracks which columns were modified by compensation

    for j_local in range(block_end - block_start):
      j = block_start + j_local
      g = j // group_size
      j_in_group = j % group_size

      # At group entry: call mx.quantize() on the current group values.
      # This gives us the definitive scale/bias AND correct int4 values
      # for all unmodified columns in this group.
      if j_in_group == 0:
        g_start = g * group_size
        group_cols = W_work[:, g_start:g_start + group_size]
        g_packed, g_scales, g_biases = mx.quantize(
            group_cols.astype(orig_dtype), group_size=group_size, bits=bits)
        mx.eval(g_packed, g_scales, g_biases)
        out_scales[:, g] = g_scales.squeeze(1)
        out_biases[:, g] = g_biases.squeeze(1)
        cur_scale_f32 = g_scales.astype(mx.float32)
        cur_bias_f32 = g_biases.astype(mx.float32)

        # Unpack int4 values: reverse the packing.
        # Packing: for nibble position k in uint32, the original column is k::8
        group_q = mx.zeros((rows, group_size), dtype=mx.int32)
        for k in range(elems_per_int):
          nibbles = ((g_packed >> (k * bits)) & max_val).astype(mx.int32)
          # nibbles has shape [rows, packed_cols_per_group]
          # These correspond to columns k, k+8, k+16, k+24 in the group
          for p in range(nibbles.shape[1]):
            col_in_group = k + p * elems_per_int
            if col_in_group < group_size:
              group_q[:, col_in_group] = nibbles[:, p]
        mx.eval(group_q)
        group_modified = [False] * group_size

      # Use mx.quantize's int4 value for unmodified columns.
      # For modified columns (compensated), re-quantize with stored scale/bias.
      if group_modified[j_in_group]:
        # Re-quantize this compensated column using GPTQ's scale/bias.
        w_col_bf16 = W_work[:, j].astype(orig_dtype)[:, None]
        q_val = mx.clip(mx.floor(
            (w_col_bf16.astype(mx.float32) - cur_bias_f32) / cur_scale_f32 + 0.5),
            0, max_val).squeeze(1).astype(mx.int32)
      else:
        q_val = group_q[:, j_in_group]

      w_col = W_work[:, j]
      d = H_inv_block[j_local, j_local]
      if d < 1e-10:
        d = mx.array(1e-10)

      q_col = q_val[:, None].astype(mx.float32)
      w_hat_col = (cur_scale_f32 * q_col + cur_bias_f32).squeeze(1)
      Q_all[:, j] = q_val

      # Error compensation for remaining columns
      err = (w_col - w_hat_col) / d
      err_block[:, j_local] = err

      if j_local < block_end - block_start - 1:
        h_compensation = H_inv_block[j_local, j_local+1:block_end-block_start]
        W_work[:, j+1:block_end] -= (
            err[:, None] * h_compensation[None, :]
        )
        # Mark columns as modified only if compensation is actually non-zero.
        # With H=I, off-diagonals are zero → no columns marked → bit-identical.
        if mx.any(h_compensation != 0).item():
          for jj in range(j+1, min(block_end, (g+1) * group_size)):
            group_modified[jj % group_size] = True

    if block_end < cols:
      W_work[:, block_end:] -= (
          err_block @ H_inv[block_start:block_end, block_end:]
      )
    mx.eval(W_work, err_block, Q_all)

  # Pack int4 values into uint32
  Q_all = mx.clip(Q_all, 0, max_val)
  packed = _pack_int4(Q_all, bits)
  mx.eval(packed, out_scales, out_biases)
  return packed, out_scales, out_biases


# ---------------------------------------------------------------------------
# Model-level GPTQ weight adjustment
# ---------------------------------------------------------------------------

def gptq_adjust_weights(model, hessians, bits=4, group_size=32, block_size=128):
  """Apply GPTQ and directly set packed int4 weights on model layers.

  Calls nn.quantize() first to create the quantized layer structure
  (QuantizedLinear, EinsumDense.q_weight, attention.qkv_proj_qw, etc.),
  then overrides packed weights/scales/biases with GPTQ's optimized values.
  """
  # Discover layers and snapshot float weights BEFORE nn.quantize().
  layers = _find_quantizable_layers(model)
  layer_info = []  # (name, module, inner, layer_type, float_weight)

  for name, module, layer_type, shape_info in layers:
    if name not in hessians:
      continue
    if layer_type == 'dense':
      inner = getattr(module, 'inner', module)
      linear = getattr(inner, '_linear', None)
      if linear is not None and hasattr(linear, 'weight') and linear.weight is not None:
        layer_info.append((name, module, inner, layer_type,
                           mx.array(linear.weight)))
    elif layer_type == 'einsum':
      inner = getattr(module, 'inner', module)
      if inner.kernel is not None:
        layer_info.append((name, module, inner, layer_type,
                           mx.array(inner.kernel)))
    elif layer_type == 'attention':
      attn = getattr(module, 'inner', module)
      if attn.q_proj is not None:
        layer_info.append((name, module, attn, layer_type,
                           (mx.array(attn.q_proj), mx.array(attn.kv_proj))))

  # Let nn.quantize() create the quantized structure.
  print(f"\nSetting up quantized layers (bits={bits}, group_size={group_size})...")
  nn.quantize(model, group_size=group_size, bits=bits)

  # Now override packed weights with GPTQ-optimized values.
  for name, module, inner_ref, layer_type, float_w in layer_info:
    H = hessians[name]

    if layer_type == 'dense':
      # inner_ref._linear is now QuantizedLinear (same object, transformed).
      linear = getattr(inner_ref, '_linear', None)
      if linear is None:
        continue
      W = float_w
      if W.shape[1] % group_size != 0:
        print(f"  [skip] {name}: not divisible by group_size")
        continue
      print(f"  [GPTQ] {name} (dense): {W.shape}", end=" ")
      packed, scales, biases = gptq_quantize_weight(
          W, H, bits, group_size, block_size)
      linear.weight = packed
      linear.scales = scales
      linear.biases = biases
      mx.eval(linear.weight, linear.scales, linear.biases)
      print("-> adjusted")

    elif layer_type == 'einsum':
      # inner_ref is the EinsumDense; now has q_weight/q_scales/q_biases.
      kernel = float_w
      d, n, h = kernel.shape
      if (n * h) % group_size != 0:
        print(f"  [skip] {name}: not divisible by group_size")
        continue
      print(f"  [GPTQ] {name} (einsum): {kernel.shape} -> [{d}, {n*h}]", end=" ")
      packed, scales, biases = gptq_quantize_weight(
          kernel.reshape(d, n * h), H, bits, group_size, block_size)
      inner_ref.q_weight = packed
      inner_ref.q_scales = scales
      inner_ref.q_biases = biases
      mx.eval(inner_ref.q_weight, inner_ref.q_scales, inner_ref.q_biases)
      print("-> adjusted")

    elif layer_type == 'attention':
      # inner_ref is the attention module; now has qkv_proj_qw/qs/qb.
      q_proj, kv_proj = float_w
      # to_quantized() does: q_proj.T and kv_proj.T, then concat along axis=0
      W_comb = mx.concatenate([q_proj.T, kv_proj.T], axis=0)
      if W_comb.shape[1] % group_size != 0:
        print(f"  [skip] {name} (attn): not divisible by group_size")
        continue
      print(f"  [GPTQ] {name} (attn): q={q_proj.shape} kv={kv_proj.shape}", end=" ")
      packed, scales, biases = gptq_quantize_weight(
          W_comb, H, bits, group_size, block_size)
      inner_ref.qkv_proj_qw = packed
      inner_ref.qkv_proj_qs = scales
      inner_ref.qkv_proj_qb = biases
      mx.eval(inner_ref.qkv_proj_qw, inner_ref.qkv_proj_qs, inner_ref.qkv_proj_qb)
      print("-> adjusted")


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def capture_activations(model, calibrate_fn, max_samples=2048,
                        debug_identity_hessian=False):
  """Run calibration and return per-layer Hessian matrices."""
  print("Capturing activations for GPTQ calibration...")
  with ActivationCapture(model, max_samples=max_samples) as capture:
    calibrate_fn(model)
  if debug_identity_hessian:
    print("[DEBUG] Computing identity Hessians (no error compensation).")
  else:
    print("Computing Hessians...")
  hessians = capture.get_hessians(debug_identity_hessian=debug_identity_hessian)
  print(f"Computed Hessians for {len(hessians)} layers.")
  return hessians


def gptq_calibrate_and_quantize(
    model, calibrate_fn,
    bits=4, group_size=32, block_size=128, max_samples=2048,
    debug_identity_hessian=False,
    hessian_save_path=None,
):
  """End-to-end GPTQ: calibrate -> GPTQ quantize -> pack int4.

  Drop-in replacement for nn.quantize(). Produces identical model structure
  and file size, but with better int4 rounding from GPTQ error compensation.

  GPTQ's packed int4 values are set directly on QuantizedLinear/EinsumDense/
  Attention layers, bypassing nn.quantize()'s rounding. This ensures GPTQ's
  error-compensated rounding decisions are preserved exactly.

  Set debug_identity_hessian=True to force H=I for all layers. This disables
  error compensation, so output should be bit-identical to nn.quantize().
  """
  hessians = capture_activations(
      model, calibrate_fn, max_samples, debug_identity_hessian)

  if hessian_save_path is not None:
    mx.save_safetensors(str(hessian_save_path), hessians)
    print(f"Saved {len(hessians)} Hessians to {hessian_save_path}")

  print(f"\nApplying GPTQ quantization (bits={bits}, group_size={group_size})...")
  gptq_adjust_weights(model, hessians, bits, group_size, block_size)
  print("GPTQ quantization complete.")
