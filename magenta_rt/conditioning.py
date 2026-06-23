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

"""Backend-agnostic helpers for shaping ``generate`` conditioning into batches.

Pure NumPy so the jax and mlx systems share one implementation (and neither
backend has to import the other). The broadcast rule mirrors NumPy/JAX
semantics: a per-batch conditioning argument may be passed either as a single
shared vector/scalar (broadcast across the batch) *or* already batched with a
leading axis of size ``N`` (used per-element). The batch size ``N`` itself is
fixed by ``style`` (the primary conditioning input); every other argument must
either be shared or match ``N``.
"""

from __future__ import annotations

import numpy as np


def broadcast_rows(values, expected_len: int, batch_size: int, name: str) -> np.ndarray:
  """Normalize a conditioning argument to a ``[batch_size, expected_len]`` int array.

  Broadcast rule:

  * rank-1 ``[expected_len]``             -> shared across the batch.
  * rank-2 ``[1, expected_len]``          -> shared across the batch.
  * rank-2 ``[batch_size, expected_len]`` -> used per-element.

  Args:
    values: The argument (list, ``[expected_len]`` vector, or ``[N, expected_len]``
      array). jax/mlx arrays are accepted via ``np.asarray``.
    expected_len: Required size of the per-element (last) axis.
    batch_size: Target batch size ``N`` (set by ``style``).
    name: Argument name, for error messages.

  Returns:
    A contiguous ``[batch_size, expected_len]`` int32 array.

  Raises:
    ValueError: on a wrong per-element length, an unsupported rank, or a leading
      axis that is neither 1 nor ``batch_size``.
  """
  arr = np.asarray(values, dtype=np.int32)
  if arr.ndim == 1:
    if arr.shape[0] != expected_len:
      raise ValueError(
          f'{name}: expected {expected_len} values, got {arr.shape[0]}'
      )
    arr = arr[None, :]
  elif arr.ndim == 2:
    if arr.shape[1] != expected_len:
      raise ValueError(
          f'{name}: expected a last axis of {expected_len}, got shape {arr.shape}'
      )
  else:
    raise ValueError(
        f'{name}: expected a rank-1 or rank-2 array, got rank {arr.ndim}'
    )
  if arr.shape[0] == 1:
    arr = np.broadcast_to(arr, (batch_size, expected_len))
  elif arr.shape[0] != batch_size:
    raise ValueError(
        f'{name}: batch size {arr.shape[0]} does not match the number of '
        f'styles ({batch_size}); pass a shared row or one row per style'
    )
  return np.ascontiguousarray(arr, dtype=np.int32)


def broadcast_scalar(value, batch_size: int, name: str, dtype) -> np.ndarray:
  """Normalize a scalar-or-``[batch_size]`` argument to a length-``batch_size`` array.

  A scalar (or length-1 vector) broadcasts across the batch; a length-``N``
  vector is used per-element.
  """
  arr = np.asarray(value)
  if arr.ndim == 0:
    arr = arr[None]
  elif arr.ndim != 1:
    raise ValueError(
        f'{name}: expected a scalar or rank-1 array, got rank {arr.ndim}'
    )
  if arr.shape[0] == 1:
    arr = np.broadcast_to(arr, (batch_size,))
  elif arr.shape[0] != batch_size:
    raise ValueError(
        f'{name}: batch size {arr.shape[0]} does not match the number of '
        f'styles ({batch_size})'
    )
  return np.asarray(arr, dtype=dtype)


def discretize_cfg(value: float, step: float, max_bin: int) -> int:
  """Map a CFG scale in [-1.0, 7.0] to a discrete conditioning token index.

  Same binning as ``magenta_rt.mlx.system.discretize_cfg`` (float64); shared
  here so the nnx and mlx_pure systems don't re-implement it per backend.

  Args:
    value: CFG scale; clamped to [-1.0, 7.0].
    step: Quantization step (0.2 for musiccoca/notes, 1.0 for drums).
    max_bin: Largest valid token index (40 for musiccoca/notes, 8 for drums).

  Returns:
    Token index in [0, max_bin].
  """
  clamped = max(-1.0, min(7.0, value))
  bin_index = int(round((clamped - (-1.0)) / step))
  return max(0, min(max_bin, bin_index))


def normalize_style_rows(tokens, num_musiccoca: int) -> list[list[int]]:
  """Normalize tokenized style(s) to a list of ``num_musiccoca``-length rows.

  Accepts a single ``[k]`` token row or a batched ``[N, k]`` array (the output
  of ``MusicCoCa.tokenize`` for a single or batched embedding); each row is
  padded with ``-1`` / truncated to ``num_musiccoca`` tokens.
  """
  rows = np.atleast_2d(np.asarray(tokens))
  out = []
  for row in rows:
    row_tokens = [int(t) for t in np.asarray(row).reshape(-1)]
    if len(row_tokens) < num_musiccoca:
      row_tokens = row_tokens + [-1] * (num_musiccoca - len(row_tokens))
    out.append(row_tokens[:num_musiccoca])
  return out


def build_conditioning_rows(
    batch_style,
    notes,
    drums,
    cfgs,
    *,
    num_musiccoca: int,
    num_notes: int,
    drum_tokens: int,
    cfg_tokens: int,
    offset: int,
) -> np.ndarray:
  """Assemble the batched ``[N, 1, C]`` conditioning block (token-CFG scheme).

  Backend-agnostic core of ``MagentaRT2System._build_conditioning``: one
  ``style|notes|drums|cfgs`` row per batch element, every channel offset by
  ``offset`` (``NUM_RESERVED_TOKENS + 1``, the +1 being the dropout token).
  ``notes`` / ``drums`` / ``cfgs`` follow the broadcast rule
  (:func:`broadcast_rows`); ``batch_style`` fixes the batch size ``N``.
  """
  style_rows = [list(s) for s in batch_style]
  for row in style_rows:
    if len(row) != num_musiccoca:
      raise ValueError(
          f'Expected {num_musiccoca} style tokens, got {len(row)}'
      )
  N = len(style_rows)
  style_arr = np.asarray(style_rows, dtype=np.int32)  # [N, num_musiccoca]

  notes = notes if notes is not None else [-1] * num_notes
  drums = drums if drums is not None else [-1] * drum_tokens

  notes_arr = broadcast_rows(notes, num_notes, N, 'notes')
  drums_arr = broadcast_rows(drums, drum_tokens, N, 'drums')
  cfgs_arr = broadcast_rows(cfgs, cfg_tokens, N, 'cfgs')

  cond_NC = np.concatenate(
      [style_arr, notes_arr, drums_arr, cfgs_arr], axis=1
  ) + offset  # [N, C]
  return cond_NC.reshape(N, 1, -1).astype(np.int32)
