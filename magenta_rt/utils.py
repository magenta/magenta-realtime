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

"""Utilities for Magenta RT."""

import functools
import logging
import pathlib
from typing import Any, Tuple
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def _globally_disable_gpu_memory_growth():
  """Prevents TF from consuming all available GPU memory by default."""
  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


@functools.cache
def load_model_cached(model_type: str, model_path: str | pathlib.Path) -> Any:
  """Loads a model from a path, caching the result."""
  # Disable verbose warnings from abseil.
  logging.getLogger('tensorflow').setLevel(logging.ERROR)
  logging.getLogger('absl').setLevel(logging.ERROR)
  if isinstance(model_path, pathlib.Path):
    model_path = str(model_path)
  _globally_disable_gpu_memory_growth()
  if model_type == 'tf':
    model = tf.saved_model.load(model_path)
  elif model_type == 'hub':
    model = hub.load(model_path, tags=['serve'])
  elif model_type == 'npy':
    model = np.load(model_path)
  else:
    raise ValueError('Unknown model type')
  return model


def rvq_quantization(
    embeddings: np.ndarray, codebooks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Performs RVQ quantization on a batch of embeddings."""
  # RVQ quantization algorithm
  # B is batch size, K is RVQ depth, V is RVQ codebook size, D is embedding dim
  # embeddings is (B, D), e.g., (B, 128) float32
  # codebooks is (K, V, D), e.g., (10, 1024, 128) float32
  # result is (B, K), e.g., (B, 10) int32
  if embeddings.ndim != 2:
    raise ValueError(f'embeddings must be a 2D array, got {embeddings.shape}')
  if codebooks.ndim != 3:
    raise ValueError(f'codebooks must be a 3D array, got {codebooks.shape}')
  if embeddings.shape[1] != codebooks.shape[2]:
    raise ValueError('residual and codebooks must have the same shape')
  (batch_size, embedding_dim) = embeddings.shape
  result = np.zeros((batch_size, codebooks.shape[0]), dtype=np.int32)
  dist_fn = lambda a, b: np.linalg.norm(a - b, axis=-1)  # L2 distance
  residual = embeddings.copy()
  for i in range(codebooks.shape[0]):
    # Compute distances between residual and each code
    # L2 distances: norm((B, 1, D) - (1, V, D), -1) -> (B, V)
    distances = dist_fn(residual[:, np.newaxis], codebooks[i : i + 1])
    assert distances.shape == (batch_size, codebooks.shape[1])
    # Find closest code: argmin((B, V), 1) -> (B,)
    nearest_neighbors = np.argmin(distances, axis=1)
    assert nearest_neighbors.shape == (batch_size,)
    result[:, i] = nearest_neighbors
    residual = residual - codebooks[i, nearest_neighbors, :]
    assert residual.shape == (batch_size, embedding_dim)
  return result, residual


def rvq_dequantization(tokens: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
  """Performs RVQ dequantization on a batch of tokens."""
  # RVQ dequantization algorithm
  # B is batch size, K is RVQ depth, V is RVQ codebook size, D is embedding dim
  # tokens is (B, <=K), e.g., (B, 10) int32
  # codebooks is (K, V, D), e.g., (10, 1024, 128) float32
  # result is (B, D), e.g., (B, 128) float32
  if tokens.ndim != 2:
    raise ValueError(f'tokens must be a 2D array, got {tokens.shape}')
  if codebooks.ndim != 3:
    raise ValueError(f'codebooks must be a 3D array, got {codebooks.shape}')
  if tokens.shape[1] > codebooks.shape[0]:
    raise ValueError('token depth must be less than or equal to codebook depth')
  if np.any(tokens < 0):
    raise IndexError(f'Negative tokens: {tokens}')
  result = np.zeros((tokens.shape[0], codebooks.shape[2]), dtype=np.float32)
  for i in range(tokens.shape[1]):
    result += codebooks[i, tokens[:, i]]
  return result


def rvq_to_llm(
    rvq_tokens: np.ndarray, rvq_codebook_size: int, offset: int = 0
) -> np.ndarray:
  """Encodes raw RVQ tokens for LLM processing (adds unique vocab offsets)."""
  if rvq_tokens.ndim == 0:
    raise ValueError(f'Invalid array shape: {rvq_tokens.shape}')
  if np.any(rvq_tokens < 0) or np.any(rvq_tokens >= rvq_codebook_size):
    raise IndexError(f'Invalid array values: {rvq_tokens}')
  rvq_depth = rvq_tokens.shape[-1]
  rvq_offsets = np.arange(rvq_depth) * rvq_codebook_size
  return (
      rvq_tokens
      + offset
      + rvq_offsets.reshape(*([1] * (rvq_tokens.ndim - 1) + [rvq_depth]))
  )


def llm_to_rvq(
    llm_tokens: np.ndarray,
    rvq_codebook_size: int,
    offset: int = 0,
    safe: bool = True,
) -> np.ndarray:
  """Decodes LLM tokens w/ vocab offsets to raw RVQ tokens."""
  if llm_tokens.ndim == 0:
    raise ValueError(f'Invalid array shape: {llm_tokens.shape}')
  rvq_depth = llm_tokens.shape[-1]
  rvq_tokens = np.maximum(llm_tokens - offset, 0)
  vocab_subset = rvq_tokens // rvq_codebook_size
  expected = np.arange(rvq_depth)
  expected = expected.reshape(*([1] * (rvq_tokens.ndim - 1) + [rvq_depth]))
  num_invalid_tokens = np.logical_or(
      llm_tokens < offset,
      vocab_subset != expected
  ).astype(np.int32).sum()
  if num_invalid_tokens > 0:
    msg = f'{num_invalid_tokens}/{llm_tokens.size} invalid tokens'
    if safe:
      raise IndexError(msg)
    else:
      warnings.warn(msg)
  return rvq_tokens % rvq_codebook_size
