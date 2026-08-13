# Copyright 2025 The MT3 Authors.
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

# Copyright 2024 The T5X Authors.
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

"""T5.1.1 layers in pure MLX (vendored building blocks for MT3).

Port of ``magenta_rt.nnx.mt3.t5_layers``; parameter shapes match the t5x
checkpoints exactly (DenseGeneral kernels are stored flattened 2-D), so
the same safetensors load into both backends. Inference-focused: dropout
layers exist for structural parity (``mlx.nn.Dropout``, inert in eval
mode) but the per-axis broadcast of the original dropout is not
replicated.
"""

from __future__ import annotations

import functools
import operator
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """Dot-product attention; q/k/v are ``[batch, length, heads, depth]``.

    No runtime query scaling — T5X folds ``1/sqrt(head_dim)`` into the
    query weights, so pretrained checkpoints carry it already.
    """
    attn_weights = mx.einsum("bqhd,bkhd->bhqk", query, key)
    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)
    attn_weights = mx.softmax(attn_weights, axis=-1)
    return mx.einsum("bhqk,bkhd->bqhd", attn_weights, value)


class T5DenseGeneral(nn.Module):
    """A linear transformation (without bias) with flexible trailing axes.

    The kernel is stored flattened 2-D ``(prod(in), prod(out))`` exactly as
    in the t5x checkpoints. Contraction axes must be the trailing axes of
    the input (always true for MT3's usage: ``-1`` or ``(-2, -1)``).
    """

    def __init__(
        self,
        in_features: Union[Iterable[int], int],
        features: Union[Iterable[int], int],
        axis: Union[Iterable[int], int] = -1,
    ):
        super().__init__()
        self.out_features = _canonicalize_tuple(features)
        self.axis = _canonicalize_tuple(axis)
        in_features_tuple = _canonicalize_tuple(in_features)
        self.kernel = mx.zeros(
            (int(np.prod(in_features_tuple)), int(np.prod(self.out_features))),
            dtype=mx.float32,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        axis = _normalize_axes(self.axis, inputs.ndim)
        if axis != tuple(range(inputs.ndim - len(axis), inputs.ndim)):
            raise ValueError(f"contraction axes must be trailing, got {axis}")
        lead = inputs.shape[: axis[0]]
        x = inputs.reshape(*lead, -1)
        y = x @ self.kernel
        return y.reshape(*lead, *self.out_features)


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    if fn_or_string == "relu":
        return nn.relu
    if fn_or_string == "gelu":
        return nn.gelu_approx  # tanh approximation, as in t5x/jax.nn.gelu
    if fn_or_string == "silu":
        return nn.silu
    if callable(fn_or_string):
        return fn_or_string
    raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")


class T5MLP(nn.Module):
    """Transformer feed-forward block (optionally gated, e.g. gated-gelu)."""

    def __init__(
        self,
        in_features: int,
        intermediate_dim: int = 2048,
        activations: Sequence[Union[str, Callable]] = ("relu",),
        intermediate_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.activations = tuple(activations)
        self.wi_layers = [
            T5DenseGeneral(in_features=in_features, features=intermediate_dim)
            for _ in self.activations
        ]
        self.dropout = nn.Dropout(intermediate_dropout_rate)
        self.wo = T5DenseGeneral(in_features=intermediate_dim, features=in_features)

    def __call__(self, inputs: mx.array) -> mx.array:
        activations = []
        for idx, act_fn in enumerate(self.activations):
            x = self.wi_layers[idx](inputs)
            activations.append(_convert_to_activation_function(act_fn)(x))
        x = functools.reduce(operator.mul, activations)
        x = self.dropout(x)
        return self.wo(x)


class T5Embed(nn.Module):
    """Integer [0, n) → d-dimensional vector lookup."""

    def __init__(self, num_embeddings: int, features: int):
        super().__init__()
        self.embedding = mx.zeros((num_embeddings, features), dtype=mx.float32)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.embedding[inputs]

    def attend(self, query: mx.array) -> mx.array:
        return query @ self.embedding.T


class T5LayerNorm(nn.Module):
    """T5 layer norm: RMSNorm without mean subtraction or bias."""

    def __init__(self, features: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.scale = mx.ones((features,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        mean2 = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(mean2 + self.epsilon) * self.scale


class T5Attention(nn.Module):
    """Multi-head dot-product attention with an optional decode cache."""

    def __init__(self, in_features: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query = T5DenseGeneral(in_features, (num_heads, head_dim))
        self.key = T5DenseGeneral(in_features, (num_heads, head_dim))
        self.value = T5DenseGeneral(in_features, (num_heads, head_dim))
        self.out = T5DenseGeneral((num_heads, head_dim), in_features, axis=(-2, -1))
        # Decode cache (allocated by init_cache; underscore = non-parameter).
        self._cached_key: Optional[mx.array] = None
        self._cached_value: Optional[mx.array] = None
        self._cache_index: int = 0

    def init_cache(self, batch_size: int, max_length: int):
        """Allocate the autoregressive KV cache."""
        shape = (batch_size, max_length, self.num_heads, self.head_dim)
        self._cached_key = mx.zeros(shape, dtype=mx.float32)
        self._cached_value = mx.zeros(shape, dtype=mx.float32)
        self._cache_index = 0

    def __call__(
        self,
        inputs_q: mx.array,
        inputs_kv: mx.array,
        mask: Optional[mx.array] = None,
        *,
        decode: bool = False,
    ) -> mx.array:
        query = self.query(inputs_q)
        key = self.key(inputs_kv)
        value = self.value(inputs_kv)

        if decode:
            if self._cached_key is None:
                raise ValueError(
                    "Autoregressive cache not initialized. Call init_cache() first."
                )
            batch, max_length = self._cached_key.shape[:2]
            if query.shape[:2] != (batch, 1):
                raise ValueError(
                    f"expected query shape ({batch}, 1, ...), got {query.shape}"
                )
            i = self._cache_index
            self._cached_key[:, i : i + 1] = key
            self._cached_value[:, i : i + 1] = value
            self._cache_index = i + 1
            key = self._cached_key
            value = self._cached_value
            cache_mask = (mx.arange(max_length) <= i).reshape(1, 1, 1, max_length)
            mask = cache_mask if mask is None else mx.logical_and(mask > 0, cache_mask)

        if mask is not None:
            attention_bias = mx.where(
                mask > 0,
                mx.zeros(mask.shape, dtype=mx.float32),
                mx.full(mask.shape, -1e10, dtype=mx.float32),
            )
        else:
            attention_bias = None

        x = dot_product_attention(query, key, value, bias=attention_bias)
        return self.out(x)


def make_attention_mask(
    query_input: mx.array,
    key_input: mx.array,
    pairwise_fn: Callable = mx.multiply,
) -> mx.array:
    """``[batch, len_q] x [batch, len_kv] -> [batch, 1, len_q, len_kv]`` mask."""
    mask = pairwise_fn(query_input[..., :, None], key_input[..., None, :])
    return mask[..., None, :, :].astype(mx.float32)


def make_causal_mask(x: mx.array) -> mx.array:
    """Causal mask ``[batch, 1, len, len]`` for 1-D inputs ``[batch, len]``."""
    idxs = mx.broadcast_to(mx.arange(x.shape[-1], dtype=mx.int32), x.shape)
    return make_attention_mask(idxs, idxs, mx.greater_equal)


def combine_masks(*masks: Optional[mx.array]) -> Optional[mx.array]:
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    mask, *others = masks
    for other in others:
        mask = mx.logical_and(mask > 0, other > 0)
    return mask.astype(mx.float32)


def make_decoder_mask(decoder_target_tokens: mx.array) -> mx.array:
    """Causal + padding self-attention mask for the decoder."""
    causal_mask = make_causal_mask(decoder_target_tokens)
    padding_mask = make_attention_mask(
        (decoder_target_tokens > 0).astype(mx.float32),
        (decoder_target_tokens > 0).astype(mx.float32),
    )
    return combine_masks(causal_mask, padding_mask)
