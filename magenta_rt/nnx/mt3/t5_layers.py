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

"""T5.1.1 layers in Flax NNX (vendored building blocks for MT3)."""

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
import functools
import operator

from flax import nnx
import jax
from jax import lax, random
from jax import numpy as jnp
import numpy as np


Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nnx.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0
)


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: DType = jnp.float32,
    float32_logits: bool = False,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
        query: queries for calculating attention with shape of `[batch, q_length,
            num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of `[batch, kv_length,
            num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of `[batch, kv_length,
            num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch, num_heads, q_length, kv_length]` This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        dtype: the dtype of the computation (default: float32)
        float32_logits: bool, if True then compute logits in float32 to avoid
            numerical issues with bfloat16.

    Returns:
        Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)

    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
            keep_prob, dtype=dtype
        )
        attn_weights = attn_weights * multiplier

    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None)
)


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class T5DenseGeneral(nnx.Module):
    """A linear transformation (without bias) with flexible axes.

    Args:
        features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
        rngs: RNG state for parameter initialization.
    """

    def __init__(
        self,
        in_features: Union[Iterable[int], int],
        features: Union[Iterable[int], int],
        axis: Union[Iterable[int], int] = -1,
        dtype: DType = jnp.float32,
        kernel_init: Initializer = nnx.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"
        ),
        rngs: nnx.Rngs = None,
    ):
        self.out_features = _canonicalize_tuple(features)
        self.axis = _canonicalize_tuple(axis)
        self.dtype = dtype
        self.kernel_init = kernel_init

        in_features_tuple = _canonicalize_tuple(in_features)
        kernel_shape = in_features_tuple + self.out_features
        kernel_param_shape = (
            np.prod(in_features_tuple),
            np.prod(self.out_features),
        )

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.kernel = nnx.Param(
            self.kernel_init(rngs.params(), kernel_param_shape, jnp.float32)
        )
        self.kernel_shape = kernel_shape

    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(self.axis, inputs.ndim)

        kernel = jnp.asarray(self.kernel[...], self.dtype)
        kernel = jnp.reshape(kernel, self.kernel_shape)

        contract_ind = tuple(range(0, len(axis)))
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    elif fn_or_string == "relu":
        return jax.nn.relu
    elif fn_or_string == "gelu":
        return functools.partial(jax.nn.gelu, approximate=True)
    elif fn_or_string == "silu":
        return jax.nn.silu
    elif callable(fn_or_string):
        return fn_or_string
    else:
        raise ValueError(
            f"don't know how to convert {fn_or_string} to an activation function"
        )


class T5MLP(nnx.Module):
    """Transformer MLP / feed-forward block.

    Args:
        in_features: Input feature dimension.
        intermediate_dim: Shared dimension of hidden layers.
        activations: Type of activations for each layer. Each element is either
            'linear', a string function name, or a function.
        kernel_init: Kernel function, passed to the dense layers.
        intermediate_dropout_rate: Dropout rate used after the intermediate layers.
        dtype: Type for the dense layer.
        rngs: RNG state for parameter initialization and dropout.
    """

    def __init__(
        self,
        in_features: int,
        intermediate_dim: int = 2048,
        activations: Sequence[Union[str, Callable]] = ("relu",),
        kernel_init: Initializer = nnx.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"
        ),
        intermediate_dropout_rate: float = 0.1,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.intermediate_dim = intermediate_dim
        self.activations = activations
        self.intermediate_dropout_rate = intermediate_dropout_rate
        self.dtype = dtype
        self.deterministic = False

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.wi_layers = nnx.List(
            [
                T5DenseGeneral(
                    in_features=in_features,
                    features=intermediate_dim,
                    dtype=dtype,
                    kernel_init=kernel_init,
                    rngs=rngs,
                )
                for idx, act_fn in enumerate(activations)
            ]
        )

        self.dropout = nnx.Dropout(
            rate=intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs
        )

        self.wo = T5DenseGeneral(
            in_features=intermediate_dim,
            features=in_features,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(
        self, inputs: Array, decode: bool = False, deterministic: Optional[bool] = None
    ) -> Array:
        """Applies Transformer MlpBlock module.

        Args:
            inputs: Input array.
            decode: Whether in decoding mode.
            deterministic: Whether to apply dropout. If None, uses self.deterministic.

        Returns:
            Output array.
        """
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )

        activations = []
        for idx, act_fn in enumerate(self.activations):
            x = self.wi_layers[idx](inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        x = functools.reduce(operator.mul, activations)
        x = self.dropout(x, deterministic=deterministic)
        output = self.wo(x)
        return output


class T5Embed(nnx.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Args:
        num_embeddings: number of embeddings.
        features: number of feature dimensions for each embedding.
        dtype: the dtype of the embedding vectors (default: float32).
        embedding_init: embedding initializer.
        one_hot: performs the gather with a one-hot contraction rather than a true
            gather. This is currently needed for SPMD partitioning.
        rngs: RNG state for parameter initialization.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: DType = jnp.float32,
        embedding_init: Initializer = default_embed_init,
        one_hot: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype
        self.one_hot = one_hot

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.embedding = nnx.Param(
            embedding_init(rngs.params(), (num_embeddings, features), jnp.float32)
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
            inputs: input data, all dimensions are considered batch dimensions.

        Returns:
            Output which is embedded input data. The output shape follows the input,
            with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")

        if self.one_hot:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(self.embedding[...], self.dtype))
        else:
            output = jnp.asarray(self.embedding[...], self.dtype)[inputs]

        return output

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
            query: array with last dimension equal the feature depth `features` of the
                embedding.

        Returns:
            An array with final dim `num_embeddings` corresponding to the batched
            inner-product of the array of query vectors against each embedding.
            Commonly used for weight-sharing between embeddings and logit transform
            in NLP models.
        """
        return jnp.dot(query, jnp.asarray(self.embedding[...], self.dtype).T)


class RelativePositionBias(nnx.Module):
    """Adds T5-style relative positional embeddings to the attention logits.

    Args:
        num_buckets: Number of buckets to bucket distances between key and query
            positions into.
        max_distance: Maximum distance before everything is lumped into the last
            distance bucket.
        num_heads: Number of heads in the attention layer. Each head will get a
            different relative position weighting.
        dtype: Type of arrays through this module.
        embedding_init: initializer for relative embedding table.
        rngs: RNG state for parameter initialization.
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        dtype: Any,
        embedding_init: Callable[..., Array] = default_embed_init,
        rngs: nnx.Rngs = None,
    ):
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.dtype = dtype

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.rel_embedding = nnx.Param(
            embedding_init(rngs.params(), (num_heads, num_buckets), jnp.float32)
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger
        buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative
        positions <=-max_distance map to the same bucket. This should allow for
        more graceful generalization to longer sequences than the model has been
        trained on.

        Args:
            relative_position: an int32 array
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32
                values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qlen, klen, bidirectional=True):
        """Produce relative position embedding attention biases.

        Args:
            qlen: attention query length.
            klen: attention key length.
            bidirectional: whether to allow positive memory-query relative position
                embeddings.

        Returns:
            output: `(1, num_heads, q_len, k_len)` attention bias
        """
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        relative_attention_bias = jnp.asarray(self.rel_embedding[...], self.dtype)

        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(
            rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype
        )

        values = lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (((1,), (0,)), ((), ())),
        )

        return values[jnp.newaxis, ...]


class T5LayerNorm(nnx.Module):
    """T5 Layer normalization operating on the last axis of the input data.

    T5 uses RMSNorm without mean subtraction or bias.

    Args:
        features: Number of features (channels).
        epsilon: Small constant for numerical stability.
        dtype: Data type for computation.
        scale_init: Initializer for the scale parameter.
        rngs: RNG state for parameter initialization.
    """

    def __init__(
        self,
        features: int,
        epsilon: float = 1e-6,
        dtype: Any = jnp.float32,
        scale_init: Initializer = nnx.initializers.ones,
        rngs: nnx.Rngs = None,
    ):
        self.epsilon = epsilon
        self.dtype = dtype

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.scale = nnx.Param(scale_init(rngs.params(), (features,), jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""
        x = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        scale = jnp.asarray(self.scale[...], self.dtype)
        return y * scale


# todo: just use nnx.MultiHeadAttention?
class T5Attention(nnx.Module):
    """Multi-head dot-product attention.

    Args:
        num_heads: number of attention heads.
        head_dim: dimension of each head.
        dtype: the dtype of the computation.
        dropout_rate: dropout rate
        kernel_init: initializer for the kernel of the Dense layers.
        float32_logits: bool, if True then compute logits in float32 to avoid
            numerical issues with bfloat16.
        rngs: RNG state for parameter initialization and dropout.
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        head_dim: int,
        dtype: DType = jnp.float32,
        dropout_rate: float = 0.0,
        kernel_init: Initializer = nnx.initializers.variance_scaling(
            1.0, "fan_in", "normal"
        ),
        float32_logits: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.float32_logits = float32_logits
        self.deterministic = False

        if rngs is None:
            rngs = nnx.Rngs(0)

        # Query scaling: T5X applies 1/sqrt(head_dim) scaling to the query weights
        # during initialization rather than to query values during forward pass.
        # This is mathematically equivalent to the standard attention scaling but
        # has advantages:
        # 1. Scaling is applied once during initialization instead of every forward pass
        # 2. More numerically stable for mixed precision training (bfloat16)
        # 3. Matches the original T5X implementation exactly for weight compatibility
        # See: https://github.com/google-research/t5x/blob/main/t5x/examples/scalable_t5/layers.py#L583
        depth_scaling = jnp.sqrt(head_dim).astype(dtype)
        query_init = lambda *args: kernel_init(*args) / depth_scaling

        self.query = T5DenseGeneral(
            in_features=in_features,
            features=(num_heads, head_dim),
            dtype=dtype,
            kernel_init=query_init,
            rngs=rngs,
        )

        self.key = T5DenseGeneral(
            in_features=in_features,
            features=(num_heads, head_dim),
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.value = T5DenseGeneral(
            in_features=in_features,
            features=(num_heads, head_dim),
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.out = T5DenseGeneral(
            in_features=(num_heads, head_dim),
            features=in_features,
            axis=(-2, -1),
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: Optional[bool] = None,
        rngs: Optional[nnx.Rngs] = None,
    ) -> Array:
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
            inputs_q: input queries of shape `[batch, q_length, q_features]`.
            inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
            mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
            bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
            decode: Whether to prepare and use an autoregressive cache.
            deterministic: Disables dropout if set to True.
                If None, uses self.deterministic.
            rngs: Optional RNG state for dropout.

        Returns:
            output of shape `[batch, length, q_features]`.
        """
        deterministic = nnx.module.first_from(
            deterministic,
            self.deterministic,
            error_msg="`deterministic` must be provided or set on the model.",
        )

        query = self.query(inputs_q)
        key = self.key(inputs_kv)
        value = self.value(inputs_kv)

        if decode:
            if not (
                hasattr(self, "cached_key")
                and hasattr(self, "cached_value")
                and hasattr(self, "cache_index")
            ):
                raise ValueError(
                    "Autoregressive cache not initialized. Call init_cache() first."
                )

            (*batch_dims, max_length, num_heads, depth_per_head) = self.cached_key[
                ...
            ].shape

            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    f"Autoregressive cache shape error, "
                    f"expected query shape {expected_shape} instead got {query.shape}."
                )

            # Note: no bounds check on cur_index vs max_length here; it would
            # break under jit tracing. Callers control the number of decode
            # steps via init_cache(max_length).
            cur_index = self.cache_index[...]

            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)

            key = lax.dynamic_update_slice(self.cached_key[...], key, indices)
            value = lax.dynamic_update_slice(self.cached_value[...], value, indices)
            self.cached_key[...] = key
            self.cached_value[...] = value
            self.cache_index[...] += 1

            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
            )

            if bias is not None:
                bias = dynamic_vector_slice_in_dim(
                    jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
                )

        if mask is not None:
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            if rngs is None:
                raise ValueError(
                    "rngs must be provided when dropout_rate > 0 and not deterministic"
                )
            dropout_rng = rngs.dropout()

        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )

        out = self.out(x)
        return out

    def init_cache(self, input_shape: tuple, dtype: DType = jnp.float32):
        """Initialize cache for fast autoregressive decoding.

        Args:
            input_shape: Shape of input sequences, typically (batch_size, max_length, features).
            dtype: Data type for cache arrays.
        """
        cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))


def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: DType = jnp.float32,
) -> Array:
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
    attention weights will be `[batch, heads, len_q, len_kv]` and this
    function will produce `[batch, 1, len_q, len_kv]`.

    Args:
        query_input: a batched, flat input of query_length size
        key_input: a batched, flat input of key_length size
        pairwise_fn: broadcasting elementwise comparison function
        extra_batch_dims: number of extra batch dims to add singleton axes for, none
            by default
        dtype: mask return dtype

    Returns:
        A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1),
        jnp.expand_dims(key_input, axis=-2),
    )

    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(
    x: Array, extra_batch_dims: int = 0, dtype: DType = jnp.float32
) -> Array:
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
    will be `[batch, heads, len, len]` and this function will produce a
    causal mask of shape `[batch, 1, len, len]`.

    Note that a causal mask does not depend on the values of x; it only depends on
    the shape. If x has padding elements, they will not be treated in a special
    manner.

    Args:
        x: input array of shape `[batch, len]`
        extra_batch_dims: number of batch dims to add singleton axes for, none by
            default
        dtype: mask return dtype

    Returns:
        A `[batch, 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        jnp.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype,
    )


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
    """Combine attention masks.

    Args:
        *masks: set of attention mask arguments to combine, some can be None.
        dtype: final mask dtype

    Returns:
        Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        map(lambda x: x.ndim == masks[0].ndim, masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
    """Combine attention biases.

    Args:
        *masks: set of attention bias arguments to combine, some can be None.

    Returns:
        Combined mask, reduced by summation, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        map(lambda x: x.ndim == masks[0].ndim, masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def make_decoder_mask(
    decoder_target_tokens: Array,
    dtype: DType,
    decoder_causal_attention: Optional[Array] = None,
    decoder_segment_ids: Optional[Array] = None,
) -> Array:
    """Compute the self-attention mask for a decoder.

    Decoder mask is formed by combining a causal mask, a padding mask and an
    optional packing mask. If decoder_causal_attention is passed, it makes the
    masking non-causal for positions that have value of 1.

    Args:
        decoder_target_tokens: decoder output tokens. [batch, length]
        dtype: dtype of the output mask.
        decoder_causal_attention: a binary mask indicating which position should
            only attend to earlier positions in the sequence. Others will attend
            bidirectionally. [batch, length]
        decoder_segment_ids: decoder segmentation info for packed examples. [batch,
            length]

    Returns:
        the combined decoder mask.
    """
    masks = []
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    if decoder_causal_attention is not None:
        inputs_mask = make_attention_mask(
            decoder_causal_attention,
            decoder_causal_attention,
            jnp.logical_and,
            dtype=dtype,
        )
        masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)

    masks.append(
        make_attention_mask(
            decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype
        )
    )

    if decoder_segment_ids is not None:
        masks.append(
            make_attention_mask(
                decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype
            )
        )

    return combine_masks(*masks, dtype=dtype)
