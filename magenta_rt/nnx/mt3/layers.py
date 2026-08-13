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

"""MT3-specific layers in Flax NNX.

The Transformer building blocks (DenseGeneral, MLP, Embed, LayerNorm,
Attention, mask helpers) are shared with the T5 port; this module adds the
fixed sinusoidal positional embedding used by MT3 in place of T5's relative
position biases.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


def sinusoidal_embeddings(
    max_length: int,
    features: int,
    min_scale: float = 1.0,
    max_scale: float = 10000.0,
) -> np.ndarray:
    """Create 1D sinusoidal position embeddings.

    Matches the ``sinusoidal`` initializer from t5x/mt3: the first half of the
    feature dimension is sine, the second half cosine (not interleaved).

    Args:
        max_length: Maximum supported position.
        features: Embedding dimension.
        min_scale: Minimum frequency-scale in sine grating.
        max_scale: Maximum frequency-scale in sine grating.

    Returns:
        Embedding table of shape [max_length, features].
    """
    pe = np.zeros((max_length, features), dtype=np.float32)
    position = np.arange(0, max_length)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, features // 2) * scale_factor)
    pe[:, : features // 2] = np.sin(position * div_term)
    pe[:, features // 2 : 2 * (features // 2)] = np.cos(position * div_term)
    return pe


class FixedEmbed(nnx.Module):
    """Fixed (not learnable) sinusoidal position embeddings.

    Args:
        features: Embedding dimension.
        max_length: The maximum supported position.
        dtype: The dtype of the returned embeddings.
    """

    def __init__(
        self,
        features: int,
        max_length: int = 2048,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.features = features
        self.max_length = max_length
        self.dtype = dtype
        self.materialize_constants()

    def materialize_constants(self) -> None:
        """(Re)compute the position table from ``max_length``/``features``.

        The table is derived rather than trained, so it is not an ``nnx.Param``
        and no checkpoint carries it. A model built with ``nnx.eval_shape``
        therefore has it abstract with nothing to restore it — and
        ``assert_fully_loaded`` will not flag it, precisely because it is not a
        parameter. See :mod:`magenta_rt.nnx.checkpoint_utils`.
        """
        self.embedding = nnx.data(
            jnp.asarray(sinusoidal_embeddings(self.max_length, self.features))
        )

    def init_cache(self):
        """Initialize the position index cache for autoregressive decoding."""
        self.position_index = nnx.Cache(jnp.zeros((), dtype=jnp.int32))

    def __call__(self, inputs: jnp.ndarray, *, decode: bool = False) -> jnp.ndarray:
        """Returns the fixed position embeddings at the given positions.

        Args:
            inputs: <int>[batch_size, seq_len] input position indices.
            decode: True if running in single-position autoregressive decode
                mode; ``inputs`` is then ignored in favor of a cached position
                index that increments on each call.

        Returns:
            The fixed position embeddings <float32>[batch_size, seq_len, features],
            or [1, features] in decode mode (broadcast over batch and length).
        """
        if decode:
            if not hasattr(self, "position_index"):
                raise ValueError("Autoregressive cache not initialized. Call init_cache() first.")
            i = self.position_index[...]
            self.position_index[...] = i + 1
            return jax.lax.dynamic_slice(
                self.embedding, (i, jnp.zeros((), dtype=jnp.int32)), (1, self.features)
            ).astype(self.dtype)

        return jnp.take(self.embedding, inputs, axis=0).astype(self.dtype)
