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

"""MT3-specific layers in pure MLX.

Adds the fixed sinusoidal positional embedding used by MT3 in place of
T5's relative position biases. The embedding table itself comes from the
framework-neutral builder in :mod:`magenta_rt.nnx.mt3.layers`'s numpy
twin below (same formula).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def sinusoidal_embeddings(
    max_length: int,
    features: int,
    min_scale: float = 1.0,
    max_scale: float = 10000.0,
) -> np.ndarray:
    """1D sinusoidal position embeddings (t5x/mt3 ``sinusoidal`` initializer).

    First half of the feature dimension is sine, second half cosine (not
    interleaved). Returns ``[max_length, features]``.
    """
    pe = np.zeros((max_length, features), dtype=np.float32)
    position = np.arange(0, max_length)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, features // 2) * scale_factor)
    pe[:, : features // 2] = np.sin(position * div_term)
    pe[:, features // 2 : 2 * (features // 2)] = np.cos(position * div_term)
    return pe


class FixedEmbed(nn.Module):
    """Fixed (not learnable) sinusoidal position embeddings."""

    def __init__(self, features: int, max_length: int = 2048):
        super().__init__()
        self.features = features
        self.max_length = max_length
        # Underscore prefix keeps the fixed table out of parameters().
        self._embedding = mx.array(sinusoidal_embeddings(max_length, features))
        self._position_index: int = 0

    def init_cache(self):
        """Reset the position index for autoregressive decoding."""
        self._position_index = 0

    def __call__(self, inputs: mx.array, *, decode: bool = False) -> mx.array:
        if decode:
            i = self._position_index
            self._position_index = i + 1
            return self._embedding[i : i + 1]  # [1, features], broadcasts
        return self._embedding[inputs]
