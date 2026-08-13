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

"""Residual vector quantizer for MusicCoCa style embeddings.

12 stages, codebook size 1024, dimension 768. Each stage picks the
codebook entry minimizing the squared Euclidean distance to the running
residual and subtracts it — the same computation as
``pretrained_vector_quantizer.tflite``.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

RVQ_DEPTH = 12
CODEBOOK_SIZE = 1024
EMBEDDING_DIM = 768


class EmbeddingQuantizer(nnx.Module):
    """768-dim embedding → 12 RVQ tokens (and back)."""

    def __init__(self):
        self.codebooks = nnx.Param(
            jnp.zeros((RVQ_DEPTH, CODEBOOK_SIZE, EMBEDDING_DIM), jnp.float32)
        )

    def tokenize(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """``[..., 768]`` → ``[..., 12]`` int32 tokens."""
        flat = embeddings.reshape(-1, EMBEDDING_DIM)
        residual = flat
        tokens = []
        for stage in range(RVQ_DEPTH):
            codebook = self.codebooks[...][stage]
            distances = (
                jnp.sum(jnp.square(residual), axis=-1, keepdims=True)
                - 2.0 * residual @ codebook.T
                + jnp.sum(jnp.square(codebook), axis=-1)
            )
            idx = jnp.argmin(distances, axis=-1)
            tokens.append(idx)
            residual = residual - codebook[idx]
        out = jnp.stack(tokens, axis=-1).astype(jnp.int32)
        return out.reshape(*embeddings.shape[:-1], RVQ_DEPTH)

    def decode(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """``[..., 12]`` tokens → ``[..., 768]`` reconstructed embedding."""
        flat = tokens.reshape(-1, RVQ_DEPTH)
        parts = [
            self.codebooks[...][stage][flat[:, stage]]
            for stage in range(RVQ_DEPTH)
        ]
        out = jnp.sum(jnp.stack(parts), axis=0)
        return out.reshape(*tokens.shape[:-1], EMBEDDING_DIM)

    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        return self.tokenize(embeddings)
