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

"""Text→audio embedding mapper (``mulan_mapper.sample`` in the export).

A one-step DiT-style sampler that maps a text embedding into the audio
embedding region of the joint space:

1. a noise vector ``[768]`` is projected to 12 tokens of width 256,
2. 8 transformer layers process the tokens, each block using RMSNorm
   followed by adaptive layer norm (scale/shift) conditioned on
   ``[c, c, text_emb]`` where ``c`` is a learned 128-dim prefix,
3. attention uses RoPE, one learned key/value "sink" per head, a
   ``30 * tanh(x / 30)`` logit cap, and tanh-approximate GELU FFNs,
4. tokens are flattened back to a 768-dim embedding.

Callers L2-normalize the output (as the Python TFLite wrapper does).

Quirk preserved from the export: the per-position attention logits scale
the query by ``1/sqrt(head_dim)`` but the sink logits use the *unscaled*
query.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .modules import Einsum

NUM_LAYERS = 8
NUM_TOKENS = 12
TOKEN_DIM = 256
NUM_HEADS = 8
HEAD_DIM = 64
FFN_DIM = 1024
CONTEXT_DIM = 1024
LOGIT_CAP = 30.0
RMS_EPS = 1e-6
ROPE_MAX_TIMESCALE = 10_000.0


def _soft_cap(x: jnp.ndarray) -> jnp.ndarray:
    return LOGIT_CAP * jnp.tanh(x / LOGIT_CAP)


def _rope(x: jnp.ndarray) -> jnp.ndarray:
    """Half-split RoPE over positions 0..T-1; x is [B, T, N, H]."""
    half = x.shape[-1] // 2
    positions = jnp.arange(x.shape[1], dtype=jnp.float32)
    inv_timescale = ROPE_MAX_TIMESCALE ** (
        -jnp.arange(half, dtype=jnp.float32) / half
    )
    angle = positions[:, None] * inv_timescale[None, :]  # [T, half]
    sin = jnp.sin(angle)[None, :, None, :]
    cos = jnp.cos(angle)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
    )


class RMSNorm(nnx.Module):
    def __init__(self, dim: int):
        self.scale = nnx.Param(jnp.ones((dim,), jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(ms + RMS_EPS) * self.scale[...]


class MapperLayer(nnx.Module):
    """One mapper block: adaLN-conditioned attention + FFN."""

    def __init__(self):
        nh = (NUM_HEADS, HEAD_DIM)
        cond_shape = (CONTEXT_DIM, 2, TOKEN_DIM)
        self.norm1 = RMSNorm(TOKEN_DIM)
        self.cond1 = Einsum("...c,ckd->...kd", cond_shape, (2, TOKEN_DIM))
        self.qkv = Einsum("...d,dknh->...knh", (TOKEN_DIM, 3, *nh))
        self.k_sink = nnx.Param(jnp.zeros((1, *nh), jnp.float32))
        self.v_sink = nnx.Param(jnp.zeros((1, *nh), jnp.float32))
        self.post = Einsum("btnh,dnh->btd", (TOKEN_DIM, *nh))
        self.norm2 = RMSNorm(TOKEN_DIM)
        self.cond2 = Einsum("...c,ckd->...kd", cond_shape, (2, TOKEN_DIM))
        self.ffn1 = Einsum("...a,ab->...b", (TOKEN_DIM, FFN_DIM), (FFN_DIM,))
        self.ffn2 = Einsum("...a,ab->...b", (FFN_DIM, TOKEN_DIM), (TOKEN_DIM,))

    @staticmethod
    def _ada_ln(h: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        # cond: [B, 1, 2, D] -> scale (offset from 1) and shift.
        return h * (cond[:, :, 0] + 1.0) + cond[:, :, 1]

    def __call__(self, x: jnp.ndarray, ctx: jnp.ndarray) -> jnp.ndarray:
        # x: [B, 12, 256]; ctx: [B, 1, 1024].
        b = x.shape[0]
        h = self._ada_ln(self.norm1(x), self.cond1(ctx))
        qkv = self.qkv(h)  # [B, T, 3, N, H]
        q = _rope(qkv[:, :, 0])
        k = _rope(qkv[:, :, 1])
        v = qkv[:, :, 2]
        # The sink logits use the unscaled query (export quirk).
        sink_logits = jnp.einsum("btnh,snh->bnts", q, self.k_sink[...])
        logits = jnp.einsum(
            "btnh,bsnh->bnts", q / jnp.sqrt(float(HEAD_DIM)), k
        )
        logits = jnp.concatenate(
            [_soft_cap(sink_logits), _soft_cap(logits)], axis=-1
        )
        attn = jax.nn.softmax(logits, axis=-1)
        v_full = jnp.concatenate(
            [jnp.broadcast_to(self.v_sink[...], (b, 1, NUM_HEADS, HEAD_DIM)), v],
            axis=1,
        )
        out = jnp.einsum("bnts,bsnh->btnh", attn, v_full)
        x = x + self.post(out)

        h = self._ada_ln(self.norm2(x), self.cond2(ctx))
        h = nnx.gelu(self.ffn1(h), approximate=True)
        return x + self.ffn2(h)


class Mapper(nnx.Module):
    """Noise + text embedding → audio-space embedding (unnormalized)."""

    def __init__(self):
        self.context = nnx.Param(jnp.zeros((128,), jnp.float32))
        self.input_proj = Einsum(
            "...a,ab->...b", (768, NUM_TOKENS * TOKEN_DIM), (NUM_TOKENS * TOKEN_DIM,)
        )
        self.output_proj = Einsum(
            "...a,ab->...b", (NUM_TOKENS * TOKEN_DIM, 768), (768,)
        )

        @nnx.vmap(in_axes=0, out_axes=0)
        def make(_) -> MapperLayer:
            return MapperLayer()

        self.layers = make(jnp.arange(NUM_LAYERS))

    def __call__(
        self, text_emb: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        # text_emb, noise: [B, 768] -> [B, 768].
        b = text_emb.shape[0]
        prefix = jnp.broadcast_to(self.context[...], (b, 128))
        ctx = jnp.concatenate([prefix, prefix, text_emb], axis=-1)[:, None]
        x = self.input_proj(noise).reshape(b, NUM_TOKENS, TOKEN_DIM)

        @nnx.scan(length=NUM_LAYERS, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(h, layer):
            return layer(h, ctx)

        x = forward(x, self.layers)
        return self.output_proj(x.reshape(b, NUM_TOKENS * TOKEN_DIM))
