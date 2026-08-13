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

"""Text→audio embedding mapper (MLX port).

Port of :mod:`magenta_rt.nnx.musiccoca.mapper`; see there for the
architecture (one-step DiT-style sampler with RoPE, adaLN conditioning,
attention sinks, and a 30·tanh(x/30) logit cap). Callers L2-normalize
the output.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

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


def _soft_cap(x: mx.array) -> mx.array:
    return LOGIT_CAP * mx.tanh(x / LOGIT_CAP)


def _rope(x: mx.array) -> mx.array:
    """Half-split RoPE over positions 0..T-1; x is [B, T, N, H]."""
    half = x.shape[-1] // 2
    positions = mx.arange(x.shape[1], dtype=mx.float32)
    inv_timescale = ROPE_MAX_TIMESCALE ** (
        -mx.arange(half, dtype=mx.float32) / half
    )
    angle = positions[:, None] * inv_timescale[None, :]  # [T, half]
    sin = mx.sin(angle)[None, :, None, :]
    cos = mx.cos(angle)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        ms = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + RMS_EPS) * self.scale


class MapperLayer(nn.Module):
    """One mapper block: adaLN-conditioned attention + FFN."""

    def __init__(self):
        super().__init__()
        nh = (NUM_HEADS, HEAD_DIM)
        cond_shape = (CONTEXT_DIM, 2, TOKEN_DIM)
        self.norm1 = RMSNorm(TOKEN_DIM)
        self.cond1 = Einsum("...c,ckd->...kd", cond_shape, (2, TOKEN_DIM))
        self.qkv = Einsum("...d,dknh->...knh", (TOKEN_DIM, 3, *nh))
        self.k_sink = mx.zeros((1, *nh), dtype=mx.float32)
        self.v_sink = mx.zeros((1, *nh), dtype=mx.float32)
        self.post = Einsum("btnh,dnh->btd", (TOKEN_DIM, *nh))
        self.norm2 = RMSNorm(TOKEN_DIM)
        self.cond2 = Einsum("...c,ckd->...kd", cond_shape, (2, TOKEN_DIM))
        self.ffn1 = Einsum("...a,ab->...b", (TOKEN_DIM, FFN_DIM), (FFN_DIM,))
        self.ffn2 = Einsum("...a,ab->...b", (FFN_DIM, TOKEN_DIM), (TOKEN_DIM,))

    @staticmethod
    def _ada_ln(h: mx.array, cond: mx.array) -> mx.array:
        # cond: [B, 1, 2, D] -> scale (offset from 1) and shift.
        return h * (cond[:, :, 0] + 1.0) + cond[:, :, 1]

    def __call__(self, x: mx.array, ctx: mx.array) -> mx.array:
        # x: [B, 12, 256]; ctx: [B, 1, 1024].
        b = x.shape[0]
        h = self._ada_ln(self.norm1(x), self.cond1(ctx))
        qkv = self.qkv(h)  # [B, T, 3, N, H]
        q = _rope(qkv[:, :, 0])
        k = _rope(qkv[:, :, 1])
        v = qkv[:, :, 2]
        # The sink logits use the unscaled query (export quirk).
        sink_logits = mx.einsum("btnh,snh->bnts", q, self.k_sink)
        logits = mx.einsum("btnh,bsnh->bnts", q / HEAD_DIM**0.5, k)
        logits = mx.concatenate(
            [_soft_cap(sink_logits), _soft_cap(logits)], axis=-1
        )
        attn = mx.softmax(logits, axis=-1)
        v_full = mx.concatenate(
            [mx.broadcast_to(self.v_sink, (b, 1, NUM_HEADS, HEAD_DIM)), v],
            axis=1,
        )
        out = mx.einsum("bnts,bsnh->btnh", attn, v_full)
        x = x + self.post(out)

        h = self._ada_ln(self.norm2(x), self.cond2(ctx))
        h = nn.gelu_approx(self.ffn1(h))  # tanh-approximate GELU
        return x + self.ffn2(h)


class Mapper(nn.Module):
    """Noise + text embedding → audio-space embedding (unnormalized)."""

    def __init__(self):
        super().__init__()
        self.context = mx.zeros((128,), dtype=mx.float32)
        self.input_proj = Einsum(
            "...a,ab->...b",
            (768, NUM_TOKENS * TOKEN_DIM),
            (NUM_TOKENS * TOKEN_DIM,),
        )
        self.output_proj = Einsum(
            "...a,ab->...b", (NUM_TOKENS * TOKEN_DIM, 768), (768,)
        )
        self.layers = [MapperLayer() for _ in range(NUM_LAYERS)]

    def __call__(self, text_emb: mx.array, noise: mx.array) -> mx.array:
        # text_emb, noise: [B, 768] -> [B, 768].
        b = text_emb.shape[0]
        prefix = mx.broadcast_to(self.context, (b, 128))
        ctx = mx.concatenate([prefix, prefix, text_emb], axis=-1)[:, None]
        x = self.input_proj(noise).reshape(b, NUM_TOKENS, TOKEN_DIM)
        for layer in self.layers:
            x = layer(x, ctx)
        return self.output_proj(x.reshape(b, NUM_TOKENS * TOKEN_DIM))
