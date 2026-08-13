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

"""Pure-MLX building blocks for MusicCoCa.

Port of :mod:`magenta_rt.nnx.musiccoca.modules`; see that module and
``magenta_rt/nnx/musiccoca/README.md`` for the architecture recovered
from the TFLite exports. Weights load from the same safetensors file
(stacked per-layer arrays are split across the layer list at load time;
see ``model.load_safetensors``).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Fill value the exporter uses for masked attention logits.
MASK_FILL = -2.3819763e38

# Soft cap applied to attention logits in both towers.
ATTN_LOGIT_CAP = 50.0


class Einsum(nn.Module):
    """A single einsum projection with optional bias (Praxis-style)."""

    def __init__(
        self,
        equation: str,
        kernel_shape: tuple[int, ...],
        bias_shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.equation = equation
        self.kernel = mx.zeros(kernel_shape, dtype=mx.float32)
        if bias_shape is not None:
            self.bias = mx.zeros(bias_shape, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.einsum(self.equation, x, self.kernel)
        if "bias" in self:
            y = y + self["bias"]
        return y


class LayerNorm(nn.Module):
    """LayerNorm with direct scale/bias (Praxis ``+1`` baked at conversion)."""

    def __init__(self, dim: int, *, eps: float = 1e-6):
        super().__init__()
        self.scale = mx.ones((dim,), dtype=mx.float32)
        self.bias = mx.zeros((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        normed = (x - mean) * mx.rsqrt(var + self.eps)
        return normed * self.scale + self.bias


class TransformerLayer(nn.Module):
    """Pre-LN encoder layer shared by the music and text towers."""

    def __init__(
        self,
        *,
        model_dim: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        ffn_dim: int = 3072,
    ):
        super().__init__()
        nh = (num_heads, head_dim)
        self.ln1 = LayerNorm(model_dim)
        self.q = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.k = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.v = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.post = Einsum("...nh,dnh->...d", (model_dim, *nh), (model_dim,))
        self.ln2 = LayerNorm(model_dim)
        self.ffn1 = Einsum("...a,ab->...b", (model_dim, ffn_dim), (ffn_dim,))
        self.ffn2 = Einsum("...a,ab->...b", (ffn_dim, model_dim), (model_dim,))

    def __call__(
        self,
        x: mx.array,
        attn_mask: Optional[mx.array] = None,
        ffn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.ln1(x)
        # Query scale (1/sqrt(head_dim)) is folded into self.q's weights.
        logits = mx.einsum("btnh,bsnh->bnts", self.q(h), self.k(h))
        logits = ATTN_LOGIT_CAP * mx.tanh(logits / ATTN_LOGIT_CAP)
        if attn_mask is not None:
            logits = mx.where(attn_mask, logits, MASK_FILL)
        attn = mx.softmax(logits, axis=-1)
        out = mx.einsum("bnts,bsnh->btnh", attn, self.v(h))
        x = x + self.post(out)

        h = self.ln2(x)
        h = nn.gelu(self.ffn1(h))  # exact (erf) GELU
        if ffn_mask is not None:
            h = h * ffn_mask
        h = self.ffn2(h)
        if ffn_mask is not None:
            h = h * ffn_mask
        return x + h


class AttentionPooler(nn.Module):
    """CoCa attentional pooler: one learned query over the sequence."""

    def __init__(
        self,
        *,
        model_dim: int = 768,
        num_heads: int = 12,
        head_dim: int = 256,
    ):
        super().__init__()
        nh = (num_heads, head_dim)
        self.query = mx.zeros(nh, dtype=mx.float32)
        self.key = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.value = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.post = Einsum("bnh,dnh->bd", (model_dim, *nh), (model_dim,))
        self.ln = LayerNorm(model_dim)

    def __call__(
        self, x: mx.array, attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        # x: [B, S, D] -> [B, D]. No logit scale or cap here; the stored
        # query constant already includes any scaling from training.
        logits = mx.einsum("nh,bsnh->bns", self.query, self.key(x))
        if attn_mask is not None:
            logits = mx.where(attn_mask, logits, MASK_FILL)
        attn = mx.softmax(logits, axis=-1)
        out = mx.einsum("bns,bsnh->bnh", attn, self.value(x))
        return self.ln(self.post(out))


class AudioEncoder(nn.Module):
    """ViT over the log-mel spectrogram: [B, 992, 128] -> [B, 768].

    16x16 patches on a 62 (time) x 8 (mel) grid give 496 tokens. The patch
    projection bias and the learned position embedding are fused into
    ``pos_emb`` (the exporter constant-folded them together).
    """

    num_layers = 12

    def __init__(self):
        super().__init__()
        self.patch_proj = Einsum("...a,ab->...b", (256, 768))
        self.pos_emb = mx.zeros((496, 768), dtype=mx.float32)
        self.layers = [TransformerLayer() for _ in range(self.num_layers)]
        self.final_ln = LayerNorm(768)
        self.pooler = AttentionPooler()

    def __call__(self, mel: mx.array) -> mx.array:
        b = mel.shape[0]
        patches = (
            mel.reshape(b, 62, 16, 8, 16)
            .transpose(0, 1, 3, 2, 4)
            .reshape(b, 496, 256)
        )
        x = self.patch_proj(patches) + self.pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        return self.pooler(x)


class TextEncoder(nn.Module):
    """Padding-masked bidirectional text tower: ids -> [B, 768]."""

    num_layers = 12
    max_length = 128

    def __init__(self, vocab_size: int = 16000):
        super().__init__()
        self.token_emb = mx.zeros((vocab_size, 768), dtype=mx.float32)
        self.pos_emb = mx.zeros((self.max_length, 768), dtype=mx.float32)
        self.layers = [TransformerLayer() for _ in range(self.num_layers)]
        self.final_ln = LayerNorm(768)
        self.pooler = AttentionPooler()

    def __call__(self, ids: mx.array, paddings: mx.array) -> mx.array:
        # ids: [B, 128] int32; paddings: [B, 128] float32 (1.0 = padded).
        x = self.token_emb[ids] + self.pos_emb
        keep = paddings < 0.5
        attn_mask = keep[:, None, None, :]  # [B, 1, T, S] broadcast
        ffn_mask = (1.0 - paddings)[..., None]
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, ffn_mask=ffn_mask)
        x = self.final_ln(x)
        return self.pooler(x, attn_mask=keep[:, None, :])
