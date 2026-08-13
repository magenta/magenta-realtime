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

"""Flax NNX building blocks for MusicCoCa, reverse engineered from the
TFLite exports (see ``convert.py`` for provenance).

Both towers share the same 12-layer pre-LN transformer with:

* separate Q/K/V projections in Praxis ``DNH`` layout, with the
  ``1/sqrt(head_dim)`` query scale folded into the Q weights at
  conversion time,
* attention-logit soft capping at ``50 * tanh(x / 50)``,
* exact (erf) GELU FFNs,
* a CoCa-style attentional pooler (one learned query, 12 heads x 256)
  followed by a LayerNorm that produces the 768-dim style embedding.

The text tower additionally applies a padding mask both to attention
logits and (multiplicatively) to the FFN hidden/output activations,
matching the exported graph.
"""

from __future__ import annotations

from typing import Optional

import einops
import jax
import jax.numpy as jnp
from flax import nnx

from .frontend import NUM_FRAMES

# Fill value the exporter uses for masked attention logits.
MASK_FILL = -2.3819763e38

# Soft cap applied to attention logits in both towers.
ATTN_LOGIT_CAP = 50.0


class Einsum(nnx.Module):
    """A single einsum projection with optional bias (Praxis-style)."""

    def __init__(
        self,
        equation: str,
        kernel_shape: tuple[int, ...],
        bias_shape: Optional[tuple[int, ...]] = None,
    ):
        self.equation = equation
        self.kernel = nnx.Param(jnp.zeros(kernel_shape, jnp.float32))
        self.bias = (
            nnx.Param(jnp.zeros(bias_shape, jnp.float32))
            if bias_shape is not None
            else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.einsum(self.equation, x, self.kernel[...])
        if self.bias is not None:
            y = y + self.bias[...]
        return y


class LayerNorm(nnx.Module):
    """LayerNorm with direct scale/bias (Praxis ``+1`` baked at conversion)."""

    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.scale = nnx.Param(jnp.ones((dim,), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((dim,), jnp.float32))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        normed = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return normed * self.scale[...] + self.bias[...]


class TransformerLayer(nnx.Module):
    """Pre-LN encoder layer shared by the music and text towers."""

    def __init__(
        self,
        *,
        model_dim: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        ffn_dim: int = 3072,
    ):
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
        x: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray] = None,
        ffn_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        h = self.ln1(x)
        # Query scale (1/sqrt(head_dim)) is folded into self.q's weights.
        logits = jnp.einsum("btnh,bsnh->bnts", self.q(h), self.k(h))
        logits = ATTN_LOGIT_CAP * jnp.tanh(logits / ATTN_LOGIT_CAP)
        if attn_mask is not None:
            logits = jnp.where(attn_mask, logits, MASK_FILL)
        attn = nnx.softmax(logits, axis=-1)
        out = jnp.einsum("bnts,bsnh->btnh", attn, self.v(h))
        x = x + self.post(out)

        h = self.ln2(x)
        h = nnx.gelu(self.ffn1(h), approximate=False)
        if ffn_mask is not None:
            h = h * ffn_mask
        h = self.ffn2(h)
        if ffn_mask is not None:
            h = h * ffn_mask
        return x + h


class AttentionPooler(nnx.Module):
    """CoCa attentional pooler: one learned query over the sequence."""

    def __init__(
        self,
        *,
        model_dim: int = 768,
        num_heads: int = 12,
        head_dim: int = 256,
    ):
        nh = (num_heads, head_dim)
        self.query = nnx.Param(jnp.zeros(nh, jnp.float32))
        self.key = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.value = Einsum("...d,dnh->...nh", (model_dim, *nh), nh)
        self.post = Einsum("bnh,dnh->bd", (model_dim, *nh), (model_dim,))
        self.ln = LayerNorm(model_dim)

    def __call__(
        self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        # x: [B, S, D] -> [B, D]. No logit scale or cap here; the stored
        # query constant already includes any scaling from training.
        logits = jnp.einsum("nh,bsnh->bns", self.query[...], self.key(x))
        if attn_mask is not None:
            logits = jnp.where(attn_mask, logits, MASK_FILL)
        attn = nnx.softmax(logits, axis=-1)
        out = jnp.einsum("bns,bsnh->bnh", attn, self.value(x))
        return self.ln(self.post(out))


def _stacked_layers(num_layers: int) -> TransformerLayer:
    """Builds ``num_layers`` identical layers with stacked (vmapped) params."""

    @nnx.vmap(in_axes=0, out_axes=0)
    def make(_) -> TransformerLayer:
        return TransformerLayer()

    return make(jnp.arange(num_layers))


def _run_layers(
    layers: TransformerLayer,
    num_layers: int,
    x: jnp.ndarray,
    attn_mask: Optional[jnp.ndarray] = None,
    ffn_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    @nnx.scan(length=num_layers, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward(h, layer):
        return layer(h, attn_mask=attn_mask, ffn_mask=ffn_mask)

    return forward(x, layers)


class AudioEncoder(nnx.Module):
    """ViT over the log-mel spectrogram: [B, 992, 128] -> [B, 768].

    16x16 patches on a 62 (time) x 8 (mel) grid give 496 tokens. The patch
    projection bias and the learned position embedding are fused into
    ``pos_emb`` (the exporter constant-folded them together).
    """

    num_layers = 12

    def __init__(self):
        self.patch_proj = Einsum("...a,ab->...b", (256, 768))
        self.pos_emb = nnx.Param(jnp.zeros((496, 768), jnp.float32))
        self.layers = _stacked_layers(self.num_layers)
        self.final_ln = LayerNorm(768)
        self.pooler = AttentionPooler()

    def __call__(self, mel: jnp.ndarray) -> jnp.ndarray:
        # 16x16 patches on a 62 (time) x 8 (mel) grid -> 496 tokens of 256:
        # split time 992 = tp(62)*ti(16) and mel 128 = mp(8)*mi(16), then group
        # patch index as (tp mp) and patch content as (ti mi).
        patches = einops.rearrange(
            mel, "b (tp ti) (mp mi) -> b (tp mp) (ti mi)", ti=16, mp=8, mi=16
        )
        x = self.patch_proj(patches) + self.pos_emb[...]
        x = _run_layers(self.layers, self.num_layers, x)
        x = self.final_ln(x)
        return self.pooler(x)

    def encode_windows(
        self,
        mel: jnp.ndarray,
        starts: jnp.ndarray,
        *,
        scan: bool = False,
        subbatch: Optional[int] = None,
    ) -> jnp.ndarray:
        """Long log-mel ``[B, T, 128]`` → per-window embeddings ``[B, N, 768]``.

        Window ``i`` is the ``NUM_FRAMES`` (992) mel frames starting at
        ``starts[i]`` (a mel-frame index), encoded by the same ViT used for a
        standalone clip — sharing the one log-mel computation across all
        overlapping windows. The caller guarantees ``starts[i] + NUM_FRAMES``
        stays within ``T`` (no out-of-range slice).

        ``scan=True`` walks the windows with ``nnx.scan`` (one window resident at
        a time — bounded peak memory for a very large spectrogram, at the cost of
        sequential execution). ``scan=False`` (default) encodes all windows as a
        single batch; ``subbatch`` chunks that batch along the window axis to cap
        memory without giving up intra-chunk parallelism.
        """
        starts = jnp.asarray(starts)
        idx = starts[:, None] + jnp.arange(NUM_FRAMES)[None, :]  # [N, 992]
        windows = mel[:, idx, :]                                 # [B, N, 992, 128]
        b, n = windows.shape[:2]

        if scan:
            # Module broadcast (in_axes None), windows scanned on axis 1.
            @nnx.scan(in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1))
            def run(carry, encoder, window):   # window: [B, 992, 128]
                return carry, encoder(window)

            _, embs = run((), self, windows)
            return embs                        # [B, N, 768]

        flat = windows.reshape(b * n, *windows.shape[2:])
        if subbatch is None:
            emb = self(flat)
        else:
            emb = jnp.concatenate(
                [self(flat[i:i + subbatch]) for i in range(0, b * n, subbatch)],
                axis=0,
            )
        return emb.reshape(b, n, -1)


class TextEncoder(nnx.Module):
    """Padding-masked bidirectional text tower: ids -> [B, 768]."""

    num_layers = 12
    max_length = 128

    def __init__(self, vocab_size: int = 16000):
        self.token_emb = nnx.Param(jnp.zeros((vocab_size, 768), jnp.float32))
        self.pos_emb = nnx.Param(
            jnp.zeros((self.max_length, 768), jnp.float32)
        )
        self.layers = _stacked_layers(self.num_layers)
        self.final_ln = LayerNorm(768)
        self.pooler = AttentionPooler()

    def __call__(
        self, ids: jnp.ndarray, paddings: jnp.ndarray
    ) -> jnp.ndarray:
        # ids: [B, 128] int32; paddings: [B, 128] float32 (1.0 = padded).
        x = self.token_emb[...][ids] + self.pos_emb[...]
        keep = paddings < 0.5
        attn_mask = keep[:, None, None, :]  # [B, 1, T, S] broadcast
        ffn_mask = (1.0 - paddings)[..., None]
        x = _run_layers(
            self.layers, self.num_layers, x,
            attn_mask=attn_mask, ffn_mask=ffn_mask,
        )
        x = self.final_ln(x)
        return self.pooler(x, attn_mask=keep[:, None, :])
