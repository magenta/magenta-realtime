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

"""Dependency-light numpy implementation of the MagentaRT-2 depthformer.

This is the executable specification for the ggml/C++ port: it reads the Flax
checkpoint directly and reproduces the JAX fp32 reference to ~1e-6. Every
non-obvious semantic here was established empirically against the JAX dumps in
tests/testdata/ggml/ rather than inferred from the source:

  * RMSNorm uses eps=1e-6 and a plain `scale` (not `1 + scale`).
  * Q/K/V projections contract as `i,ihd->hd`; the output projection contracts
    the head axes as `hd,ohd->o` (both operands are square at 1024, so shape
    alone cannot disambiguate this).
  * Query scaling is Praxis-style:
    `1.442695041 * (1/sqrt(dh)) * softplus(per_dim_scale)`.
  * Attention-sink logits are computed with the **unscaled** query, while real
    keys use the scaled query. This is the single easiest thing to get wrong.
  * The FFN activation (GELU, tanh approximation) is folded into `ffn_layer1`'s
    output; `ffn_layer2` applies no activation.
  * KV caches hold `max_past_horizon + 1` frames, NOT `max_past_horizon`. The
    JAX local attention concatenates the current frame onto its horizon-sized
    buffer and attends over both, so a horizon of 41 means 42 attended
    positions. Sizing the cache at 41 evicts the oldest frame one step early,
    which is invisible until the buffer first wraps at step 41.
  * Cache order does not matter (the model is NoPE), so a plain circular buffer
    plus a validity mask is equivalent to the reference's shifting buffer.
"""

import numpy as np

EPS = 1e-6
R_SOFTPLUS_0 = 1.442695041


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def rms_norm(x: np.ndarray, scale: np.ndarray, eps: float = EPS) -> np.ndarray:
    x32 = x.astype(np.float32)
    return x32 / np.sqrt((x32 ** 2).mean(-1, keepdims=True) + eps) * scale


def layer_norm(x, scale, bias, eps: float = EPS):
    x32 = x.astype(np.float32)
    mu = x32.mean(-1, keepdims=True)
    var = x32.var(-1, keepdims=True)
    return (x32 - mu) / np.sqrt(var + eps) * scale + bias


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation (matches JAX's default `nn.gelu`)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def query_scale(per_dim_scale: np.ndarray, units_per_head: int) -> np.ndarray:
    softplus = np.log1p(np.exp(per_dim_scale.astype(np.float32)))
    return R_SOFTPLUS_0 * (1.0 / np.sqrt(units_per_head)) * softplus


def attend(q, k_cache, v_cache, valid, sink_k, sink_v, pds):
    """One decode step of attention over a right-aligned KV ring buffer.

    Args:
      q: [heads, dh] unscaled query.
      k_cache/v_cache: [window, heads, dh].
      valid: [window] bool; False slots are masked out.
      sink_k/sink_v: [heads, dh] or None.
      pds: [dh] per-dim scale.
    Returns:
      [heads, dh] context vector.
    """
    heads, dh = q.shape
    qs = q * query_scale(pds, dh)
    logits = np.einsum("hd,khd->hk", qs, k_cache)               # [h, window]
    logits = np.where(valid[None, :], logits, -np.inf)
    values = v_cache
    if sink_k is not None:
        # Sink logits deliberately use the *unscaled* query.
        sink_logit = np.einsum("hd,hd->h", q, sink_k)[:, None]  # [h, 1]
        logits = np.concatenate([sink_logit, logits], axis=1)
        values = np.concatenate([sink_v[None], v_cache], axis=0)
    p = softmax(logits, axis=-1)
    return np.einsum("hk,khd->hd", p, values)


class Cache:
    """Circular buffer of the last `slots` frames.

    Pass `slots = max_past_horizon + 1`; see the module docstring.
    """

    def __init__(self, window, heads, dh):
        self.k = np.zeros((window, heads, dh), np.float32)
        self.v = np.zeros((window, heads, dh), np.float32)
        self.n = 0
        self.window = window

    def push(self, k, v):
        self.k[:-1] = self.k[1:]
        self.v[:-1] = self.v[1:]
        self.k[-1] = k
        self.v[-1] = v
        self.n = min(self.n + 1, self.window)

    @property
    def valid(self):
        m = np.zeros(self.window, bool)
        if self.n:
            m[-self.n:] = True
        return m


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class DepthformerNumpy:
    def __init__(self, weights: dict, cfg: dict):
        self.W = weights
        self.cfg = cfg

    def w(self, path: str) -> np.ndarray:
        return self.W["params/depthformer/" + path]

    # ---- encoder ---------------------------------------------------------
    def encode(self, tokens: np.ndarray) -> np.ndarray:
        """Conditioning tokens [144] -> encoder output [256]."""
        p = "encoder/body/layers_1/"
        n_mulan = self.cfg["n_mulan"]
        mulan = self.w(p + "branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding")
        adapter = self.w(p + "branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel")
        reg = self.w(p + "branched_regular_embedder/regular_embedder/embedding")

        a = mulan[tokens[:n_mulan] + self.cfg["mulan_offsets"]].sum(axis=0) @ adapter
        b = reg[tokens[n_mulan:] + self.cfg["regular_offsets"]].astype(np.float32).mean(axis=0)
        y = (a + b) / 2.0  # sl.CombinationMode.MEAN
        return layer_norm(y, self.w("encoder/body/encoder_ln/scale"),
                          self.w("encoder/body/encoder_ln/bias"))

    # ---- attention sublayer ---------------------------------------------
    def _attention_block(self, prefix, x, kv_source, cache):
        """Pre-norm -> attention -> output projection -> post-norm -> residual."""
        pre = rms_norm(x, self.w(prefix + "pre_norm/scale"))
        a = prefix + "attention/"
        q = np.einsum("i,ihd->hd", pre, self.w(a + "query_projection/kernel"))
        if kv_source is not None:  # cross-attention projects the encoder output
            k = np.einsum("i,ihd->hd", kv_source, self.w(a + "key_projection/kernel"))
            v = np.einsum("i,ihd->hd", kv_source, self.w(a + "value_projection/kernel"))
            cache.push(k, v)
        else:
            k = np.einsum("i,ihd->hd", pre, self.w(a + "key_projection/kernel"))
            v = np.einsum("i,ihd->hd", pre, self.w(a + "value_projection/kernel"))
            cache.push(k, v)

        sink_key = "params/depthformer/" + a + "sink_key_embeddings"
        if sink_key in self.W:
            sink_k = self.w(a + "sink_key_embeddings")[0]
            sink_v = self.w(a + "sink_value_embeddings")[0]
        else:
            sink_k = sink_v = None

        ctx = attend(q, cache.k, cache.v, cache.valid, sink_k, sink_v,
                     self.w(a + "per_dim_scale"))
        o = np.einsum("hd,ohd->o", ctx, self.w(prefix + "output_projection/kernel"))
        return x + rms_norm(o, self.w(prefix + "post_norm/scale"))

    def _ffn_block(self, prefix, x):
        pre = rms_norm(x, self.w(prefix + "pre_norm/scale"))
        h = gelu(pre @ self.w(prefix + "ffn_layer1/kernel") + self.w(prefix + "ffn_layer1/bias"))
        o = h @ self.w(prefix + "ffn_layer2/kernel") + self.w(prefix + "ffn_layer2/bias")
        return x + rms_norm(o, self.w(prefix + "post_norm/scale"))

    # ---- temporal body ---------------------------------------------------
    def temporal_step(self, x, enc_out, caches):
        for i in range(self.cfg["t_layers"]):
            p = f"decoder/temporal_body/transformer/x_layers_{i}/"
            x = self._attention_block(p + "self_attention/", x, None, caches["self"][i])
            x = self._attention_block(p + "cross_attention/", x, enc_out, caches["cross"][i])
            x = self._ffn_block(p + "ffn/", x)
        return x

    # ---- depth body ------------------------------------------------------
    def depth_step(self, depth_input, caches):
        """One RVQ step: [1024] -> logits [vocab] (pre soft-cap)."""
        p = "decoder/depth_body/"
        x = depth_input @ self.w(p + "depth_input_adapter/kernel")
        for i in range(self.cfg["d_layers"]):
            q = f"{p}transformer/x_layers_{i}/"
            x = self._attention_block(q + "self_attention/", x, None, caches[i])
            x = self._ffn_block(q + "ffn/", x)
        x = layer_norm(x, self.w(p + "final_ln/scale"), self.w(p + "final_ln/bias"))
        return x @ self.w(p + "to_logits/kernel") + self.w(p + "to_logits/bias")

    def embed_token(self, token: int) -> np.ndarray:
        """Shared decoder embedding with the sqrt(D) scale folded in."""
        emb = self.w("decoder/decoder_embedding/embedding/embedding")
        return emb[token] * np.sqrt(self.cfg["t_model_dims"])


def soft_cap(logits: np.ndarray, cap: float = 30.0) -> np.ndarray:
    return np.tanh(logits / cap) * cap


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def large_negative(dtype=np.float32) -> float:
    """Matches depthformer.get_large_negative_number: -0.7 * finfo.max."""
    return float(np.asarray(-0.7 * np.finfo(dtype).max, dtype=dtype))


def apply_cfg(logits: np.ndarray, scales) -> np.ndarray:
    """Classifier-free guidance.

    `logits` is [arity, vocab] laid out as [fully_conditional, negative_1, ...];
    returns [vocab]. The update is
        out = cond + sum_i scale_i * (cond - negative_i)
    matching _sample_categorical_with_temperature.
    """
    cond = logits[0]
    out = cond.astype(np.float32).copy()
    for i, scale in enumerate(scales, start=1):
        out += scale * (cond - logits[i])
    return out


def filter_logits(logits, valid_range=None, top_k=None):
    """Deterministic part of sampling: valid-range mask then top-k mask."""
    out = logits.astype(np.float32).copy()
    neg = large_negative()
    if valid_range is not None:
        idx = np.arange(out.shape[-1])
        out = np.where((idx >= valid_range[0]) & (idx < valid_range[1]), out, neg)
    if top_k is not None:
        k = int(np.clip(top_k, 1, out.shape[-1]))
        kth = np.sort(out, axis=-1)[..., -k]
        out = np.where(out >= kth, out, neg)
    return out


def sample_token(logits, rvq_index, cfg, rng: np.random.Generator):
    """Gumbel-max sampling for one RVQ step.

    Note the temperature scales the *noise* rather than dividing the logits,
    matching the JAX implementation:  logits += gumbel * temperature.

    Bit-exact agreement with JAX is not attainable here (different PRNG), so
    parity testing uses teacher forcing, as tests/test_bitlevel_parity.py does.
    """
    lo = cfg["num_reserved_tokens"] + rvq_index * cfg["codebook_size"]
    hi = lo + cfg["codebook_size"]
    filtered = filter_logits(logits, valid_range=(lo, hi), top_k=cfg.get("top_k"))
    u = rng.random(filtered.shape, dtype=np.float32)
    gumbel = -np.log(-np.log(np.clip(u, 1e-20, 1.0)))
    return int(np.argmax(filtered + gumbel * cfg.get("temperature", 1.0)))
