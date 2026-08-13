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

"""MusicCoCa for the jax backend — functional ground-truth implementation.

Unlike the depthformer/codec stack (flax.linen + sequence_layers),
MusicCoCa is a stateless embedder, so this module is plain ``jax.numpy``
functions over a params pytree: every component is a pure function, the
12-layer towers and the 8-layer mapper run as ``jax.lax.scan`` over their
stacked per-layer weights (mirroring the original Praxis repeat layer),
and the public entry points jit cleanly with ``params`` as a pytree
argument.

Weights come from the safetensors file produced by
``python -m magenta_rt.nnx.musiccoca.convert`` — shared with the nnx and
mlx_pure backends. See ``magenta_rt/nnx/musiccoca/README.md`` for the
architecture recovered from the TFLite exports.

Example:

    from magenta_rt.jax import musiccoca
    style_model = musiccoca.MusicCoCa()
    tokens = style_model.tokenize(style_model.embed_text('staccato funk'))
"""

from __future__ import annotations

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from .. import musiccoca as musiccoca_base
from .. import paths

# Filename shared with the nnx backend's converter.
WEIGHTS_FILENAME = "musiccoca_nnx.safetensors"

# Fill value the exporter uses for masked attention logits.
MASK_FILL = -2.3819763e38
# Soft cap applied to tower attention logits.
ATTN_LOGIT_CAP = 50.0

# Frontend constants (see nnx.musiccoca.frontend).
SAMPLE_RATE = 16_000
CLIP_SAMPLES = 160_000
FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_LENGTH = 2048
NUM_FRAMES = 992
PREEMPHASIS = 0.97
MEL_FLOOR = 1e-3

# Text constants.
MAX_TEXT_LENGTH = 128
TEXT_SOS_ID = 1

# Mapper constants.
MAPPER_TOKENS = 12
MAPPER_DIM = 256
MAPPER_HEADS = 8
MAPPER_HEAD_DIM = 64
MAPPER_LOGIT_CAP = 30.0
ROPE_MAX_TIMESCALE = 10_000.0

Params = dict


# -----------------------------------------------------------------------------
# Parameter loading / initialization
# -----------------------------------------------------------------------------


def load_params(path: str | pathlib.Path) -> Params:
    """Loads the shared converter safetensors into a nested params dict."""
    import safetensors.numpy

    flat = safetensors.numpy.load_file(str(path))
    return traverse_util.unflatten_dict(
        {tuple(k.split(".")): jnp.asarray(v) for k, v in flat.items()}
    )


def init_params() -> Params:
    """Zero-initialized params with the checkpoint's schema (for tests)."""
    shapes: dict[str, tuple[int, ...]] = {
        "frontend.mel_matrix": (1025, 128),
        "frontend.window": (400,),
        "quantizer.codebooks": (12, 1024, 768),
        "mapper.context": (128,),
        "mapper.input_proj.kernel": (768, 3072),
        "mapper.input_proj.bias": (3072,),
        "mapper.output_proj.kernel": (3072, 768),
        "mapper.output_proj.bias": (768,),
        "audio.patch_proj.kernel": (256, 768),
        "audio.pos_emb": (496, 768),
        "text.token_emb": (16000, 768),
        "text.pos_emb": (128, 768),
    }
    for tower in ("audio", "text"):
        for name in ("q", "k", "v", "post"):
            shapes[f"{tower}.layers.{name}.kernel"] = (12, 768, 12, 64)
        for name in ("q", "k", "v"):
            shapes[f"{tower}.layers.{name}.bias"] = (12, 12, 64)
        shapes[f"{tower}.layers.post.bias"] = (12, 768)
        for name in ("ln1", "ln2"):
            shapes[f"{tower}.layers.{name}.scale"] = (12, 768)
            shapes[f"{tower}.layers.{name}.bias"] = (12, 768)
        shapes[f"{tower}.layers.ffn1.kernel"] = (12, 768, 3072)
        shapes[f"{tower}.layers.ffn1.bias"] = (12, 3072)
        shapes[f"{tower}.layers.ffn2.kernel"] = (12, 3072, 768)
        shapes[f"{tower}.layers.ffn2.bias"] = (12, 768)
        shapes[f"{tower}.final_ln.scale"] = (768,)
        shapes[f"{tower}.final_ln.bias"] = (768,)
        shapes[f"{tower}.pooler.query"] = (12, 256)
        for name in ("key", "value"):
            shapes[f"{tower}.pooler.{name}.kernel"] = (768, 12, 256)
            shapes[f"{tower}.pooler.{name}.bias"] = (12, 256)
        shapes[f"{tower}.pooler.post.kernel"] = (768, 12, 256)
        shapes[f"{tower}.pooler.post.bias"] = (768,)
        shapes[f"{tower}.pooler.ln.scale"] = (768,)
        shapes[f"{tower}.pooler.ln.bias"] = (768,)
    for name, shape in {
        "norm1.scale": (256,),
        "norm2.scale": (256,),
        "cond1.kernel": (1024, 2, 256),
        "cond1.bias": (2, 256),
        "cond2.kernel": (1024, 2, 256),
        "cond2.bias": (2, 256),
        "qkv.kernel": (256, 3, 8, 64),
        "k_sink": (1, 8, 64),
        "v_sink": (1, 8, 64),
        "post.kernel": (256, 8, 64),
        "ffn1.kernel": (256, 1024),
        "ffn1.bias": (1024,),
        "ffn2.kernel": (1024, 256),
        "ffn2.bias": (256,),
    }.items():
        shapes[f"mapper.layers.{name}"] = (8, *shape)
    return traverse_util.unflatten_dict(
        {tuple(k.split(".")): jnp.zeros(v, jnp.float32) for k, v in shapes.items()}
    )


# -----------------------------------------------------------------------------
# Components (pure functions)
# -----------------------------------------------------------------------------


def log_mel_spectrogram(p: Params, waveform: jnp.ndarray) -> jnp.ndarray:
    """Waveform ``[B, 160000]`` → log-mel features ``[B, 992, 128]``."""
    x = waveform.astype(jnp.float32)
    shifted = jnp.pad(x, ((0, 0), (1, 0)))[:, :-1]
    x = x - PREEMPHASIS * shifted

    num_frames = (x.shape[-1] - FRAME_LENGTH) // FRAME_HOP + 1
    starts = jnp.arange(num_frames) * FRAME_HOP
    idx = starts[:, None] + jnp.arange(FRAME_LENGTH)[None, :]
    frames = x[:, idx] * p["window"]

    spectrum = jnp.fft.rfft(frames, n=FFT_LENGTH, axis=-1)
    power = jnp.square(jnp.abs(spectrum))
    mel = power @ p["mel_matrix"] + MEL_FLOOR
    return jnp.log(mel)[:, :NUM_FRAMES]


def _layer_norm(p: Params, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * p["scale"] + p["bias"]


def _transformer_layer(p, x, attn_mask=None, ffn_mask=None):
    h = _layer_norm(p["ln1"], x)
    # Query scale (1/sqrt(head_dim)) is folded into the q weights.
    q = jnp.einsum("btd,dnh->btnh", h, p["q"]["kernel"]) + p["q"]["bias"]
    k = jnp.einsum("btd,dnh->btnh", h, p["k"]["kernel"]) + p["k"]["bias"]
    v = jnp.einsum("btd,dnh->btnh", h, p["v"]["kernel"]) + p["v"]["bias"]
    logits = jnp.einsum("btnh,bsnh->bnts", q, k)
    logits = ATTN_LOGIT_CAP * jnp.tanh(logits / ATTN_LOGIT_CAP)
    if attn_mask is not None:
        logits = jnp.where(attn_mask, logits, MASK_FILL)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bnts,bsnh->btnh", attn, v)
    out = jnp.einsum("btnh,dnh->btd", out, p["post"]["kernel"])
    x = x + out + p["post"]["bias"]

    h = _layer_norm(p["ln2"], x)
    h = jax.nn.gelu(h @ p["ffn1"]["kernel"] + p["ffn1"]["bias"], approximate=False)
    if ffn_mask is not None:
        h = h * ffn_mask
    h = h @ p["ffn2"]["kernel"] + p["ffn2"]["bias"]
    if ffn_mask is not None:
        h = h * ffn_mask
    return x + h


def _attention_pooler(p, x, attn_mask=None):
    # x: [B, S, D] -> [B, D]. No logit scale/cap; the stored query
    # constant already includes any scaling from training.
    k = jnp.einsum("bsd,dnh->bsnh", x, p["key"]["kernel"]) + p["key"]["bias"]
    v = jnp.einsum("bsd,dnh->bsnh", x, p["value"]["kernel"]) + p["value"]["bias"]
    logits = jnp.einsum("nh,bsnh->bns", p["query"], k)
    if attn_mask is not None:
        logits = jnp.where(attn_mask, logits, MASK_FILL)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bns,bsnh->bnh", attn, v)
    out = jnp.einsum("bnh,dnh->bd", out, p["post"]["kernel"]) + p["post"]["bias"]
    return _layer_norm(p["ln"], out)


def _run_tower(p, x, attn_mask=None, ffn_mask=None, pool_mask=None):
    def body(h, layer_p):
        return _transformer_layer(layer_p, h, attn_mask, ffn_mask), None

    x, _ = jax.lax.scan(body, x, p["layers"])
    x = _layer_norm(p["final_ln"], x)
    return _attention_pooler(p["pooler"], x, pool_mask)


def encode_audio(p: Params, mel: jnp.ndarray) -> jnp.ndarray:
    """Log-mel ``[B, 992, 128]`` → embeddings ``[B, 768]``."""
    b = mel.shape[0]
    patches = (
        mel.reshape(b, 62, 16, 8, 16)
        .transpose(0, 1, 3, 2, 4)
        .reshape(b, 496, 256)
    )
    # Patch-projection bias and position embedding are fused in pos_emb.
    x = patches @ p["patch_proj"]["kernel"] + p["pos_emb"]
    return _run_tower(p, x)


def embed_audio(params: Params, waveform: jnp.ndarray) -> jnp.ndarray:
    """Mono clips ``[B, 160000]`` @ 16 kHz → embeddings ``[B, 768]``."""
    return encode_audio(params["audio"], log_mel_spectrogram(params["frontend"], waveform))


def embed_text(
    params: Params, ids: jnp.ndarray, paddings: jnp.ndarray
) -> jnp.ndarray:
    """Token ids/paddings ``[B, 128]`` → embeddings ``[B, 768]``."""
    p = params["text"]
    x = p["token_emb"][ids] + p["pos_emb"]
    keep = paddings < 0.5
    return _run_tower(
        p,
        x,
        attn_mask=keep[:, None, None, :],
        ffn_mask=(1.0 - paddings)[..., None],
        pool_mask=keep[:, None, :],
    )


def tokenize(params: Params, embeddings: jnp.ndarray) -> jnp.ndarray:
    """``[..., 768]`` → ``[..., 12]`` int32 RVQ tokens."""
    flat = embeddings.reshape(-1, embeddings.shape[-1])

    def body(residual, codebook):
        distances = (
            jnp.sum(jnp.square(residual), axis=-1, keepdims=True)
            - 2.0 * residual @ codebook.T
            + jnp.sum(jnp.square(codebook), axis=-1)
        )
        idx = jnp.argmin(distances, axis=-1)
        return residual - codebook[idx], idx

    _, tokens = jax.lax.scan(body, flat, params["quantizer"]["codebooks"])
    tokens = jnp.transpose(tokens).astype(jnp.int32)  # [M, 12]
    return tokens.reshape(*embeddings.shape[:-1], -1)


def decode_tokens(params: Params, tokens: jnp.ndarray) -> jnp.ndarray:
    """``[..., 12]`` tokens → ``[..., 768]`` reconstructed embedding."""
    codebooks = params["quantizer"]["codebooks"]
    flat = tokens.reshape(-1, codebooks.shape[0])
    out = jnp.take_along_axis(
        codebooks, flat.T[:, :, None], axis=1
    ).sum(axis=0)
    return out.reshape(*tokens.shape[:-1], -1)


def _rope(x: jnp.ndarray) -> jnp.ndarray:
    """Half-split RoPE over positions 0..T-1; x is [B, T, N, H]."""
    half = x.shape[-1] // 2
    positions = jnp.arange(x.shape[1], dtype=jnp.float32)
    inv_timescale = ROPE_MAX_TIMESCALE ** (
        -jnp.arange(half, dtype=jnp.float32) / half
    )
    angle = positions[:, None] * inv_timescale[None, :]
    sin = jnp.sin(angle)[None, :, None, :]
    cos = jnp.cos(angle)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _rms_norm(scale, x, eps=1e-6):
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * scale


def _ada_ln(h, cond):
    # cond: [B, 1, 2, D] -> scale (offset from 1) and shift.
    return h * (cond[:, :, 0] + 1.0) + cond[:, :, 1]


def _mapper_soft_cap(x):
    return MAPPER_LOGIT_CAP * jnp.tanh(x / MAPPER_LOGIT_CAP)


def _mapper_layer(p, x, ctx):
    b = x.shape[0]
    cond = jnp.einsum("bsc,ckd->bskd", ctx, p["cond1"]["kernel"]) + p["cond1"]["bias"]
    h = _ada_ln(_rms_norm(p["norm1"]["scale"], x), cond)
    qkv = jnp.einsum("btd,dknh->btknh", h, p["qkv"]["kernel"])
    q = _rope(qkv[:, :, 0])
    k = _rope(qkv[:, :, 1])
    v = qkv[:, :, 2]
    # The sink logits use the unscaled query (export quirk).
    sink_logits = jnp.einsum("btnh,snh->bnts", q, p["k_sink"])
    logits = jnp.einsum(
        "btnh,bsnh->bnts", q / jnp.sqrt(float(MAPPER_HEAD_DIM)), k
    )
    logits = jnp.concatenate(
        [_mapper_soft_cap(sink_logits), _mapper_soft_cap(logits)], axis=-1
    )
    attn = jax.nn.softmax(logits, axis=-1)
    v_full = jnp.concatenate(
        [jnp.broadcast_to(p["v_sink"], (b, 1, MAPPER_HEADS, MAPPER_HEAD_DIM)), v],
        axis=1,
    )
    out = jnp.einsum("bnts,bsnh->btnh", attn, v_full)
    x = x + jnp.einsum("btnh,dnh->btd", out, p["post"]["kernel"])

    cond = jnp.einsum("bsc,ckd->bskd", ctx, p["cond2"]["kernel"]) + p["cond2"]["bias"]
    h = _ada_ln(_rms_norm(p["norm2"]["scale"], x), cond)
    h = jax.nn.gelu(h @ p["ffn1"]["kernel"] + p["ffn1"]["bias"], approximate=True)
    return x + h @ p["ffn2"]["kernel"] + p["ffn2"]["bias"]


def map_text_embedding(
    params: Params, text_emb: jnp.ndarray, noise: jnp.ndarray
) -> jnp.ndarray:
    """Maps text embeddings toward audio space; L2-normalized."""
    p = params["mapper"]
    b = text_emb.shape[0]
    prefix = jnp.broadcast_to(p["context"], (b, 128))
    ctx = jnp.concatenate([prefix, prefix, text_emb], axis=-1)[:, None]
    x = noise @ p["input_proj"]["kernel"] + p["input_proj"]["bias"]
    x = x.reshape(b, MAPPER_TOKENS, MAPPER_DIM)

    def body(h, layer_p):
        return _mapper_layer(layer_p, h, ctx), None

    x, _ = jax.lax.scan(body, x, p["layers"])
    x = x.reshape(b, MAPPER_TOKENS * MAPPER_DIM)
    mapped = x @ p["output_proj"]["kernel"] + p["output_proj"]["bias"]
    return mapped / jnp.linalg.norm(mapped, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Text preprocessing + high-level system class
# -----------------------------------------------------------------------------


def encode_text(vocab, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Lowercased SentencePiece ids + paddings, shaped ``[128]`` each."""
    labels = vocab.EncodeAsIds(text.lower())[: MAX_TEXT_LENGTH - 1]
    ids = [TEXT_SOS_ID] + labels
    num_tokens = len(ids)
    ids = ids + [0] * (MAX_TEXT_LENGTH - num_tokens)
    paddings = np.ones(MAX_TEXT_LENGTH, dtype=np.float32)
    paddings[:num_tokens] = 0.0
    return np.array(ids, dtype=np.int32), paddings


class MusicCoCa(musiccoca_base.MusicCoCaBase):
    """High-level MusicCoCa backed by the functional jax implementation.

    Same interface and resource directory as the TFLite-backed
    :class:`magenta_rt.musiccoca.MusicCoCa`; expects ``spm.model`` and the
    safetensors produced by ``python -m magenta_rt.nnx.musiccoca.convert``.
    """

    def __init__(
        self,
        resource_dir: str | pathlib.Path | None = None,
        lazy: bool = True,
    ):
        super().__init__(
            musiccoca_base.MusicCoCaConfiguration(
                sample_rate=16000,
                clip_length=10.0,
                embedding_dim=768,
                rvq_depth=12,
                rvq_codebook_size=1024,
            )
        )
        self._resource_dir = pathlib.Path(resource_dir or paths.musiccoca_dir())
        self._embed_audio = jax.jit(embed_audio)
        self._embed_text = jax.jit(embed_text)
        self._tokenize = jax.jit(tokenize)
        self._map_text = jax.jit(map_text_embedding)
        if not lazy:
            self._vocab  # pylint: disable=pointless-statement
            self._params  # pylint: disable=pointless-statement

    @functools.cached_property
    def _vocab(self):
        import sentencepiece

        spm_path = self._resource_dir / "spm.model"
        if not spm_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {spm_path}")
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load(str(spm_path))
        return sp

    @functools.cached_property
    def _params(self) -> Params:
        path = self._resource_dir / WEIGHTS_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `python -m"
                " magenta_rt.nnx.musiccoca.convert` to create it from the"
                " TFLite resources."
            )
        return load_params(path)

    def _embed_batch_text(
        self,
        batch_text: list[str],
        use_mapper: bool = False,
        seed: int = 0,
    ) -> np.ndarray:
        encoded = [encode_text(self._vocab, s) for s in batch_text]
        ids = np.stack([e[0] for e in encoded])
        paddings = np.stack([e[1] for e in encoded])
        emb = np.asarray(self._embed_text(self._params, ids, paddings))
        if use_mapper:
            # Matches the TFLite wrapper: one fixed-seed Gaussian noise
            # vector shared across the batch.
            rng = np.random.RandomState(seed)
            noise = rng.randn(self.config.embedding_dim).astype(np.float32)
            noise = np.broadcast_to(noise, emb.shape)
            emb = np.asarray(self._map_text(self._params, emb, noise))
        return emb.astype(np.float32)

    def _embed_batch_clips(
        self,
        batch_clips: np.ndarray,
    ) -> np.ndarray:
        clips = np.asarray(batch_clips, dtype=np.float32)
        if clips.shape[-1] != CLIP_SAMPLES:
            pad = CLIP_SAMPLES - clips.shape[-1]
            if pad < 0:
                clips = clips[..., :CLIP_SAMPLES]
            else:
                clips = np.pad(clips, ((0, 0), (0, pad)))
        emb = self._embed_audio(self._params, clips)
        return np.asarray(emb, dtype=np.float32)

    def tokenize(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        if embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"Embedding dimension must be {self.config.embedding_dim},"
                f" got {embeddings.shape[-1]}."
            )
        tokens = self._tokenize(
            self._params, np.asarray(embeddings, dtype=np.float32)
        )
        return np.asarray(tokens, dtype=np.int32)
