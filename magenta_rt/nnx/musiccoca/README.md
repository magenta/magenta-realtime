# `magenta_rt.nnx.musiccoca`

A pure `flax.nnx` MusicCoCa, reverse engineered from the TFLite exports
shipped with Magenta-RT. Numerically matches the TFLite models (max abs
error ~1e-5 on embeddings; RVQ tokens match exactly) and exposes the same
high-level interface as `magenta_rt.musiccoca.MusicCoCa`.

## Quick start

Convert the TFLite weights once (requires `tensorflow` for the flatbuffer
schema):

```sh
python -m magenta_rt.nnx.musiccoca.convert
```

Then:

```python
from magenta_rt.nnx.musiccoca import MusicCoCa

style_model = MusicCoCa()
embedding = style_model.embed('staccato funk')          # [768]
tokens = style_model.tokenize(embedding)                # [12] RVQ tokens
mapped = style_model.embed('staccato funk', use_mapper=True)
```

Or drive the nnx module directly (jit-able, batched):

```python
from magenta_rt.nnx.musiccoca import from_safetensors
from magenta_rt import paths

module = from_safetensors(paths.musiccoca_dir() / 'musiccoca_nnx.safetensors')
emb = module.embed_audio(waveform)        # [B, 160000] @ 16 kHz -> [B, 768]
emb = module.embed_text(ids, paddings)    # [B, 128] each -> [B, 768]
tokens = module.tokenize(emb)             # [B, 12]
```

## Architecture (as recovered from the exports)

The five TFLite files are `jax2tf` exports of Praxis modules; the tensor
names inside the flatbuffers retain the original module paths, which is
what made the reverse engineering tractable (see `convert.py`).

| Component | File | Structure |
|---|---|---|
| Log-mel frontend | `audio_preprocessor.tflite` | pre-emphasis 0.97 → 400/160 frames, periodic Hann → 2048-pt rFFT power → 128 mel bins (`log(mel + 1e-3)`) → `[992, 128]` |
| Music tower | `music_encoder.tflite` | ViT: 16×16 patches (62 time × 8 mel = 496 tokens) → Linear 256→768 + learned pos emb → 12 pre-LN layers (12 heads × 64) → final LN → attentional pooler → `[768]` |
| Text tower | `text_encoder.tflite` | 16k SentencePiece vocab, learned pos emb (128 positions), 12 layers with padding-masked bidirectional attention; FFN activations multiplied by `(1 - padding)` → pooler → `[768]` |
| Quantizer | `pretrained_vector_quantizer.tflite` | 12-stage residual VQ, codebooks `[1024, 768]`, greedy nearest neighbor |
| Mapper | `mapper.tflite` | one-step DiT-style sampler: noise `[768]` → 12×256 tokens, 8 layers (RMSNorm + adaLN conditioned on `[c, c, text_emb]`, RoPE, learned KV sink per head, `30·tanh(x/30)` cap, tanh GELU) → `[768]`, then L2 normalize |

Shared transformer details (both towers):

* attention-logit soft cap `50 · tanh(x / 50)`;
* query scale `1/sqrt(64)` folded into the Q weights at conversion time
  (the text export already had it folded; the music tower's is folded by
  `convert.py`);
* exact (erf) GELU in the FFNs; LayerNorm ε = 1e-6;
* CoCa attentional pooler: one learned query, 12 heads × 256, no logit
  scale/cap, followed by LayerNorm.

The pooled embeddings are **not** L2-normalized (matching the TFLite
behavior); only the mapper output is.

## Tests

```sh
pytest tests/nnx/test_musiccoca.py                       # unit, no resources
pytest tests/nnx/parity/test_musiccoca_tflite.py -m checkpoint  # vs TFLite
```
