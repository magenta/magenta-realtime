# Models

Magenta RealTime 2 comprises three components:
1. [MusicCoCa](https://arxiv.org/abs/2508.04651): a text / audio style embedding model
2. [SpectroStream](https://arxiv.org/abs/2508.05207): a codec model for audio encoding and decoding
3. [Depthformer](https://deepmind.google/blog/pushing-the-frontiers-of-audio-generation/): a transformer-based model that generates SpectroStream tokens

## Hardware requirements

MRT2 offers two model sizes:

- **`mrt2_small`** (230M parameters) — runs real-time on any Apple Silicon Mac, including Air models.
- **`mrt2_base`** (2.4B parameters) — higher quality; requires a Pro/Max chip for real-time streaming.

The table below shows which devices support **real-time streaming** (generating audio faster than playback):

| Device | `mrt2_small` (230M) | `mrt2_base` (2.4B) |
|---|---|---|
| M5 Max | ✅ | ✅ |
| M3 Max | ✅ | ✅ |
| M2 Max | ✅ | ✅ |
| M4 Pro | ✅ | ✅ |
| M2 Pro | ✅ | ❌ |
| M1 Pro | ✅ | ❌ |
| M4 Air | ✅ | ❌ |
| M3 Air | ✅ | ❌ |
| M1 Air | ✅ | ❌ |

## Download models

Use the `mrt models` CLI to fetch models (automatically saved in `~/Documents/Magenta/magenta-rt-v2/`)

```bash
# Download resource models:
# MusicCoCa and SpectroStream
mrt models init

# Download Depthformer models
# exported in mlxfn format
mrt models download
```

Expected directory layout:

```
~/Documents/Magenta/magenta-rt-v2/
├── resources/
│   ├── musiccoca/
│   │   ├── audio_preprocessor.tflite
│   │   ├── music_encoder.tflite
│   │   ├── pretrained_vector_quantizer.tflite
│   │   ├── spm.model
│   │   └── text_encoder.tflite
│   └── spectrostream/
│       ├── spectrostream_encoder.mlxfn
│       ├── decoder.safetensors
│       ├── encoder.safetensors
│       └── quantizer.safetensors
├── models/
│   └── <model_name>/
│       ├── <model_name>.mlxfn
│       └── <model_name>_state.safetensors
└── checkpoints/
    └── <model_name>.safetensors
```

### Reusing the HuggingFace cache

Assets are loaded with `~/Documents/Magenta/magenta-rt-v2/` taking precedence, then
the standard HuggingFace cache (`~/.cache/huggingface/hub`) as a fallback. If you have
already pulled the repo with `hf download google/magenta-realtime-2`, those files are
reused automatically at load time — no re-download — so you don't have to run the
`mrt` download commands at all.

To have the `mrt` CLI populate (and reuse) that shared cache instead of the
`magenta-rt-v2/` folder, pass `--use-hf-cache` (or set `MAGENTA_RT_USE_HF_CACHE=1`):

```bash
mrt models init --use-hf-cache
mrt models download --use-hf-cache
```

The default behavior is unchanged (downloads into `magenta-rt-v2/`). The flag applies to
the HuggingFace source only; the GCS source always downloads into the local folder.
`MAGENTA_HOME` (and `--download-path`) still override locations, and outputs, locally
exported `.mlxfn` models, and GCS downloads always live under `magenta-rt-v2/`.

## Download raw checkpoints

One may find raw model checkpoints (safetensors before exporting to mlxfn) useful for research purposes. Thus we offer them via `mrt checkpoints download` CLI.
