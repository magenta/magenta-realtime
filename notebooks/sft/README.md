# SFT quick guide

Fine-tune MRT2 on your own music with LoRA/DoRA, on Apple Silicon
(MLX) or JAX (NNX). This page is the whole happy path — **export → fine-tune →
listen → use**. For the internals (data pipeline, CFG conditioning, freeze,
checkpoint conversion) see the [package README](../../magenta_rt/sft/README.md).

## What you need

- A pretrained checkpoint, e.g. `checkpoints/mrt2_small.safetensors`.
- A directory of audio to fine-tune toward (any common format).
- Install this repo so `magenta_rt` (and the `mrt` command) resolve —
  `pip install -e ".[mlx,sft]"` (use `[jax,sft]` for the NNX trainer). Run the
  commands below from the repo root.

The two trainers are peers — same flags, same outputs:

| Script                         | Backend          | Notes                                                 |
|--------------------------------|------------------|-------------------------------------------------------|
| [`train_mlx.py`](train_mlx.py) | MLX              | Apple Silicon GPU (Metal) — the fast path on a Mac.   |
| [`train_nnx.py`](train_nnx.py) | JAX / `flax.nnx` | CPU on Mac, GPU/TPU elsewhere; also `--export_linen`. |

Examples below use MLX; swap in `train_nnx.py` (prefix `JAX_PLATFORMS=cpu` on a
Mac) for the JAX trainer.

## 1. Export a dataset

Precompute SpectroStream codes + MusicCoCa style tokens from your audio into a
fast-to-load dataset:

```sh
mrt sft export \
    --sources ~/Datasets/my_audio --out datasets/my_sft \
    --backend mlx_pure --num-samples 1024 \
    --val-fraction 0.1 --val-num-samples 128
```

- `--num-samples` — how many salient excerpts to draw (`--duration` seconds each, default 10).
- `--val-fraction 0.1 --val-num-samples 128` — hold out whole *files* into `datasets/my_sft_val` for an honest eval signal (optional).
- `--style-prompt "electronic breakbeat"` — **single-text-prompt recipe**: condition the whole dataset on one fixed text prompt instead of each clip's own audio style. Removes the per-clip style fingerprint so the LoRA learns *that style* anchored to the prompt. Recommended when fine-tuning toward one genre/vibe.
- `--transcribe` — add MT3 piano-roll conditioning channels (slow; optional). Needs the MT3 checkpoint once: `mrt mt3 download mt3`.

## 2. Fine-tune with LoRA / DoRA

```sh
python notebooks/sft/train_mlx.py \
    --model_name mrt2_small --checkpoint checkpoints/mrt2_small.safetensors \
    --data_dir datasets/my_sft --valid_dir datasets/my_sft_val \
    --output_dir runs/my_run \
    --lora_rank 16 --lora_alpha 16 --lora_dora --lora_all_linears \
    --total_steps 2000 --batch_size 2 --learning_rate 1e-3 \
    --sample_every_steps 100 --valid_freq 250
```

The most-used knobs:

| Flag                                               | What it does                                                                                     |
|----------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `--lora_rank 16`                                   | Adapter rank. 0 = full (encoder-frozen) SFT. Higher = more capacity.                             |
| `--lora_alpha 16`                                  | Adapter scale (effective `alpha/rank`). 1× rank is a good start.                                 |
| `--lora_dora`                                      | Use **DoRA** (recommended) — resists the energy-collapse that plain LoRA hits at full strength.  |
| `--lora_all_linears`                               | Widen the adapter set (adds attention/FFN targets; per-backend specifics in the package README). |
| `--total_steps`, `--batch_size`, `--learning_rate` | The usual. 1e-3 with LoRA is reasonable; lower (2–3e-4) for longer runs.                         |
| `--sample_every_steps 100`                         | Generate an audio clip every N steps so you can hear progress (step 0 = pre-SFT baseline).       |
| `--valid_freq 250`                                 | Eval on the held-out set every N steps (needs `--valid_dir`); drives early-stop.                 |
| `--crop_length_seconds 2`                          | Training crop length.                                                                            |

## 3. Listen while it trains

Everything for a run lands under `--output_dir`:

| Artifact                         | What                                                                           |
|----------------------------------|--------------------------------------------------------------------------------|
| `tb/`                            | TensorBoard events: loss, grad/adapter norms, lr, and the generated **audio**. |
| `samples/step_*.wav`             | Generated clips on disk — just open them.                                      |
| `loss_curve.png`                 | Train/eval loss + a `gen/frac_silent` line.                                    |
| `config.yaml`, `train_log.jsonl` | The resolved config + per-step metrics.                                        |

Watch it live in the browser (scrub the audio slider to hear it evolve):

```sh
tensorboard --logdir runs/my_run --samples_per_plugin audio=200
```

Watch `gen/frac_silent`: it should stay near 0. If it climbs toward 1 while the
loss keeps falling, the model is *energy-collapsing* — switch on `--lora_dora`
and/or lower the inference strength in step 4.

## 4. Use the fine-tuned model

Audition a checkpoint by ear (baseline vs SFT), trying the **strength** knob —
`s≈0.6–0.8` is usually the sweet spot, full `1.0` can over-drive:

```sh
mrt sft generate \
    --model mrt2_small --checkpoint checkpoints/mrt2_small.safetensors \
    --data-dir datasets/my_sft_val --out runs/my_run/eval \
    --adapters runs/my_run/sft_mlx_adapters_step_2000.safetensors \
    --lora-rank 16 --lora-alpha 32 --dora \
    --lora-strength 0.7 --seed 7
```

For a single portable checkpoint that runs on **any** inference backend with no
LoRA-aware code, fold the adapters into the base weights. The NNX trainer does
this directly with `--export_linen runs/my_run/sft_merged.safetensors` (it
merges then writes a Linen-format checkpoint).

## Notes

- No `--checkpoint` / random weights + `TinyPOCSpec` is for the automated tests
  (`pytest tests/sft`), not a fine-tuning starting point.
- Shared, backend-neutral trainer glue lives in the package at
  [`magenta_rt.sft.trainer_common`](../../magenta_rt/sft/trainer_common.py).
- Add `--use_wandb --wandb_project my-sft` on either trainer to mirror scalars +
  audio to Weights & Biases.
