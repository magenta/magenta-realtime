# `magenta_rt.sft`

Supervised fine-tuning (SFT) for Magenta-RT V2. Two trainers — one per
inference backend — share **everything except the framework-native train
step**: the same `SFTConfig`, the same grain/AudioTree data pipeline, the
same LoRA/DoRA math (merge is bit-exact across backends), the same logging.

* `notebooks/sft/train_mlx.py` — **MLX**, Apple Silicon GPU (Metal). The
  fast path on a Mac.
* `notebooks/sft/train_nnx.py` — **NNX**, JAX (CPU on Mac; GPU/TPU
  elsewhere). Also exports a Linen-format checkpoint for any inference
  backend.

The trainers live under `notebooks/sft/` (not in the package) so
hyperparameters can be edited without round-tripping through the package
surface; their shared, backend-neutral glue lives in the package at
`magenta_rt.sft.trainer_common` (CLI parsing, LR schedule, dataset
factories, logging setup). New here? Jump to **[Quick start](#quick-start)**,
then **[Watch your run](#watch-your-run-tensorboard--wavs)**.

## Layout

| File | Purpose |
|---|---|
| `configs.py`     | `SFTConfig` dataclass; `TinyPOCSpec` (mrt2-shaped, ~1M random-weight params) — the checkpoint-free test model. |
| `data.py`        | `grain.MapDataset` pipeline (pure numpy in, batched `AudioTree` out). |
| `transforms.py`  | AudioTree grain transforms + `augment_batch` / `to_source_target` boundary. |
| `export.py`      | Offline dataset precompute: directories of audio → `TreeWriter` AudioTree dataset (codes + MusicCoCa + optional MT3 piano-roll). |
| `embed_prompt.py`| Subprocess CLI that embeds one text prompt with MusicCoCa (SentencePiece is C++ and deadlocks grain in-process; isolation avoids it). |
| `pianoroll.py`   | Rasterize MT3 note transcriptions → 25 Hz piano-roll conditioning channels. |
| `freeze.py`      | `Frozen(nnx.Variable)` + `freeze_module(...)` for encoder freezing (NNX). |
| `lora_nnx.py`    | NNX `LoRAAdapter` + `inject_lora` / `merge_lora_into_base` / `set_lora_strength`; DoRA-capable. |
| `lora_mlx.py`    | Idiomatic-MLX LoRA twin (`LoRALinear` / `LoRAEinsumDense`, `inject_lora` / `mark_lora_trainable` / `merge_lora_into_base` / `set_lora_strength`); DoRA-capable. |
| `checkpoint.py`  | NNX ↔ Linen-format safetensors loader + exporter (round-trip bit-exact). |
| `earlystop.py`   | `EarlyStopper` (min-delta + patience). |
| `wandb_writer.py`| Optional Weights & Biases `MetricWriter` for the `MultiWriter` stack. |
| `tb_writer.py`   | TF-free TensorBoard writer (`tensorboardX`) for the MLX trainer (`clu`/TensorFlow aborts co-resident with Metal). |
| `trainer_common.py` | Backend-neutral trainer glue: the `argbind` CLI over `SFTConfig` (`TrainCLI`), warmup→rsqrt LR schedule, dataset factories, logging setup. |

## Quick start

A fine-tune is three steps: **export a dataset → fine-tune from a pretrained
checkpoint → watch the audio improve.**

### 1. Export a dataset from a directory of audio

```sh
mrt sft export \
    --sources ~/Datasets/my_audio --out datasets/my_sft \
    --backend mlx_pure --model mrt2_small --num-samples 1024
```

Precomputes SpectroStream codes + MusicCoCa style tokens into a `TreeWriter`
dataset (options + a single-text-prompt recipe under
[Exporting a dataset](#exporting-a-dataset-exportpy)).

### 2. Fine-tune with LoRA / DoRA

MLX (Apple Silicon GPU — the fast path on a Mac):

```sh
python notebooks/sft/train_mlx.py \
    --model_name mrt2_small --checkpoint checkpoints/mrt2_small.safetensors \
    --data_dir datasets/my_sft --output_dir runs/my_run \
    --lora_rank 16 --lora_alpha 16 --lora_dora --lora_all_linears \
    --total_steps 2000 --batch_size 2 --learning_rate 1e-3 \
    --sample_every_steps 100
```

```
[sft-mlx] device: Device(gpu, 0)   spec: mrt2_small
[sft-mlx] TensorBoard logdir: runs/my_run/tb  (tensorboard --logdir runs/my_run --samples_per_plugin audio=200)
[sft-mlx] froze encoder. trainable (adapters): ...
  [sft-mlx] sample baseline  rms=... frac_silent=... -> runs/my_run/samples/step_00000.wav
  step  100/2000  loss=...  grad_norm=...  adapter_norm=...  lr=...  eta ...
  ...
```

Swap `train_mlx.py` → `train_nnx.py` (prefix `JAX_PLATFORMS=cpu` on a Mac) for
the JAX trainer — same flags, plus `--export_linen <path>` to write a
Linen-format checkpoint that loads on every inference backend. Knobs:
`--lora_all_linears` widens the adapter set (per-backend targets
[below](#lora--dora-lora_nnxpy--lora_mlxpy)); drop `--lora_dora` for plain LoRA;
drop `--lora_rank` for full (encoder-frozen) SFT.

Every `SFTConfig` field is a flag — `--help` lists them all (the CLI is built
from the dataclass with
[argbind](https://pypi.org/project/argbind-dbraun/)). Boolean flags take either
the bare form or an explicit value: `--lora_dora` and `--lora_dora True` are the
same, and you turn one off with `--freeze_encoder False` (there is no
`--no-flag` spelling).

argbind also gives every run a reproducible config: `--args.save run.yml` writes
the exact arguments used, and `--args.load run.yml` replays them (command-line
flags still win over the file, so you can replay a run with one value changed).

### 3. Watch it train

See [Watch your run](#watch-your-run-tensorboard--wavs) — loss curves and the
periodically-generated audio, live in the browser.

## Watch your run (TensorBoard + WAVs)

The point of SFT here is *audio quality you can hear improve*, so both trainers
periodically generate a clip from held-out conditioning and log it — the local
equivalent of a "generate during training" callback. Turn it on with
`--sample_every_steps N` (needs a real preset + `--checkpoint`); step 0 logs the
pre-SFT **baseline** so later steps have a reference.

Each run writes, under `--output_dir`:

| Artifact | What |
|---|---|
| `tb/` | TensorBoard events: `train/loss`, `grad_norm`, `adapter_norm`, `lr`, `gen/rms`, `gen/frac_silent`, and the generated **audio** (`gen/audio`). |
| `samples/step_*.wav` | The generated clips on disk (MLX trainer) — open them directly, no server needed. |
| `loss_curve.png` | Train/eval loss + a `gen/frac_silent` twin axis (MLX trainer). |
| `train_log.jsonl` | Per-step raw metrics. |
| `config.yaml` | The fully-resolved `SFTConfig` for this run. |

Launch TensorBoard and scrub the audio slider to hear the model evolve:

```sh
tensorboard --logdir <output_dir> --samples_per_plugin audio=200
# compare several runs at once:
tensorboard --logdir <parent_of_output_dirs> --samples_per_plugin audio=200
```

The default events dir is `<output_dir>/tb` (override with `--tensorboard_dir`).
The NNX trainer logs through `clu.metric_writers`; the MLX trainer uses the
**TF-free** `tensorboardX` shim (`tb_writer.py`) because `clu`/TensorFlow aborts
when co-resident with Metal/MLX. Add `--use_wandb` on either to mirror scalars
and audio to Weights & Biases.

## Tests

```sh
pytest tests/sft -v
```

The fast suite is checkpoint-free: it trains `TinyPOCSpec` (a tiny random-weight
mrt2-shaped model) on a synthetic dataset from
`tests/sft/test_utils.write_fake_tree_dataset`,
so no downloads or real audio are needed. It covers the data pipeline
(shape/dtype), freeze retype accounting, LoRA/DoRA zero-init identity + fuse
round-trip + strength=0 collapse, loss decrease over a short run, and the orbax
save/restore round-trip. Real-checkpoint tests are
marked `checkpoint`/`slow` and skip when the weights aren't present.

## Data pipeline (`data.py`)

The pipeline is built on the real
[audiotree](https://github.com/DBraun/audiotree) package: one
`audiotree.AudioTree` container — channel-major `waveform` `[B, C, T]`,
frame-major `codes` `[B, T, D]`, conditioning arrays in `metadata` — flows
through a numpy/`grain` pipeline of transforms, and the trainer reaches for a
backend codec only at the `augment_batch` / `to_source_target` boundary
(mirroring `audiotree.transforms` + `encode_with_codec`).

```python
ds = create_audiotree_dataset(        # yields batched audiotree.AudioTree
    data_dir="...",                  # a TreeWriter export (manifest.json)
    batch_size=B,
    crop_length_seconds=2,           # → 50 frames @ 25 Hz
    input_configs=spec.input_configs,
    target_config=spec.target_tokens_config,
    seed=0,
    num_workers=8,                   # 0 disables mp_prefetch (single-process)
)
for batch in ds:                     # AudioTree: extras['source'] + codes/waveform
    # Pre-tokenized data -> no codec. For audio data, pass a SpectroStream via
    # codec= to encode samples->tokens on device (the on-the-fly path).
    source, target = to_source_target(  # np.int32 [B,T,144] / [B,T,Q], cast for you
        batch, spec.target_tokens_config, asarray=jnp.asarray,
    )
    ...
```

Design points:

* **`.seed(seed)` right after the source** — every downstream random
  transform (`.shuffle()`, `.random_map(...)`) inherits a derived seed.
  No per-transform `seed=` kwargs.
* **Pure numpy in transforms** — `mlx.core.array` / `jnp.asarray` cast
  happens at the *consumer* boundary. Same pipeline serves NNX, JAX,
  and MLX trainers.
* **`to_iter_dataset(ReadOptions(0, 0))`** — in-pipeline thread pool
  disabled. Parallelism comes from `mp_prefetch` workers, each running
  the pipeline synchronously.
* **`mp_prefetch(per_worker_buffer_size=2)`** — small constant buffer
  per worker. Memory is `num_workers × 2 × batch_size × bytes`.
* **`grain.DatasetIterator` state is checkpointable** — pairs with
  `ocp.CheckpointManager` for resumable training (NNX trainer; the MLX
  trainer's exact resume is a follow-up).

Example schema expected by `data.py` (per-example, no offsets applied):

```
soundstream_tokens             : [T, target_rvq]  int32 in [0, codebook_size)
mulan_tokens_25hz              : [T, 12]   int32 in [-1, musiccoca_codebook_size)
pianoroll_with_onsets_tokens   : [T, 128]  int32 in [0, 4)
drum_pianoroll_tokens          : [T, 1]    int32 in [0, 2)  (NOT produced by
                                 the export — an intent directive, not an onset
                                 raster; conditioned on the dropout token)
cfg_conditioning_tokens        : [T, 2]    int32 in [0, 41)
cfg_conditioning_drums_tokens  : [T, 1]    int32 in [0, 9)
audio                          : [T * 1920, ch] f32 (optional, instead of
                                 soundstream_tokens — on-the-fly encode path)
```

`mulan_tokens_25hz` carries *quantized* MusicCoCa style tokens — the export
step quantizes raw MusicCoCa embeddings to 12 RVQ tokens per frame before
writing (the trainer conditions on tokens, not embeddings).

Produced offline by `magenta_rt.sft.export` (below), or by any pipeline
emitting this schema.

### On-disk format: audiotree `TreeWriter` export

`create_audiotree_dataset(data_dir, ...)` reads one format — an audiotree
`TreeWriter` export (`manifest.json` + one memmap per leaf); both
notebooks (`train_nnx.py` / `train_mlx.py`) consume it through
`trainer_common.make_dataset`, so a single export serves both trainers. The export
is written as an **`AudioTree` pytree** — SpectroStream codes in `codes`,
conditioning channels in `metadata` — so audiotree's `TreeDataSource`
reconstructs each record as an `AudioTree` directly (no adapter class.
Memmaps give cheap random access without per-item file open/decompress,
and `TreeDataSource` is pickle-safe for `mp_prefetch` workers. The records
are plain numpy at read time, so a single export feeds the NNX, JAX, and
MLX trainers alike.

One constraint: every example in a `TreeWriter` dataset shares a fixed
per-leaf shape, so excerpts are exported at a fixed duration **at least as
long as the training crop** (`crop_length_seconds * 25` frames);
`AudioTreeRandomCrop` still picks a random window within each.
`write_fake_tree_dataset(...)` writes a synthetic dataset in this format
(see `tests/sft/test_tree_data.py`).

### Exporting a dataset (`export.py`)

`export_tree_dataset` is an audiotree prerender pipeline:
`audiotree.sources.create_audio_dataset` + `ExcerptConfig` draw
`num_samples` salient fixed-`duration` excerpts from directories of audio
(multi-try loudness search; files shuffle and repeat with fresh excerpt
positions), grain `mp_prefetch` workers decode in parallel, and the model
encode runs as a pipeline `.map()` stage **after** `.batch()` in the main
process (the workers data-parallelize the decode while the accelerator
consumes whole batches). The raw waveform is **not** saved; each record
carries full-depth `codes` `[1, T, 64]` (the target prep truncates to
`rvq_truncation_level` at train time), broadcast style tokens
`extras['mulan_tokens_25hz']` `[1, T, 12]`, the raw
`extras['musiccoca_embedding']` `[1, 768]` (static per-example metadata;
pass `tree_exclude_prefixes=["extras.musiccoca_embedding"]` to
`create_audiotree_dataset` to leave it on disk during training), and
`filepath` / `offset` provenance. `profile=True` prints grain's per-stage
execution summary at the end (`ExecutionTrackingMode.STAGE_TIMING`).

```python
from flax import nnx
from magenta_rt.nnx import MagentaRT2Sampler
from magenta_rt.nnx.musiccoca import MusicCoCa
from magenta_rt.sft import export_tree_dataset, mt3_transcriber

mrt = MagentaRT2Sampler.from_preset("mrt2_small", rngs=nnx.Rngs(0))
mrt.load_checkpoint("checkpoints/<name>.safetensors")
export_tree_dataset(
    "~/Datasets/my_audio", "datasets/my_sft",
    codec=mrt.spectrostream,          # waveform_to_codes([B, C, T] @ 48 kHz)
    style_model=MusicCoCa(),          # any MusicCoCaBase backend
    transcriber=mt3_transcriber(),    # optional: MT3 piano-roll channels
    num_samples=1024,
    duration=10.0,                    # one MusicCoCa clip per record
)
```

`mrt sft export` is the CLI driver
(`--backend mlx_pure|nnx`, `--transcribe` to enable MT3, `--profile`).

**Leak-free file-level train/val split.** Pass `--val-fraction 0.1
--val-num-samples 256` to hold out whole *files* (not just different
excerpts of the same files) for validation: the driver discovers the file
set, deterministically partitions it (`--split-seed`), and writes the train
export to `<out>` and the val export to `<out>_val`, each manifest recording
its `source_files` for auditing. This is the honest generalization signal —
an excerpt-level split (different `seed` over a shared file pool) leaks each
file's style into both sets, so a flat eval loss there can't distinguish
memorization from an easy val set. `discover_audio_files(...)` /
`split_audio_files(...)` expose the split for programmatic use, and
`export_tree_dataset(..., files=[...])` draws excerpts from an explicit file
list.

With a `transcriber` (the vendored MT3 via `mt3_transcriber()` —
`backend="nnx"` ([JAX](../nnx/mt3/README.md)) or `"mlx_pure"`
([Apple Silicon](../mlx_pure/mt3/README.md)), both decoding identical
tokens — or any `(mono_16k_samples) -> transcription` callable), each
excerpt also gets the note piano-roll conditioning channel —
`pianoroll_with_onsets_tokens` `[T, 128]` (0 off / 1 on / 2 onset) —
rasterized at 25 Hz by `magenta_rt.sft.pianoroll`. The drum channel
(`drum_pianoroll_tokens`) is **not** synthesized: it is a per-frame intent
directive (`-1` let-the-model-decide / `0` don't-play-drums / `1`
please-play-drums), not an onset raster, so it can't be labeled from a
transcription — training conditions it on the dropout token. Without a
transcriber (MT3 dominates export time), the export is still trainable
against the full mrt2 source spec:
`prepare_source_tokens` fills any configured-but-missing conditioning
channel with its **learned unconditional (dropout) token** — the same
`-1 → dropout-token` mapping inference uses for "no notes given". (The
CFG conditioning channels are likewise derived at training time.)

### CFG conditioning channels (`PrepareCFG`)

The mrt2 source is six channels, and two of them —
`cfg_conditioning_tokens` `[T, 2]` (musiccoca/notes) and
`cfg_conditioning_drums_tokens` `[T, 1]` (drums) — carry the
*classifier-free-guidance strength* the model is conditioned on (scales in
`[-1, 7]`, discretized by `magenta_rt.conditioning.discretize_cfg`; the
inference systems hold them constant over time). They are a property of
the training recipe, not of the audio, so `export_tree_dataset` does not
store them. Instead, `create_audiotree_dataset` synthesizes them in the
grain pipeline via `PrepareCFG` for any `input_configs` entry whose key
starts with `cfg_conditioning` that the data doesn't already carry
(data that has them passes through untouched):

* **Default — sampled**: a fresh guidance scale per example, drawn
  uniformly over the channel's token range and held constant across the
  example's frames. This exercises the pretrained model's full
  guidance-token range, so fine-tuning doesn't collapse the token ↔
  strength association the checkpoint already carries.
* **Fixed**: pin every example to specific scales —

  ```python
  ds = create_audiotree_dataset(
      data_dir, ...,
      cfg_fixed_scales={
          "cfg_conditioning_tokens": (3.0, 1.0),   # musiccoca, notes
          "cfg_conditioning_drums_tokens": 1.0,
      },
  )
  ```

  Use this to specialize the model at the guidance strengths you will
  actually run at inference (the values above are the system defaults).

Provenance note: the original mrt2 training recipe for these channels is
not public (the local `training` branch predates the 6-channel spec), so
sampled-uniform is the conservative default for SFT-from-pretrained.

### Style augmentation in embedding space (`StyleEmbeddingJitter`)

Because the export stores the raw `musiccoca_embedding` per window, the
pipeline can augment *style* far more faithfully than token corruption:
`create_audiotree_dataset(..., style_jitter_std=0.05)` perturbs the
embedding with Gaussian noise (scaled to the embedding's per-dim RMS) and
**re-quantizes** it with MusicCoCa's RVQ (a pure-numpy tokenizer,
`rvq_tokenize`, so it runs inside `mp_prefetch` workers — codebooks
lazy-load per process from the converted `musiccoca_nnx.safetensors`).
Small jitter lands in the same RVQ cells (identical tokens); larger
jitter shifts the deepest levels first — exactly the topology of the real
style space. Off by default; a no-op for data without embeddings. Applied
before the sticky time-augmentation.

### End-to-end real-model test

`tests/sft/test_real_sft.py` (marks `checkpoint` + `slow`, ~3 min on CPU
JAX) runs the entire chain against the real model: synthesized audio →
`export_tree_dataset` with the real SpectroStream, nnx MusicCoCa, and
real MT3 transcription → `create_audiotree_dataset` with the full
6-channel `mrt2_small` `input_configs` (PrepareCFG filling the guidance
channels) → 30 LoRA steps on the pretrained checkpoint → loss decreases.
This is the test that sees real vocabulary sizes and the real data path,
which the checkpoint-free `TinyPOCSpec` tests cannot.

## LoRA / DoRA (`lora_nnx.py` / `lora_mlx.py`)

```sh
python notebooks/sft/train_nnx.py \
    --lora_rank 16 --lora_alpha 16 \
    --learning_rate 1e-2 \
    --total_steps 30
```

```
[sft] base params: 659,270
[sft] LoRA: wrapped 6 Linears, rank=8, alpha=16.0, adapter params=15,360
[sft] trainable (MRTLoRAParam): 15,360  frozen: 0
```

`inject_lora(model, rank=R, alpha=A)` walks the model, wraps every attention
`q_proj` / `kv_proj` (`default_targets`) in a `LoRAAdapter` that shares the
base `Linear` via `self.base` — no weight duplication. Pass
`targets=all_linear_targets` to additionally wrap FFN `Linear`s. Adapter
weights land in `MRTLoRAParam(nnx.Param)`, and the trainer flips both
`nnx.Optimizer(model, tx, wrt=MRTLoRAParam)` and
`nnx.DiffState(0, MRTLoRAParam)` to that filter — base weights stay
materialized for the forward but never receive grads or Adam state.

Why a custom adapter instead of `flax.nnx.LoRA`? The decoder transformers
are built under `nnx.vmap`, so `q_proj.kernel` has shape
`(num_layers, in, out)`. Upstream `nnx.LoRA` hardcodes `(in, rank)` for its
adapter weights, which mismatches inside the `nnx.scan` body. `LoRAAdapter`
matches whatever leading axes the base `kernel` has, so the same code wraps
vmapped and unvmapped Linears.

Two correctness properties tested:

* **Bit-exact at init**: `B` is zero-initialized, so a freshly injected
  model produces *byte-identical* forward outputs as the unwrapped model.
* **Base weights untouched during LoRA training**: after N train steps with
  `wrt=MRTLoRAParam`, every non-LoRA `Param`'s value is unchanged
  (verified by byte-hash multiset comparison).

### Merge for inference

```python
n = merge_lora_into_base(model)   # base.kernel += scale * A @ B; unwraps
```

Folds each adapter into its base `Linear.kernel` and replaces the wrapper.
The resulting model has the *original* `EncoderDecoder` structure, so the
existing Linen-format safetensors export works without any LoRA-aware code
downstream (NNX / MLX / MLX-pure / CoreML inference all see plain Linears).

### DoRA and the runtime strength knob

Both backends support **DoRA** (weight-decomposed LoRA) and a runtime
**strength** scalar — the two knobs that matter most for SFT audio quality,
because plain LoRA at full strength tends to *energy-collapse* the free-running
output (quiet / mostly-silent generation even while the loss falls).

* **DoRA** (`--lora_dora`) splits each adapted weight into a learned per-output
  **magnitude** and a **direction** (`W' = m · (W + αΔ) / ‖W + αΔ‖`), so the
  adapter sets direction and magnitude independently. This resists the
  effective-weight-norm runaway that drives the collapse. Recommended over plain
  LoRA. (`eps` lives *inside* the sqrt — a singular gradient at zero rows
  otherwise NaNs the first step.) Available on both `lora_nnx` and `lora_mlx`;
  the fuse/merge round-trip stays bit-exact (tiny fp32 reorder only).

* **`set_lora_strength(model, s)`** scales every adapter delta by `s` at
  *inference* without retraining: `base + s·(α/r)·BA` (DoRA scales the
  direction term the same way). `s=0` collapses exactly to the base model;
  `s=1` is the trained adapter; **`s≈0.6–0.8` is usually the sweet spot** —
  enough of the fine-tuned character without the full-strength collapse. This is
  the knob to reach for first, *before* lowering training `alpha`.
  `mrt sft generate --lora-strength 0.7` exposes it for A/B-ing a
  checkpoint by ear.

`gen/frac_silent` (logged every `sample_every_steps`, see below) is the curve
that makes the collapse visible: a healthy run sits near 0, a collapsing one
climbs toward 1 even as `train/loss` keeps dropping.

### Why LoRA matters for the 16 GB MacBook

Full SFT on `mrt2_small` (~700 M params, smallest real preset) won't
fit in 16 GB CPU JAX once optimizer state + grads + activations land. LoRA
keeps the forward identical but eliminates gradient and Adam-state memory
for the base weights:

| | Full SFT (encoder frozen) | LoRA (rank 16, attention QKV) |
|---|---|---|
| Trainable params | ~250 M (decoder)         | ~5 M (adapters)      |
| Adam moments (fp32) | ~2 GB                  | ~40 MB               |
| Forward activations | identical              | identical            |
| Checkpoint size  | 2.7 GB                    | tens of MB           |
| Fits in 16 GB CPU? | no                      | yes                  |

## Encoder freeze (`freeze.py`)

```python
freeze_module(model.encoder)                       # nnx.Param → Frozen
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)  # skips Frozen
grad_fn = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))
```

`Frozen` is a trivial `nnx.Variable` subclass. Because it isn't
`nnx.Param`, both `nnx.Optimizer(..., wrt=nnx.Param)` and
`nnx.value_and_grad(..., argnums=nnx.DiffState(0, nnx.Param))` filter
it out. Encoder Adam moments are not allocated, and encoder grads are
not materialized at all.

The retype shows up in `nnx.tabulate(model)` so the freeze decision is
visible in the model state itself, not hidden in an external mask.

For dynamic toggling (e.g. unfreeze halfway through training), prefer
`optax.transforms.freeze(mask)` or
`optax.transforms.selective_transform(adam, freeze_mask=mask)` at the
optimizer layer instead — rebuilding the mask + `optimizer.init(params)`
is the natural fit there.

## Trainer drivers

The two trainers are **peers**, not a reference + a smoke driver. The NNX
trainer runs on JAX (CPU on Mac via `JAX_PLATFORMS=cpu`; GPU/TPU elsewhere);
the MLX trainer is the path to **Apple Silicon GPU** (Metal) fine-tuning. Both
share `sft.data`, `SFTConfig`, and `sft.trainer_common`, and
both support LoRA, **DoRA**, the runtime [strength knob](#dora-and-the-runtime-strength-knob),
pretrained-checkpoint loading, gradient accumulation, periodic [audio
sampling](#watch-your-run-tensorboard--wavs), eval + early-stopping, and
TensorBoard/W&B logging. The adapter math and merge are identical across
backends, so a LoRA-merged checkpoint runs on every backend regardless of which
trainer produced it.

Two capabilities are still **NNX-only**: exact resume (optimizer +
data-iterator state) and the Linen-interchange export. The MLX trainer saves
adapters/weights every `save_every_steps` but does not yet auto-restore them
to continue a run; merge with `set_lora_strength`/`merge_lora_into_base` and
run inference on any backend instead.

Differences live in the framework-native specifics:

| | NNX trainer | MLX trainer |
|---|---|---|
| Hardware    | JAX: CPU on Mac, GPU/TPU elsewhere | Apple Silicon GPU (Metal) |
| Model       | `magenta_rt.nnx.depthformer.EncoderDecoder.from_config(...)` | `spec.build_decoder()` from `magenta_rt.mlx_pure.configs` |
| Loss/grads  | `nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, …), has_aux=True)` | `nn.value_and_grad(model, loss_fn)` (model is implicit) |
| Encoder freeze | `Frozen(nnx.Variable)` retype via `freeze_module(model.encoder)` | `model.encoder.freeze()` (built-in MLX `Module.freeze()`) |
| LoRA targets | default attention QKV; `--lora_all_linears` adds FFN | default FFN; `--lora_all_linears` adds attention output proj (see below) |
| Pretrained load | `load_nnx_depthformer_from_safetensors` | `MagentaRT2Sampler.load_from_safetensors` (codec skipped) |
| JIT         | `@nnx.jit` train_step | `mx.compile(...)` fused step; `mx.eval(...)` per step materializes |
| Grad accumulation | `optax.MultiSteps` | manual accumulator loop (`gradient_accumulation_steps`) |
| Checkpoints | `orbax.checkpoint.CheckpointManager` (async, best-key, **exact resume**) | `save_weights` safetensors every `save_every_steps` (resume: follow-up) |
| Linen export | `--export_linen` (`export_nnx_to_linen_safetensors`) | follow-up (merge + reuse the NNX exporter, or run MLX-native) |
| Logging     | `clu.metric_writers` (TensorBoard) + `ReportProgress` | `absl.logging` + TF-free `tensorboardX` (`tb_writer.py`) |
| Audio in browser | `write_audios` → TensorBoard | `write_audios` → TensorBoard (+ WAVs to `samples/`) |

### MLX LoRA targets differ from NNX (and why)

The NNX depthformer stores attention `q`/`kv` as vmap-stacked `nnx.Linear`, so
its default LoRA target is `{q_proj, kv_proj}`. In `mlx_pure` those projections
are **bare `mx.array` attributes** applied by a matmul helper — not modules — so
they can't be wrapped without perturbing the (parity-sensitive) inference hot
path. The idiomatic-MLX wrap points are instead the **FFN linears** (each a plain
`nn.Linear` inside a `Dense`, so the adapter sits inside the activation) and the
attention **`output_projection`** (an `EinsumDense`, `"...nh,dnh->...d"`). So
`lora_mlx.default_targets` selects the FFN linears and `all_linear_targets` adds
the attention output projections. The fuse/merge round-trip is bit-exact
(`test_lora_mlx.py`), so this divergence in *which* layers are adapted does not
affect cross-backend interchange. QKV-LoRA on the bare arrays is a documented
follow-up.

## NNX-specific trainer (`notebooks/sft/train_nnx.py`)

* **Loss**: `optax.softmax_cross_entropy_with_integer_labels(logits, target).mean()`
  over `[B, T, Q]`.
* **Optimizer chain**: `clip_by_global_norm → adam(0.9, 0.95) →
  scale_by_learning_rate(warmup+rsqrt)`, with the schedule threaded through
  `optax.inject_hyperparams` so the current LR is readable from optimizer
  state (`current_learning_rate(optimizer)`, logged as
  `train/learning_rate`). Mirrors the existing JAX/MLX trainers on the
  `training` branch.
* **Gradient accumulation**: `--gradient_accumulation_steps k` wraps the
  whole transform in `optax.MultiSteps` (the tunix recipe) — gradients
  accumulate over `k` micro-batches, parameters and the LR schedule advance
  once per accumulated update. No loop logic involved.
* **`nnx.value_and_grad(loss, has_aux=True)`** returns `(loss, metrics_dict)`;
  the jitted step donates the optimizer buffers
  (`nnx.jit(..., donate_argnames=("optimizer",))`).
* **`nnx.cached_partial`**: the train/eval steps are bound to
  `(model, optimizer)` once, so each step skips the nnx graph
  traversal/flatten (tunix's `cache_nnx_graph` pattern). Module structure
  must be final before binding — `inject_lora` / `freeze_module` run first.
* **`model.train()` / `model.eval()`** discipline at train-loop entry and
  around eval / sample forwards. NNX cascades `deterministic` down to
  any `nnx.Dropout` modules.
* **Checkpoints**: `orbax.checkpoint.CheckpointManager` with async +
  best-key tracking. `save_state(...)` / `restore_state(...)` round-trip
  both model and optimizer state; in LoRA mode the model side saves
  **adapters only** (`lora_only=True`, KBs–MBs instead of the full model —
  the optimizer state is already adapter-only via `wrt=MRTLoRAParam`).
* **Exact resume** (`config.resume`, default on): checkpoints also store
  the grain data-iterator state (`iterator.get_state()`, a JSON dict), and
  `train()` auto-restores the latest checkpoint in `output_dir` — model,
  optimizer, *and* position in the shuffled/repeated data stream, so a
  resumed run produces bit-identical losses to an uninterrupted one
  (`test_exact_resume_continues_data_stream`).
* **Audio samples** (`--sample_every_steps N` + a real preset/checkpoint):
  `AudioSampleWriter` periodically generates a few seconds of audio from
  the live model and writes it to TensorBoard/W&B. It keeps a *separate*
  sampler (depthformer + SpectroStream codec) and syncs just the trainable
  state before each sample — adapters only in LoRA mode — because arming
  streaming on the training model itself would invalidate the
  `nnx.cached_partial` train step. Conditioning comes from a held-out
  batch of the run's own prepared source tokens. Costs a second model
  instance in memory; off by default.
* **Logging**: `clu.metric_writers.create_default_writer` (TensorBoard)
  + `periodic_actions.ReportProgress` with per-phase `.timed(...)`
  context managers (`data`, `train`, …).

## Pretrained checkpoints

```sh
python notebooks/sft/train_nnx.py \
    --model_name mrt2_small \
    --checkpoint checkpoints/mrt2_small.safetensors \
    --lora_rank 16 --lora_alpha 16 \
    --export_linen checkpoints/sft_mrt2_small.safetensors \
    --total_steps 200
```

* **Loader** (`magenta_rt.sft.checkpoint.load_nnx_depthformer_from_safetensors`)
  reuses `magenta_rt.nnx.load_weights.load_system_state_dict` against an
  `{"depthformer": ...}` shim so the NNX trainer's bare `EncoderDecoder`
  can consume the same `.safetensors` files the JAX/MLX inference paths
  do.
* **Exporter** (`export_nnx_to_linen_safetensors`) walks `nnx.state(model)`
  and emits Linen-format flat keys (`params/depthformer/.../x_layers_i/...`).
  SpectroStream params + any unchanged Linen keys are pulled through
  verbatim from `source_checkpoint_path` when provided, so the result is
  a drop-in replacement that loads in any inference backend (JAX, MLX,
  MLX-pure, NNX, CoreML).
* The exporter is `O(num_tensors)` and writes ~700 MB for
  `mrt2_small`. Use `merge_lora_into_base(model)` first if you
  trained with LoRA — then the export looks identical to a full-SFT
  result, and downstream inference doesn't need LoRA-aware code.
* Verified round-trip: `build → forward → export → load-into-fresh-model
  → forward` is bit-exact (see `test_linen_safetensors_round_trip`).

## Validation, NaN guard, early stopping

```sh
python notebooks/sft/train_nnx.py \
    --data_dir /path/to/train_dataset \
    --valid_dir /path/to/valid_dataset \
    --valid_freq 500 --valid_batches 16 \
    --early_stop_patience 5
```

Every `valid_freq` steps the trainer flips `model.eval()`, averages loss
over `valid_batches` held-out batches, and either updates the
best-known value or decrements `early_stop_patience`. `nan_check` (on
by default; disable with `--no-nan_check`) short-circuits before a
non-finite loss has a chance to corrupt the checkpoint.

## Memory + W&B telemetry (opt-in)

```sh
python notebooks/sft/train_nnx.py --log_memory
python notebooks/sft/train_mlx.py --log_memory \
    --use_wandb --wandb_project my-sft --wandb_name run-1
```

`--log_memory` prints active and peak device memory each log step
(host-side numbers on CPU JAX; MLX active + peak from `mx.get_*_memory`).
`--use_wandb` adds a `WandbWriter` alongside the default TB writer in
the NNX trainer's `MultiWriter`; the MLX trainer logs directly to W&B
via the same shim.

## Dataset inspection

```sh
mrt sft inspect datasets/my_sft --records 3 --plot 0
```

Prints the manifest metadata and per-record stats (codes / style tokens /
provenance / pianoroll density when present), and can plot a record's
pianoroll.

## Known limits / follow-ups

* **`mrt2_small` (~700 M params)** loads + LoRA-injects in ~20s on
  16 GB CPU JAX. Full forward+backward will be slow there; real training
  targets GPU / TPU.
* **Documented follow-ups** (in rough priority order; also listed in the
  PR doc):
  * loss masking for zero-padded short examples (`AudioTreeRandomCrop`
    zero-pads examples shorter than the crop into real loss — fixed-window
    exports are unaffected);
  * per-RVQ-level token accuracy in eval (depth-0 accuracy is a much
    sharper signal than mean loss);
  * dropout wiring from `ModelSpec.dropout_prob` (see the TODO at the top
    of `configs.py`) for larger-data SFT;
  * activation checkpointing (`nnx.remat` on decoder layers) and
    multi-host sharding;
  * a `mrt sft train` CLI entry point (training currently runs via the
    notebook scripts);
  * dataset mixing/weighting across multiple export directories;
  * throughput metrics (steps/s, tokens/s) in the logs;
  * **cache whole-song MT3 transcriptions and index into them per
    excerpt** instead of re-transcribing each ~31 s chunk. The export
    draws many salient excerpts (random offsets, `repeat`) from a handful
    of source songs, so the same audio is sent through MT3 repeatedly —
    and MT3 dominates export time (see *Exporting a dataset*).
    Transcribing each whole song **once** into per-frame channels and
    slicing `[offset_frame : offset_frame + frames]` per excerpt (in
    `_transcribe_channels`, `export.py`) would remove the bulk of that
    cost; indexing into the full-song transcription is also slightly
    *more* correct (no per-window boundary truncation). Needs a per-song
    transcription cache keyed by source file, using each excerpt's
    `offset` metadata to index in;
  * MLX-side exact resume + Linen-interchange export (the two remaining
    NNX-only trainer capabilities — see *Trainer drivers*).
* **Streaming `DecodeState` workaround.** `EncoderDecoder.from_config`
  declares `previous_frame` / `rng_state` slots as `nnx.data(None)`
  placeholders, which `nnx.jit` can't trace. `build_model` calls
  `model.decoder.init_streaming(batch_size=1)` once to seed them with
  real arrays; they're inert during the full-sequence training forward.
* **No dropout in the NNX modules yet.** SFT-from-pretrained uses
  `p=0` anyway, but pretraining-from-scratch will need
  `nnx.Dropout` added to `nnx/transformer.py` and `nnx/attention.py`,
  wired from `ModelSpec.dropout_prob`. The trainer already calls
  `model.train()` / `model.eval()` in the right places, so the
  dropout add is contained. If the encoder is frozen, also call
  `model.encoder.eval()` once after `model.train()` so encoder dropout
  stays off.
* **Multi-host sharding** is deliberately skipped — the pipeline is
  single-process numpy. Adding `Mesh` + `NamedSharding` + a `shard_dp`
  helper is contained to the trainer driver when the time comes.
