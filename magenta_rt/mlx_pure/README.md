# `magenta_rt.mlx_pure`

A pure-MLX inference library for Magenta-RT. **Zero runtime
dependence on `sequence_layers.mlx`** — the package can be imported
and run with `sequence_layers` absent from `sys.modules`. CI
enforces this via `tests/mlx_pure/test_no_sl_dependency.py`.

> Looking for the original sl-backed implementation? See
> `magenta_rt.mlx`. The two trees produce numerically equivalent
> results within bf16 tolerance — `mlx_pure` mirrors the module
> structure so weights load natively 1:1 via
> `mlx_pure.load_weights.load_from_safetensors`. Codec audio output is
> sample-equivalent on the v1v3 production config (verified in
> `tests/mlx_pure/parity/test_e2e_audio_diff.py`).

## Quick start

The fastest way to make audio with a real checkpoint:

```sh
python -m magenta_rt.mlx_pure.generate \
    --restore --model mrt2_small \
    --checkpoint mrt2_small.safetensors \
    --num-steps 100 --num-cfgs 2 \
    --temperature 1.3 --top-k 40 --cfg-musiccoca 3.0 --cfg-notes 1.0 \
    --output outputs/disco_funk.wav
```

Add `--bits 4` for nearest-rounding int4 quantization of the
depthformer's Dense / EinsumDense layers, or
`--bits 4 --quantize-method gptq --gptq-cal-steps 8` to run a short
GPTQ calibration loop first (better quality, ~few seconds extra
startup). Group size defaults to 32 at 4 bits and 64 elsewhere — set
`--quantize-group-size` to override.

`--restore` loads the checkpoint natively via
`mlx_pure.load_weights.load_from_safetensors` (no sl dependency); `--bridge`
instead builds the sl-backed system, loads the checkpoint via
`magenta_rt.mlx.load_weights`, then mirrors weights into the pure tree via
`mlx_pure.load_weights.load_weights_from_combinator`. Drop the
`--checkpoint`
flag (or point it at a missing path) to run with random weights for
end-to-end smoke testing — the custom loading pipeline auto-falls-back to
materialize-and-randomize.

## System API (`MagentaRT2System`)

The same system shape as the `jax` / `mlx` backends — `embed_style` →
`generate` → `(AudioTree, state)` — with the state threaded functionally
(the depthformer `SamplerState`):

```python
from magenta_rt import MagentaRT2MlxPure  # = mlx_pure.system.MagentaRT2System

mrt = MagentaRT2MlxPure(size="mrt2_small", bits=4)  # optional int4/int8
embedding = mrt.embed_style("disco funk")
audio, state = mrt.generate(style=embedding, frames=25)   # 1 second
audio, state = mrt.generate(style=embedding, frames=25, state=state)  # continue
```

CFG uses the trained conditioning tokens (the `cfgs` channels), like the
jax/mlx systems; the logit-mixing CFG path (`--num-cfgs`) and GPTQ
calibration stay on the `mlx_pure.generate` research script.
`temperature`/`top_k` must be shared scalars (per-element raises). Note the
SpectroStream codec's streaming buffers live on the codec module, so
interleaved streams on one system share codec state — use separate systems
for fully independent streams.

For a streaming inference loop in your own code:

```python
import mlx.core as mx
from magenta_rt.mlx_pure import model

# Concise preset compilation builder instantiating depthformer + spectrostream natively
mrt = model.MagentaRT2Sampler.from_preset("mrt2_small")

# Arms the codec for streaming (per-conv left-context buffers,
# InverseSTFT OverlapAddCache, decoder lookahead countdown).
# ParallelChannels handles its per-group state by reshaping groups
# into the batch axis so the inner conv's own cache carries it.
# Pass codec_streaming=False for tests that drive
# ``codes_to_waveform`` directly.
state = mrt.make_initial_state(batch_size=1)

for source_tokens in stream_of_source_tokens:
    waveform, state = mrt.step(
        state, source_tokens=source_tokens,
        temperature=1.3, top_k=40,
    )
    play(waveform)
```

For codec-only usage (`enable_streaming` / `disable_streaming`,
`step_codes_to_waveform`, `channel_splits=2`), see
[`spectrostream/README.md`](spectrostream/README.md).

## Streaming-cache lifecycle

Streaming state lives in two places: the depthformer's KV caches
(carried in the `SamplerState` from `make_initial_state`) and the
codec's conv left-context + InverseSTFT overlap buffers (on the
`SpectroStream` module, armed by `enable_streaming`). Both default to
**lazy allocation** — `make_initial_state()` returns un-allocated
caches and the first `step()` sizes and fills them on demand. That is
all you need for live generation: construct, then keep stepping.

For an **allocated-but-neutral** state — every cache pre-allocated to
its final shape/dtype, carrying *zero generation content* — call
`MagentaRT2Sampler.init_cache(state, batch=B)` after `make_initial_state`:

```python
mrt = model.MagentaRT2Sampler.from_preset("mrt2_small")
state = mrt.make_initial_state(batch_size=1)
mrt.init_cache(state, batch=1)   # allocate every cache, content-neutral
```

`init_cache` eagerly zero-allocates and sink-primes the depthformer KV
caches directly (`DepthformerDecoder.init_cache` → `Transformer.init_cache`
→ each attention layer's `init_cache` → `LocalKVCache.init_cache`). The
codec's conv / overlap buffers can't be sized from the architecture
without a conv-stack spatial walk, so it allocates them with a few
zero-input codec passes and then zeros them in place via
`SpectroStream.reset_caches`, draining the decoder lookahead countdown
to 0. The result has the exact shapes and dtypes the lazy path would
produce, with no `previous_frame` / KV / codec / RNG / step-counter
content baked in.

This is what `mlx_pure.export` ships in the companion
`_state.safetensors`: a host can stream straight from the snapshot
without inheriting warmup-generation content (the only artifact is the
decoder's ~`lookahead_length`-frame opening transient, which a prefill
erases). It is also useful for any consumer that needs a fixed-shape
streaming state without running a warmup step.

> **C++ engine compatibility.** The traced signature follows the same
> calling convention the C++ engine (`core/src/mlx_engine.cpp`, behind
> the `examples/` hosts) binds against positionally — identical to
> `magenta_rt.mlx.export`: 9 leading args including `cfg_drums`,
> `[1, 1, 141]` conditioning rows, in-trace CFG-scale discretization,
> and contiguous `state_<i>` safetensors keys (zero-element conv
> buffers are excluded from the flat state). Validated via a Python
> re-import exercising the exact C++ argument construction; an
> end-to-end run through the real engine is still outstanding, and the
> engine's `previous_frame` shape probe doesn't match this export's
> batch-first slot (prefill/`tokens_out` need the engine to read the
> manifest `role` instead). See the warning in `mlx_pure/export.py`.

Related helpers:

* `SpectroStream.reset_caches()` — zero the codec's streaming buffers
  *in place*, keeping them allocated and the convs streaming-armed.
  Contrast `disable_streaming()`, which drops the caches to `None`.
* `conv.reset_streaming_caches(module)` — the conv-subtree helper
  `reset_caches` builds on; zeros every `Conv2D` / `Conv2DTranspose`
  left-context buffer in `module`.
* `LocalKVCache.init_cache(...)` / `Conv2DCache.reset()` — the
  leaf-level primitives.

## Public API

| Module | What it provides |
|--------|------------------|
| `mlx_pure.layers` | `Dense`, `EinsumDense` (both with `to_quantized()`), and `LayerNorm` / `RMSNorm` — thin `mlx.nn` subclasses that fix the output dtype (and, for `LayerNorm`, upcast to fp32) to match sl's normalization layers. `Embedding` uses `mlx.nn` directly. |
| `mlx_pure.attention` | `LocalSelfAttention`, `StreamingCrossAttention` (with sink-embedding support) |
| `mlx_pure.transformer` | `TransformerBlock`, `Transformer`, `MultiChannelEmbedding`, `Encoder`. `Transformer.init_cache` eagerly allocates the per-layer self/cross KV caches. |
| `mlx_pure.depthformer` | `DepthformerDecoder`, `EncoderDecoder`, `SamplerState`, `TemporalCaches`. `init_cache(state, batch=…)` on both decoders eagerly allocates + sink-primes the temporal KV caches (see [Streaming-cache lifecycle](#streaming-cache-lifecycle)). `soft_cap_logits=30` applied to depth-body logits before sampling. |
| `mlx_pure.spectrostream` | Subpackage — see [`spectrostream/README.md`](spectrostream/README.md). |
| `mlx_pure.signal` | `STFT`, `InverseSTFT` (full-seq + streaming `step` via `OverlapAddCache`), `hann_window`, `inverse_stft_window_fn`, `frame`, `overlap_and_add` |
| `mlx_pure.conv` | `Conv2D`, `Conv2DTranspose`, `AveragePooling2D`, `Upsample2D`, `ParallelChannels` with per-instance `_streaming` flag and `Conv2DCache`; `enable_streaming` / `disable_streaming` / `reset_streaming_caches` helpers walk a module subtree |
| `mlx_pure.cache` | `KVCache`, `LocalKVCache`, `OverlapAddCache` (mlx-lm-style). `LocalKVCache.init_cache` / `Conv2DCache.reset` are the leaf-level eager-alloc / in-place-zero primitives. |
| `mlx_pure.sample_utils` | `sample_categorical_with_temperature` (top-k / top-p / CFG) |
| `mlx_pure.configs` | `ModelSpec`, `TokensConfig`, the full `MagentaRT2Model*` registry (incl. `mrt2_small`), `MODEL_REGISTRY`, `get_model_class`, `ScaledEmbedding` |
| `mlx_pure.model` | `MagentaRT2Sampler` orchestrator. `init_cache(state, batch=…)` eagerly allocates every streaming cache (depthformer + codec) as a content-neutral state — see [Streaming-cache lifecycle](#streaming-cache-lifecycle). |
| `mlx_pure.musiccoca` | Pure-MLX MusicCoCa (style embedder: log-mel frontend, music/text towers, RVQ tokenizer, text→audio mapper), reverse engineered from the TFLite exports — see [`magenta_rt/nnx/musiccoca/README.md`](../nnx/musiccoca/README.md) for the architecture. Loads the shared `musiccoca_nnx.safetensors`; `MusicCoCa` is a drop-in for the TFLite-backed class. |
| `mlx_pure.mt3` | Pure-MLX MT3 transcription (T5.1.1 over log-mel frames; eager greedy decode with KV cache + early EOS exit). Shares the framework-neutral core (`magenta_rt/mt3/`) and the converted safetensors with the nnx port; greedy tokens match nnx exactly (gated parity test). |
| `mlx_pure.quantize` | `quantize_in_place` (nearest-rounding int4 / int8) and `gptq_calibrate_and_quantize` (calibration-driven int4 / int8 with per-block error compensation) |
| `mlx_pure.load_weights` | `load_from_safetensors` (Linen safetensors checkpoint → pure tree, the standalone production load path) plus the per-subsystem parameter loading helpers (`load_transformer_weights`, `load_encoder_embedding_weights`, `load_decoder_embedder_weights`, `load_decoder_tail_weights`, `load_depthformer_weights`). SpectroStream-side helpers re-exported from `mlx_pure.spectrostream.load_weights`. Also: `mirror_params` for narrow leaf-by-name copies, `init_random_params` for untrained-weights demos. |
| `mlx_pure.export` | `mx.exporter` wrapper for the streaming step → `.mlxfn` (+ companion `_state.safetensors` and `.json` shape manifest). Threads all per-step state: depthformer (`SamplerState` + `LocalKVCache.{keys,values,offset}`) and codec (decoder Conv2D / Conv2DTranspose left-context buffers + the InverseSTFT `OverlapAddCache`). The shipped `_state.safetensors` is a content-neutral `init_cache` snapshot (allocated, no generation content). Bit-exact vs eager across streaming steps. Drives `mrt mlx-pure export`. |
| `mlx_pure.generate` | CLI mirror of `mlx/generate.py` (`--restore`, `--tiny`) |
| `mlx_pure.system` | `MagentaRT2System` (exported as `magenta_rt.MagentaRT2MlxPure`), `MagentaRT2State` (= `depthformer.SamplerState`) — the jax/mlx-shaped `embed_style` → `generate` → `(AudioTree, state)` API |

## Locked-feature subset

`mlx_pure` only implements the configurations that ship in
`magenta_rt.mlx.model.MODEL_REGISTRY`. Anything outside that subset
raises `NotImplementedError` with a precise message at construction
time, so future configs that flip a guarded flag fail loudly:

* `use_rope=False` (NoPE)
* `param_dtype=fp32`, `compute_dtype=bf16`
* separate Q + combined KV projections (no GQA, no ringbuffer)
* `ffn_gated=False`, `gelu_approx`, `ffn_use_bias=True`
* `use_repeat_layers=False`
* `use_local_attention=True`
* `norm_type=rms_normalization`, `norm_policy=primer_hybrid`
* `attention_per_dim_scale=True`, no soft-cap-on-attention, no bias
* `max_future_horizon=0` (fully causal)
* sink-embedding count: 1 (temporal) / 0 (depth)
* `soft_cap_logits=30.0` applied before depth-body sampling

## Bit-exact parity with sl (verified)

With `load_weights_from_combinator` mirroring weights from a built sl
`MagentaRT2Sampler`:

* **Encoder body** (multi-channel embedding + LayerNorm) — diff 0.
* **RVQ codes_to_embeddings** — diff 0.
* **Temporal transformer** (24 layers × 1024 dims × bf16) — diff 0.
* **Depth transformer + final_ln + to_logits** — diff 0; argmax
  with `soft_cap_logits=30` matches sl's sampler at 12/12 codebooks
  per step, across multiple consecutive streaming steps.
* **SpectroStream codec** — see
  [`spectrostream/README.md`](spectrostream/README.md#bit-exact-parity-with-sl)
  for `codes_to_waveform` / `waveform_to_codes` numbers.

Reproduce via the parity test suite (`pytest tests/mlx_pure/parity`).

## Running the tests

The parity tests live under `tests/mlx_pure/`. Default CI run:

```sh
pytest tests/mlx_pure -m "not slow and not checkpoint and not gptq"
```

Markers (registered in `pyproject.toml`):

| Marker | Use |
|--------|-----|
| `bf16` | mixed-precision (param fp32 / compute bf16) parity tests |
| `checkpoint` | requires a real safetensors file in `checkpoints/` |
| `slow` | end-to-end / multi-second tests |
| `gptq` | requires GPTQ-quantized weights |

The parity tests `import sequence_layers.mlx`. The repo's
`pyproject.toml` declares it as an editable local dependency, so one
`uv pip install -e .` (or `uv sync`) is the only setup step.

The **runtime** `mlx_pure` package itself does not depend on
`sequence_layers` (enforced by `test_no_sl_dependency.py`).

## Loading a real checkpoint

Easiest route is the CLI:

```sh
python -m magenta_rt.mlx_pure.generate --restore \
    --model mrt2_small \
    --checkpoint mrt2_small.safetensors \
    --num-steps 100 --output outputs/track.wav
```

If you want to load weights programmatically in your own code:

```python
from magenta_rt.mlx_pure import model

# 1. Construct the fully pure, un-deferred pipeline straightaway via the class compiler
mrt = model.MagentaRT2Sampler.from_preset("mrt2_small", int16_outputs=False)

# 2. Load parameters natively straight from disk via the object delegator
mrt.load_from_safetensors("checkpoints/mrt2_small.safetensors", model_name="mrt2_small")
```

`.load_from_safetensors` is a zero-dependency standalone native checkpoint deserializer method delegating array mapping directly into `mlx_pure` module properties without requiring external framework packages.

## Remaining work

### Real per-module `step(state, x) → (y, state)` API

The streaming codec currently uses module-internal mutable state
(`_streaming` flag + lazy `_streaming_cache` smuggled through
`__call__`). The sl-style explicit pytree-of-state `step` interface
composes better, but every conv site has to thread the state out.
Cost-benefit not yet worth it.

(The `.mlxfn` export now *does* have an end-to-end integration test —
`tests/mlx_pure/parity/test_export_streaming_parity.py` — which exports
a `.mlxfn`, re-imports it, and asserts bit-exact agreement with the
eager streaming path across multiple steps.)

## Notable sl bugs encountered (worked around)

* **`SpectroStream.Config.make()`** with the *tiny* config raises
  `AttributeError: 'Serial' object has no attribute
  'get_accumulated_input_latency'` (latency-tracking protocol
  regression in this sl checkout). The production build path
  does not hit it; `test_e2e_audio_diff.py` uses the production path.
* **`MultivariateDecoder.layer_with_emits`** (`mlx/depthformer.py`)
  calls `self.depth_body(...)` (callable on a Serial — raises
  `TypeError`). Production only exercises the streaming
  `step_with_emits` path; per-module parity is covered piecewise.
* **`einops.rearrange` on `mx.array`** — sl's `_flatten_batch_time`
  uses einops, which has no MLX backend. Same impact as above.
* **`RVQ.codes_to_embeddings(use_unique_codes=True)`** uses
  non-existent `mx.mod`. Worked around by testing pure's behaviour
  internally rather than against sl.

## See also

* `tests/mlx_pure/parity/conftest.py` — fixtures and `assert_close`
  helper (a thin wrapper around `np.testing.assert_allclose`).
* `scripts/bench_mlx_vs_mlxpure.py` — side-by-side latency
  benchmark for `magenta_rt.mlx` vs `magenta_rt.mlx_pure` on the
  mrt2_small real checkpoint. mlx_pure is currently ~1.23×
  faster (11.8 vs 9.6 steps/s, batch=3, Apple Silicon CPU).
