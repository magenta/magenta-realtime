# `magenta_rt.nnx`

A `flax.nnx` inference library for Magenta-RT. Runs on JAX
(CUDA / TPU / CPU) with a streaming `step` API; loads safetensors
checkpoints. Parameters live at `param_dtype` (default fp32) and
matmul / einsum activations live at `dtype` (default
bf16); RMSNorm always reduces in fp32. See
[Inspecting dtype flow](#inspecting-dtype-flow) to confirm with
`nnx.tabulate`.

## Quick start

```sh
python -m magenta_rt.nnx.generate \
    --model mrt2_small \
    --checkpoint mrt2_small.safetensors \
    --num-steps 100 --num-cfgs 2 \
    --temperature 1.0 --top-k 40 --cfg-musiccoca 3.0 --cfg-notes 1.0 \
    --jit \
    --output outputs/disco_funk.wav
```

Pass `--skip-restore` to run with random weights for an end-to-end smoke
test (no checkpoint needed).

## System API (`MagentaRT2System`)

The same system shape as the `jax` / `mlx` backends — `embed_style` →
`generate` → `(AudioTree, state)`:

```python
from magenta_rt import MagentaRT2Nnx  # = magenta_rt.nnx.system.MagentaRT2System

mrt = MagentaRT2Nnx(size="mrt2_small")
embedding = mrt.embed_style("disco funk")
audio_tree, state = mrt.generate(style=embedding, frames=25)   # 1 second
audio_tree, state = mrt.generate(style=embedding, frames=25, state=state)  # continue

# Batched: N styles -> N parallel streams in one call (waveform [N, 2, T]).
embeddings = mrt.embed_styles(["disco funk", "ambient drone"])
audio_tree, state = mrt.generate(style=embeddings, frames=25)   # N = 2
```

CFG uses the trained conditioning tokens (the `cfgs` channels), like the
jax/mlx systems; the logit-mixing CFG path (`--num-cfgs` with stacked
negative rows) stays on the `nnx.generate` research script below. One
backend caveat vs jax/mlx: `temperature`/`top_k` must be shared scalars
(per-element raises). The returned `state` is a `MagentaRT2State` carrying the
stream pytree (KV / codec caches, decode state, sampling rng), like the
jax/mlx `sl.State`: streams are independent values, so `state=None` starts a
fresh stream while any prior handle stays valid — multiple streams can be
interleaved on one system.

The step splits the model once per fresh stream into a constant parameter
partition (held on the system) and the stream partition, then threads only the
stream through a `jax.jit` step — params are passed as a non-donated constant
and the stream is donated for in-place cache updates, while `merge` / `split`
run at trace time. The per-step cost is then just array dispatch (markedly
faster than re-splitting the whole module each call, as `nnx.jit` would), and
nothing duplicates the parameter tree, which is what lets `mrt2_base` (bf16)
stream on a 16 GB GPU. Build base in bf16
(`from_preset(param_dtype=jnp.bfloat16, …)` + `load_checkpoint(host=True)`),
and give XLA a bit more of the card than its 75% default — the peak is
~10.9 GB and the default pool fragments just over it:

```sh
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python your_base_script.py
```

For batch generation, also pass `generate(..., scan=True)` (see below).

## Streaming API in your own code

```python
import jax.numpy as jnp
from flax import nnx
from magenta_rt.nnx import MagentaRT2Sampler

rngs = nnx.Rngs(0)
mrt = MagentaRT2Sampler.from_preset("mrt2_small", rngs=rngs)
mrt.load_checkpoint("checkpoints/pianorollbaseline_…")

@nnx.jit
def step(mrt, source_tokens, temperature=1.3, top_k=40):
    waveform = mrt.step(
        source_tokens=source_tokens,
        temperature=temperature, top_k=top_k,
    )
    return waveform.waveform  # [B, 2, 1920] channel-major AudioTree audio

mrt.init_streaming(batch_size=1, rngs=rngs)  # arms streaming caches and variables
for source_tokens in stream_of_source_tokens:
    audio_data = step(
        mrt,
        source_tokens=source_tokens,
        temperature=1.3, top_k=40,
    )
    play(audio_data)

mrt.disable_streaming()           # deallocates caches and disables streaming
```

## Public API

| Module              | Surface                                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| `nnx.attention`     | `LocalSelfAttention`, `StreamingCrossAttention`                                                                  |
| `nnx.cache`         | `LocalKVCache`, `OverlapAddCache`                                                                                |
| `nnx.configs`       | `ModelSpec`, `TokensConfig`, `MODEL_REGISTRY`, `get_model_class`, `MagentaRT2Model*`                             |
| `nnx.conv`          | `Conv2D`, `Conv2DTranspose`, `AveragePooling2D`, `Upsample2D`, `ParallelChannels`, `remove_cache`                |
| `nnx.depthformer`   | `DepthformerDecoder`, `EncoderDecoder`, `DecodeState`                                                            |
| `nnx.generate`      | CLI driver (`--skip-restore`, `--checkpoint`)                                                                    |
| `nnx.load_weights`  | `load_from_jax_safetensors`                                                                                      |
| `nnx.model`         | `MagentaRT2Sampler`                                                                                              |
| `nnx.sample_utils`  | `sample_categorical_with_temperature`                                                                            |
| `nnx.system`        | `MagentaRT2System` (exported as `magenta_rt.MagentaRT2Nnx`), `MagentaRT2State`                                   |
| `nnx.signal`        | `STFT`, `InverseSTFT`, `hann_window`, `inverse_stft_window_fn`, `frame`, `overlap_and_add`                       |
| `nnx.spectrostream` | `SpectroStream`, `SpectroStreamEncoder`, `SpectroStreamDecoder`, `ResidualVectorQuantizer`, `Conv2DResidualUnit` |
| `nnx.transformer`   | `TransformerBlock`, `Transformer`, `MultiChannelEmbedding`, `Encoder`                                            |

## Locked-feature subset

`nnx` only implements the configurations that ship in
`MODEL_REGISTRY`. Anything outside this subset raises
`NotImplementedError` at construction:

* `use_rope=False` (NoPE)
* `param_dtype=fp32` (kernel / scale storage), `dtype=bf16`
  (Linear / Einsum activations); RMSNorm reductions stay at fp32.
  See [Inspecting dtype flow](#inspecting-dtype-flow)
* separate Q + combined KV projections (no GQA, no ringbuffer)
* `ffn_gated=False`, `gelu_approx`, `ffn_use_bias=True`
* `norm_type=rms_normalization`, `norm_policy=primer_hybrid`
* `attention_per_dim_scale=True`, no soft-cap-on-attention, no bias
* `max_future_horizon=0` (fully causal)
* sink-embedding count: 1 (temporal) / 0 (depth)
* `soft_cap_logits=30.0` applied before depth-body sampling

## Inspecting dtype flow

`nnx.tabulate` prints the per-layer dtype of inputs, outputs, and
parameters — handy for confirming what's actually computed at which
precision. A production-shape temporal block:

```python
import jax.numpy as jnp
from flax import nnx
from magenta_rt.nnx.transformer import TransformerBlock

block = TransformerBlock(
    model_dim=1024, num_heads=8, units_per_head=128, ffn_dim=4096,
    max_past_horizon=64, num_sinks=1,
    param_dtype=jnp.float32, dtype=jnp.bfloat16,
    rngs=nnx.Rngs(0),
)
x = jnp.ones((1, 8, 1024), dtype=jnp.bfloat16)
print(nnx.tabulate(block, x, depth=3, console_kwargs={"width": 200}))
```

The printed table shows the textbook mixed-precision pattern:
kernels at `float32`, every `Linear` / `Einsum` output at `bfloat16`,
every `RMSNorm` output back at `float32` (so the mean-square
reduction stays accurate). Production `ModelSpec` defaults to
`dtype=bfloat16`; pass `dtype=jnp.float32` for
fp32 throughout.

## Tests

```sh
pytest tests/nnx -m "not slow and not checkpoint"
pytest tests/nnx -m "checkpoint"   # needs checkpoints/*.safetensors
```
