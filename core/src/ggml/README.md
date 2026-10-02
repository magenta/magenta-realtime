# MagentaRT-2 → ggml port

Status report for the ggml/C++ port of Magenta RealTime 2 (`mrt2_small`), so the
model can run anywhere ggml runs rather than only on Apple silicon via MLX.

**Bottom line:** the depthformer LLM and the SpectroStream codec are both ported
and numerically verified against the JAX fp32 reference. The LLM is competitive
with (often faster than) the MLX path. **The codec is correct but slow**, and it
is the reason the end-to-end comparison against `benchmark_mlxfn` loses.

---

## 1. Status

| Component | Ported | Parity vs JAX fp32 | Notes |
| --- | --- | --- | --- |
| Depthformer encoder | ✅ | `2.62e-06` | embedders + LayerNorm (no transformer layers) |
| Temporal decoder (12 layers) | ✅ | `3.24e-05` | self + cross attention, sinks, windowed KV |
| Depth decoder (2 layers × 12 RVQ) | ✅ | `3.05e-05` … `1.18e-04` | teacher-forced |
| Streaming (frame 1, caches reused) | ✅ | `1.07e-05` | |
| Deep streaming (45 frames) | ✅ | passes | regression test, catches the cache-wrap bug |
| Sampling (temp / top-k / CFG) | ✅ | n/a | RNG parity not attainable across JAX/C++ |
| SpectroStream decoder | ✅ | `1.19e-06` (CPU) | incl. inverse STFT; **NaN on Metal** (§5) |

Tolerance is `1e-3`, matching `tests/test_bitlevel_parity.py`.

---

## 2. Results

### Parity

```
-- encoder --                encoder           2.6226e-06  PASS
-- temporal decoder --       temporal_outputs  3.2425e-05  PASS
-- depth decoder --          rvq=0 … rvq=11    3.05e-05 … 1.18e-04  PASS
-- streaming (step 1) --     temporal_outputs  1.0729e-05  PASS
-- codec (CPU) --            full incl. ISTFT  1.1919e-06  PASS
```

`tests/test_ggml_numpy_parity.py`: 4 passed.

### Speed — per frame, 25 Hz ⇒ 40 ms realtime budget

| Config | LLM only | LLM + codec | Correct? |
| --- | --- | --- | --- |
| **MLX `benchmark_mlxfn`** (8-bit LLM, fp32 codec, Metal) | — | **18.02** | reference |
| ggml Metal fp32 | 19.50 | 56.67 | codec NaN |
| ggml Metal Q8_0 | 14.15 | 51.51 | codec NaN |
| ggml CPU fp32 | 23.34 | **95.51** | ✅ |
| ggml CPU Q8_0 | **9.63** | 82.95 | ✅ (LLM quantized) |

The LLM alone beats MLX's *full* step at Q8_0 on CPU (9.63 vs 18.02). The codec
adds ~72 ms on CPU / ~37 ms on Metal, against roughly ~4 ms inside MLX's 18 ms.

Note the workload is **dispatch-bound, not FLOP-bound** (batch 1, 13 sequential
graph launches per frame), which is why CPU is competitive with Metal and why
Q8_0 on CPU wins outright — ggml's int8 SIMD kernels are excellent.

### Quantization quality

Weight-only, teacher-forced, 45 frames / 540 tokens:

| | temporal err | greedy tokens = fp32 | JAX token still in top-40 |
| --- | --- | --- | --- |
| fp32 | 5.3e-05 | 540/540 | 540/540 |
| f16 | 2.6e-02 | 539/540 | 540/540 |
| Q8_0 | 4.0e-01 | 518/540 (95.9%) | 535/540 |

Listening test on 10 s of generated audio ranked **f32 > f16 > q8**.

Absolute logit error overstates the damage (logits are soft-capped to ±30 and
top-1 gaps are usually wider). f16 is essentially decision-identical.

---

## 3. What was built

```
magenta_rt/ggml/
  convert.py            Flax safetensors → GGUF (374 tensors, bit-exact round-trip)
  numpy_model.py        Depthformer spec in numpy — the C++ was written from this
  numpy_codec.py        SpectroStream decoder spec in numpy
core/src/ggml/
  mrt_ggml.{h,cpp}      Model load, KV + codec state, all graph builders
  parity_main.cpp       Stage-by-stage parity harness vs JAX references
  benchmark_main.cpp    Latency benchmark (mrt_bench)
  generate_main.cpp     Free-running generation with sampling (mrt_generate)
  CMakeLists.txt        Standalone; pulls ggml from a llama.cpp checkout
scripts/
  generate_ggml_reference.py   JAX fp32 per-layer references (no model edits needed)
  decode_tokens.py             Decode RVQ tokens → WAV via the JAX codec
  ggml_latency_benchmark.sh    Convert + build + benchmark
tests/
  test_ggml_numpy_parity.py    numpy-vs-JAX, incl. the 45-frame deep-stream test
  testdata/ggml/               ~96 MB of reference tensors (3 sets)
```

**Writing the numpy model first was the decision that made this tractable.** Four
semantics (§4) were impossible to read off the source and would each have been a
silent wrong-output bug; resolving them in numpy took minutes, whereas each
ggml debug cycle costs a rebuild.

---

## 4. Non-obvious things the references taught us

Each of these produces a model that *runs* and emits garbage:

1. **Attention sink logits use the *unscaled* query**, while real keys use the
   scaled one ([`attention.py:329`](../../../magenta_rt/_vendor/sequence-layers/sequence_layers/mlx/attention.py#L329)).
   Output was off by 10.8 until found.
2. **`output_projection` contracts the head axes** (`hd,ohd->o`). Both readings
   are shape-legal because 1024 = 8×128 — only parity distinguishes them.
3. **GELU (tanh) is folded into `ffn_layer1`'s output**; `ffn_layer2` applies none.
4. **Query scaling is Praxis-style**: `1.442695041 · (1/√dh) · softplus(per_dim_scale)`.
5. **KV cache holds `max_past_horizon + 1` frames, not `max_past_horizon`.** JAX's
   `LocalDotProductSelfAttention` concatenates the current frame onto its
   horizon-sized buffer, so a horizon of 41 means 42 attended positions. Sizing
   it at 41 is correct for 41 steps and then diverges hard (`4e-1`) at step 41 —
   invisible to any short test. `TestDeepStreamParity` guards this.
6. **The model is NoPE**, so attention is order-invariant over cached keys: a
   plain circular buffer + validity mask replaces the reference's shifting buffer.
7. **The encoder has zero transformer layers** — `body=sl.Identity.Config()`
   ([`model.py:362`](../../../magenta_rt/mlx/model.py#L362)); `encoder_size` only
   supplies the width. GGUF records `encoder.block_count = 0`.
8. **JAX's `conv_transpose` flips the kernel**, MLX's does not (documented in
   [`spectrostream/load_weights.py`](../../../magenta_rt/mlx/spectrostream/load_weights.py#L103)).
   Wrong by 200× until applied.
9. **`semicausal` padding** = `(max(ek−stride,0), ek−1−left)`, which degenerates
   to fully causal at stride 1 (all the decoder's regular convs).
10. **Codec channel groups share weights** (`share_scope=True`): `decoder_1..6`
    plus the output conv run twice over 512-channel halves, then concat.
11. **`keep_dc=True`** ⇒ the inverse STFT pads the Nyquist bin at the *end*;
    window is Griffin & Lim eq. 7.

### ggml-specific constraints hit

- **GGUF caps tensor names at 64 chars**; Flax paths reach 73 → compact name map.
- **ggml has no FFT op.** The inverse STFT is a precomputed real-IDFT matmul
  (`[481×960]`, ~0.5 MFLOP/frame/channel) — portable to every backend.
- **`ggml_conv_transpose_2d_p0` takes one stride for both axes**, but the codec's
  ratios are anisotropic → transposed conv = zero-upsample + conv with the
  original kernel. ggml has no interleave op, so zero-upsampling concatenates a
  zeroed copy on a spare axis and reshapes.
- **Metal requires `src[0]` to be F32 for ADD/SUB/MUL/DIV.** With `--dtype f16`
  or `q8_0`, the sink embeddings, norms/biases and the embedding table stay fp32;
  only matmul/conv weights convert.
- **Dequantization timing is free on Metal** — `mul_mv_q8_0_f32` (weights
  unpacked in-register, fp32 activations) is bit-identical to pre-dequantizing.
  On **CPU** it is not: quantized storage selects the integer path, which also
  quantizes *activations* (`vec_dot_type`), costing ~1.8× more error. That same
  mechanism is why CPU Q8_0 is the fastest config.
- **CPU fp32 is ~60× looser than Metal fp32 in absolute terms** (2.05e-3 vs
  3.24e-5 on `temporal_outputs`) — but that is 2.2e-5 *relative*; accumulation
  order, not a port error. It marginally exceeds the 1e-3 absolute gate.

---

## 5. Known issues

1. **Metal `kernel_im2col_f32` produces NaN on the codec's conv shapes.** The
   same graph is exact on CPU; `GGML_METAL_GRAPH_OPTIMIZE=0` and
   `GGML_METAL_CONCURRENCY=0` don't help. Bisected to group 1's `decoder_6`, all
   NaN at the last time frame in a narrow frequency band. This is a ggml backend
   issue, not the port — but it means the *fast* path is the broken one.
2. **Codec performance.** `ggml_conv_2d` is im2col + matmul, materializing a
   `[positions × 9·C]` matrix per conv (~3 GMAC/frame over 13 conv blocks).
   `ggml_conv_2d_direct` exists and would likely close much of the gap to MLX —
   **not yet tried, and the single highest-value next step.**
3. **Codec streaming is validated on frame 0 only.** Per-conv causal history is
   implemented but not verified across a long run. Given §4.5, this should get a
   multi-frame test before multi-frame ggml audio is trusted.
4. **Q8_0 fails the fp32 parity gate** (expected). Judge it by token agreement
   and listening, not by `1e-3`.

---

## 6. How to run

All commands are run from the **repository root**, with the venv active.

```bash
# convert (choose --dtype f32 | f16 | q8_0)
python -m magenta_rt.ggml.convert \
  --checkpoint ~/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors \
  --output ~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small_f32.gguf \
  --model mrt2_small --dtype f32

# build (NOT -B build: the repo root's build/ belongs to the top-level project)
cmake -B core/src/ggml/build -S core/src/ggml -DGGML_METAL=ON
cmake --build core/src/ggml/build -j --target mrt_parity mrt_bench mrt_generate

# parity (append `cpu` to force the CPU backend)
./core/src/ggml/build/mrt_parity <model.gguf> tests/testdata/ggml/mrt2_small
./core/src/ggml/build/mrt_parity <model.gguf> tests/testdata/ggml/mrt2_small_codec cpu

# benchmark: [frames] [cpu] [codec] [rebuild]
./scripts/ggml_latency_benchmark.sh              # Metal, LLM only
./scripts/ggml_latency_benchmark.sh cpu codec    # CPU, full step

# generate audio: ggml emits tokens, the JAX codec decodes them
./core/src/ggml/build/mrt_generate <model.gguf> tests/testdata/ggml/mrt2_small \
    outputs/ggml/tokens.bin 250 0
python scripts/decode_tokens.py --tokens outputs/ggml/tokens.bin \
    --output outputs/ggml/out.wav
```

Regenerate references (needs JAX):

```bash
python scripts/generate_ggml_reference.py --model=mrt2_small --num-steps=2
python scripts/generate_ggml_reference.py --model=mrt2_small --num-steps=45 --minimal \
    --outdir=tests/testdata/ggml/mrt2_small_long
```

Requires a llama.cpp checkout for ggml sources; default
`~/code/personal_projects/llama.cpp/ggml`, override with `-DGGML_SOURCE_DIR=`.

---

## 7. Suggested next steps

1. Try **`ggml_conv_2d_direct`** for the codec — the one change most likely to
   make the end-to-end comparison competitive.
2. **File / work around the Metal im2col NaN** so the GPU path is usable.
3. Add a **multi-frame codec streaming test** (§5.3).
4. Re-run the dequantization A/B **with CFG enabled** (batch > 1): Metal switches
   from `mul_mv` to `mul_mm`, which was never measured here.
