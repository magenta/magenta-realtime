# `magenta_rt.mlx_pure.spectrostream`

Pure-MLX SpectroStream codec — encoder, decoder, residual vector
quantizer, and the streaming `step` machinery that lets the decoder
emit audio chunk-by-chunk. Mirrors the sl version's module structure
1:1 so weights bridge in via `bridge_spectrostream` / its narrower
siblings.

## Surface

| Symbol | Role |
|--------|------|
| `SpectroStream` | Top-level codec. `codes_to_waveform`, `waveform_to_codes`, plus `step_codes_to_waveform` and `enable_streaming` / `disable_streaming` for chunked decoding. |
| `SpectroStreamEncoder` / `SpectroStreamDecoder` | Conv-stack halves. `channel_splits=2` (stereo prod config) reshapes the per-channel prefix into the batch axis via `ParallelChannels`. |
| `SpectroStreamSTFT` / `SpectroStreamInverseSTFT` | Frontend / backend STFT layers. The inverse exposes streaming `step` backed by `OverlapAddCache`. |
| `ResidualVectorQuantizer` | RVQ embeddings + `codes_to_embeddings` / quantize path. |
| `Conv2DResidualUnit` | Building block — exposed for narrow parity tests. |
| `bridge_spectrostream`, `bridge_quantizer`, `bridge_spectrostream_encoder`, `bridge_spectrostream_decoder` | Per-subsystem sl → pure weight bridges. Re-exported at `magenta_rt.mlx_pure.load_weights` for convenience. |

## Streaming

`SpectroStream` has two modes:

```python
codec.disable_streaming()              # full-sequence forwards
audio = codec.codes_to_waveform(codes)

codec.enable_streaming()               # per-step state across calls
for chunk in code_chunks:
    audio_chunk = codec.step_codes_to_waveform(chunk)
```

Concatenated `step_codes_to_waveform` chunks are bit-equal (within
the bf16 floor) to a single `codes_to_waveform` over the joined
codes — verified in `tests/mlx_pure/parity/test_codec_streaming.py`.

`enable_streaming` walks the module subtree and flips three pieces
of state:

* Per-conv left-context buffers (`Conv2DCache` on every `Conv2D` /
  `Conv2DTranspose` in the decoder).
* The `OverlapAddCache` inside `SpectroStreamInverseSTFT`.
* The `SpectroStreamDecoder` lookahead countdown that hides the first
  N frames until the analysis window is full.

`ParallelChannels` handles its per-group state by reshaping groups
into the batch axis so the inner conv's own cache carries it — no
per-group cache list, no swap-in / swap-out.

## Bit-exact parity with sl

* `codes_to_waveform`: max diff `2.4e-05` over 46080 samples (bf16
  floor); streaming chunks match full-seq.
* `waveform_to_codes`: int codes match exactly (832/832 on 13 STFT
  frames × 64 RVQ levels).

Reproduce via `pytest tests/mlx_pure/parity/test_soundstream.py
tests/mlx_pure/parity/test_codec_streaming.py
tests/mlx_pure/parity/test_e2e_audio_diff.py`.
