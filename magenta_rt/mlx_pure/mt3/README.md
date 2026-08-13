# `magenta_rt.mlx_pure.mt3` — MT3 in pure MLX

MLX port of the [`magenta_rt.nnx.mt3`](../../nnx/mt3/README.md)
transcription model (see that README for the architecture and checkpoint
table). The framework-neutral core — event vocabulary, `NoteSequence`
decoding, checkpoint download/conversion, configs, mel filterbank — is
shared via `magenta_rt/mt3/`; this package adds the MLX network
(`t5_layers.py`, `model.py`), spectrogram frontend, and an **eager**
greedy decoder (python loop with a KV cache that exits as soon as every
sequence in the batch hits EOS — no compiled `while_loop` needed).

```sh
mrt mt3 download mt3   # once; shared with the nnx port (or python -m magenta_rt.mt3.download mt3)
```

```python
from magenta_rt.mlx_pure.mt3 import load_model, transcribe

model = load_model("mt3")
ns = transcribe(model, samples_16khz)
ns.write_midi("out.mid")
```

Parity (gated tests, mark `checkpoint`): encode/logits within ~5e-4 of
the nnx port (itself bit-exact vs the original Linen network) and
**greedy tokens match exactly**, so transcriptions are identical across
backends. Use it in the SFT export via
`magenta_rt.sft.mt3_transcriber(backend="mlx_pure")`.
