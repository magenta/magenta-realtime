# `magenta_rt.nnx.mt3` — Multi-Task Multitrack Music Transcription

A JAX/NNX port of [magenta/mt3](https://github.com/magenta/mt3) for
inference (the network includes Dropout layers so it can also be
fine-tuned). In magenta-rt it supplies the **piano-roll conditioning
channels** for SFT data: `magenta_rt.sft.export` transcribes each audio
window with MT3 alongside the SpectroStream + MusicCoCa encoding.

## Architecture

T5.1.1 encoder-decoder (`model.py`) with two differences from standard T5:

- Fixed sinusoidal absolute position embeddings (`layers.FixedEmbed`)
  instead of relative position biases.
- The encoder consumes continuous inputs — log mel spectrogram frames
  (`[batch, 256 frames, 512 mel bins]`) — through a linear projection
  instead of a token embedding.

Transformer building blocks are vendored in `t5_layers.py` (T5 attention /
MLP / LayerNorm / embedding in NNX).

The decoder emits tokens from an event vocabulary (`event_codec.py`,
`vocabularies.py`): time shifts (10 ms steps), pitch, velocity, program,
drum, and tie events. `note_sequences.py` and `run_length_encoding.py`
decode token streams into a lightweight `NoteSequence` (no `note_seq`
protobuf dependency) with MIDI export via `pretty_midi`. The
framework-neutral pieces (event codec, vocabularies, note sequences,
download, configs, numpy spectrogram helpers) live in `magenta_rt/mt3/`,
shared with the `mlx_pure` port.

## Pretrained checkpoints

Downloaded from the public `gs://mt3` bucket and converted to safetensors
under `magenta_rt.paths.mt3_dir()` (default
`~/Documents/Magenta/magenta-rt-v2/resources/mt3`; `download.py`, no
gcloud needed):

| model_type        | Task                            | Params | inputs_length | velocity bins | ties |
|-------------------|---------------------------------|--------|---------------|---------------|------|
| `mt3`             | multitrack (Slakh, etc.)        | 46M    | 256           | 1             | yes  |
| `ismir2021`       | piano-only with velocities      | 46M    | 512           | 127           | no   |
| `ismir2022_small` | multitrack + mixture augment    | 46M    | 256           | 1             | yes  |
| `ismir2022_base`  | multitrack + mixture augment    | 114M   | 256           | 1             | yes  |

```sh
mrt mt3 download mt3   # or: python -m magenta_rt.mt3.download mt3
```

## Usage

```python
from magenta_rt.nnx.mt3 import load_model, transcribe

model = load_model("mt3")        # multitrack; or "ismir2021" for piano
ns = transcribe(model, samples)  # mono float audio at 16 kHz
ns.write_midi("out.mid")
```

`transcribe` splits audio into non-overlapping segments of
`config.inputs_length` frames (hop 128 @ 16 kHz → 2.048 s for `mt3`),
computes log mel spectrograms, greedily decodes each segment
(jit-compiled `lax.while_loop` with KV cache), and stitches the
per-segment events into one `NoteSequence`, carrying notes across segment
boundaries via tie events.

## Porting notes

- Originally ported (and verified bit-exact against the original Linen
  `network.Transformer` with pretrained `mt3` weights) in DBraun's JAX
  monorepo; vendored here with that project's dependencies inlined
  (`t5_layers.py`) and the checkpoint cache moved to
  `magenta_rt.paths.mt3_dir()`. Parity test:
  `tests/nnx/parity/test_mt3_pretrained.py` (gated, mark `checkpoint`).
- The spectrogram frontend replicates `tf.signal` semantics (periodic
  Hann, no centering, `pad_end=True`, HTK mel triangles computed in the
  mel domain, `log(max(x, 1e-5))`) rather than librosa conventions;
  agreement with TF is checked in `tests/nnx/test_mt3.py`.
