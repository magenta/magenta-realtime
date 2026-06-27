# Inference

**JAX:**
```bash
# Generate 4 seconds of audio
mrt jax generate
```

**MLX:**
```bash
# Generate 4 seconds of audio
mrt mlx generate --bits=8
```

To print MusicCoCa tokens for a prompt directly without generating audio:

```python
>>> from magenta_rt.musiccoca import MusicCoCa
>>> m = MusicCoCa()
>>> emb = m.embed_text('a jazz piano trio')
>>> emb.shape
(768,)
>>> tokens = m.tokenize(emb)
>>> tokens.shape
(12,)
>>> tokens.tolist()
[555, 402, 72, 993, 482, 932, 402, 153, 33, 374, 519, 147]

# Get tokens from an audio file (returns a [1, 12] batched token list)
>>> from audiotree import AudioTree
>>> wav = AudioTree.from_file("jazz_piano_trio.wav")
>>> emb_audio = m.embed_audio(wav)
>>> emb_audio.shape
(1, 768)
>>> m.tokenize(emb_audio).shape
(1, 12)
>>> m.tokenize(emb_audio).tolist()
[[737, 949, 213, 432, 684, 652, 643, 850, 241, 512, 992, 964]]

# Collapsing the batch axis to get a flat 1D list of tokens
>>> emb_audio_single = emb_audio[0]
>>> emb_audio_single.shape
(768,)
>>> m.tokenize(emb_audio_single).tolist()
[737, 949, 213, 432, 684, 652, 643, 850, 241, 512, 992, 964]
```

## Bulk generation

Bulk-generate 60s audio clips from MusicCoCa prompts for listener evaluation:

```bash
python scripts/bulk_generate.py --size=mrt2_base
```

Outputs are saved to `outputs/eval_audio/<size>/`.
