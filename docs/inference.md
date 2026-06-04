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
from magenta_rt.musiccoca import MusicCoCa
m = MusicCoCa()
print(m.tokenize(m.embed('a jazz piano trio')).tolist())

# Get tokens from audio
from magenta_rt.audio import Waveform
wav = Waveform.from_file("jazz_piano_trio.wav")
print(m.tokenize(m.embed(wav)).tolist())
```

## Bulk generation

Bulk-generate 60s audio clips from MusicCoCa prompts for listener evaluation:

```bash
python scripts/bulk_generate.py --size=mrt2_base
```

Outputs are saved to `outputs/eval_audio/<size>/`.
