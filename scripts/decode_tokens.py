# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode depthformer RVQ tokens to audio with the JAX SpectroStream decoder.

Lets the ggml port produce audio for listening tests before the codec itself is
ported: the codec is identical across precision variants, so it is a fair shared
back end when comparing fp32 / f16 / Q8_0 token streams.

Usage:
  python scripts/decode_tokens.py --tokens out.tokens.bin --output out.wav
"""

import argparse
import wave
from pathlib import Path

import flax.linen as nn
import flax.traverse_util as flaxtu
import jax
import numpy as np
import safetensors.flax as safetensors_flax
from jax import numpy as jnp

import magenta_rt  # noqa: F401 — activates vendored sequence_layers
import sequence_layers.jax as sl
from magenta_rt import paths
from magenta_rt.jax import model, spectrostream, system


class SpectroStreamDecoder(nn.Module):
    """Wraps the codec's codes -> embeddings -> waveform path."""

    cfg: spectrostream.SpectroStream.Config

    def setup(self):
        # Name must be "soundstream" so the checkpoint's param subtree matches.
        self.soundstream = self.cfg.make()

    def __call__(self, codes: sl.Sequence) -> sl.Sequence:
        x = self.soundstream.quantizer.codes_to_embeddings_layer.layer(
            codes, training=False)
        return self.soundstream.embeddings_to_waveform_layer.layer(x, training=False)


def to_int16(samples: np.ndarray, gain: float = 0.5) -> np.ndarray:
    """Mirrors system._float_samples_to_int16 (gain guards against clipping)."""
    s = np.clip(gain * samples, -1.0, 1.0)
    return np.round((np.iinfo(np.int16).max + 0.5) * s - 0.5).astype(np.int16)


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 48000) -> None:
    """audio: [num_samples, channels] float32 waveform (pre-gain)."""
    pcm = to_int16(audio)
    with wave.open(path, "wb") as f:
        f.setnchannels(pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def decode(tokens: np.ndarray, model_name: str = "mrt2_small") -> np.ndarray:
    """tokens: [num_frames, num_codebooks] in the depthformer unique scheme."""
    exp = model.get_model_class(model_name)()
    exp.compute_dtype = jnp.float32
    rvq = exp.spectrostream.rvq_truncation_level
    cfg = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=rvq, use_unique_codes=False,
    )

    codes = system.convert_from_unique_codes(
        tokens.astype(np.int32), codebook_size=cfg.quantizer.num_embeddings)
    seq = sl.Sequence.from_values(codes[np.newaxis].astype(np.int32))

    flat = safetensors_flax.load_file(str(paths.resolve_checkpoint(f"{model_name}.safetensors")))
    params = flaxtu.unflatten_dict({tuple(k.split("/")): v for k, v in flat.items()})

    decoder = SpectroStreamDecoder(cfg)
    out = decoder.apply(params, seq, rngs={"params": jax.random.PRNGKey(0)})
    return np.asarray(jax.device_get(out.values))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", required=True, help="int32 .bin of shape [frames, codebooks]")
    ap.add_argument("--num-codebooks", type=int, default=12)
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    toks = np.fromfile(args.tokens, dtype=np.int32).reshape(-1, args.num_codebooks)
    print(f"Decoding {toks.shape[0]} frames ({toks.shape[0] / 25.0:.2f} s)...")
    audio = decode(toks, args.model)
    print(f"audio {audio.shape} range [{audio.min():.3f}, {audio.max():.3f}]")
    write_wav(args.output, audio)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
