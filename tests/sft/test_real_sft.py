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

"""End-to-end *real* SFT: mrt2_small + LoRA on a real export.

The one test that exercises the whole training chain against the real
model and the real data path — everything the toy-config POC tests
can't see (real vocabulary sizes, the full 6-channel source including
the PrepareCFG-synthesized guidance channels, real RVQ codes, the
pretrained checkpoint loader):

  synthesize audio
    → export_tree_dataset (real SpectroStream + real nnx MusicCoCa +
      real MT3 transcription)
    → create_audiotree_dataset (crop / sticky / PrepareCFG / source prep)
    → LoRA fine-tune mrt2_small for ~30 steps
    → assert the loss decreases.

Gated on the local mrt2_small checkpoint, the converted MusicCoCa
safetensors, and the MT3 checkpoint; marked ``checkpoint`` and ``slow``
(runs ~3 minutes on a CPU-JAX MacBook). Opt in with::

    pytest tests/sft/test_real_sft.py -m "checkpoint and slow"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("audiotree")

from magenta_rt import paths

pytestmark = [pytest.mark.checkpoint, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = paths.resolve_checkpoint("mrt2_small.safetensors")
MUSICCOCA_WEIGHTS = paths.musiccoca_dir() / "musiccoca_nnx.safetensors"
MT3_WEIGHTS = paths.mt3_dir() / "mt3_mt3.safetensors"

# Make notebooks/sft importable (train_nnx + its sibling utils).
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "sft"))


def _skip_unless_resources():
    for path, hint in [
        (CHECKPOINT, "mrt2_small checkpoint"),
        (MUSICCOCA_WEIGHTS, "MusicCoCa safetensors (nnx.musiccoca.convert)"),
        (MT3_WEIGHTS, "MT3 checkpoint (magenta_rt.mt3.download)"),
    ]:
        if not path.exists():
            pytest.skip(f"{hint} not found at {path}")


def _synthesize_arpeggio(seconds: float, sr: int = 48_000) -> np.ndarray:
    """Looping harmonics-rich arpeggio — enough structure to overfit on."""

    def tone(pitch, start, dur):
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        t = np.arange(int(dur * sr)) / sr
        x = sum((0.6**k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        return start, (x * np.exp(-2.5 * t) * 0.3).astype(np.float32)

    audio = np.zeros(int(seconds * sr), np.float32)
    pattern = [(60, 0.0), (64, 0.5), (67, 1.0), (72, 1.5)]
    bar = 2.0
    for bar_start in np.arange(0.0, seconds - bar, bar):
        for pitch, offset in pattern:
            start, x = tone(pitch, bar_start + offset, 1.0)
            i = int(start * sr)
            audio[i : i + len(x)] += x[: len(audio) - i]
    return audio


def test_real_lora_sft_loss_decreases(tmp_path):
    _skip_unless_resources()

    import jax.numpy as jnp
    import soundfile
    from flax import nnx

    import train_nnx as sft  # notebooks/sft/train_nnx.py
    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.model import MODEL_REGISTRY
    from magenta_rt.nnx.musiccoca import MusicCoCa
    from magenta_rt.sft import create_audiotree_dataset, to_source_target
    from magenta_rt.sft.configs import SFTConfig
    from magenta_rt.sft.export import export_tree_dataset, mt3_transcriber
    from magenta_rt.sft.lora_nnx import MRTLoRAParam, inject_lora

    spec = MODEL_REGISTRY["mrt2_small"]()

    # --- 1. Real export: codec + MusicCoCa + MT3 on synthesized audio ----
    audio = _synthesize_arpeggio(seconds=42.0)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    soundfile.write(audio_dir / "loop.wav", np.stack([audio, audio], axis=1), 48_000)

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0)
    )
    mrt.load_checkpoint(CHECKPOINT)
    data_dir = export_tree_dataset(
        audio_dir,
        tmp_path / "dataset",
        codec=mrt.spectrostream,
        style_model=MusicCoCa(),
        transcriber=mt3_transcriber("mt3"),
        num_samples=4,  # four 10 s salient excerpts
        duration=10.0,
    )
    del mrt  # free the inference stack before building the trainable model

    # --- 2. Training pipeline over the export (full 6-channel source) ----
    # PrepareCFG synthesizes the guidance channels the export doesn't store.
    ds = create_audiotree_dataset(
        data_dir,
        batch_size=2,
        crop_length_seconds=1,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config,
        seed=0,
        tree_exclude_prefixes=["extras.musiccoca_embedding"],
    )
    it = iter(ds)

    # --- 3. LoRA fine-tune the real depthformer -------------------------
    model = sft.build_model(spec, seed=0, checkpoint_path=str(CHECKPOINT))
    n_wrapped = inject_lora(model, rank=4, alpha=8.0, seed=0)
    assert n_wrapped > 0
    config = SFTConfig(learning_rate=1e-2, warmup_steps=0)
    optimizer = sft.build_optimizer(model, config, wrt=MRTLoRAParam)
    train_step = sft.make_train_step(diff_filter=MRTLoRAParam)

    losses = []
    for _ in range(30):
        source, target = to_source_target(
            next(it), spec.target_tokens_config, asarray=jnp.asarray
        )
        assert source.shape[-1] == spec.input_num_channels  # all 6 channels
        metrics = train_step(model, optimizer, source, target)
        losses.append(float(metrics["loss"]))

    assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
    first, last = np.mean(losses[:5]), np.mean(losses[-5:])
    assert last < first - 0.05, (
        f"loss did not decrease on the real model: first5={first:.4f} "
        f"last5={last:.4f}\nlosses={np.round(losses, 4).tolist()}"
    )


def test_real_audio_sample_writer(tmp_path):
    """The opt-in audio sampler generates finite stereo audio from the live
    LoRA model + codec without disturbing the training graph."""
    _skip_unless_resources()

    import jax.numpy as jnp
    from flax import nnx

    import train_nnx as sft
    from magenta_rt.nnx.model import MODEL_REGISTRY
    from magenta_rt.sft.configs import SFTConfig
    from magenta_rt.sft.lora_nnx import MRTLoRAParam, inject_lora

    spec = MODEL_REGISTRY["mrt2_small"]()
    model = sft.build_model(spec, seed=0, checkpoint_path=str(CHECKPOINT))
    inject_lora(model, rank=2, alpha=4.0, seed=0)
    config = SFTConfig(sample_every_steps=1, lora_rank=2, lora_alpha=4.0)

    writer_calls = {}
    writer_scalars = {}

    class _Writer:
        def write_audios(self, step, audios, *, sample_rate):
            writer_calls.update(audios)
            writer_calls["sample_rate"] = sample_rate

        def write_scalars(self, step, scalars):
            writer_scalars.update(scalars)

    sample_writer = sft.AudioSampleWriter(
        model=model, diff_filter=MRTLoRAParam, config=config,
        model_name="mrt2_small", checkpoint_path=str(CHECKPOINT),
    )
    assert sample_writer.available

    rng = np.random.RandomState(0)
    # Conditioning shaped like the pipeline's prepared source (2 s crop).
    source = jnp.asarray(
        rng.randint(6, 40, (2, 50, spec.input_num_channels)).astype(np.int32)
    )
    sample_writer.set_source(source)
    sample_writer(_Writer(), step=1)

    audio = writer_calls["sample/audio"]
    assert writer_calls["sample_rate"] == 48_000
    seconds = sample_writer.SAMPLE_SECONDS
    assert audio.shape == (1, min(50, seconds * 25) * 1920, 2)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0
    assert np.abs(audio).max() > 0  # not silence
    # Energy diagnostics are logged alongside the clip (gen/frac_silent flags
    # the energy-collapse failure mode).
    assert {"gen/rms", "gen/frac_silent"} <= set(writer_scalars)
    assert np.isfinite(writer_scalars["gen/rms"])
