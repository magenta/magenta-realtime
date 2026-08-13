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

"""fp32 SpectroStream codec parity: ``magenta_rt.mlx_pure`` vs JAX/Linen.

The depthformer logit-parity tests pin *token generation* against the
Linen reference; this pins the **codec** — the ``codes -> waveform``
path that turns those tokens into the audio you actually hear (the RVQ
table lookup, the transpose-conv decoder stack, the InverseSTFT +
overlap-add). A bug here produces bad audio while every logit test
stays green.

Both pipelines run at fp32, which is the codec's *natural* config —
JAX's ``stft_soundstream_..._config`` defaults to
``compute_dtype=float32``, ``mlx_pure.SpectroStream`` defaults to
``compute_dtype=float32``, and the shipped checkpoint's soundstream
params are fp32. At fp32 the two backends agree to ~1e-4
(cross-framework round-off).

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``; auto-skips
when JAX is unavailable or the checkpoint file is absent.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_NAME = "mrt2_small.safetensors"
_NUM_CODEBOOKS = 12  # mrt2_small rvq truncation level
_CODEBOOK_SIZE = 1024


@pytest.fixture
def smallm4air_checkpoint():
    from magenta_rt import paths as _paths
    p = pathlib.Path(_paths.resolve_checkpoint(_CHECKPOINT_NAME))
    if not p.exists():
        pytest.skip(f"checkpoint not found: {p}")
    return p


def _rvq_codes(num_frames: int = 8, seed: int = 0) -> np.ndarray:
    """Deterministic ``[1, num_frames, 12]`` block of per-codebook RVQ
    indices in ``[0, codebook_size)``."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, _CODEBOOK_SIZE, size=(1, num_frames, _NUM_CODEBOOKS)).astype(np.int32)


def _jax_codes_to_waveform(checkpoint_path: Path, codes: np.ndarray) -> np.ndarray:
    """JAX/Linen ``SoundStream.codes_to_waveform`` at fp32. Returns the
    waveform as fp32 numpy, shape ``[1, samples, channels]``."""
    import jax.numpy as jnp
    import sequence_layers.jax as sl
    from magenta_rt.jax import model as jm, system as jsys, spectrostream as jss
    from magenta_rt.jax.system import _load_jax_weights as load_jax_weights

    spec = jm.get_model_class("mrt2_small")()
    mrt = jsys.MagentaRT2Sampler.Config(
        depthformer=spec.depthformer_config(),
        spectrostream=jss.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=spec.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        ),
    ).make()
    params = load_jax_weights(checkpoint_path)
    codes_seq = sl.Sequence.from_values(jnp.asarray(codes))
    # ``method`` may be a plain function whose first arg is the bound module.
    wav = mrt.apply(
        params, codes_seq,
        method=lambda module, cs: module.soundstream.codes_to_waveform(
            cs, training=False,
        ),
    )
    return np.asarray(wav.values, np.float32)


@pytest.mark.checkpoint
@pytest.mark.slow
def test_spectrostream_codes_to_waveform_jax_parity(smallm4air_checkpoint):
    """``codes -> waveform`` through the full SpectroStream decoder must
    match the JAX/Linen codec at fp32."""
    pytest.importorskip("jax")
    pytest.importorskip("sequence_layers.jax")
    import mlx.core as mx
    from magenta_rt.mlx_pure import model as pure_model

    codes = _rvq_codes()
    jax_wav = _jax_codes_to_waveform(smallm4air_checkpoint, codes)

    mrt = pure_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False,
    )
    mrt.load_from_safetensors(smallm4air_checkpoint)
    # codes_to_waveform is the non-streaming forward; SpectroStream
    # defaults to compute_dtype=float32, so no dtype override needed.
    pure_wav = np.asarray(
        mrt.spectrostream.codes_to_waveform(mx.array(codes)).astype(mx.float32)
    )
    # mlx_pure audio is channel-major [1, C, T]; the sl-backed jax codec
    # keeps [1, T, C] — compare in the jax layout.
    pure_wav = pure_wav.swapaxes(1, 2)

    assert jax_wav.shape == pure_wav.shape, (
        f"shape mismatch: jax {jax_wav.shape} vs mlx_pure (transposed) {pure_wav.shape}"
    )
    # fp32 cross-framework round-off through the conv decoder +
    # InverseSTFT is ~1e-4 on a waveform whose peak is ~0.7; 1e-3 keeps
    # a comfortable margin while still catching a real codec divergence
    # (wrong conv cache, transposed kernel, overlap-add bug).
    diff = float(np.abs(jax_wav - pure_wav).max())
    assert diff < 1e-3, (
        f"codes_to_waveform diverges: max|diff|={diff:.6f} "
        f"(|val|max={np.abs(jax_wav).max():.4f})"
    )
