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

"""fp32 SpectroStream codec parity: ``magenta_rt.nnx`` vs JAX/Linen.

The depthformer logit-parity test pins *token generation* against the
Linen reference; this pins the **codec** — the ``codes -> waveform``
path that turns those tokens into the audio you actually hear (the RVQ
table lookup, the transpose-conv decoder stack, the InverseSTFT +
overlap-add). A bug here produces bad audio while every logit test
stays green.

The comparison is at fp32. Note ``nnx.MagentaRT2Sampler.from_preset`` builds
the codec at the spec's ``dtype`` (bf16); this test flips it back to
fp32 via ``set_attributes(dtype=float32)`` so the comparison is a clean
implementation-vs-implementation check. JAX's SoundStream config and
the shipped checkpoint's soundstream params are already fp32. At fp32
the two backends agree to ~1e-4 (cross-framework round-off).

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``; auto-skips
when JAX is unavailable or the checkpoint file is absent.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pytest
from flax import nnx


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
    import jax.numpy as jnp
    from magenta_rt.nnx import model as nnx_model

    codes = _rvq_codes()
    jax_wav = _jax_codes_to_waveform(smallm4air_checkpoint, codes)

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0)
    )
    mrt.load_checkpoint(smallm4air_checkpoint)
    # from_preset builds the codec at the spec's dtype (bf16); flip it to
    # fp32 for a clean implementation-vs-implementation comparison.
    mrt.spectrostream.set_attributes(dtype=jnp.float32, raise_if_not_found=False)
    # codes_to_waveform is the non-streaming forward.
    mrt.spectrostream.set_attributes(streaming=False, raise_if_not_found=False)
    nnx_wav = np.asarray(
        mrt.spectrostream.codes_to_waveform(jnp.asarray(codes)), np.float32
    )
    # nnx audio is channel-major [1, C, T]; the sl-backed jax codec keeps
    # [1, T, C] — compare in the jax layout.
    nnx_wav = nnx_wav.swapaxes(1, 2)

    assert jax_wav.shape == nnx_wav.shape, (
        f"shape mismatch: jax {jax_wav.shape} vs nnx (transposed) {nnx_wav.shape}"
    )
    # fp32 cross-framework round-off through the conv decoder +
    # InverseSTFT is ~1e-4 on a waveform whose peak is ~0.7; 1e-3 keeps
    # a comfortable margin while still catching a real codec divergence
    # (wrong conv cache, transposed kernel, overlap-add bug).
    diff = float(np.abs(jax_wav - nnx_wav).max())
    assert diff < 1e-3, (
        f"codes_to_waveform diverges: max|diff|={diff:.6f} "
        f"(|val|max={np.abs(jax_wav).max():.4f})"
    )


@pytest.mark.checkpoint
@pytest.mark.slow
def test_streaming_matches_nonstreaming_codec(smallm4air_checkpoint):
    """The streaming ``step_codes_to_waveform`` must reconstruct the same audio
    as the (jax-parity-tested) non-streaming ``codes_to_waveform``.

    The streaming decoder runs the conv stack + InverseSTFT through per-frame
    caches instead of one full-sequence forward, and emits audio delayed by
    exactly one codec frame (``decoder_lookahead=1``). After undoing that
    one-frame latency the two paths must agree to fp32 round-off. This is the
    one codec route the jax parity test above does not exercise; pinning it here
    guards against a streaming conv-cache or overlap-add regression (the
    streaming path is otherwise transitively correct vs jax: streaming == this
    non-streaming == jax non-streaming).
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from magenta_rt.nnx import model as nnx_model

    num_frames = 16
    codes = _rvq_codes(num_frames=num_frames, seed=1)

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0)
    )
    mrt.load_checkpoint(smallm4air_checkpoint)
    ss = mrt.spectrostream
    ss.set_attributes(dtype=jnp.float32, raise_if_not_found=False)

    # Non-streaming reference (full-sequence forward).
    ss.set_attributes(streaming=False, raise_if_not_found=False)
    ss.remove_cache()
    w_ns = np.asarray(ss.codes_to_waveform(jnp.asarray(codes)), np.float32)  # [1, C, T]

    # Streaming: arm caches, decode one input frame at a time.
    ss.set_attributes(streaming=True, raise_if_not_found=False)
    ss.init_cache(batch=1, dtype=jnp.float32)
    chunks = [
        np.asarray(ss.step_codes_to_waveform(jnp.asarray(codes[:, t : t + 1])), np.float32)
        for t in range(num_frames)
    ]
    w_s = np.concatenate(chunks, axis=-1)  # [1, C, T'] — one frame longer (the lookahead)

    # Streaming lags the non-streaming forward by exactly one codec frame; drop
    # that leading frame and compare the common span.
    samples_per_frame = w_s.shape[-1] // num_frames
    aligned_s = w_s[..., samples_per_frame:]
    n = min(aligned_s.shape[-1], w_ns.shape[-1])
    assert n > 0
    diff = float(np.abs(aligned_s[..., :n] - w_ns[..., :n]).max())
    assert diff < 1e-3, (
        f"streaming step_codes_to_waveform diverges from non-streaming by "
        f"max|diff|={diff:.6f} after undoing the 1-frame lookahead "
        f"(|val|max={np.abs(w_ns).max():.4f}) — a streaming conv-cache or "
        f"InverseSTFT overlap-add regression."
    )
