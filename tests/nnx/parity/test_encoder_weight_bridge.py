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

"""SpectroStream ENCODER weight-bridge parity: ``magenta_rt.nnx`` vs Linen.

The mrt2 depthformer checkpoint ships only the SpectroStream decoder +
quantizer (under ``soundstream``); the encoder lives in a standalone Linen
safetensors (``resources/spectrostream/encoder.safetensors``, resolvable via
:func:`magenta_rt.paths.resolve_encoder_weights`). Before the fix the nnx
codec never loaded it, so ``waveform_to_codes`` ran on a randomly-initialised
encoder and produced garbage codes — silently corrupting SFT dataset exports
(``mrt sft export --backend nnx``).

This pins the bridge two ways:

* **Structural** — after :func:`load_encoder_weights`, *zero* encoder params
  remain at their construction-time init values (all 48 Linen tensors land
  somewhere). An un-updated param is a missing / mis-routed weight.
* **Numerical** — ``waveform_to_codes`` on a real audio array must produce the
  EXACT same quantized RVQ code indices as the already-parity-verified
  ``mlx_pure`` direct-Linen loader (``load_spectrostream_from_linen``), which
  reads the same encoder file. Quantized codes are argmin-discrete: if the
  encoder embeddings agree to float round-off they snap to identical indices,
  so the bar is an *exact* match (not a tolerance).

Gated by ``@pytest.mark.checkpoint`` + ``@pytest.mark.slow``; auto-skips when
the standalone encoder resource is absent or ``mlx`` is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def encoder_resource():
    from magenta_rt import paths

    p = paths.resolve_encoder_weights()
    if not p.exists():
        pytest.skip(f"encoder weights not found: {p}")
    return p


def _deterministic_audio(num_samples: int = 24000, seed: int = 0) -> np.ndarray:
    """A reproducible ``[1, T]`` mono waveform (sine + a little noise)."""
    rng = np.random.RandomState(seed)
    t = np.arange(num_samples) / 48000.0
    audio = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.randn(num_samples)
    return audio.astype(np.float32)[None, :]


def _collect_encoder_params(encoder) -> dict:
    """Flatten every ``nnx.Param`` leaf of the encoder to numpy."""
    from flax import nnx

    state = nnx.state(encoder, nnx.Param)
    flat = nnx.traversals.flatten_mapping(dict(state))
    return {k: np.array(v[...]) for k, v in flat.items()}


@pytest.mark.checkpoint
@pytest.mark.slow
def test_encoder_bridge_updates_every_param(encoder_resource):
    """Every encoder param must change value after the bridge runs.

    The standalone encoder has exactly 48 Linen tensors; the nnx encoder
    has 48 params. A non-zero un-updated count means a weight was missed or
    mis-routed (the encoder would silently run partly on random init).
    """
    from flax import nnx

    from magenta_rt.nnx import model as nnx_model
    from magenta_rt.nnx.spectrostream.load_weights import (
        _read_linen_encoder, load_encoder_weights,
    )

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0),
    )
    encoder = mrt.spectrostream.encoder

    before = _collect_encoder_params(encoder)
    assert len(before) == 48, f"expected 48 encoder params, got {len(before)}"

    enc_tree = _read_linen_encoder(str(encoder_resource))
    load_encoder_weights(encoder, enc_tree)

    after = _collect_encoder_params(encoder)
    unupdated = [k for k in before if np.array_equal(before[k], after[k])]
    assert not unupdated, (
        f"{len(unupdated)} encoder params left at init: "
        f"{['.'.join(map(str, k)) for k in unupdated]}"
    )


@pytest.mark.checkpoint
@pytest.mark.slow
def test_encoder_bridge_waveform_to_codes_exact(encoder_resource):
    """nnx ``waveform_to_codes`` must produce the EXACT same RVQ codes as the
    parity-verified ``mlx_pure`` direct-Linen loader (same encoder file)."""
    pytest.importorskip("mlx")
    import jax.numpy as jnp
    import mlx.core as mx
    from flax import nnx

    audio = _deterministic_audio()

    # --- mlx_pure reference (already JAX-parity-verified) ---
    from magenta_rt.mlx_pure import configs as mpc
    from magenta_rt.mlx_pure.spectrostream.load_weights import (
        load_spectrostream_from_linen,
    )

    ref_codec = mpc.get_model_class("mrt2_small")().build_spectrostream()
    load_spectrostream_from_linen(ref_codec)
    ref_codes = np.array(ref_codec.waveform_to_codes(mx.array(audio)))

    # --- nnx under test: fallback auto-loads the encoder from the resource ---
    from magenta_rt import paths
    from magenta_rt.nnx import model as nnx_model

    mrt = nnx_model.MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0),
    )
    ckpt = paths.resolve_checkpoint("mrt2_small.safetensors")
    if not ckpt.exists():
        pytest.skip(f"checkpoint not found: {ckpt}")
    mrt.load_checkpoint(ckpt)
    # Compare at fp32 (the build default for mlx_pure); flip nnx off bf16.
    mrt.spectrostream.set_attributes(dtype=jnp.float32, raise_if_not_found=False)
    mrt.spectrostream.set_attributes(streaming=False, raise_if_not_found=False)
    nnx_codes = np.array(
        mrt.spectrostream.waveform_to_codes(jnp.asarray(audio))
    )

    assert nnx_codes.shape == ref_codes.shape, (
        f"shape mismatch: nnx {nnx_codes.shape} vs mlx_pure {ref_codes.shape}"
    )
    # Quantized RVQ indices are argmin-discrete; a correct encoder snaps to the
    # exact same codebook entries as the reference.
    mismatch = int((nnx_codes != ref_codes).sum())
    assert mismatch == 0, (
        f"{mismatch}/{nnx_codes.size} RVQ codes differ from the mlx_pure "
        f"reference (encoder bridge is wrong)"
    )
