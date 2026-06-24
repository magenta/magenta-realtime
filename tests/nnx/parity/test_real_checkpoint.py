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

"""Real-checkpoint smoke for ``magenta_rt.nnx``.

Loads the ``mrt2_small`` safetensors checkpoint via the JAX/nnx
bridge and exercises the load path on its public components:

* ``ResidualVectorQuantizer.codes_to_embeddings`` — bit-exact against
  a hand-sum of the raw safetensors values.
* Encoder forward, decoder-embedder lookup, SpectroStream
  ``codes_to_waveform`` — finite output of the expected shape.

Skipped by default; opt in with ``pytest -m checkpoint``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from magenta_rt import paths
import safetensors.flax as safetensors_flax
from flax import nnx

from magenta_rt.nnx import model as nnx_model


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = paths.resolve_checkpoint("mrt2_small.safetensors")


pytestmark = pytest.mark.checkpoint


@pytest.fixture(scope="module")
def bridged_system():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT}")
    spec = nnx_model.get_model_class("mrt2_small")()
    mrt = nnx_model.MagentaRT2Sampler.from_preset("mrt2_small", int16_outputs=False, rngs=nnx.Rngs(0))
    mrt.load_checkpoint(CHECKPOINT)
    return mrt, spec


def test_rvq_codes_to_embeddings_bit_exact(bridged_system):
    """RVQ lookup is a plain table read; bridge copies bytes verbatim."""
    mrt, _ = bridged_system
    raw = safetensors_flax.load_file(str(CHECKPOINT))
    ref_emb = jnp.asarray(raw["params/soundstream/quantizer/embedding"])

    codes = jnp.array([[[3, 7, 5, 11, 0, 1, 8, 4, 9, 6, 2, 10]]], dtype=jnp.int32)
    got = mrt.spectrostream.quantizer.codes_to_embeddings(codes)
    expected = sum(ref_emb[q, codes[0, 0, q]] for q in range(codes.shape[-1]))
    np.testing.assert_array_equal(np.asarray(got[0, 0]), np.asarray(expected))


def test_encoder_forward_finite(bridged_system):
    mrt, spec = bridged_system
    musiccoca = [679, 132, 480, 389, 160, 1010]
    notes = [-1] * (spec.input_num_channels - len(musiccoca))
    src = jnp.asarray(
        (np.array(musiccoca + notes, dtype=np.int32) + 7).reshape(1, 1, -1),
    )
    out = mrt.depthformer.encoder(src)
    arr = np.asarray(jnp.asarray(out).astype(jnp.float32))
    assert arr.shape == (1, 1, spec.encoder_size.model_dims)
    assert np.isfinite(arr).all()


def test_decoder_embedder_finite(bridged_system):
    mrt, spec = bridged_system
    sos = jnp.zeros(
        (1, 1, spec.target_tokens_config.rvq_truncation_level), dtype=jnp.int32,
    )
    out = mrt.depthformer.decoder.embedder(sos)
    arr = np.asarray(jnp.asarray(out).astype(jnp.float32))
    assert arr.shape == (
        1, 1, spec.target_tokens_config.rvq_truncation_level,
        spec.decoder_temporal_size.model_dims,
    )
    assert np.isfinite(arr).all()


def test_codes_to_waveform_finite(bridged_system):
    mrt, _ = bridged_system
    codes = jnp.array([[[3, 7, 5, 11, 0, 1, 8, 4, 9, 6, 2, 10]]], dtype=jnp.int32)
    rvq_seq = jnp.asarray(np.tile(np.array(codes), (1, 4, 1)).astype(np.int32))
    mrt.spectrostream.set_attributes(streaming=False, raise_if_not_found=False)
    audio = mrt.spectrostream.codes_to_waveform(rvq_seq)
    arr = np.asarray(jnp.asarray(audio).astype(jnp.float32))
    # Channel-major [B, C, T] audio out of the codec.
    assert arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 2
    assert np.isfinite(arr).all()
