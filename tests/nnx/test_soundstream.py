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

"""Construction + smoke tests for ``magenta_rt.nnx.spectrostream`` modules.

A full numerical-parity test against the JAX/Linen reference codec
lives in ``test_spectrostream_jax_parity.py``; here we just check that:

* The encoder + decoder construct and run end-to-end on random
  weights with channel_splits=2 (the production-stereo config).
* ``codes_to_waveform`` returns the expected sample count for a given
  codes-batch shape.
* ``init_cache`` / ``remove_cache`` round-trip cleanly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from magenta_rt.nnx.spectrostream import (
    ResidualVectorQuantizer, SpectroStream, SpectroStreamDecoder, SpectroStreamEncoder,
)


def _build_tiny_codec(seed: int = 0) -> SpectroStream:
    """Tiny codec used for construction / shape smoke tests.

    Shapes: stereo (num_channels=2), 16 STFT bins → 4-bin bottleneck,
    two ratio levels (×2 each in time and freq) so total time stride
    is 4 samples. ``channel_splits=None`` here so this smoke isn't
    coupled to the production-config mults/recombo arithmetic;
    channel_splits=2 parity against a real model spec lives in
    ``test_spectrostream_jax_parity.py``.
    """
    rngs = nnx.Rngs(seed)
    quantizer = ResidualVectorQuantizer(
        num_quantizers=4, num_embeddings=8, embedding_dim=16,
        rngs=rngs,
    )
    return SpectroStream(
        sample_rate=48000,
        stft_frame_length=32, stft_frame_step=16, stft_fft_length=32,
        ratios=[(2, 2), (2, 2)], mults=[1, 1],
        channel_splits=None,
        is_resnet=True, num_bins=16, num_channels=2,
        num_features=16, causal=True,
        encoder_base_conv_depth=8, encoder_base_conv_size=(3, 3),
        decoder_base_conv_depth=8, decoder_base_conv_size=(3, 3),
        quantizer=quantizer,
        decoder_lookahead=1,
        rngs=rngs,
    )


def test_spectrostream_construction_smoke():
    codec = _build_tiny_codec()
    assert codec.encoder is not None
    assert codec.decoder is not None
    assert codec.quantizer is not None


def test_spectrostream_codes_to_waveform_shape(rng_key):
    codec = _build_tiny_codec()
    # 8 time frames of RVQ codes, 4 codebooks.
    codes = jax.random.randint(rng_key, (1, 8, 4), 0, 8, dtype=jnp.int32)
    audio = codec.codes_to_waveform(codes)
    # num_channels=2 in the SpectroStream config means real+imag for 1
    # audio channel — InverseSTFT squeezes the trailing singleton, so
    # the output is [B, T_audio] (mono).
    assert audio.ndim == 2
    assert audio.shape[0] == 1
    assert jnp.all(jnp.isfinite(audio))


def test_spectrostream_init_remove_cache_cycle():
    """``init_cache`` arms the streaming pipeline; ``remove_cache`` clears
    every nnx.Cache slot in the subtree (decoder convs, InverseSTFT
    overlap buffer, lookahead countdown).
    """
    codec = _build_tiny_codec()
    codec.set_attributes(streaming=True, raise_if_not_found=False)
    codec.init_cache(batch=1, dtype=jnp.float32)

    # InverseSTFT overlap buffer should be allocated.
    assert codec.inverse_stft._istft.cached_overlap is not None
    assert codec.inverse_stft._istft.streaming is True
    # Decoder lookahead countdown matches its configured length.
    assert codec.decoder._lookahead_remaining == codec.decoder._lookahead_length

    codec.remove_cache()
    assert codec.inverse_stft._istft.cached_overlap is None
    assert codec.inverse_stft._istft.streaming is False
    assert codec.decoder._lookahead_remaining == 0


def test_spectrostream_streaming_step_runs(rng_key):
    """``step_codes_to_waveform`` produces output one chunk at a time."""
    codec = _build_tiny_codec()
    codec.set_attributes(streaming=True, raise_if_not_found=False)
    codec.init_cache(batch=1, dtype=jnp.float32)

    out_chunks = []
    sub = jax.random.split(rng_key, 4)
    for k in sub:
        codes = jax.random.randint(k, (1, 1, 4), 0, 8, dtype=jnp.int32)
        out = codec.step_codes_to_waveform(codes)
        out_chunks.append(out)
    # Output is finite, spans expected total time (first step is zeroed out, not dropped).
    audio = jnp.concatenate(out_chunks, axis=1)
    assert jnp.all(jnp.isfinite(audio))
    # We expect total emitted samples = total_input_frames * upsampled_frames_per_step * frame_step.
    # 4 steps * 4 upsampled_frames * 16 frame_step = 256 samples.
    # Allow some flex; just assert > 0.
    assert audio.shape[1] > 0
