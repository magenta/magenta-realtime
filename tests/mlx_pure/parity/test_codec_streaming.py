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

"""Streaming-codec parity: enable_streaming + per-step
``step_codes_to_waveform`` chunks should concatenate to the same audio
as a single non-streaming ``codes_to_waveform`` on the joined codes.

Pins the per-conv ``Conv2DCache`` left-context buffers, the
``ParallelChannels`` per-group cache stash, the SpectroStreamDecoder
lookahead countdown, and the InverseSTFT OverlapAddCache.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from magenta_rt.mlx_pure.spectrostream import (
    ResidualVectorQuantizer, SpectroStream,
)


def _build_tiny_soundstream(*, channel_splits: int | None = None,
                            decoder_lookahead: int = 0) -> SpectroStream:
    return SpectroStream(
        stft_frame_length=64, stft_frame_step=32, stft_fft_length=64,
        ratios=((1, 2), (2, 1)), mults=(2, 1),
        is_resnet=True, activation_fn=nn.elu,
        num_bins=32, num_channels=2, num_features=16,
        causal=True, keep_dc=True,
        encoder_base_conv_depth=8, encoder_base_conv_size=3,
        decoder_base_conv_depth=8, decoder_base_conv_size=3,
        channel_splits=channel_splits,
        channel_recombo_block=-2 if channel_splits else -1,
        decoder_lookahead=decoder_lookahead,
        quantizer=ResidualVectorQuantizer(
            num_quantizers=2, num_embeddings=4, embedding_dim=16,
        ),
    )


def _randomize(ss: SpectroStream, seed: int = 0) -> None:
    """Walk the codec module tree and replace zero-init params with
    small random samples so the streaming-vs-non-streaming comparison
    isn't a trivial all-zero match."""
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(ss.parameters()))
    key = mx.random.key(seed)
    new = {}
    for i, (name, arr) in enumerate(sorted(flat.items())):
        sub = mx.random.split(key, len(flat) + 1)[i]
        if arr.size > 0 and bool(mx.all(arr == 0).item()):
            new[name] = (mx.random.normal(arr.shape, key=sub) * 0.05).astype(arr.dtype)
    if new:
        ss.update(tree_unflatten(list(new.items())))


def _streaming_concat_matches_non_streaming(ss: SpectroStream, codes: mx.array,
                                             *, atol: float, rtol: float):
    ss.disable_streaming()
    ref = ss.codes_to_waveform(codes)

    ss.enable_streaming()
    chunks = []
    T = codes.shape[1]
    for t in range(T):
        chunk = ss.step_codes_to_waveform(codes[:, t : t + 1])
        mx.eval(chunk)
        chunks.append(chunk)
    # Audio is time-last ([B, T] mono or [B, C, T]); concat on the last axis.
    streamed = mx.concatenate(chunks, axis=-1)

    common = min(ref.shape[-1], streamed.shape[-1])
    np.testing.assert_allclose(
        np.array(ref[..., :common].astype(mx.float32)),
        np.array(streamed[..., :common].astype(mx.float32)),
        atol=atol, rtol=rtol,
    )


def test_streaming_codec_no_split_no_lookahead():
    ss = _build_tiny_soundstream()
    _randomize(ss, seed=1)
    codes = mx.random.randint(0, 4, (1, 8, 2), dtype=mx.int32, key=mx.random.key(2))
    _streaming_concat_matches_non_streaming(ss, codes, atol=1e-4, rtol=1e-4)


def test_streaming_codec_with_channel_splits():
    """Hits the ParallelChannels per-group cache-stash path."""
    ss = _build_tiny_soundstream(channel_splits=2)
    _randomize(ss, seed=3)
    codes = mx.random.randint(0, 4, (1, 8, 2), dtype=mx.int32, key=mx.random.key(4))
    _streaming_concat_matches_non_streaming(ss, codes, atol=1e-4, rtol=1e-4)


def test_streaming_codec_with_lookahead():
    """Hits the SpectroStreamDecoder lookahead countdown across calls."""
    ss = _build_tiny_soundstream(channel_splits=2, decoder_lookahead=1)
    _randomize(ss, seed=5)
    # Need T larger than the lookahead's STFT-frame-equivalent (4 frames
    # at total_time_stride=2) so streaming actually emits non-zero audio.
    codes = mx.random.randint(0, 4, (1, 12, 2), dtype=mx.int32, key=mx.random.key(6))
    _streaming_concat_matches_non_streaming(ss, codes, atol=1e-4, rtol=1e-4)
