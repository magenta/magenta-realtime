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

"""Top-level streaming inference orchestrator.

Stitches :class:`mlx_pure.depthformer.EncoderDecoder` (token
generation) together with a SpectroStream codec (token → audio waveform).

Both an injected callable ``codes_to_waveform`` (external delegator to
sibling backend variants) and a fully-pure SpectroStream module
are accepted — pass either ``codes_to_waveform=fn`` or
``spectrostream=mlx_pure.spectrostream.SpectroStream(...)``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from typing import Union
from pathlib import Path

from . import depthformer
from .spectrostream import SpectroStream, ResidualVectorQuantizer
from .configs import get_model_class
from .transformer import Transformer, Encoder


_OUTPUT_GAIN = 0.5
"""Output gain applied to reduce clipping artifacts. Matches the ``gain``
in ``magenta_rt.mlx.system._float_samples_to_int16``."""


def _apply_output_gain(samples: mx.array) -> mx.array:
    """Apply the ``0.5`` output gain and clip to ``[-1, 1]``.

    This is the gain+clip half of sl's ``_float_samples_to_int16`` and is
    applied *unconditionally* — in the sl pipeline the gain+clip+int16
    step is the last layer of the chain and is never gated, so the float
    output must carry the same gain+clip to stay at parity. Only the
    final int16 cast is conditional (see :func:`_float_to_int16`).
    """
    return mx.clip(_OUTPUT_GAIN * samples, -1.0, 1.0)


def _float_to_int16(samples: mx.array) -> mx.array:
    """Cast gain+clipped ``[-1, 1]`` float samples to int16.

    Mirrors the cast half of ``magenta_rt.mlx.system._float_samples_to_int16``
    exactly: ``round((iinfo(int16).max + 0.5) * samples - 0.5)``. Assumes
    ``samples`` has already been through :func:`_apply_output_gain`.
    """
    int16_max = float(mx.iinfo(mx.int16).max)
    samples = mx.round((int16_max + 0.5) * samples - 0.5)
    return samples.astype(mx.int16)


def convert_from_unique_codes(codes: mx.array, *, num_reserved_tokens: int, codebook_size: int) -> mx.array:
    """Convert depthformer's globally-indexed codes back to per-codebook indices.

    Equivalent to ``magenta_rt.mlx.system.convert_from_unique_codes``:
    ``code_q -> (code_q - num_reserved_tokens - q * codebook_size)``.
    """
    Q = codes.shape[-1]
    offsets = mx.array(
        [num_reserved_tokens + q * codebook_size for q in range(Q)],
        dtype=codes.dtype,
    )
    return codes - offsets


class MagentaRT2Sampler(nn.Module):
    """End-to-end streaming inference pipeline.

    Steps per frame:
        1. Encode the source frame.
        2. Run :meth:`depthformer.EncoderDecoder.step` to sample new
           codebook tokens.
        3. Convert global token indices back to per-codebook indices.
        4. Look up RVQ embeddings, decode to waveform via the codec.
        5. Convert waveform to int16 (optional).
    """

    def __init__(
        self,
        *,
        depthformer_model: depthformer.EncoderDecoder,
        num_reserved_tokens: int,
        codebook_size: int,
        spectrostream: Optional[SpectroStream] = None,
        codes_to_waveform: Optional[Callable[[mx.array], mx.array]] = None,
        int16_outputs: bool = True,
    ):
        super().__init__()
        if (spectrostream is None) == (codes_to_waveform is None):
            raise ValueError(
                "must provide exactly one of `spectrostream` or `codes_to_waveform`"
            )
        self.depthformer = depthformer_model
        self.spectrostream = spectrostream
        self._codes_to_waveform = codes_to_waveform
        self.num_reserved_tokens = num_reserved_tokens
        self.codebook_size = codebook_size
        self.int16_outputs = int16_outputs

    @classmethod
    def from_preset(
        cls,
        model_name: str,
        *,
        int16_outputs: bool = True,
        build_spectrostream: bool = True,
    ) -> MagentaRT2Sampler:
        """Ergonomic factory instantiating the pre-wired encoder-decoder
        and codec pipeline directly from a registered preset name.
        """
        spec = get_model_class(model_name)()
        enc_dec = spec.build_decoder()
        target_cfg = spec.target_tokens_config
        target_num_reserved = target_cfg.num_extra_tokens
        target_codebook_size = target_cfg.codebook_size
        target_num_codebooks = target_cfg.rvq_truncation_level

        if build_spectrostream:
            ss = spec.build_spectrostream()
            codes_to_waveform = None
        else:
            ss = None
            codes_to_waveform = lambda codes: codes

        return cls(
            depthformer_model=enc_dec,
            spectrostream=ss,
            codes_to_waveform=codes_to_waveform,
            num_reserved_tokens=target_num_reserved,
            codebook_size=target_codebook_size,
            int16_outputs=int16_outputs,
        )

    def load_from_safetensors(self, checkpoint_path: Union[str, Path], *, model_name: str = "mrt2_small") -> None:
        """Populate all module properties natively straight from an unflattened
        safetensors checkpoint dictionary on disk.

        Uses the DIRECT loader: the depthformer is streamed Linen -> bf16 leaf by
        leaf (no fp32 sl twin held alongside the populated pure model), which
        keeps the base (2.4 B) load peak at ~model size instead of ~model + the
        9.6 GB fp32 sl depthformer. The result is bit-identical to the sl-bridge
        path (see tests/mlx_pure/parity/test_direct_loader_parity.py). The codec
        still mirrors via the sl bridge inside ``load_from_safetensors_direct``.
        """
        from .load_weights import load_from_safetensors_direct
        load_from_safetensors_direct(self, checkpoint_path, model_name=model_name)

    def make_initial_state(
        self, batch_size: int, *, seed: int = 42,
        codec_streaming: bool = True,
    ) -> depthformer.SamplerState:
        # Arm the SpectroStream codec for streaming so that per-conv
        # left-context buffers and the InverseSTFT OverlapAddCache
        # start empty for each new generation. Set
        # ``codec_streaming=False`` to leave the codec in the
        # non-streaming forward path (useful for tests that drive
        # ``codes_to_waveform`` directly on the same SpectroStream
        # instance).
        if self.spectrostream is not None:
            if codec_streaming:
                self.spectrostream.enable_streaming()
            else:
                self.spectrostream.disable_streaming()
        return self.depthformer.make_initial_state(batch_size, seed=seed)

    def init_cache(
        self, state: depthformer.SamplerState, *, batch: int,
        dtype: Optional[mx.Dtype] = None,
    ) -> None:
        """Eagerly allocate every streaming cache so a content-neutral
        state is ready *without* a warmup-generation loop.

        * Depthformer: the temporal KV caches held in ``state`` are
          zero-allocated and sink-primed directly (see
          :meth:`DepthformerDecoder.init_cache`).
        * Codec: the conv left-context buffers and the InverseSTFT
          overlap buffer can't be sized from the architecture without a
          conv-stack spatial walk, so they're allocated by one
          zero-input codec pass and then zeroed in place by
          :meth:`SpectroStream.reset_caches` — net result: allocated,
          content-neutral, lookahead countdown drained.

        Call after :meth:`make_initial_state`. The result is the
        allocated-but-neutral streaming state that ``mlx_pure.export``
        ships in ``_state.safetensors`` (no generation contamination).
        """
        self.depthformer.init_cache(state, batch=batch, dtype=dtype)
        if self.spectrostream is not None:
            self.spectrostream.enable_streaming()
            num_codebooks = self.depthformer.decoder.num_codebooks
            zero_codes = mx.zeros((batch, 1, num_codebooks), dtype=mx.int32)
            # Step zero-input codec frames until every lazy cache is
            # allocated. The decoder's lookahead drops the first
            # ~lookahead_length output frames, so the opening pass(es)
            # can yield nothing for the InverseSTFT to cache — keep
            # going until its overlap buffer exists too. Capped well
            # above any realistic lookahead.
            for _ in range(16):
                self.spectrostream.step_codes_to_waveform(zero_codes)
                istft = self.spectrostream._istft_cache
                if istft is not None and istft.buffer is not None:
                    break
            else:
                raise RuntimeError(
                    "init_cache: InverseSTFT overlap cache never allocated "
                    "after 16 zero-input codec steps"
                )
            self.spectrostream.reset_caches()

    def step_with_codes(
        self,
        state: depthformer.SamplerState,
        *,
        source_tokens: mx.array,
        **sampling_kwargs,
    ) -> tuple[mx.array, mx.array, depthformer.SamplerState]:
        """One streaming step, also returning the sampled RVQ codes.

        Returns ``(waveform_chunk, codes, new_state)`` where ``codes`` is the
        ``[B, 1, Q]`` per-codebook (non-unique) indices that produced the
        chunk — what the system layer stacks into ``AudioTree.codes``.
        """
        encoded = self.depthformer.encode(source_tokens)
        codes, state = self.depthformer.step(state, source_frame=encoded, **sampling_kwargs)
        codes = convert_from_unique_codes(
            codes,
            num_reserved_tokens=self.num_reserved_tokens,
            codebook_size=self.codebook_size,
        )
        if self.spectrostream is not None:
            waveform = self.spectrostream.step_codes_to_waveform(codes)
            if waveform.ndim == 2:
                waveform = waveform[:, None, :]  # mono [B, T] -> [B, 1, T]
        else:
            waveform = self._codes_to_waveform(codes)
        # Gain+clip is unconditional (matches sl's always-on output layer);
        # only the int16 cast is gated by int16_outputs.
        waveform = _apply_output_gain(waveform)
        if self.int16_outputs:
            waveform = _float_to_int16(waveform)
        return waveform, codes, state

    def step(
        self,
        state: depthformer.SamplerState,
        *,
        source_tokens: mx.array,
        **sampling_kwargs,
    ) -> tuple[mx.array, depthformer.SamplerState]:
        """One streaming step. Returns (waveform_chunk, new_state)."""
        waveform, _codes, state = self.step_with_codes(
            state, source_tokens=source_tokens, **sampling_kwargs,
        )
        return waveform, state
