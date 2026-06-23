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

Stitches together :class:`~magenta_rt.nnx.depthformer.EncoderDecoder`
(token generation) and a :class:`~magenta_rt.nnx.spectrostream.SpectroStream`
codec (token → audio waveform).

Streaming is armed by:

  model.set_attributes(streaming=True)
  model.init_cache(batch=B, dtype=...)

The bare nnx primitive set; ``MagentaRT2Sampler`` provides
``init_streaming(batch_size, rngs=nnx.Rngs(0))`` as sugar around both calls.
"""

from __future__ import annotations

from typing import Optional, Union

import jax.numpy as jnp
from flax import nnx

from . import depthformer
from . import spectrostream as _ss
from audiotree import AudioTree
from .configs import get_model_class


_OUTPUT_GAIN = 0.5


def _apply_output_gain(samples: jnp.ndarray) -> jnp.ndarray:
    """Apply the 0.5 output gain and clip to [-1, 1].

    This is the gain+clip half of sl's ``_float_samples_to_int16`` and must
    be applied *unconditionally* (the sl pipeline's gain+clip+int16 layer is
    never gated), so the float output carries the same gain+clip as the
    int16 path. Only the final int16 cast is conditional.
    """
    return jnp.clip(_OUTPUT_GAIN * samples, -1.0, 1.0)


def _float_to_int16(samples: jnp.ndarray) -> jnp.ndarray:
    """Cast gain+clipped [-1, 1] float samples to int16 (matches
    ``magenta_rt.mlx.system._float_samples_to_int16``)."""
    int16_max = float(jnp.iinfo(jnp.int16).max)
    return jnp.round((int16_max + 0.5) * samples - 0.5).astype(jnp.int16)


def _convert_from_unique_codes(
    codes: jnp.ndarray, *, num_reserved_tokens: int, codebook_size: int,
) -> jnp.ndarray:
    """Convert depthformer's globally-indexed codes back to per-codebook indices."""
    Q = codes.shape[-1]
    offsets = jnp.array(
        [num_reserved_tokens + q * codebook_size for q in range(Q)],
        dtype=codes.dtype,
    )
    return codes - offsets


class MagentaRT2Sampler(nnx.Module):
    """End-to-end streaming inference pipeline."""

    def __init__(
        self,
        *,
        depthformer_model: depthformer.EncoderDecoder,
        num_reserved_tokens: int,
        codebook_size: int,
        spectrostream: Optional[_ss.SpectroStream] = None,
        int16_outputs: bool = True,
    ):
        self.depthformer = depthformer_model
        self.spectrostream = spectrostream
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
        param_dtype=None,
        dtype=None,
        rngs: nnx.Rngs = None,
    ) -> MagentaRT2Sampler:
        """Ergonomic factory instantiating the pre-wired encoder-decoder
        and codec pipeline directly from a registered preset name.

        ``param_dtype`` / ``dtype`` override the preset spec's storage /
        compute dtypes. The big use case is bf16 inference for ``mrt2_base``:
        the fp32 spec builds a 9.6 GB parameter tree that OOMs a 16 GB GPU
        (and cannot co-reside with a bf16 training model for periodic audio
        sampling), whereas ``param_dtype=jnp.bfloat16`` halves it to 4.8 GB.
        Both default to ``None`` = leave the spec's own dtypes untouched.
        """
        spec = get_model_class(model_name)()
        # The spec is a frozen config object (not a dataclass instance with
        # ``replace``); ``object.__setattr__`` is how the trainer mutates it.
        if param_dtype is not None:
            object.__setattr__(spec, "param_dtype", param_dtype)
        if dtype is not None:
            object.__setattr__(spec, "dtype", dtype)
        enc_dec = depthformer.EncoderDecoder.from_config(spec, rngs=rngs)
        target_cfg = spec.target_tokens_config
        num_codebooks = target_cfg.rvq_truncation_level

        if build_spectrostream:
            quantizer = _ss.ResidualVectorQuantizer(
                num_quantizers=64,
                num_embeddings=1024, embedding_dim=256,
                use_unique_codes=False,
                truncation_level=num_codebooks,
                param_dtype=spec.param_dtype, rngs=rngs,
            )
            ss = _ss.SpectroStream(
                sample_rate=48_000,
                stft_frame_length=960, stft_frame_step=480, stft_fft_length=960,
                ratios=((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1)),
                mults=(2, 1, 2, 1, 1, 2, 1),
                is_resnet=True,
                num_bins=480, num_channels=4,
                channel_splits=2, channel_recombo_block=-2,
                num_features=256,
                causal=True,
                encoder_base_conv_depth=32, encoder_base_conv_size=7,
                decoder_base_conv_depth=64, decoder_base_conv_size=7,
                keep_dc=True,
                decoder_lookahead=1,
                quantizer=quantizer,
                param_dtype=spec.param_dtype, dtype=spec.dtype,
                rngs=rngs,
            )
        else:
            ss = None

        return cls(
            depthformer_model=enc_dec,
            spectrostream=ss,
            num_reserved_tokens=target_cfg.num_extra_tokens,
            codebook_size=target_cfg.codebook_size,
            int16_outputs=int16_outputs,
        )

    @property
    def sample_rate(self) -> Union[int, None]:
        if self.spectrostream is not None:
            return self.spectrostream.sample_rate
        return None

    def load_checkpoint(self, checkpoint_path, *, host: bool = False) -> None:
        """Populate all system variables from a safetensors checkpoint path.

        ``host=True`` loads via host RAM (per-leaf cast on transfer) instead of
        a single on-device load — needed when this model must share the GPU
        with another (e.g. a bf16 ``mrt2_base`` sampler beside the trainer).
        """
        from .load_weights import load_from_jax_safetensors
        load_from_jax_safetensors(self, checkpoint_path, host=host)

    def init_cache(self, *, batch: int = 1, dtype=jnp.float32) -> None:
        """Walks depthformer + spectrostream init_cache; prepares streaming."""
        self.depthformer.init_cache(batch=batch, dtype=dtype)
        if self.spectrostream is not None:
            self.spectrostream.init_cache(batch=batch, dtype=dtype)

    def remove_cache(self) -> None:
        self.depthformer.remove_cache()
        if self.spectrostream is not None:
            self.spectrostream.remove_cache()

    def init_streaming(
        self,
        batch_size: int,
        *,
        rngs: nnx.Rngs,
        codec_streaming: bool = True,
    ) -> None:
        """Arm everything for a fresh streaming session.

        Sets ``streaming=True`` recursively and arms internal state variables.

        Pass ``codec_streaming=False`` to leave the codec in
        non-streaming mode (useful for tests that drive
        ``codes_to_waveform`` on the same SpectroStream instance).
        """
        if self.spectrostream is not None and not codec_streaming:
            self.spectrostream.set_attributes(streaming=False, raise_if_not_found=False)
            self.spectrostream.remove_cache()
        else:
            self.set_attributes(streaming=True, raise_if_not_found=False)
        self.init_cache(batch=batch_size)
        self.depthformer.init_streaming(batch_size, rngs=rngs)

    def disable_streaming(self) -> None:
        """Deallocate caches and set streaming=False recursively."""
        self.set_attributes(streaming=False, raise_if_not_found=False)
        self.remove_cache()

    def step(
        self,
        *,
        source_tokens: jnp.ndarray,
        **sampling_kwargs,
    ) -> AudioTree:
        """One streaming step.

        Args:
            source_tokens: Input token conditioning of shape ``[B, 1, C]``.
            **sampling_kwargs: Optional kwargs for token sampling.

        Returns:
            An ``AudioTree`` containing the generated audio chunk of shape
            ``[B, 2, frame_step]`` (stereo, channel-major) under ``waveform``,
            and the generated codes of shape ``[B, 1, Q]`` under ``codes``.
        """
        codes = self.depthformer.step(
            source_tokens=source_tokens, **sampling_kwargs,
        )
        codes = _convert_from_unique_codes(
            codes,
            num_reserved_tokens=self.num_reserved_tokens,
            codebook_size=self.codebook_size,
        )
        if self.spectrostream is not None:
            waveform = self.spectrostream.step_codes_to_waveform(codes)
            if waveform.ndim == 2:
                waveform = waveform[:, None, :]  # mono [B, T] -> [B, 1, T]
            # Gain+clip is unconditional (matches sl's always-on output
            # layer); only the int16 cast is gated by int16_outputs.
            waveform = _apply_output_gain(waveform)
            if self.int16_outputs:
                waveform = _float_to_int16(waveform)
        else:
            waveform = None
        return AudioTree(
            waveform=waveform,
            sample_rate=self.sample_rate,
            codes=codes,
        )
