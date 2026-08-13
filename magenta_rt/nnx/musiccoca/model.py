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

"""MusicCoCa in pure flax.nnx.

``MusicCoCaModule`` is the nnx assembly of the five reverse-engineered
components (log-mel frontend, music/text towers, RVQ, mapper); weights
load from the safetensors file produced by ``convert.py``.

``MusicCoCa`` wraps it in the same high-level interface as
:class:`magenta_rt.musiccoca.MusicCoCa` (``embed`` / ``tokenize``),
including SentencePiece text preprocessing, so the two are drop-in
interchangeable.
"""

from __future__ import annotations

import functools
import pathlib

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ... import musiccoca as musiccoca_base
from ... import paths
from .convert import DEFAULT_FILENAME
from .frontend import CLIP_SAMPLES, LogMelFrontend
from .mapper import Mapper
from .modules import AudioEncoder, TextEncoder
from .quantizer import EmbeddingQuantizer

MAX_TEXT_LENGTH = 128
TEXT_SOS_ID = 1


def encode_text(vocab, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Lowercased SentencePiece ids + paddings, shaped ``[128]`` each."""
    labels = vocab.EncodeAsIds(text.lower())[: MAX_TEXT_LENGTH - 1]
    ids = [TEXT_SOS_ID] + labels
    num_tokens = len(ids)
    ids = ids + [0] * (MAX_TEXT_LENGTH - num_tokens)
    paddings = np.ones(MAX_TEXT_LENGTH, dtype=np.float32)
    paddings[:num_tokens] = 0.0
    return np.array(ids, dtype=np.int32), paddings


class MusicCoCa(nnx.Module, musiccoca_base.MusicCoCaBase):
    """The full MusicCoCa model as a single nnx module tree and high-level wrapper.

    Same interface and resource directory as the TFLite-backed
    :class:`magenta_rt.musiccoca.MusicCoCa`; expects ``spm.model`` and the
    ``musiccoca_nnx.safetensors`` produced by
    ``python -m magenta_rt.nnx.musiccoca.convert``.
    """

    def __init__(
        self,
        resource_dir: str | pathlib.Path | None = None,
        *,
        load_weights: bool = True,
    ):
        nnx.Module.__init__(self)
        musiccoca_base.MusicCoCaBase.__init__(
            self,
            musiccoca_base.MusicCoCaConfiguration(
                sample_rate=16000,
                clip_length=10.0,
                embedding_dim=768,
                rvq_depth=12,
                rvq_codebook_size=1024,
            )
        )
        self._resource_dir = pathlib.Path(resource_dir or paths.musiccoca_dir())
        self.frontend = LogMelFrontend()
        self.audio = AudioEncoder()
        self.text = TextEncoder()
        self.quantizer = EmbeddingQuantizer()
        self.mapper = Mapper()

        if load_weights:
            path = self._resource_dir / DEFAULT_FILENAME
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; run `python -m"
                    " magenta_rt.nnx.musiccoca.convert` to create it from the"
                    " TFLite resources. Pass load_weights=False for a"
                    " randomly-initialized model."
                )
            load_safetensors(self, path)

    @functools.cached_property
    def _vocab(self):
        import sentencepiece

        spm_path = self._resource_dir / "spm.model"
        if not spm_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {spm_path}")
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load(str(spm_path))
        return sp

    # ------------------------------------------------------------------
    # Raw nnx ops — operate on ``jnp`` arrays; ``nnx.jit`` them yourself.
    # ------------------------------------------------------------------

    def encode_clips(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """Embed mono audio clips through the audio tower.

        Args:
            waveform: Mono clips ``[B, 160000]`` (10 s at 16 kHz) ``jnp`` array.

        Returns:
            Style embeddings of shape ``[B, 768]``.
        """
        return self.audio(self.frontend(waveform))

    def encode_tokens(self, ids: jnp.ndarray, paddings: jnp.ndarray) -> jnp.ndarray:
        """Embed tokenized text through the text tower.

        Args:
            ids: SentencePiece token ids ``[B, 128]``.
            paddings: Matching padding mask ``[B, 128]`` (1.0 = padded).

        Returns:
            Style embeddings of shape ``[B, 768]``.
        """
        return self.text(ids, paddings)

    def map_text_embedding(
        self, text_emb: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        """Map text embeddings toward the audio sub-space, L2-normalized.

        Args:
            text_emb: Text style embeddings ``[B, 768]``.
            noise: Gaussian noise ``[B, 768]`` driving the mapper.

        Returns:
            L2-normalized mapped embeddings of shape ``[B, 768]``.
        """
        mapped = self.mapper(text_emb, noise)
        return mapped / jnp.linalg.norm(mapped, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # High-level host-side API — str / ``AudioTree`` / NumPy.
    # ------------------------------------------------------------------

    def _embed_batch_text(
        self,
        batch_text: list[str],
        use_mapper: bool = False,
        seed: int = 0,
    ) -> np.ndarray:
        encoded = [encode_text(self._vocab, s) for s in batch_text]
        ids = np.stack([e[0] for e in encoded])
        paddings = np.stack([e[1] for e in encoded])
        emb = np.asarray(self.encode_tokens(ids, paddings))
        if use_mapper:
            # Matches the TFLite wrapper: one fixed-seed Gaussian noise
            # vector shared across the batch.
            rng = np.random.RandomState(seed)
            noise = rng.randn(self.config.embedding_dim).astype(np.float32)
            noise = np.broadcast_to(noise, emb.shape)
            emb = np.asarray(self.map_text_embedding(emb, noise))
        return emb.astype(np.float32)

    def _embed_batch_clips(
        self,
        batch_clips: np.ndarray,
    ) -> np.ndarray:
        clips = np.asarray(batch_clips, dtype=np.float32)
        if clips.shape[-1] != CLIP_SAMPLES:
            pad = CLIP_SAMPLES - clips.shape[-1]
            if pad < 0:
                clips = clips[..., :CLIP_SAMPLES]
            else:
                clips = np.pad(clips, ((0, 0), (0, pad)))
        emb = self.encode_clips(clips)
        return np.asarray(emb, dtype=np.float32)

    def embed_audio_windows(
        self,
        audio,
        *,
        hop_seconds: float,
        num_windows: int,
        start_seconds: float = 0.0,
        mono_strategy: str = "average",
        scan: bool = False,
        window_subbatch=None,
        frame_rate: float = 25.0,
    ) -> np.ndarray:
        """Mel-once windowed style embeddings ``[B, num_windows, 768]``.

        Overrides :meth:`magenta_rt.musiccoca.MusicCoCaBase.embed_audio_windows`
        with the efficient path: the log-mel frontend (the part shared by all
        overlapping windows) runs ONCE over the whole signal, then the audio ViT
        encodes each ``NUM_FRAMES``-frame mel window (LEADING ``[t, t+10s]``).
        ``scan=True`` streams the windows through ``nnx.scan`` (bounded memory).
        """
        from .frontend import FRAME_HOP, NUM_FRAMES

        a = audio.to_mono(strategy=mono_strategy).resample(self.config.sample_rate)
        mono = jnp.asarray(np.asarray(a.waveform, dtype=np.float32)[:, 0, :])  # [B, S]
        mel = self.frontend(mono, num_frames=None)  # [B, T_mel, 128]
        t_mel = int(mel.shape[1])

        # mel-frame rate = sample_rate / FRAME_HOP (100 Hz). Snap window starts
        # onto the mel grid and dedup identical starts (cheap for a coarse hop).
        mel_frame_period = FRAME_HOP / self.config.sample_rate
        mel_start = round(start_seconds / mel_frame_period)
        snapped_times = np.floor((np.arange(num_windows) / frame_rate) / hop_seconds) * hop_seconds
        starts = mel_start + np.round(snapped_times / mel_frame_period).astype(np.int64)
        starts = np.minimum(starts, t_mel - NUM_FRAMES)
        uniq, inverse = np.unique(starts, return_inverse=True)
        emb_u = np.asarray(
            self.audio.encode_windows(
                mel, jnp.asarray(uniq), scan=scan, subbatch=window_subbatch
            )
        )  # [B, U, 768]
        return emb_u[:, inverse, :]

    def tokenize(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        if embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"Embedding dimension must be {self.config.embedding_dim},"
                f" got {embeddings.shape[-1]}."
            )
        tokens = self.quantizer.tokenize(np.asarray(embeddings, dtype=np.float32))
        return np.asarray(tokens, dtype=np.int32)


# Backwards compatibility alias
MusicCoCaModule = MusicCoCa


def load_safetensors(
    model: MusicCoCa, path: str | pathlib.Path
) -> MusicCoCa:
    """Loads ``convert.py`` output into the module tree (keys = attr paths)."""
    import safetensors.numpy

    flat = safetensors.numpy.load_file(str(path))
    for key, value in flat.items():
        obj = model
        *parents, leaf = key.split(".")
        for part in parents:
            obj = getattr(obj, part)
        param = getattr(obj, leaf)
        if param[...].shape != value.shape:
            raise ValueError(
                f"{key}: checkpoint shape {value.shape} != "
                f"module shape {param[...].shape}"
            )
        param.set_value(jnp.asarray(value))
    return model


def from_safetensors(path: str | pathlib.Path) -> MusicCoCa:
    return load_safetensors(MusicCoCa(load_weights=False), path)

