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

"""MusicCoCa in pure MLX.

``MusicCoCaModule`` is the MLX assembly of the five reverse-engineered
components; weights load from the safetensors file produced by
``python -m magenta_rt.nnx.musiccoca.convert`` (shared with the nnx
backend — stacked per-layer arrays are split across the layer lists
here, since MLX has no scan-over-stacked-params idiom).

``MusicCoCa`` wraps it in the same high-level interface as
:class:`magenta_rt.musiccoca.MusicCoCa`.
"""

from __future__ import annotations

import functools
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ... import musiccoca as musiccoca_base
from ... import paths
from .frontend import CLIP_SAMPLES, LogMelFrontend
from .mapper import Mapper
from .modules import AudioEncoder, TextEncoder
from .quantizer import EmbeddingQuantizer

# Filename shared with the nnx backend's converter.
WEIGHTS_FILENAME = "musiccoca_nnx.safetensors"

MAX_TEXT_LENGTH = 128
TEXT_SOS_ID = 1


class MusicCoCaModule(nn.Module):
    """The full MusicCoCa model as a single MLX module tree."""

    def __init__(self):
        super().__init__()
        self.frontend = LogMelFrontend()
        self.audio = AudioEncoder()
        self.text = TextEncoder()
        self.quantizer = EmbeddingQuantizer()
        self.mapper = Mapper()

    def embed_audio(self, waveform: mx.array) -> mx.array:
        """Mono clips ``[B, 160000]`` @ 16 kHz → embeddings ``[B, 768]``."""
        return self.audio(self.frontend(waveform))

    def embed_text(self, ids: mx.array, paddings: mx.array) -> mx.array:
        """Token ids/paddings ``[B, 128]`` → embeddings ``[B, 768]``."""
        return self.text(ids, paddings)

    def tokenize(self, embeddings: mx.array) -> mx.array:
        return self.quantizer.tokenize(embeddings)

    def map_text_embedding(
        self, text_emb: mx.array, noise: mx.array
    ) -> mx.array:
        """Maps text embeddings toward audio space; L2-normalized."""
        mapped = self.mapper(text_emb, noise)
        norm = mx.sqrt(mx.sum(mx.square(mapped), axis=-1, keepdims=True))
        return mapped / norm


def load_safetensors(
    module: MusicCoCaModule, path: str | pathlib.Path
) -> MusicCoCaModule:
    """Loads the shared converter output, splitting stacked layer weights.

    Keys like ``audio.layers.q.kernel`` hold all 12 layers stacked on a
    leading axis; they expand to ``audio.layers.{i}.q.kernel`` to match
    the MLX layer lists. All other keys map 1:1 to module paths.
    """
    flat = mx.load(str(path))
    weights: list[tuple[str, mx.array]] = []
    for key, value in flat.items():
        tower, _, rest = key.partition(".layers.")
        if rest:
            for i in range(value.shape[0]):
                weights.append((f"{tower}.layers.{i}.{rest}", value[i]))
        else:
            weights.append((key, value))
    module.load_weights(weights, strict=True)
    mx.eval(module.parameters())
    return module


def from_safetensors(path: str | pathlib.Path) -> MusicCoCaModule:
    return load_safetensors(MusicCoCaModule(), path)


def encode_text(vocab, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Lowercased SentencePiece ids + paddings, shaped ``[128]`` each."""
    labels = vocab.EncodeAsIds(text.lower())[: MAX_TEXT_LENGTH - 1]
    ids = [TEXT_SOS_ID] + labels
    num_tokens = len(ids)
    ids = ids + [0] * (MAX_TEXT_LENGTH - num_tokens)
    paddings = np.ones(MAX_TEXT_LENGTH, dtype=np.float32)
    paddings[:num_tokens] = 0.0
    return np.array(ids, dtype=np.int32), paddings


class MusicCoCa(musiccoca_base.MusicCoCaBase):
    """High-level MusicCoCa backed by the pure-MLX implementation.

    Same interface and resource directory as the TFLite-backed
    :class:`magenta_rt.musiccoca.MusicCoCa`; expects ``spm.model`` and the
    safetensors produced by ``python -m magenta_rt.nnx.musiccoca.convert``.
    """

    def __init__(
        self,
        resource_dir: str | pathlib.Path | None = None,
        lazy: bool = True,
    ):
        super().__init__(
            musiccoca_base.MusicCoCaConfiguration(
                sample_rate=16000,
                clip_length=10.0,
                embedding_dim=768,
                rvq_depth=12,
                rvq_codebook_size=1024,
            )
        )
        self._resource_dir = pathlib.Path(resource_dir or paths.musiccoca_dir())
        if not lazy:
            self._vocab  # pylint: disable=pointless-statement
            self._module  # pylint: disable=pointless-statement

    @functools.cached_property
    def _vocab(self):
        import sentencepiece

        spm_path = self._resource_dir / "spm.model"
        if not spm_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {spm_path}")
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load(str(spm_path))
        return sp

    @functools.cached_property
    def _module(self) -> MusicCoCaModule:
        path = self._resource_dir / WEIGHTS_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `python -m"
                " magenta_rt.nnx.musiccoca.convert` to create it from the"
                " TFLite resources."
            )
        return from_safetensors(path)

    def _embed_batch_text(
        self,
        batch_text: list[str],
        use_mapper: bool = False,
        seed: int = 0,
    ) -> np.ndarray:
        encoded = [encode_text(self._vocab, s) for s in batch_text]
        ids = mx.array(np.stack([e[0] for e in encoded]))
        paddings = mx.array(np.stack([e[1] for e in encoded]))
        emb = self._module.embed_text(ids, paddings)
        if use_mapper:
            # Matches the TFLite wrapper: one fixed-seed Gaussian noise
            # vector shared across the batch.
            rng = np.random.RandomState(seed)
            noise = rng.randn(self.config.embedding_dim).astype(np.float32)
            noise = mx.array(np.broadcast_to(noise, (len(batch_text), 768)))
            emb = self._module.map_text_embedding(emb, noise)
        return np.array(emb, dtype=np.float32)

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
        emb = self._module.embed_audio(mx.array(clips))
        return np.array(emb, dtype=np.float32)

    def tokenize(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        if embeddings.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"Embedding dimension must be {self.config.embedding_dim},"
                f" got {embeddings.shape[-1]}."
            )
        tokens = self._module.tokenize(
            mx.array(np.asarray(embeddings, dtype=np.float32))
        )
        return np.array(tokens, dtype=np.int32)
