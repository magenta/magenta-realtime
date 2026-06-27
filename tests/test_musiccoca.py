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

"""Tests for the MusicCoCa module."""

import unittest
from pathlib import Path

import numpy as np

from audiotree import AudioTree

from magenta_rt import musiccoca
from magenta_rt import paths


# ---------------------------------------------------------------------------
# Mock-only tests
# ---------------------------------------------------------------------------


class MusicCoCaMockTest(unittest.TestCase):

  def test_embed_text(self):
    model = musiccoca.MockMusicCoCa()
    a, b, c = "metal", "rock", "metal"

    # A single string -> [dim]; a sequence -> [B, dim]. __call__ dispatches a
    # str to embed_text too.
    for embed in (model.embed_text, model.__call__):
      single = embed(a)
      self.assertIsInstance(single, np.ndarray)
      self.assertEqual(single.shape, (768,))

      batch = embed([a, b, c])
      self.assertEqual(batch.shape, (3, 768))
      self.assertTrue(np.array_equal(batch[0], batch[2]))
      self.assertFalse(np.array_equal(batch[0], batch[1]))

  def test_embed_audio(self):
    sr = 16000
    model = musiccoca.MockMusicCoCa(
        musiccoca.MusicCoCaConfiguration(sample_rate=sr, clip_length=1.0)
    )
    noise = np.random.rand(2, sr * 10).astype(np.float32)  # [C, T] channel-major
    a = AudioTree.create(noise[:, : sr * 1], sr)
    b = AudioTree.create(noise[:, sr * 1 : sr * 2], sr)

    # The AudioTree carries the batch axis: a [3, C, T] tree -> [3, dim].
    # Rows 0 and 2 (both `a`) match; row 1 (`b`) differs.
    batched = AudioTree(
        waveform=np.stack([a.waveform[0], b.waveform[0], a.waveform[0]], axis=0),
        sample_rate=sr,
    )
    embedded = model.embed_audio(batched)
    self.assertEqual(embedded.shape, (3, 768))
    self.assertTrue(np.array_equal(embedded[0], embedded[2]))
    self.assertFalse(np.array_equal(embedded[0], embedded[1]))
    # __call__ dispatches an AudioTree to embed_audio (batch axis preserved).
    self.assertEqual(model(a).shape, (1, 768))

    # Framing: a batch-1 ([1, C, T]) tree keeps the leading batch axis.
    clip_length_samples = model.config.clip_length_samples
    for num_samples in [0, sr // 2, sr * 2, round(sr * 2.5)]:
      w = AudioTree.create(noise[:, :num_samples], sr)
      for hop_length in [0.5, 1.0, 1.5]:
        hop_length_samples = round(hop_length * sr)
        for pool_across_time in [False, True]:
          for pad_end in [False, True]:
            embedded = model.embed_audio(
                w,
                hop_length=hop_length,
                pool_across_time=pool_across_time,
                pad_end=pad_end,
            )
            if pool_across_time:
              expected_shape = (1, 768)
            elif pad_end:
              expected_shape = (
                  1, int(np.ceil(num_samples / hop_length_samples)), 768)
            else:
              num_frames = max(
                  0,
                  (num_samples - clip_length_samples) // hop_length_samples + 1,
              )
              expected_shape = (1, num_frames, 768)
            self.assertEqual(embedded.shape, expected_shape)

  def test_tokenize(self):
    model = musiccoca.MockMusicCoCa()
    embedding = model.embed_text("metal")
    tokens = model.tokenize(embedding)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (12,))
    tokens = model.tokenize(np.array([embedding, embedding]))
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (2, 12))
    tokens = model.tokenize(np.array([[embedding, embedding]]))
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (1, 2, 12))


# ---------------------------------------------------------------------------
# TFLite integration tests (require real models in ~/.magenta/…/musiccoca)
# ---------------------------------------------------------------------------

_RESOURCE_DIR = paths.musiccoca_dir()
_REQUIRED_FILES = [
    'spm.model',
    'text_encoder.tflite',
    'audio_preprocessor.tflite',
    'music_encoder.tflite',
    'pretrained_vector_quantizer.tflite',
    'mapper.tflite',
]
_HAS_MODELS = all((_RESOURCE_DIR / f).exists() for f in _REQUIRED_FILES)

# Golden reference tokens (verified against musiccoca_tokens.py).
_GOLDEN_TOKENS = {
    'a jazz piano trio': [555, 402,  72, 993, 482, 932, 402, 153,  33, 374, 519, 147],
    'heavy metal guitar riff': [322, 825, 598, 131, 826, 656, 592, 759,  89, 410, 778, 642],
    'ambient electronic': [796, 540,  12, 691, 843, 202, 225, 516, 324, 836,  41, 574],
    'disco funk': [660, 597, 668, 315, 857, 217, 930, 175, 655, 343, 534, 137],
}

# Golden reference tokens with mapper (use_mapper=True, seed=0, L2 normalized).
_GOLDEN_TOKENS_MAPPER = {
    'a jazz piano trio': [555, 333, 237, 468, 451, 244, 82, 624, 719, 466, 168, 601],
    'heavy metal guitar riff': [270, 797, 578, 144, 653, 254, 942, 760, 89, 390, 437, 303],
    'ambient electronic': [369, 867, 202, 38, 413, 255, 213, 905, 340, 893, 300, 725],
    'disco funk': [660, 1016, 208, 530, 589, 702, 376, 54, 203, 444, 36, 246],
}


@unittest.skipUnless(_HAS_MODELS, f'MusicCoCa TFLite models not found in {_RESOURCE_DIR}')
class MusicCoCaTFLiteTest(unittest.TestCase):
  """Integration tests using real TFLite models."""

  @classmethod
  def setUpClass(cls):
    cls.model = musiccoca.MusicCoCa(resource_dir=_RESOURCE_DIR)

  # -- Text embedding -------------------------------------------------------

  def test_text_embedding_shape(self):
    """Text embedding should produce a 768-dim float32 vector."""
    emb = self.model.embed_text('a jazz piano trio')
    self.assertEqual(emb.shape, (768,))
    self.assertEqual(emb.dtype, np.float32)

  def test_text_embedding_batch(self):
    """Batch text embedding should stack correctly."""
    prompts = list(_GOLDEN_TOKENS.keys())
    emb = self.model.embed_text(prompts)
    self.assertEqual(emb.shape, (len(prompts), 768))

  def test_text_embedding_deterministic(self):
    """Same text should always produce the same embedding."""
    emb1 = self.model.embed_text('a jazz piano trio')
    emb2 = self.model.embed_text('a jazz piano trio')
    np.testing.assert_array_equal(emb1, emb2)

  def test_text_embedding_different_prompts(self):
    """Different prompts should produce different embeddings."""
    emb_a = self.model.embed_text('a jazz piano trio')
    emb_b = self.model.embed_text('heavy metal guitar riff')
    self.assertFalse(np.array_equal(emb_a, emb_b))

  # -- Tokenization ---------------------------------------------------------

  def test_tokenize_golden(self):
    """Tokens should match golden reference values from musiccoca_tokens.py."""
    for prompt, expected in _GOLDEN_TOKENS.items():
      with self.subTest(prompt=prompt):
        emb = self.model.embed_text(prompt)
        tokens = self.model.tokenize(emb)
        self.assertEqual(tokens.tolist(), expected)

  def test_tokenize_batch(self):
    """Batch tokenization should produce per-prompt results matching singles."""
    prompts = list(_GOLDEN_TOKENS.keys())
    embeddings = self.model.embed_text(prompts)
    tokens_batch = self.model.tokenize(embeddings)
    self.assertEqual(tokens_batch.shape, (len(prompts), 12))
    for i, prompt in enumerate(prompts):
      self.assertEqual(tokens_batch[i].tolist(), _GOLDEN_TOKENS[prompt])

  def test_tokenize_values_in_range(self):
    """All token values should be valid codebook indices."""
    emb = self.model.embed_text('classical string quartet')
    tokens = self.model.tokenize(emb)
    self.assertTrue(np.all(tokens >= 0))
    self.assertTrue(np.all(tokens < self.model.config.rvq_codebook_size))

  # -- Mapper ---------------------------------------------------------------

  def test_mapper_embedding_shape(self):
    """Mapper embedding should produce a 768-dim float32 vector."""
    emb = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    self.assertEqual(emb.shape, (768,))
    self.assertEqual(emb.dtype, np.float32)

  def test_mapper_embedding_is_l2_normalized(self):
    """Mapper embedding should be L2 normalized."""
    emb = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    np.testing.assert_almost_equal(np.linalg.norm(emb), 1.0, decimal=5)

  def test_mapper_embedding_deterministic(self):
    """Same text + seed should always produce the same mapper embedding."""
    emb1 = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    emb2 = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    np.testing.assert_array_equal(emb1, emb2)

  def test_mapper_embedding_differs_from_unmapped(self):
    """Mapper embedding should differ from plain text embedding."""
    emb_plain = self.model.embed_text('disco funk')
    emb_mapped = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    self.assertFalse(np.array_equal(emb_plain, emb_mapped))

  def test_mapper_different_seeds(self):
    """Different seeds should produce different mapper embeddings."""
    emb_s0 = self.model.embed_text('disco funk', use_mapper=True, seed=0)
    emb_s1 = self.model.embed_text('disco funk', use_mapper=True, seed=1)
    self.assertFalse(np.array_equal(emb_s0, emb_s1))

  def test_mapper_tokenize_golden(self):
    """Mapper tokens should match golden reference values."""
    for prompt, expected in _GOLDEN_TOKENS_MAPPER.items():
      with self.subTest(prompt=prompt):
        emb = self.model.embed_text(prompt, use_mapper=True, seed=0)
        tokens = self.model.tokenize(emb)
        self.assertEqual(tokens.tolist(), expected)

  # -- Audio embedding ------------------------------------------------------

  def test_audio_embedding_shape(self):
    """Audio embedding should produce a 768-dim float32 vector."""
    sr = self.model.config.sample_rate
    waveform = AudioTree.create(np.random.randn(sr * 10).astype(np.float32), sr)
    emb = self.model.embed_audio(waveform)[0]
    self.assertEqual(emb.shape, (768,))
    self.assertEqual(emb.dtype, np.float32)
    self.assertTrue(np.all(np.isfinite(emb)))

  def test_audio_embedding_deterministic(self):
    """Same audio should always produce the same embedding."""
    sr = self.model.config.sample_rate
    waveform = AudioTree.create(np.random.randn(sr * 10).astype(np.float32), sr)
    emb1 = self.model.embed_audio(waveform)[0]
    emb2 = self.model.embed_audio(waveform)[0]
    np.testing.assert_array_equal(emb1, emb2)

  def test_audio_tokenize(self):
    """Audio → embed → tokenize should produce valid tokens."""
    sr = self.model.config.sample_rate
    waveform = AudioTree.create(np.random.randn(sr * 10).astype(np.float32), sr)
    emb = self.model.embed_audio(waveform)[0]
    tokens = self.model.tokenize(emb)
    self.assertEqual(tokens.shape, (12,))
    self.assertTrue(np.all(tokens >= 0))
    self.assertTrue(np.all(tokens < self.model.config.rvq_codebook_size))

  # -- End-to-end pipeline --------------------------------------------------

  def test_embed_and_tokenize_pipeline(self):
    """Full pipeline: embed text → average → tokenize."""
    emb1 = self.model.embed_text('jazz')
    emb2 = self.model.embed_text('piano')
    blended = np.mean([emb1, emb2], axis=0)
    tokens = self.model.tokenize(blended)
    self.assertEqual(tokens.shape, (12,))
    self.assertTrue(np.all(tokens >= 0))
    self.assertTrue(np.all(tokens < self.model.config.rvq_codebook_size))


if __name__ == "__main__":
  unittest.main()
