# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from magenta_rt import asset
from magenta_rt import audio
from magenta_rt import musiccoca


class MusicCoCaTest(absltest.TestCase):

  def test_musiccoca_savedmodel(self):
    style_model = musiccoca.MusicCoCa(skip_cache=True)

    # Single
    embedding = style_model.embed("metal")
    self.assertIsInstance(embedding, np.ndarray)
    self.assertEqual(embedding.shape, (768,))
    tokens = style_model.tokenize(embedding)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (12,))

    # Batch
    embeddings = style_model.embed(["metal", "rock"])
    self.assertIsInstance(embeddings, np.ndarray)
    self.assertEqual(embeddings.shape, (2, 768))
    tokens = style_model.tokenize(embeddings)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (2, 12))

  def test_musiccoca_against_reference(self):
    # Test text
    with open(asset.fetch("testdata/musiccoca_mv212/inputs.txt")) as f:
      input_text = f.read().strip().splitlines()
    style_model = musiccoca.MusicCoCa(skip_cache=True)
    embeddings = style_model.embed(input_text)
    embeddings_ref = np.load(
        asset.fetch("testdata/musiccoca_mv212/embeddings.npy")
    )
    error = np.abs(embeddings - embeddings_ref)
    self.assertLess(np.max(error), 1e-5)

    # Test tokens
    tokens = style_model.tokenize(embeddings_ref)
    tokens_ref = np.load(asset.fetch("testdata/musiccoca_mv212/tokens.npy"))
    np.testing.assert_array_equal(tokens, tokens_ref)

    # Test quantizer against SavedModel.
    quantizer_sm = tf.saved_model.load(
        asset.fetch(
            "savedmodels/musiccoca_mv212_quant", skip_cache=True, is_dir=True
        )
    )
    tokens_sm = quantizer_sm(tf.constant(embeddings_ref)).cpu().numpy()
    np.testing.assert_array_equal(tokens_sm[:, 0], tokens_ref)

    # Test audio
    samples = np.load(asset.fetch("testdata/musiccoca_mv212/audio.npy"))
    waveforms = [audio.Waveform(s, 16000) for s in samples]
    embeddings = style_model.embed(waveforms)
    embeddings_ref = np.load(
        asset.fetch("testdata/musiccoca_mv212/embeddings_audio.npy")
    )
    error = np.abs(embeddings - embeddings_ref)
    self.assertLess(np.max(error), 1e-5)

  def test_musiccoca_audio_embedding(self):
    style_model = musiccoca.MusicCoCa(skip_cache=True)
    waveform = audio.Waveform(np.random.randn(160000, 2), 16000)
    embeddings = style_model([waveform, "static", waveform, "noise"])
    self.assertIsInstance(embeddings, np.ndarray)
    self.assertEqual(embeddings.shape, (4, 768))


if __name__ == "__main__":
  absltest.main()
