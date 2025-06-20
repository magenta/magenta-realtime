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

from . import audio
from . import musiccoca


class MusicCoCaTest(absltest.TestCase):

  def test_embed_text(self):
    model = musiccoca.MockMusicCoCa()

    a = "metal"
    b = "rock"
    c = a[:]

    for fn in [
        model.embed_batch_text,
        model.embed,
        model.__call__,
    ]:
      if fn != model.embed_batch_text:
        embedded = fn(a)
        self.assertIsInstance(embedded, np.ndarray)
        self.assertEqual(embedded.shape, (768,))

      embedded = fn([a, b, c])
      self.assertEqual(embedded.shape, (3, 768))
      self.assertTrue(np.array_equal(embedded[0], embedded[2]))
      self.assertFalse(np.array_equal(embedded[0], embedded[1]))

    with self.assertRaises(TypeError):
      model.embed_batch_text(a)

  def test_embed_audio(self):
    sr = 16000
    model = musiccoca.MockMusicCoCa(
        musiccoca.MusicCoCaConfiguration(
            sample_rate=sr,
            clip_length=1.0,
        )
    )

    noise = np.random.rand(sr * 10, 2).astype(np.float32)

    a = audio.Waveform(noise[: sr * 1], 16000)
    b = audio.Waveform(noise[sr * 1 : sr * 2], 16000)
    a_copy = audio.Waveform(a.samples.copy(), 16000)

    # Check basic use.
    for fn in [
        model.embed_batch_audio,
        model.embed,
        model.__call__,
    ]:
      if fn != model.embed_batch_audio:
        embedded = fn(a)
        self.assertIsInstance(embedded, np.ndarray)
        self.assertEqual(embedded.shape, (768,))

      embedded = fn([a, b, a_copy])
      self.assertEqual(embedded.shape, (3, 768))
      self.assertTrue(np.array_equal(embedded[0], embedded[2]))
      self.assertFalse(np.array_equal(embedded[0], embedded[1]))

    # Test framing.
    clip_length_samples = model.config.clip_length_samples
    for num_samples in [0, sr // 2, sr * 2, round(sr * 2.5)]:
      w = audio.Waveform(noise[:num_samples], 16000)
      for hop_length in [0.5, 1.0, 1.5]:
        hop_length_samples = round(hop_length * sr)
        for pool_across_time in [False, True]:
          for pad_end in [False, True]:
            embedded = model.embed(
                w,
                hop_length=hop_length,
                pool_across_time=pool_across_time,
                pad_end=pad_end,
            )
            if pool_across_time:
              expected_shape = (768,)
            elif pad_end:
              expected_shape = (np.ceil(num_samples / hop_length_samples), 768)
            else:
              num_frames = max(
                  0,
                  (num_samples - clip_length_samples) // hop_length_samples + 1,
              )
              expected_shape = (num_frames, 768)
            self.assertEqual(embedded.shape, expected_shape)

    with self.assertRaises(NotImplementedError):
      model.embed_batch_audio([
          audio.Waveform(noise[: sr * 1], 16000),
          audio.Waveform(noise[: sr * 2], 16000),
      ])

  def test_tokenize(self):
    model = musiccoca.MockMusicCoCa()
    embedding = model.embed("metal")
    tokens = model.tokenize(embedding)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (12,))
    tokens = model.tokenize(np.array([embedding, embedding]))
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (2, 12))
    tokens = model.tokenize(np.array([[embedding, embedding]]))
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (1, 2, 12))


if __name__ == "__main__":
  absltest.main()
