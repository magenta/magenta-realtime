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
from . import spectrostream


class SpectroStreamTest(absltest.TestCase):

  def test_codec(self):
    ss_model = spectrostream.MockSpectroStream()

    a = audio.Waveform(np.random.rand(32000, 2).astype(np.float32), 16000)
    b = audio.Waveform(np.random.rand(32000, 2).astype(np.float32), 16000)

    tokens = ss_model.encode(a)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (50, 64))

    rt = ss_model.decode(tokens)
    self.assertIsInstance(rt, audio.Waveform)
    self.assertEqual(rt.sample_rate, 48000)
    self.assertEqual(rt.num_samples, 96000)
    self.assertEqual(rt.num_channels, 2)

    tokens = ss_model.encode([a, b])
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (2, 50, 64))

    rt = ss_model.decode(tokens)
    self.assertIsInstance(rt, list)
    self.assertLen(rt, 2)
    for w in rt:
      self.assertEqual(w.sample_rate, 48000)
      self.assertEqual(w.num_samples, 96000)
      self.assertEqual(w.num_channels, 2)


if __name__ == "__main__":
  absltest.main()
