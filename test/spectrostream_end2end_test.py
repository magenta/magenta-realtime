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
from magenta_rt import spectrostream


class SpectroStreamTest(absltest.TestCase):

  def test_spectrostream_savedmodel(self):
    spectrostream_model = spectrostream.SpectroStream(skip_cache=True)
    waveform = audio.Waveform(
        np.random.rand(16000, 2).astype(np.float32), 16000
    )
    tokens = spectrostream_model.encode(waveform)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(
        tokens.shape,
        (
            25,
            64,
        ),
    )
    rt = spectrostream_model.decode(tokens)
    self.assertIsInstance(rt, audio.Waveform)
    self.assertEqual(rt.num_samples, 48000)
    self.assertEqual(rt.num_channels, 2)
    rt_lowq = spectrostream_model.decode(tokens[:, :16])
    self.assertIsInstance(rt_lowq, audio.Waveform)
    self.assertEqual(rt_lowq.num_samples, 48000)
    self.assertEqual(rt_lowq.num_channels, 2)
    self.assertFalse(np.array_equal(rt_lowq.samples, rt.samples))

  def test_spectrostream_against_reference(self):
    # Load test audio
    waveforms = [
        audio.Waveform(w.swapaxes(0, 1), 48000)
        for w in np.load(asset.fetch('testdata/ssv2_48k_stereo/audio.npy'))
    ]
    self.assertEqual([w.num_samples for w in waveforms], [96000, 96000])
    self.assertEqual([w.num_channels for w in waveforms], [2, 2])

    # Encode and compare against reference tokens
    with tf.device('/cpu:0'):
      spectrostream_model = spectrostream.SpectroStream(skip_cache=True)
      tokens = spectrostream_model.encode(waveforms)
      tokens_ref = np.load(asset.fetch('testdata/ssv2_48k_stereo/tokens.npy'))
      self.assertEqual(tokens.shape, tokens_ref.shape)
      np.testing.assert_array_equal(tokens, tokens_ref)

      # Decode and compare against reference round trip
      rt = np.array(
          [w.samples.swapaxes(0, 1) for w in spectrostream_model.decode(tokens)]
      )
      rt_ref = np.load(asset.fetch('testdata/ssv2_48k_stereo/audio_rt.npy'))
      self.assertEqual(rt.shape, rt_ref.shape)
      self.assertEqual(rt.dtype, rt_ref.dtype)
      self.assertLess(np.max(np.abs(rt - rt_ref)), 1e-4)


if __name__ == '__main__':
  absltest.main()
