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

import io
import tempfile

from absl.testing import absltest
import numpy as np

from . import audio


class TestWaveform(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_rate = 44100
    self.samples = np.random.rand(100, 2).astype(np.float32)
    self.waveform = audio.Waveform(self.samples, self.sample_rate)

  def test_init(self):
    self.assertEqual(self.waveform.sample_rate, self.sample_rate)
    self.assertTrue(np.array_equal(self.waveform.samples, self.samples))

  def test_sample_rate_property(self):
    self.assertEqual(self.waveform.sample_rate, self.sample_rate)

  def test_sample_rate_setter_raises_exception(self):
    with self.assertRaises(AttributeError):
      self.waveform.sample_rate = 48000

  def test_samples_property(self):
    self.assertTrue(np.array_equal(self.waveform.samples, self.samples))

  def test_samples_setter_valid_input(self):
    new_samples = np.random.rand(50, 2).astype(np.float32)
    self.waveform.samples = new_samples
    self.assertTrue(np.array_equal(self.waveform.samples, new_samples))

  def test_samples_mono(self):
    samples = np.random.rand(100).astype(np.float32)
    waveform = audio.Waveform(samples, self.sample_rate)
    self.assertEqual(waveform.num_channels, 1)
    self.assertEqual(waveform.samples.shape, (100, 1))

  def test_samples_setter_invalid_ndim(self):
    invalid_samples = np.random.rand(100, 1, 1).astype(np.float32)
    with self.assertRaises(ValueError):
      self.waveform.samples = invalid_samples

  def test_samples_setter_invalid_shape(self):
    invalid_samples = np.random.rand(100, 0).astype(np.float32)
    with self.assertRaises(ValueError):
      self.waveform.samples = invalid_samples

  def test_samples_setter_invalid_dtype(self):
    invalid_samples = np.random.rand(100, 2).astype(np.int16)
    with self.assertRaises(TypeError):
      self.waveform.samples = invalid_samples

  def test_shape(self):
    self.assertEqual(self.waveform.num_samples, 100)
    self.assertLen(self.waveform, 100)
    self.assertEqual(self.waveform.num_channels, 2)

  def test_resample(self):
    new_sample_rate = 22050
    new_waveform = self.waveform.resample(new_sample_rate)
    self.assertEqual(new_waveform.sample_rate, new_sample_rate)
    self.assertEqual(
        new_waveform.num_samples,
        self.waveform.num_samples // 2,
    )
    self.assertEqual(new_waveform.num_channels, self.waveform.num_channels)

  def test_as_mono(self):
    mono_waveform = self.waveform.as_mono()
    self.assertEqual(mono_waveform.sample_rate, self.sample_rate)
    self.assertEqual(mono_waveform.num_samples, self.waveform.num_samples)
    self.assertEqual(mono_waveform.num_channels, 1)
    self.assertTrue(
        np.array_equal(
            mono_waveform.samples, self.samples.mean(axis=1, keepdims=True)
        )
    )

  def test_peak_normalize(self):
    silence = audio.Waveform(
        np.zeros((self.sample_rate * 10, 2), dtype=np.float32),
        sample_rate=self.sample_rate,
    )
    self.assertEqual(silence.peak_amplitude, 0)
    silence_norm = silence.peak_normalize()
    self.assertEqual(silence.peak_amplitude, 0)
    self.assertEqual(silence_norm.peak_amplitude, 0)

  def test_amp_to_db(self):
    self.assertAlmostEqual(audio.amp_to_db(0.0), float("-inf"))
    self.assertAlmostEqual(audio.amp_to_db(0.01), -40.0)
    self.assertAlmostEqual(audio.amp_to_db(0.1), -20.0)
    self.assertAlmostEqual(audio.amp_to_db(1), 0.0)
    self.assertAlmostEqual(audio.amp_to_db(10), 20.0)
    self.assertAlmostEqual(audio.amp_to_db(0.01, amp_ref=0.1), -20.0)
    with self.assertRaises(ValueError):
      audio.amp_to_db(-1.0)
    with self.assertRaises(ValueError):
      audio.amp_to_db(1.0, amp_ref=0.0)
    with self.assertRaises(ValueError):
      audio.amp_to_db(1.0, amp_ref=-1.0)

  def test_db_to_amp(self):
    self.assertAlmostEqual(audio.db_to_amp(float("-inf")), 0.0)
    self.assertAlmostEqual(audio.db_to_amp(-40.0), 0.01)
    self.assertAlmostEqual(audio.db_to_amp(-20.0), 0.1)
    self.assertAlmostEqual(audio.db_to_amp(0.0), 1.0)
    self.assertAlmostEqual(audio.db_to_amp(20.0), 10.0)
    self.assertAlmostEqual(audio.db_to_amp(-20.0, amp_ref=0.1), 0.01)
    with self.assertRaises(ValueError):
      audio.db_to_amp(0.0, amp_ref=0.0)
    with self.assertRaises(ValueError):
      audio.db_to_amp(0.0, amp_ref=-1.0)

  def test_from_file_str(self):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
      self.waveform.write(f.name, subtype="FLOAT")
      waveform = audio.Waveform.from_file(f.name)
      self.assertEqual(waveform.sample_rate, self.sample_rate)
      self.assertEqual(waveform.samples.shape, self.samples.shape)
      print(np.sum(np.abs(waveform.samples - self.samples)))
      self.assertTrue(np.array_equal(waveform.samples, self.samples))

  def test_from_file_io(self):
    memfile = io.BytesIO()
    self.waveform.write(memfile, format="WAV", subtype="FLOAT")
    memfile.seek(0)
    waveform = audio.Waveform.from_file(memfile)
    self.assertEqual(waveform.sample_rate, self.sample_rate)
    self.assertEqual(waveform.samples.shape, self.samples.shape)
    self.assertTrue(np.array_equal(waveform.samples, self.samples))


if __name__ == "__main__":
  absltest.main()
