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

from magenta_rt import audio
from magenta_rt import system


class MagentaRTEnd2EndTest(absltest.TestCase):

  def test_magenta_rt(self):
    mrt = system.MagentaRT(tag="base", skip_cache=True, lazy=True)
    mrt.warm_start()
    waveform, state = mrt.generate_chunk(max_decode_frames=10)
    self.assertEqual(waveform.samples.shape, (21120, 2))
    self.assertEqual(state.context_tokens.shape, (250, 16))
    waveform, state = mrt.generate_chunk()
    self.assertIsInstance(waveform, audio.Waveform)
    self.assertIsInstance(state, system.MagentaRTState)
    self.assertEqual(waveform.sample_rate, 48000)
    self.assertEqual(waveform.samples.shape, (97920, 2))
    self.assertEqual(state.context_tokens.shape, (250, 16))

  # TODO(chrisdonahue): add integration test against real outputs


if __name__ == "__main__":
  absltest.main()
