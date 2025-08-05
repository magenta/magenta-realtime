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

from typing import Optional, Tuple

from absl.testing import absltest
import numpy as np

from . import audio
from . import musiccoca
from . import spectrostream
from . import system


class TestMagentaRT(absltest.TestCase):

  def test_magenta_rt_vocab_size(self):
    config = system.MagentaRTConfiguration()
    self.assertEqual(config.vocab_reserved_tokens["PAD"], 0)
    self.assertEqual(config.vocab_reserved_tokens["MASK"], 1)
    self.assertLen(config.vocab_reserved_tokens, 2)
    self.assertEqual(config.vocab_codec_offset, 2)
    self.assertEqual(config.vocab_codec_size, 16384)
    self.assertEqual(config.vocab_codec_offset + config.vocab_codec_size, 16386)
    self.assertEqual(config.vocab_style_offset, 17410)
    self.assertEqual(config.vocab_style_size, 6144)
    self.assertEqual(config.vocab_style_offset + config.vocab_style_size, 23554)
    self.assertEqual(config.vocab_size, 23554)
    self.assertEqual(config.vocab_size_pretrained, 29698)

  def test_magenta_rt_configuration_valid(self):
    config = system.MagentaRTConfiguration()
    self.assertEqual(config.chunk_length, 2.0)
    self.assertEqual(config.context_tokens_shape, (250, 16))

  def test_magenta_rt_configuration(self):
    sys = system.MockMagentaRT()
    self.assertEqual(sys.sample_rate, 48000)
    self.assertEqual(sys.codec.frame_rate, 25.0)
    self.assertEqual(sys.chunk_length, 2.0)
    self.assertEqual(sys.num_channels, 2)
    self.assertEqual(sys.config.chunk_length_samples, 96000)
    self.assertEqual(sys.config.chunk_length_frames, 50)

  def test_magenta_rt_configuration_invalid_sample_rate(self):
    with self.assertRaises(ValueError):
      system.MockMagentaRT(
          config=system.MagentaRTConfiguration(chunk_length=1.05),
          codec_config=spectrostream.SpectroStreamConfiguration(sample_rate=10),
      )

  def test_magenta_rt_configuration_invalid_frame_rate(self):
    with self.assertRaises(ValueError):
      system.MockMagentaRT(
          config=system.MagentaRTConfiguration(chunk_length=2.0),
          codec_config=spectrostream.SpectroStreamConfiguration(
              frame_rate=25.1
          ),
      )

  def test_magenta_rt_configuration_custom_values(self):
    sys = system.MockMagentaRT(
        config=system.MagentaRTConfiguration(
            chunk_length=1.0,
            codec_sample_rate=44100,
            crossfade_length=0.5,
            codec_frame_rate=30.0,
        ),
        codec_config=spectrostream.SpectroStreamConfiguration(
            sample_rate=44100, frame_rate=30.0, num_channels=1
        ),
    )
    self.assertEqual(sys.sample_rate, 44100)
    self.assertEqual(sys.codec.frame_rate, 30.0)
    self.assertEqual(sys.chunk_length, 1.0)
    self.assertEqual(sys.num_channels, 1)
    self.assertEqual(sys.config.chunk_length_samples, 44100)
    self.assertEqual(sys.config.chunk_length_frames, 30)

  def test_mock_magenta_rt_system_init(self):
    sys = system.MockMagentaRT()
    self.assertIsInstance(sys.config, system.MagentaRTConfiguration)
    self.assertEqual(sys.sample_rate, 48000)
    self.assertEqual(sys.codec.frame_rate, 25.0)

  def test_mock_magenta_rt_system_embed_style(self):
    sys = system.MockMagentaRT()
    style = sys.embed_style("test")
    self.assertIsInstance(style, np.ndarray)
    self.assertEqual(style.shape, (768,))

  def test_mock_magenta_rt_system_live_continuation(self):
    sys = system.MockMagentaRT()
    result, state = sys.generate_chunk(seed=42)
    self.assertIsInstance(result, audio.Waveform)
    self.assertIsInstance(state, system.MagentaRTState)
    self.assertEqual(
        result.samples.shape,
        (sys.config.chunk_length_samples, sys.num_channels),
    )
    self.assertEqual(result.sample_rate, sys.sample_rate)
    self.assertEqual(state.shape, sys.config.context_tokens_shape)

  def test_mock_magenta_rt_system_live_continuation_no_seed(self):
    sys = system.MockMagentaRT()
    result, state = sys.generate_chunk()
    self.assertIsInstance(result, audio.Waveform)
    self.assertIsInstance(state, system.MagentaRTState)

  def test_magenta_rt_system_abstract(self):
    class MyMagentaRT(system.MagentaRTBase):

      def __init__(self, config: system.MagentaRTConfiguration):
        super().__init__(
            config=config,
            codec=spectrostream.MockSpectroStream(),
            style_model=musiccoca.MockMusicCoCa(),
        )

      def generate_chunk(
          self,
          state: Optional[system.MagentaRTState] = None,
          style: Optional[musiccoca.StyleEmbedding] = None,
          seed: Optional[int] = None,
      ) -> Tuple[audio.Waveform, system.MagentaRTState]:
        return audio.Waveform(np.zeros((1, 2)), 1), np.zeros((1, 1))

    config = system.MagentaRTConfiguration()
    sys = MyMagentaRT(config)
    self.assertIsInstance(sys.config, system.MagentaRTConfiguration)
    sys.embed_style("a")
    sys.generate_chunk()


if __name__ == "__main__":
  absltest.main()
