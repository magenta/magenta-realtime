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

from . import decode

mock = absltest.mock


class DecodeTest(absltest.TestCase):

  def test_constrained_decoding_semantic(self):
    in_logits = np.ones(5122)
    expected_logits = np.full_like(in_logits, -np.inf)
    expected_logits[4098:5122] = 1.0

    for i in range(250):
      state = mock.Mock(cur_index=[i])
      new_logits = decode.constrained_logit_callback_fn(
          in_logits, state, split_point=250
      )
      np.testing.assert_equal(new_logits, expected_logits)

  def test_constrained_decoding_acoustic(self):
    in_logits = np.ones(5122)

    for i in range(250, 2250):
      expected_logits = np.full_like(in_logits, -np.inf)
      expected_logits[
          2 + (i - 250) % 4 * 1024 : 2 + ((i - 250) % 4 + 1) * 1024
      ] = 1.0
      state = mock.Mock(cur_index=[i])
      new_logits = decode.constrained_logit_callback_fn(
          in_logits, state, split_point=250
      )
      np.testing.assert_equal(new_logits, expected_logits)

  def test_constrained_decoding_style(self):
    in_logits = np.ones(17410)

    for i in range(12):
      expected_logits = np.ones(17410) * -np.inf
      expected_logits[5122 + i * 1024 : 5122 + (i + 1) * 1024] = 1.0
      state = mock.Mock(cur_index=[i])
      new_logits = decode.constrained_logit_callback_fn(
          in_logits, state, split_point=250, style_depth=12
      )
      np.testing.assert_equal(new_logits, expected_logits)

    # Test semantic with style offset.
    expected_logits = np.ones(17410) * -np.inf
    expected_logits[4098:5122] = 1.0
    state = mock.Mock(cur_index=[13])
    new_logits = decode.constrained_logit_callback_fn(
        in_logits, state, split_point=250, style_depth=12
    )
    np.testing.assert_equal(new_logits, expected_logits)


if __name__ == "__main__":
  absltest.main()
