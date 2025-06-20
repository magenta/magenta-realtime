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

import warnings

from absl.testing import absltest
import numpy as np

from . import utils


class UtilsTest(absltest.TestCase):

  def test_rvq_quantization(self):
    embeddings = np.random.randn(2, 128).astype(np.float32)
    codebooks = np.random.randn(10, 1024, 128).astype(np.float32)
    tokens, residual = utils.rvq_quantization(embeddings, codebooks)
    self.assertIsInstance(tokens, np.ndarray)
    self.assertEqual(tokens.shape, (2, 10))
    self.assertEqual(tokens.dtype, np.int32)
    self.assertEqual(residual.shape, (2, 128))
    self.assertEqual(residual.dtype, np.float32)

    # (4, 2)
    residual = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    # (3, 4, 2)
    codebooks = np.array([
        [[0.000, 0.000], [0.000, 0.500], [0.500, 0.000], [0.500, 0.500]],
        [[0.250, 0.250], [0.000, 0.000], [0.000, 0.250], [0.250, 0.000]],
        [[0.125, 0.000], [0.125, 0.125], [0.000, 0.000], [0.000, 0.125]],
    ])
    tokens, residual = utils.rvq_quantization(residual, codebooks)
    self.assertEqual(
        tokens.tolist(), [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
    )
    self.assertEqual(
        residual.tolist(),
        [
            [0.000, 0.000],
            [0.000, 0.125],
            [0.125, 0.000],
            [0.125, 0.125],
        ],
    )

  def test_rvq_dequantization(self):
    tokens = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [3, 0, 1],
    ])
    codebooks = np.array([
        [[0.000, 0.000], [0.000, 0.500], [0.500, 0.000], [0.500, 0.500]],
        [[0.250, 0.250], [0.000, 0.000], [0.000, 0.250], [0.250, 0.000]],
        [[0.125, 0.000], [0.125, 0.125], [0.000, 0.000], [0.000, 0.125]],
    ])
    for depth, v in zip([3, 2, 1, 0], [0.875, 0.750, 0.500, 0.000]):
      embeddings = utils.rvq_dequantization(tokens[:, :depth], codebooks)
      self.assertEqual(
          embeddings.tolist(),
          [
              [0.0, 0.0],
              [0.0, v],
              [v, 0.0],
              [v, v],
          ],
      )
    self.assertEqual(
        utils.rvq_dequantization(np.array([[0]]), codebooks).tolist(),
        [[0.0, 0.0]],
    )
    with self.assertRaises(IndexError):
      utils.rvq_dequantization(np.array([[-1]]), codebooks)
    self.assertEqual(
        utils.rvq_dequantization(np.array([[3]]), codebooks).tolist(),
        [[0.5, 0.5]],
    )
    with self.assertRaises(IndexError):
      utils.rvq_dequantization(np.array([[4]]), codebooks)

  def test_rvq_llm_conversion(self):
    cases = [
        # RVQ tokens, RVQ codebook size, offset, corresponding LLM tokens
        ([0, 0, 0, 0], 1024, 0, [0, 1024, 2048, 3072]),
        ([0, 0, 0, 0], 2, 4, [4, 6, 8, 10]),
        ([0, 1, 2, 3], 1024, 0, [0, 1025, 2050, 3075]),
        ([6, 4, 2, 0], 8, 0, [6, 12, 18, 24]),
        (
            [[0, 0, 0, 0], [0, 1, 2, 3]],
            1024,
            0,
            [[0, 1024, 2048, 3072], [0, 1025, 2050, 3075]],
        ),
        ([3], 4, 0, [3]),
        ([], 1024, 0, []),
    ]
    for rvq_tokens, rvq_codebook_size, offset, llm_tokens in cases:
      rvq_tokens = np.array(rvq_tokens, dtype=np.int32)
      llm_tokens = np.array(llm_tokens, dtype=np.int32)
      self.assertEqual(
          utils.rvq_to_llm(rvq_tokens, rvq_codebook_size, offset).tolist(),
          llm_tokens.tolist(),
      )
      self.assertEqual(
          utils.llm_to_rvq(llm_tokens, rvq_codebook_size, offset).tolist(),
          rvq_tokens.tolist(),
      )

      # Should fail if wrong input is passed.
      if not np.array_equal(rvq_tokens, llm_tokens):
        with self.assertRaises(IndexError):
          utils.rvq_to_llm(llm_tokens, rvq_codebook_size)
        with self.assertRaises(IndexError):
          utils.llm_to_rvq(rvq_tokens, rvq_codebook_size)

    # Basic failure cases
    with self.assertRaises(ValueError):
      utils.rvq_to_llm(np.array(0, dtype=np.int32), 1024)
    with self.assertRaises(IndexError):
      utils.rvq_to_llm(np.array([2]), 2)
    with self.assertRaises(IndexError):
      utils.rvq_to_llm(np.array([-1]), 2)
    with self.assertRaises(ValueError):
      utils.llm_to_rvq(np.array(0, dtype=np.int32), 1024)
    for case in [[3, 0], [3], [0, 0], [-1]]:
      with warnings.catch_warnings(record=True) as w:
        utils.llm_to_rvq(np.array(case), 2, safe=False)
        self.assertLen(w, 1)
        self.assertTrue(issubclass(w[-1].category, UserWarning))
      with self.assertRaises(IndexError):
        utils.llm_to_rvq(np.array(case), 2)


if __name__ == "__main__":
  absltest.main()
