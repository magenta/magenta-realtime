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

"""Unit tests for GPTQ quantization (magenta_rt.mlx.gptq).

Tests bit-level parity between gptq_quantize_weight(W, H=I) and mx.quantize(W).
With H=I (identity Hessian), GPTQ performs no error compensation, so the output
should be identical to standard nearest-rounding quantization.

Usage:
  pytest tests/test_gptq.py -v
"""

import unittest

import mlx.core as mx

from magenta_rt.mlx.gptq import gptq_quantize_weight, _pack_int4


class TestPackInt4(unittest.TestCase):
  """Test int4 packing/unpacking roundtrip."""

  def test_pack_known_values(self):
    """Pack known int4 values and verify uint32 output."""
    # 8 values [0..7] packed into one uint32, LSB-first
    Q = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    packed = _pack_int4(Q, bits=4)
    mx.eval(packed)
    # Expected: 0x76543210
    self.assertEqual(packed[0, 0].item(), 0x76543210)

  def test_pack_unpack_roundtrip(self):
    """Pack random int4 values and verify roundtrip."""
    mx.random.seed(99)
    Q = mx.random.randint(0, 16, shape=(32, 64)).astype(mx.int32)
    packed = _pack_int4(Q, bits=4)
    mx.eval(packed)

    # Unpack
    elems_per_int = 8
    Q_unpacked = mx.zeros_like(Q)
    for k in range(elems_per_int):
      nibbles = ((packed >> (k * 4)) & 0xF).astype(mx.int32)
      for p in range(nibbles.shape[1]):
        col = k + p * elems_per_int
        if col < Q.shape[1]:
          Q_unpacked[:, col] = nibbles[:, p]
    mx.eval(Q_unpacked)
    self.assertTrue(mx.array_equal(Q, Q_unpacked).item())


class TestGPTQIdentityParity(unittest.TestCase):
  """gptq_quantize_weight(W, H=I) must be bit-identical to mx.quantize(W).

  This is the fundamental sanity check: with no error compensation (H=I),
  GPTQ should reproduce the exact same packed weights, scales, and biases
  as MLX's native quantization.
  """

  def _assert_parity(self, rows, cols, group_size=32, seed=42):
    """Helper: verify bit-identical output for a given matrix size."""
    mx.random.seed(seed)
    W = mx.random.normal((rows, cols)).astype(mx.bfloat16)
    H = mx.eye(cols)

    packed_gptq, scales_gptq, biases_gptq = gptq_quantize_weight(
        W, H, bits=4, group_size=group_size)
    packed_ref, scales_ref, biases_ref = mx.quantize(
        W, group_size=group_size, bits=4)
    mx.eval(packed_gptq, packed_ref,
            scales_gptq, scales_ref,
            biases_gptq, biases_ref)

    self.assertTrue(
        mx.array_equal(scales_gptq, scales_ref).item(),
        f'Scales mismatch for [{rows}, {cols}] (group_size={group_size})')
    self.assertTrue(
        mx.array_equal(biases_gptq, biases_ref).item(),
        f'Biases mismatch for [{rows}, {cols}] (group_size={group_size})')
    self.assertTrue(
        mx.array_equal(packed_gptq, packed_ref).item(),
        f'Packed weights mismatch for [{rows}, {cols}] (group_size={group_size})')

  def test_small_square(self):
    """[128, 128] matrix with group_size=32."""
    self._assert_parity(128, 128, group_size=32)

  def test_large_square(self):
    """[4096, 4096] matrix with group_size=32 — matches real model layer size."""
    self._assert_parity(4096, 4096, group_size=32)

  def test_rectangular_wide(self):
    """[1024, 4096] — typical FFN up-projection shape."""
    self._assert_parity(1024, 4096, group_size=32)

  def test_rectangular_tall(self):
    """[4096, 1024] — typical FFN down-projection shape."""
    self._assert_parity(4096, 1024, group_size=32)

  def test_group_size_64(self):
    """Verify parity with group_size=64."""
    self._assert_parity(256, 256, group_size=64)

  def test_multiple_seeds(self):
    """Verify parity across different random seeds to catch edge cases."""
    for seed in [0, 1, 7, 13, 42, 99, 123, 256]:
      with self.subTest(seed=seed):
        self._assert_parity(64, 128, group_size=32, seed=seed)


if __name__ == '__main__':
  unittest.main(verbosity=2)
