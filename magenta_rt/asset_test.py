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

import pathlib
import tempfile

from absl.testing import absltest

from . import asset


class TestAsset(absltest.TestCase):

  def test_cache_dir(self):
    cache_dir = asset.get_cache_dir()
    self.assertTrue(cache_dir.exists())
    self.assertTrue(cache_dir.is_dir())
    with tempfile.TemporaryDirectory() as tmp_dir:
      asset.set_cache_dir(tmp_dir)
      self.assertEqual(asset.get_cache_dir(), pathlib.Path(tmp_dir))

  def test_get_path_gcp(self):
    self.assertEqual(
        asset.get_path_gcp("foo/bar/baz"), "gs://magenta-rt-public/foo/bar/baz"
    )

  def test_get_path_hf(self):
    self.assertEqual(
        asset.get_path_hf("foo/bar/baz"),
        "hf://google/magenta-realtime/foo/bar/baz",
    )


if __name__ == "__main__":
  absltest.main()
