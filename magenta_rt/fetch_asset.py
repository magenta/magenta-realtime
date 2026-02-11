# Copyright 2026 Google LLC
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

"""Binary to download an asset from GCP to cache dir.

Example usage:
  python -m magenta_rt.fetch_asset --asset=README.md
"""

from absl import app
from absl import flags
from absl import logging

from . import asset

_ASSET = flags.DEFINE_string(
    'asset',
    None,
    'Path to the asset to download.',
    required=True,
)
_SOURCE = flags.DEFINE_string(
    'source',
    'gcp',
    'Source to fetch the asset from.',
)
_IS_DIR = flags.DEFINE_bool(
    'is_dir',
    False,
    'Whether the asset is a directory.',
)


def main(unused_argv):
  is_archive = _ASSET.value.endswith('.tar')
  result_path = asset.fetch(
      _ASSET.value,
      is_dir=is_archive or _IS_DIR.value,
      extract_archive=is_archive,
      override_cache=True,
      source=_SOURCE.value,
  )
  logging.info('Fetched %s to %s', _ASSET.value, result_path)


if __name__ == '__main__':
  app.run(main)
