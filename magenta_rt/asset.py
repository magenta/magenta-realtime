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

"""Utilities for fetching assets from different sources."""

from concurrent import futures
import functools
import os
import pathlib
import shutil
from typing import NamedTuple, Optional

from absl import logging
from google.auth import credentials
from google.cloud import storage
import requests
import tqdm

from . import path as mrt_path


class BucketInfo(NamedTuple):
  name: str
  is_public: bool


GCP_BUCKET = BucketInfo('magenta-rt-public', True)
HF_REPO_NAME = 'google/magenta-realtime'
DEFAULT_SOURCE = os.environ.get('MAGENTA_RT_DEFAULT_ASSET_SOURCE', 'hf')

if 'MAGENTA_RT_CACHE_DIR' in os.environ:
  _CACHE_DIR = pathlib.Path(os.environ['MAGENTA_RT_CACHE_DIR'])
else:
  _CACHE_DIR = mrt_path.DEFAULT_CACHE_DIR
_HF_TOKEN = os.environ.get('HF_TOKEN', None)


def get_cache_dir() -> pathlib.Path:
  """Gets the asset cache directory for Magenta RT."""
  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  return _CACHE_DIR


def set_cache_dir(cache_dir: pathlib.Path | str):
  """Manually sets the asset cache directory for Magenta RT."""
  global _CACHE_DIR
  if isinstance(cache_dir, str):
    cache_dir = pathlib.Path(cache_dir)
  _CACHE_DIR = cache_dir


@functools.cache
def _get_bucket(bucket_info: BucketInfo) -> storage.Bucket:
  """Returns a storage client for the given bucket."""
  if bucket_info.is_public:
    # Retrieve anonymously to avoid authentication issues.
    client_kwargs = {
        'project': None,
        'credentials': credentials.AnonymousCredentials(),
    }
  else:
    # Inherit the default environment credentials.
    client_kwargs = {}
  storage_client = storage.Client(**client_kwargs)
  return storage_client.bucket(bucket_info.name)


def _fetch_single_gcp(blob: storage.Blob, output_path: pathlib.Path) -> None:
  """Downloads a blob to a local file."""
  logging.info(
      'Downloading gs://%s/%s to %s',
      blob.bucket.name,
      blob.name,
      output_path,
  )
  try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(output_path)
  except Exception as e:  # pylint: disable=broad-except
    output_path.unlink(missing_ok=True)
    logging.exception('Failed to download %s', blob.name)
    raise e
  assert output_path.exists()


def _iter_fetches_gcp(
    cache_dir: pathlib.Path,
    asset_relative_path: pathlib.PurePath,
    is_dir: bool,
    bucket_info: BucketInfo,
):
  """Fetches an asset (file or directory) from GCP bucket."""
  bucket = _get_bucket(bucket_info)
  if is_dir:
    blobs = bucket.list_blobs(prefix=str(asset_relative_path))
  else:
    blobs = [bucket.blob(str(asset_relative_path))]
  for blob in blobs:
    yield (
        _fetch_single_gcp,
        (blob, cache_dir / pathlib.Path(blob.name)),
    )


def get_path_gcp(path: str, bucket_info: Optional[BucketInfo] = None) -> str:
  """Returns the GCP path for a given asset."""
  if bucket_info is None:
    bucket_info = GCP_BUCKET
  return f'gs://{bucket_info.name}/{path}'


def _fetch_single_hf(
    repo_name: str,
    repo_path: pathlib.PurePath,
    cache_dir: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
  """Downloads a blob to a local file."""
  import huggingface_hub  # pylint: disable=g-import-not-at-top

  logging.info(
      'Downloading %s:%s to %s',
      repo_name,
      repo_path,
      output_path,
  )
  try:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    huggingface_hub.hf_hub_download(
        repo_id=repo_name,
        filename=str(repo_path),
        local_dir=str(cache_dir),
        token=_HF_TOKEN,
    )
  except requests.exceptions.HTTPError as e:
    output_path.unlink(missing_ok=True)
    if e.response.status_code == 429:
      logging.exception(
          'Please authenticate with HuggingFace to avoid rate limits. Run'
          ' `huggingface-cli login` or set $HF_TOKEN.'
      )
    raise e
  except Exception as e:  # pylint: disable=broad-except
    output_path.unlink(missing_ok=True)
    logging.exception('Failed to download %s', repo_path)
    raise e
  assert output_path.exists()


def _iter_fetches_hf(
    cache_dir: pathlib.Path,
    asset_relative_path: pathlib.PurePath,
    is_dir: bool,
    repo_name: str,
):
  """Fetches an asset (file or directory) from HuggingFace repo."""
  import huggingface_hub  # pylint: disable=g-import-not-at-top
  import huggingface_hub.utils  # pylint: disable=g-import-not-at-top

  huggingface_hub.utils.disable_progress_bars()

  asset_cache_path = cache_dir / asset_relative_path
  if is_dir:
    repo_path = pathlib.PurePath(repo_name) / asset_relative_path
    fs = huggingface_hub.HfFileSystem(token=_HF_TOKEN)
    for entry in fs.find(
        get_path_hf(str(asset_relative_path), repo_name), withdirs=False
    ):
      entry = pathlib.PurePath(entry).relative_to(repo_path)
      yield (
          _fetch_single_hf,
          (
              repo_name,
              asset_relative_path / entry,
              cache_dir,
              asset_cache_path / entry,
          ),
      )
  else:
    yield (
        _fetch_single_hf,
        (repo_name, asset_relative_path, cache_dir, asset_cache_path),
    )


def get_path_hf(path: str, repo_name: Optional[str] = None) -> str:
  """Returns the HF path for a given asset."""
  if repo_name is None:
    repo_name = HF_REPO_NAME
  return f'hf://{repo_name}/{path}'


def fetch(
    path: str,
    *,
    is_dir: bool = False,
    extract_archive: bool = False,
    override_cache: bool = False,
    skip_cache: bool = False,
    source: Optional[str] = None,
    bucket_info: Optional[BucketInfo] = None,
    hf_repo_name: Optional[str] = None,
    parallelism: int = 1,
) -> str:
  """Fetches a file from GCP, or from the cache if it was previously fetched."""
  # Set defaults
  if source is None:
    source = DEFAULT_SOURCE
  if bucket_info is None:
    bucket_info = GCP_BUCKET
  if hf_repo_name is None:
    hf_repo_name = HF_REPO_NAME

  # Skip cache entirely and return a direct fetch path.
  if skip_cache:
    if extract_archive:
      raise ValueError('extract_archive not supported when skip_cache is True.')
    if source == 'gcp':
      direct_path = get_path_gcp(path, bucket_info)
    else:
      raise ValueError(f'Direct fetching unsupported for {source}')
    logging.info('Skipping cache, loading from %s', direct_path)
    return direct_path

  # Local cache directories
  assets_cache_dir = get_cache_dir() / 'assets'
  asset_relative_path = pathlib.PurePath(path)
  asset_cache_path = assets_cache_dir / asset_relative_path
  if extract_archive:
    if asset_relative_path.suffix != '.tar':
      raise ValueError(
          f'Path {asset_relative_path} does not have a .tar extension, but'
          ' extract_archive is True.'
      )
    if not is_dir:
      raise NotImplementedError('Archives are only supported for directories.')
    asset_cache_path = asset_cache_path.parent / asset_relative_path.stem

  # Remove the asset from the cache if override is requested.
  if override_cache:
    if asset_cache_path.is_dir():
      shutil.rmtree(asset_cache_path)
    else:
      asset_cache_path.unlink(missing_ok=True)

  # Check if the asset is already in the cache.
  if not asset_cache_path.exists():
    # Download the asset from the source.
    if source == 'gcp':
      iter_fn = functools.partial(_iter_fetches_gcp, bucket_info=bucket_info)
    elif source == 'hf':
      iter_fn = functools.partial(_iter_fetches_hf, repo_name=hf_repo_name)
    else:
      raise ValueError(f'Unsupported source: {source}')
    futures_list = []
    with futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
      for fn, args in iter_fn(
          assets_cache_dir,
          asset_relative_path,
          False if extract_archive else is_dir,
      ):
        futures_list.append(executor.submit(fn, *args))
      if not futures_list:
        raise AssertionError(f'Asset not found: {asset_relative_path}')
      for future in tqdm.tqdm(
          futures.as_completed(futures_list),
          total=len(futures_list),
          desc=f'Downloading from {source}: {asset_relative_path}',
      ):
        future.result()

    # Extract the archive
    if extract_archive:
      asset_archive_path = assets_cache_dir / asset_relative_path
      assert asset_archive_path.exists()
      assert not asset_cache_path.exists()
      shutil.unpack_archive(asset_archive_path, extract_dir=asset_cache_path)
      assert asset_cache_path.is_dir()
      asset_archive_path.unlink()
  else:
    logging.info('Using cached %s', asset_cache_path)
  assert asset_cache_path.exists()

  return str(asset_cache_path)
