# Copyright 2026 The CLU Authors.
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

# Adapted from CommonLoopUtils (https://github.com/google/CommonLoopUtils),
# clu/metric_writers/logging_writer.py, with modifications by David Braun for
# magenta-rt: dropped the point-cloud surface.

"""A ``MetricWriter`` that writes everything to the absl INFO log."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from absl import logging

from .interface import Array, MetricWriter, Scalar


class LoggingWriter(MetricWriter):
    """Writes all values to the INFO log."""

    def __init__(self, collection: str | None = None):
        self._collection_str = f" collection={collection}" if collection else ""

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        logging.info(
            "[%d]%s Got raw tensors: %s.",
            step,
            self._collection_str,
            {k: v.shape for k, v in values.items()},
        )

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        values = [
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(scalars.items())
        ]
        logging.info("[%d]%s %s", step, self._collection_str, ", ".join(values))

    def write_images(self, step: int, images: Mapping[str, Array]):
        logging.info(
            "[%d]%s Got images: %s.",
            step,
            self._collection_str,
            {k: v.shape for k, v in images.items()},
        )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        logging.info(
            "[%d]%s Got videos: %s.",
            step,
            self._collection_str,
            {k: v.shape for k, v in videos.items()},
        )

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        logging.info(
            "[%d]%s Got audios: %s.",
            step,
            self._collection_str,
            {k: v.shape for k, v in audios.items()},
        )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        logging.info("[%d]%s Got texts: %s.", step, self._collection_str, texts)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        num_buckets = num_buckets or {}
        for key, value in arrays.items():
            histo, bins = _compute_histogram(
                np.asarray(value), num_buckets=num_buckets.get(key)
            )
            if histo is not None:
                logging.info(
                    "[%d]%s Histogram for %r = {%s}",
                    step,
                    self._collection_str,
                    key,
                    _histogram_as_string(histo, bins),
                )

    def write_hparams(self, hparams: Mapping[str, Any]):
        logging.info("[Hyperparameters]%s %s", self._collection_str, hparams)

    def flush(self):
        logging.flush()

    def close(self):
        self.flush()


def _compute_histogram(array: np.ndarray, num_buckets: int | None = None):
    """Computes ``(counts, bin_edges)`` for ``array`` (matching TensorBoard)."""
    num_buckets = num_buckets or 30
    if num_buckets < 2:
        num_buckets = 2
    if array.size == 0:
        return None, None
    range_max = np.max(array)
    range_min = np.min(array)
    if np.isclose(range_max, range_min, rtol=1e-5, atol=1e-8):
        histo = np.asarray([array.size], dtype=np.int64)
        bins = np.asarray([range_max - 0.5, range_max + 0.5], dtype=np.float64)
    else:
        histo, bins = np.histogram(
            array, bins=num_buckets, range=(range_min, range_max)
        )
        bins = np.asarray(bins, dtype=np.float64)
    return histo, bins


def _histogram_as_string(histo: np.ndarray, bins: np.ndarray) -> str:
    items = [
        f"[{bins[i]:.3g}, {bins[i + 1]:.3g}): {count}"
        for i, count in enumerate(histo[:-1])
    ]
    items.append(f"[{bins[-2]:.3g}, {bins[-1]:.3g}]: {histo[-1]}")
    return ", ".join(items)
