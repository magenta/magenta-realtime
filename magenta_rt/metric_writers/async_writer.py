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
# clu/metric_writers/async_writer.py, with modifications by David Braun for
# magenta-rt: reimplemented on concurrent.futures instead of clu.asynclib + wrapt.

"""Writers that perform writes on a background thread.

- Write order is preserved (a single worker thread runs writes sequentially).
- Call ``flush()`` (or use ``ensure_flushes()``) to guarantee all writes land.
- An exception raised in the background thread is re-raised on the main thread
  on the next ``write_*``/``flush`` call.

This is a dependency-light port of ``clu.metric_writers.async_writer`` that uses
``concurrent.futures`` instead of ``clu.asynclib`` + ``wrapt``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from concurrent import futures
from typing import Any

from .interface import Array, MetricWriter, Scalar
from .multi_writer import MultiWriter


class AsyncWriter(MetricWriter):
    """Runs another writer's operations on a background thread.

    With the default ``num_workers=1`` all writes run in order on a single
    worker. Using more workers is only safe if the wrapped writer is
    thread-safe and tolerates out-of-order events.
    """

    def __init__(self, writer: MetricWriter, *, num_workers: int | None = 1):
        super().__init__()
        self._writer = writer
        self._pool = futures.ThreadPoolExecutor(
            max_workers=num_workers or 1, thread_name_prefix="AsyncWriter"
        )
        self._futures: list[futures.Future] = []

    def _submit(self, fn, *args, **kwargs) -> None:
        # Surface any prior background error before enqueueing more work.
        self._reraise_pending()
        self._futures.append(self._pool.submit(fn, *args, **kwargs))

    def _reraise_pending(self) -> None:
        """Re-raises the first background exception, dropping completed futures."""
        still_pending = []
        for fut in self._futures:
            if fut.done():
                fut.result()  # raises if the background call failed
            else:
                still_pending.append(fut)
        self._futures = still_pending

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        self._submit(self._writer.write_summaries, step, values, metadata)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        self._submit(self._writer.write_scalars, step, scalars)

    def write_images(self, step: int, images: Mapping[str, Array]):
        self._submit(self._writer.write_images, step, images)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        self._submit(self._writer.write_videos, step, videos)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        self._submit(self._writer.write_audios, step, audios, sample_rate=sample_rate)

    def write_texts(self, step: int, texts: Mapping[str, str]):
        self._submit(self._writer.write_texts, step, texts)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        self._submit(self._writer.write_histograms, step, arrays, num_buckets)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._submit(self._writer.write_hparams, hparams)

    def flush(self):
        try:
            # Wait for all enqueued writes, raising the first failure (if any).
            for fut in self._futures:
                fut.result()
            self._futures = []
        finally:
            self._writer.flush()

    def close(self):
        try:
            self.flush()
        finally:
            self._pool.shutdown(wait=True)
            self._writer.close()


class AsyncMultiWriter(MultiWriter):
    """Fans out to multiple writers, each running on its own background thread."""

    def __init__(self, writers: Sequence[MetricWriter], *, num_workers: int | None = 1):
        super().__init__([AsyncWriter(w, num_workers=num_workers) for w in writers])


@contextlib.contextmanager
def ensure_flushes(*writers: MetricWriter):
    """Context manager that flushes one or more writers on exit."""
    try:
        yield writers[0]
    finally:
        for writer in writers:
            writer.flush()
