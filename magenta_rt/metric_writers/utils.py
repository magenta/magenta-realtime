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
# clu/metric_writers/utils.py, with modifications by David Braun for magenta-rt:
# pathlib instead of etils, a pluggable default TensorBoard back end, and no
# absl FLAGS dependency.

"""Helpers for constructing writers and writing typed values."""

from __future__ import annotations

import collections
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import values
from .async_writer import AsyncMultiWriter
from .interface import MetricWriter
from .logging_writer import LoggingWriter
from .multi_writer import MultiWriter


def _is_scalar(value: Any) -> bool:
    if isinstance(value, values.Scalar) or isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return value.ndim == 0 or value.size <= 1
    return False


def write_values(
    writer: MetricWriter,
    step: int,
    metrics: Mapping[str, values.Value | values.ArrayType | values.ScalarType],
):
    """Writes a mapping of typed values, dispatching by value type.

    Each value may be a ``values.Value`` (``Scalar``, ``Image``, ``Audio``, ...)
    or a bare scalar/array (treated as a scalar). Values of the same type are
    batched into a single ``write_*`` call.
    """
    writes = collections.defaultdict(dict)
    histogram_num_buckets = collections.defaultdict(int)
    for k, v in metrics.items():
        if isinstance(v, values.Summary):
            key = (writer.write_summaries, frozenset({"metadata": v.metadata}.items()))
            writes[key][k] = v.value
        elif _is_scalar(v):
            value = v.value if isinstance(v, values.Scalar) else v
            writes[(writer.write_scalars, frozenset())][k] = value
        elif isinstance(v, values.Image):
            writes[(writer.write_images, frozenset())][k] = v.value
        elif isinstance(v, values.Text):
            writes[(writer.write_texts, frozenset())][k] = v.value
        elif isinstance(v, values.HyperParam):
            writes[(writer.write_hparams, frozenset())][k] = v.value
        elif isinstance(v, values.Histogram):
            writes[(writer.write_histograms, frozenset())][k] = v.value
            histogram_num_buckets[k] = v.num_buckets
        elif isinstance(v, values.Audio):
            key = (
                writer.write_audios,
                frozenset({"sample_rate": v.sample_rate}.items()),
            )
            writes[key][k] = v.value
        else:
            raise ValueError(f"Metric {k!r} has unsupported value: {v!r}")

    for (fn, extra_args), vals in writes.items():
        if fn == writer.write_histograms:
            writer.write_histograms(step, vals, num_buckets=histogram_num_buckets)
        else:
            fn(step, vals, **dict(extra_args))


def create_default_writer(
    logdir: str | os.PathLike | None = None,
    *,
    just_logging: bool = False,
    asynchronous: bool = True,
    collection: str | None = None,
) -> MultiWriter:
    """Creates the default writer: console logging plus (optionally) TensorBoard.

    Args:
        logdir: Directory for TensorBoard event files. If ``None``, only logging
            is written.
        just_logging: If ``True``, use only a ``LoggingWriter`` (e.g. on
            non-primary hosts).
        asynchronous: If ``True``, wrap writers so writes happen off the main
            thread.
        collection: Optional grouping label forwarded to the writers.

    Returns:
        A ``MultiWriter`` (or ``AsyncMultiWriter``) over the selected back ends.
    """
    if just_logging:
        if asynchronous:
            return AsyncMultiWriter([LoggingWriter(collection=collection)])
        return MultiWriter([LoggingWriter(collection=collection)])

    writers: list[MetricWriter] = [LoggingWriter(collection=collection)]
    if logdir is not None:
        # Import lazily so the TensorBoard back end (torch or TF) is only loaded
        # when a logdir-backed writer is actually requested.
        from .tensorboard_writer import create_summary_writer

        logdir = Path(logdir)
        if collection is not None:
            logdir /= collection
        writers.append(create_summary_writer(os.fspath(logdir)))

    if asynchronous:
        return AsyncMultiWriter(writers)
    return MultiWriter(writers)
