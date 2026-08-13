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
# clu/metric_writers/interface.py, with modifications by David Braun for
# magenta-rt: dropped the point-cloud surface and modernized typing.

"""The ``MetricWriter`` interface.

A ``MetricWriter`` unifies reporting of model metrics across logging back ends
(console logging, TensorBoard, Weights & Biases, ...). Compose writers with
``MultiWriter`` to fan out, or ``AsyncWriter`` to write off the main thread.

This is a TF-free port of ``clu.metric_writers.interface``.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, Union

import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]
Scalar = Union[int, float, np.number, np.ndarray, jnp.ndarray]


class MetricWriter(abc.ABC):
    """Interface for writing metrics to a logging back end."""

    @abc.abstractmethod
    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Array],
        metadata: Mapping[str, Any] | None = None,
    ):
        """Saves an arbitrary tensor summary.

        Useful for custom plugins or constructing a summary directly.

        Args:
            step: Step at which the values occurred.
            values: Mapping from tensor key to tensor.
            metadata: Optional summary metadata.
        """

    @abc.abstractmethod
    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Writes scalar values for the step.

        Args:
            step: Step at which the scalar values occurred.
            scalars: Mapping from metric name to value.
        """

    @abc.abstractmethod
    def write_images(self, step: int, images: Mapping[str, Array]):
        """Writes images for the step.

        Args:
            step: Step at which the images occurred.
            images: Mapping from key to images of shape ``[N, H, W, C]`` or
                ``[H, W, C]`` (C is 1 or 3).
        """

    @abc.abstractmethod
    def write_videos(self, step: int, videos: Mapping[str, Array]):
        """Writes videos for the step.

        Args:
            step: Step at which the videos occurred.
            videos: Mapping from key to videos of shape ``[N, T, H, W, C]`` or
                ``[T, H, W, C]``.
        """

    @abc.abstractmethod
    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        """Writes audios for the step.

        Args:
            step: Step at which the audios occurred.
            audios: Mapping from key to audios of shape ``[N, T, C]`` with
                floating-point values in ``[-1, +1]``.
            sample_rate: Sample rate for the audios.
        """

    @abc.abstractmethod
    def write_texts(self, step: int, texts: Mapping[str, str]):
        """Writes text snippets for the step."""

    @abc.abstractmethod
    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, Array],
        num_buckets: Mapping[str, int] | None = None,
    ):
        """Writes histograms for the step.

        Args:
            step: Step at which the arrays were generated.
            arrays: Mapping from name to arrays to summarize.
            num_buckets: Optional mapping from name to bucket count.
        """

    @abc.abstractmethod
    def write_hparams(self, hparams: Mapping[str, Any]):
        """Writes hyperparameters. Do not call twice."""

    @abc.abstractmethod
    def flush(self):
        """Tells the writer to flush any cached values."""

    @abc.abstractmethod
    def close(self):
        """Flushes and closes the writer. Use after this is undefined."""
