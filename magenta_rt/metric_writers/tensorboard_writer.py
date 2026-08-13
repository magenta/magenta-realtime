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
# clu/metric_writers/{torch_tensorboard_writer,summary_writer}.py, with
# modifications by David Braun for magenta-rt: a shared add_*-API base, plus
# tensorboardX (TF-free default), torch, and TensorFlow back ends with
# auto-selection.

"""TensorBoard ``MetricWriter`` back ends.

Three interchangeable back ends are provided:

- ``TensorboardXWriter`` — writes event files via ``tensorboardX`` (pure Python,
  **no TensorFlow, no torch**). This is the core default.
- ``TorchTensorboardWriter`` — writes via ``torch.utils.tensorboard`` (no TF, but
  requires torch).
- ``TFSummaryWriter`` — writes via ``tf.summary`` (requires TensorFlow).

Each imports its heavy dependency lazily inside ``__init__``, so importing this
module pulls in none of tensorboardX/torch/TensorFlow. Use
``create_summary_writer`` to pick a back end; the default ``"auto"`` prefers the
TF-free ``tensorboardX`` writer, then torch, then TensorFlow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from absl import logging

from .interface import Array, MetricWriter, Scalar


class _SummaryWriterBackend(MetricWriter):
    """Shared implementation over the ``add_*`` SummaryWriter API.

    Both ``tensorboardX.SummaryWriter`` and ``torch.utils.tensorboard.
    SummaryWriter`` expose the same ``add_scalar``/``add_image``/... API; this
    base implements ``MetricWriter`` against ``self._writer``. Subclasses only
    construct the concrete writer.
    """

    _writer: Any

    def write_summaries(self, step, values, metadata=None):
        logging.log_first_n(
            logging.WARNING,
            "%s does not support writing raw summaries.",
            1,
            type(self).__name__,
        )

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for key, value in scalars.items():
            self._writer.add_scalar(key, np.asarray(value).item(), global_step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self._writer.add_image(
                key, np.asarray(value), global_step=step, dataformats="HWC"
            )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        logging.log_first_n(
            logging.WARNING,
            "%s does not support writing videos.",
            1,
            type(self).__name__,
        )

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        # clu's contract passes a batch ``[N, frames, channels]``; tensorboardX's
        # ``add_audio`` wants a single 2-D ``[frames, channels]`` clip, so unbatch
        # (a multi-clip batch is written as ``key/0``, ``key/1``, …).
        for key, value in audios.items():
            value = np.asarray(value)
            clips = value if value.ndim == 3 else value[None]
            for i, clip in enumerate(clips):
                tag = key if len(clips) == 1 else f"{key}/{i}"
                self._writer.add_audio(
                    tag, np.asarray(clip), global_step=step, sample_rate=sample_rate
                )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        for key, value in texts.items():
            self._writer.add_text(key, value, global_step=step)

    def write_histograms(self, step, arrays, num_buckets=None):
        for tag, values in arrays.items():
            bins = None if num_buckets is None else num_buckets.get(tag)
            self._writer.add_histogram(
                tag, np.asarray(values), global_step=step, bins="auto", max_bins=bins
            )

    def write_hparams(self, hparams: Mapping[str, Any]):
        self._writer.add_hparams(dict(hparams), {})

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


class TensorboardXWriter(_SummaryWriterBackend):
    """Writes TensorBoard summaries via ``tensorboardX`` (no TF, no torch)."""

    def __init__(self, logdir: str):
        super().__init__()
        try:
            from tensorboardX import SummaryWriter
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "TensorboardXWriter requires tensorboardX. Install it, or use a "
                "different backend via create_summary_writer(backend=...)."
            ) from e
        self._writer = SummaryWriter(logdir=str(logdir))


class TorchTensorboardWriter(_SummaryWriterBackend):
    """Writes TensorBoard summaries via ``torch.utils.tensorboard`` (no TF)."""

    def __init__(self, logdir: str):
        super().__init__()
        try:
            from torch.utils import tensorboard
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "TorchTensorboardWriter requires torch. Install torch, or use "
                "create_summary_writer(backend='tensorboardx'/'tf')."
            ) from e
        self._writer = tensorboard.SummaryWriter(log_dir=str(logdir))


class TFSummaryWriter(MetricWriter):
    """Writes TensorBoard summaries via ``tf.summary`` (requires TensorFlow).

    TensorFlow is imported lazily here so it is never pulled in merely by
    importing ``magenta_rt.metric_writers``.
    """

    def __init__(self, logdir: str):
        super().__init__()
        import tensorflow as tf

        self._tf = tf
        self._writer = tf.summary.create_file_writer(str(logdir))

    def write_summaries(self, step, values, metadata=None):
        with self._writer.as_default():
            for key, value in values.items():
                md = None if metadata is None else metadata.get(key)
                self._tf.summary.write(key, value, step=step, metadata=md)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        with self._writer.as_default():
            for key, value in scalars.items():
                self._tf.summary.scalar(key, np.asarray(value).item(), step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        with self._writer.as_default():
            for key, value in images.items():
                value = np.asarray(value)
                if value.ndim == 3:
                    value = value[None]
                self._tf.summary.image(
                    key, value, step=step, max_outputs=value.shape[0]
                )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        logging.log_first_n(
            logging.WARNING, "TFSummaryWriter does not support writing videos.", 1
        )

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        with self._writer.as_default():
            for key, value in audios.items():
                self._tf.summary.audio(
                    key,
                    np.asarray(value),
                    sample_rate=sample_rate,
                    step=step,
                    max_outputs=np.asarray(value).shape[0],
                )

    def write_texts(self, step: int, texts: Mapping[str, str]):
        with self._writer.as_default():
            for key, value in texts.items():
                self._tf.summary.text(key, value, step=step)

    def write_histograms(self, step, arrays, num_buckets=None):
        num_buckets = num_buckets or {}
        with self._writer.as_default():
            for key, value in arrays.items():
                self._tf.summary.histogram(
                    key, np.asarray(value), step=step, buckets=num_buckets.get(key)
                )

    def write_hparams(self, hparams: Mapping[str, Any]):
        from tensorboard.plugins.hparams import api as hp

        with self._writer.as_default():
            hp.hparams(dict(hparams))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


# Back ends tried, in order, by ``create_summary_writer(backend="auto")``.
_AUTO_BACKENDS = (
    ("tensorboardx", TensorboardXWriter),
    ("torch", TorchTensorboardWriter),
    ("tf", TFSummaryWriter),
)
_BACKENDS = {name: cls for name, cls in _AUTO_BACKENDS}


def create_summary_writer(logdir: str, *, backend: str = "auto") -> MetricWriter:
    """Creates a TensorBoard writer for ``logdir``.

    Args:
        logdir: Event-file directory.
        backend: ``"tensorboardx"`` (TF-free, core default), ``"torch"``
            (TF-free, needs torch), ``"tf"`` (TensorFlow), or ``"auto"`` (try
            tensorboardX, then torch, then TensorFlow).
    """
    if backend in _BACKENDS:
        return _BACKENDS[backend](logdir)
    if backend == "auto":
        last_error: ImportError | None = None
        for _, cls in _AUTO_BACKENDS:
            try:
                return cls(logdir)
            except ImportError as e:
                last_error = e
        raise ImportError(
            "No TensorBoard backend available. Install one of: tensorboardX "
            "(recommended, TF-free), torch, or tensorflow."
        ) from last_error
    raise ValueError(f"Unknown tensorboard backend {backend!r}")
