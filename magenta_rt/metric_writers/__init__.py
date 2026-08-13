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
# clu/metric_writers/__init__.py, with modifications by David Braun for
# magenta-rt: lazy back-end exports so importing the package pulls in no
# TensorBoard back end (tensorboardX / torch / TensorFlow).

"""TensorFlow-free metric writers — a drop-in replacement for ``clu.metric_writers``.

Exposes the same public surface magenta-rt relies on (``MetricWriter``,
``MultiWriter``, ``AsyncMultiWriter``, ``LoggingWriter``, ``ensure_flushes``,
``create_default_writer``) plus interchangeable TensorBoard back ends
(``TensorboardXWriter`` — TF-free, core default; ``TorchTensorboardWriter`` —
TF-free, needs torch; ``TFSummaryWriter`` — TensorFlow) in place of clu's
TensorFlow ``SummaryWriter``. Importing this package pulls in none of
tensorboardX/torch/TensorFlow; a back end is loaded lazily on first use,
preferring the TF-free ``tensorboardX`` writer.
"""

from . import values
from .async_writer import AsyncMultiWriter, AsyncWriter, ensure_flushes
from .interface import MetricWriter
from .logging_writer import LoggingWriter
from .multi_writer import MultiWriter
from .utils import create_default_writer, write_values

__all__ = [
    "MetricWriter",
    "MultiWriter",
    "AsyncWriter",
    "AsyncMultiWriter",
    "LoggingWriter",
    "ensure_flushes",
    "create_default_writer",
    "write_values",
    "values",
    "TensorboardXWriter",
    "TorchTensorboardWriter",
    "TFSummaryWriter",
    "create_summary_writer",
]

# Names served lazily so the TensorBoard back ends (tensorboardX / torch /
# TensorFlow) are not imported merely by importing this package.
_LAZY = {
    "TensorboardXWriter",
    "TorchTensorboardWriter",
    "TFSummaryWriter",
    "create_summary_writer",
}


def __getattr__(name: str):
    if name in _LAZY:
        from . import tensorboard_writer

        return getattr(tensorboard_writer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
