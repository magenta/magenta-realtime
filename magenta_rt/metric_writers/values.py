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
# clu/values.py, with minor modifications by David Braun for magenta-rt.

"""Typed values a metric may produce, dispatched by ``write_values``.

A ``Metric.compute()`` may return a plain scalar/array or one of these typed
wrappers, which ``write_values`` routes to the matching ``MetricWriter`` method.
Port of ``clu.values`` (no external dependencies beyond numpy/jax).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np

ArrayType = Union[np.ndarray, jnp.ndarray]
ScalarType = Union[int, float, np.number, np.ndarray, jnp.ndarray]


@runtime_checkable
class Value(Protocol):
    """A metric computation return value carrying a writer-routable type."""

    value: Any


@dataclasses.dataclass
class Summary(Value):
    value: ArrayType
    metadata: Any


@dataclasses.dataclass
class Scalar(Value):
    value: ScalarType


@dataclasses.dataclass
class Image(Value):
    """Image(s) of shape ``[N, H, W, C]`` or ``[H, W, C]`` (C is 1 or 3)."""

    value: ArrayType


@dataclasses.dataclass
class Audio(Value):
    """Audio of shape ``[N, T, C]`` with values in ``[-1, +1]``."""

    value: ArrayType
    sample_rate: int


@dataclasses.dataclass
class Text(Value):
    value: str


@dataclasses.dataclass
class Histogram(Value):
    value: ArrayType  # array of counts
    num_buckets: int


@dataclasses.dataclass
class HyperParam(Value):
    value: Any
