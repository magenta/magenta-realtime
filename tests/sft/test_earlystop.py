# Copyright 2026 Google LLC
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

"""Unit tests for the SFT early-stopping decision (patience + min-delta)."""

from __future__ import annotations

from magenta_rt.sft import EarlyStopper


def test_improvement_resets_patience_and_tracks_best():
    s = EarlyStopper(min_delta=1e-4, patience=3)
    assert s.best == float("inf") and s.patience_left == 3
    assert s.update(1.0) is False and s.best == 1.0 and s.patience_left == 3
    assert s.update(0.5) is False and s.best == 0.5 and s.patience_left == 3


def test_stops_after_patience_without_improvement():
    s = EarlyStopper(min_delta=1e-4, patience=3)
    s.update(1.0)  # best=1.0, left=3
    assert s.update(1.0) is False and s.patience_left == 2  # no improvement
    assert s.update(1.0) is False and s.patience_left == 1
    assert s.update(1.0) is True and s.patience_left == 0  # patience exhausted


def test_improvement_resets_the_counter_mid_run():
    s = EarlyStopper(min_delta=1e-4, patience=2)
    s.update(1.0)  # best=1.0, left=2
    assert s.update(1.0) is False and s.patience_left == 1  # no improvement
    assert s.update(0.1) is False and s.patience_left == 2  # improvement -> reset
    assert s.update(0.1) is False and s.patience_left == 1
    assert s.update(0.1) is True  # exhausted again


def test_min_delta_is_required_to_count_as_improvement():
    # A drop smaller than min_delta does NOT reset patience.
    s = EarlyStopper(min_delta=0.1, patience=1)
    s.update(1.0)  # best=1.0, left=1
    assert s.update(0.95) is True  # 0.95 + 0.1 = 1.05, not < 1.0 -> stop
    # A drop larger than min_delta IS an improvement.
    s2 = EarlyStopper(min_delta=0.1, patience=1)
    s2.update(1.0)
    assert s2.update(0.85) is False and s2.best == 0.85  # 0.85 + 0.1 = 0.95 < 1.0
