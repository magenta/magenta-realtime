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

"""Min-delta + patience early stopping on a validation metric.

Extracted from the SFT trainers so the patience/min-delta decision — exactly
the kind of off-by-one-prone logic worth a unit test — is testable in isolation
rather than buried in the train loop.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EarlyStopper:
  """Stops after ``patience`` validation rounds without improvement.

  Lower is better. A round *improves* when ``value + min_delta < best``; that
  resets the patience counter and records the new best. Otherwise the counter
  decrements, and :meth:`update` returns ``True`` (stop) once it hits zero.
  """

  min_delta: float = 1e-4
  patience: int = 5
  best: float = float("inf")
  patience_left: int = dataclasses.field(init=False)

  def __post_init__(self):
    self.patience_left = self.patience

  def update(self, value: float) -> bool:
    """Record a validation ``value``; return ``True`` if training should stop."""
    if value + self.min_delta < self.best:
      self.best = value
      self.patience_left = self.patience
      return False
    self.patience_left -= 1
    return self.patience_left <= 0
