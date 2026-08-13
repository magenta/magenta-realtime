# Copyright 2025 The MT3 Authors.
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

"""Encode and decode events.

Ported from https://github.com/magenta/mt3 (event_codec.py).
"""

import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: str
    value: int


class Codec:
    """Encode and decode events.

    Useful for declaring what certain ranges of a vocabulary should be used for.
    This is intended to be used from Python before encoding or after decoding with
    GenericTokenVocabulary. This class is more lightweight and does not include
    things like EOS or UNK token handling.

    To ensure that 'shift' events are always the first block of the vocab and
    start at 0, that event type is required and specified separately.
    """

    def __init__(
        self,
        max_shift_steps: int,
        steps_per_second: float,
        event_ranges: List[EventRange],
    ):
        """Define Codec.

        Args:
            max_shift_steps: Maximum number of shift steps that can be encoded.
            steps_per_second: Shift steps will be interpreted as having a duration
                of 1 / steps_per_second.
            event_ranges: Other supported event types and their ranges.
        """
        self.steps_per_second = steps_per_second
        self._shift_range = EventRange(type="shift", min_value=0, max_value=max_shift_steps)
        self._event_ranges = [self._shift_range] + event_ranges
        # Ensure all event types have unique names.
        assert len(self._event_ranges) == len(set(er.type for er in self._event_ranges))

    @property
    def num_classes(self) -> int:
        return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

    def is_shift_event_index(self, index: int) -> bool:
        return (self._shift_range.min_value <= index) and (index <= self._shift_range.max_value)

    @property
    def max_shift_steps(self) -> int:
        return self._shift_range.max_value

    def encode_event(self, event: Event) -> int:
        """Encode an event to an index."""
        offset = 0
        for er in self._event_ranges:
            if event.type == er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"Event value {event.value} is not within valid range "
                        f"[{er.min_value}, {er.max_value}] for type {event.type}"
                    )
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event.type}")

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Return [min_id, max_id] for an event type."""
        offset = 0
        for er in self._event_ranges:
            if event_type == er.type:
                return offset, offset + (er.max_value - er.min_value)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event type: {event_type}")

    def decode_event_index(self, index: int) -> Event:
        """Decode an event index to an Event."""
        offset = 0
        for er in self._event_ranges:
            if offset <= index <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + index - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"Unknown event index: {index}")
