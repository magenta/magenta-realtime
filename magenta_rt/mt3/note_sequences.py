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

"""Helper functions that operate on note sequences.

Ported from https://github.com/magenta/mt3 (note_sequences.py), with the
``note_seq.NoteSequence`` protobuf replaced by lightweight dataclasses and
MIDI export via ``pretty_midi``.
"""

import dataclasses
from typing import List, MutableMapping, MutableSet, Optional, Sequence, Tuple

from . import event_codec
from . import run_length_encoding
from . import vocabularies

DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01

# Quantization can result in zero-length notes; enforce a minimum duration.
MIN_NOTE_DURATION = 0.01


@dataclasses.dataclass
class Note:
    """A single note, mirroring the fields of note_seq.NoteSequence.Note."""

    start_time: float
    end_time: float
    pitch: int
    velocity: int
    program: int = 0
    is_drum: bool = False
    instrument: int = 0


@dataclasses.dataclass
class NoteSequence:
    """A sequence of notes, mirroring note_seq.NoteSequence."""

    notes: List[Note] = dataclasses.field(default_factory=list)
    total_time: float = 0.0

    def add_note(self, **kwargs) -> Note:
        note = Note(**kwargs)
        self.notes.append(note)
        return note

    def to_pretty_midi(self):
        """Convert to a ``pretty_midi.PrettyMIDI`` object."""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        instruments = {}
        for note in self.notes:
            key = (note.instrument, note.program, note.is_drum)
            if key not in instruments:
                instruments[key] = pretty_midi.Instrument(
                    program=note.program, is_drum=note.is_drum, name=f"instrument {note.instrument}"
                )
            instruments[key].notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_time,
                    end=note.end_time,
                )
            )
        pm.instruments.extend(instruments[k] for k in sorted(instruments.keys()))
        return pm

    def write_midi(self, path: str) -> None:
        """Write the note sequence to a MIDI file."""
        self.to_pretty_midi().write(path)


def assign_instruments(ns: NoteSequence) -> None:
    """Assign instrument numbers to notes; modifies NoteSequence in place."""
    program_instruments = {}
    for note in ns.notes:
        if note.program not in program_instruments and not note.is_drum:
            num_instruments = len(program_instruments)
            note.instrument = num_instruments if num_instruments < 9 else num_instruments + 1
            program_instruments[note.program] = note.instrument
        elif note.is_drum:
            note.instrument = 9
        else:
            note.instrument = program_instruments[note.program]


def validate_note_sequence(ns: NoteSequence) -> None:
    """Raise ValueError if NoteSequence contains invalid notes."""
    for note in ns.notes:
        if note.start_time >= note.end_time:
            raise ValueError(
                "note has start time >= end time: %f >= %f" % (note.start_time, note.end_time)
            )
        if note.velocity == 0:
            raise ValueError("note has zero velocity")


@dataclasses.dataclass
class NoteEventData:
    pitch: int
    velocity: Optional[int] = None
    program: Optional[int] = None
    is_drum: Optional[bool] = None
    instrument: Optional[int] = None


def note_sequence_to_onsets(
    ns: NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract note onsets and pitches from a NoteSequence."""
    # Sort by pitch to use as a tiebreaker for subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    return (
        [note.start_time for note in notes],
        [NoteEventData(pitch=note.pitch) for note in notes],
    )


def note_sequence_to_onsets_and_offsets(
    ns: NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
        ns: NoteSequence from which to extract onsets and offsets.

    Returns:
        times: A list of note onset and offset times.
        values: A list of NoteEventData objects where velocity is zero for note
            offsets.
    """
    # Sort by pitch and put offsets before onsets as a tiebreaker for subsequent
    # stable sort.
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    times = [note.end_time for note in notes] + [note.start_time for note in notes]
    values = [NoteEventData(pitch=note.pitch, velocity=0) for note in notes] + [
        NoteEventData(pitch=note.pitch, velocity=note.velocity) for note in notes
    ]
    return times, values


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches & programs from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
        ns: NoteSequence from which to extract onsets and offsets.

    Returns:
        times: A list of note onset and offset times.
        values: A list of NoteEventData objects where velocity is zero for note
            offsets.
    """
    # Sort by program and pitch and put offsets before onsets as a tiebreaker for
    # subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    times = [note.end_time for note in notes if not note.is_drum] + [
        note.start_time for note in notes
    ]
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
        for note in notes
        if not note.is_drum
    ] + [
        NoteEventData(
            pitch=note.pitch, velocity=note.velocity, program=note.program, is_drum=note.is_drum
        )
        for note in notes
    ]
    return times, values


@dataclasses.dataclass
class NoteEncodingState:
    """Encoding state for note transcription, keeping track of active pitches."""

    # velocity bin for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(default_factory=dict)


def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: event_codec.Codec,
) -> Sequence[event_codec.Event]:
    """Convert note event data to a sequence of events."""
    if value.velocity is None:
        # onsets only, no program or velocity
        return [event_codec.Event("pitch", value.pitch)]
    else:
        num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
        velocity_bin = vocabularies.velocity_to_bin(value.velocity, num_velocity_bins)
        if value.program is None:
            # onsets + offsets + velocities only, no programs
            if state is not None:
                state.active_pitches[(value.pitch, 0)] = velocity_bin
            return [
                event_codec.Event("velocity", velocity_bin),
                event_codec.Event("pitch", value.pitch),
            ]
        else:
            if value.is_drum:
                # drum events use a separate vocabulary
                return [
                    event_codec.Event("velocity", velocity_bin),
                    event_codec.Event("drum", value.pitch),
                ]
            else:
                # program + velocity + pitch
                if state is not None:
                    state.active_pitches[(value.pitch, int(value.program))] = velocity_bin
                return [
                    event_codec.Event("program", value.program),
                    event_codec.Event("velocity", velocity_bin),
                    event_codec.Event("pitch", value.pitch),
                ]


def note_encoding_state_to_events(state: NoteEncodingState) -> Sequence[event_codec.Event]:
    """Output program and pitch events for active notes plus a final tie event."""
    events = []
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: k[::-1]):
        if state.active_pitches[(pitch, program)]:
            events += [event_codec.Event("program", program), event_codec.Event("pitch", pitch)]
    events.append(event_codec.Event("tie", 0))
    return events


@dataclasses.dataclass
class NoteDecodingState:
    """Decoding state for note transcription."""

    current_time: float = 0.0
    # velocity to apply to subsequent pitch events (zero for note-off)
    current_velocity: int = DEFAULT_VELOCITY
    # program to apply to subsequent pitch events
    current_program: int = 0
    # onset time and velocity for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = dataclasses.field(
        default_factory=dict
    )
    # pitches (with programs) to continue from previous segment
    tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(default_factory=set)
    # whether or not we are in the tie section at the beginning of a segment
    is_tie_section: bool = False
    # partially-decoded NoteSequence
    note_sequence: NoteSequence = dataclasses.field(default_factory=NoteSequence)


def decode_note_onset_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
    """Process note onset event and update decoding state."""
    if event.type == "pitch":
        state.note_sequence.add_note(
            start_time=time,
            end_time=time + DEFAULT_NOTE_DURATION,
            pitch=event.value,
            velocity=DEFAULT_VELOCITY,
        )
        state.note_sequence.total_time = max(
            state.note_sequence.total_time, time + DEFAULT_NOTE_DURATION
        )
    else:
        raise ValueError("unexpected event type: %s" % event.type)


def _add_note_to_sequence(
    ns: NoteSequence,
    start_time: float,
    end_time: float,
    pitch: int,
    velocity: int,
    program: int = 0,
    is_drum: bool = False,
) -> None:
    end_time = max(end_time, start_time + MIN_NOTE_DURATION)
    ns.add_note(
        start_time=start_time,
        end_time=end_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum,
    )
    ns.total_time = max(ns.total_time, end_time)


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
    """Process note event and update decoding state."""
    if time < state.current_time:
        raise ValueError("event time < current time, %f < %f" % (time, state.current_time))
    state.current_time = time
    if event.type == "pitch":
        pitch = event.value
        if state.is_tie_section:
            # "tied" pitch
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(
                    "inactive pitch/program in tie section: %d/%d"
                    % (pitch, state.current_program)
                )
            if (pitch, state.current_program) in state.tied_pitches:
                raise ValueError(
                    "pitch/program is already tied: %d/%d" % (pitch, state.current_program)
                )
            state.tied_pitches.add((pitch, state.current_program))
        elif state.current_velocity == 0:
            # note offset
            if (pitch, state.current_program) not in state.active_pitches:
                raise ValueError(
                    "note-off for inactive pitch/program: %d/%d" % (pitch, state.current_program)
                )
            onset_time, onset_velocity = state.active_pitches.pop((pitch, state.current_program))
            _add_note_to_sequence(
                state.note_sequence,
                start_time=onset_time,
                end_time=time,
                pitch=pitch,
                velocity=onset_velocity,
                program=state.current_program,
            )
        else:
            # note onset
            if (pitch, state.current_program) in state.active_pitches:
                # The pitch is already active; this shouldn't really happen but we'll
                # try to handle it gracefully by ending the previous note and starting
                # a new one.
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, state.current_program)
                )
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time=onset_time,
                    end_time=time,
                    pitch=pitch,
                    velocity=onset_velocity,
                    program=state.current_program,
                )
            state.active_pitches[(pitch, state.current_program)] = (time, state.current_velocity)
    elif event.type == "drum":
        # drum onset (drums have no offset)
        if state.current_velocity == 0:
            raise ValueError("velocity cannot be zero for drum event")
        offset_time = time + DEFAULT_NOTE_DURATION
        _add_note_to_sequence(
            state.note_sequence,
            start_time=time,
            end_time=offset_time,
            pitch=event.value,
            velocity=state.current_velocity,
            is_drum=True,
        )
    elif event.type == "velocity":
        # velocity change
        num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
        velocity = vocabularies.bin_to_velocity(event.value, num_velocity_bins)
        state.current_velocity = velocity
    elif event.type == "program":
        # program change
        state.current_program = event.value
    elif event.type == "tie":
        # end of tie section; end active notes that weren't declared tied
        if not state.is_tie_section:
            raise ValueError("tie section end event when not in tie section")
        for pitch, program in list(state.active_pitches.keys()):
            if (pitch, program) not in state.tied_pitches:
                onset_time, onset_velocity = state.active_pitches.pop((pitch, program))
                _add_note_to_sequence(
                    state.note_sequence,
                    start_time=onset_time,
                    end_time=state.current_time,
                    pitch=pitch,
                    velocity=onset_velocity,
                    program=program,
                )
        state.is_tie_section = False
    else:
        raise ValueError("unexpected event type: %s" % event.type)


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
    """Begin the tied pitches section at the start of a segment."""
    state.tied_pitches = set()
    state.is_tie_section = True


def flush_note_decoding_state(state: NoteDecodingState) -> NoteSequence:
    """End all active notes and return resulting NoteSequence."""
    for onset_time, _ in state.active_pitches.values():
        state.current_time = max(state.current_time, onset_time + MIN_NOTE_DURATION)
    for pitch, program in list(state.active_pitches.keys()):
        onset_time, onset_velocity = state.active_pitches.pop((pitch, program))
        _add_note_to_sequence(
            state.note_sequence,
            start_time=onset_time,
            end_time=state.current_time,
            pitch=pitch,
            velocity=onset_velocity,
            program=program,
        )
    assign_instruments(state.note_sequence)
    return state.note_sequence


class NoteEncodingSpecType(run_length_encoding.EventEncodingSpec):
    pass


# encoding spec for modeling note onsets only
NoteOnsetEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_onset_event,
    flush_decoding_state_fn=lambda state: state.note_sequence,
)


# encoding spec for modeling onsets and offsets
NoteEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)


# encoding spec for modeling onsets and offsets, with a "tie" section at the
# beginning of each segment listing already-active notes
NoteEncodingWithTiesSpec = NoteEncodingSpecType(
    init_encoding_state_fn=NoteEncodingState,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=note_encoding_state_to_events,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=begin_tied_pitches_section,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)
