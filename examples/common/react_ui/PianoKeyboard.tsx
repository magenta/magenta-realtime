/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { useRef, useCallback } from 'react';

interface PianoKeyboardProps {
  activeNotes: number[];
  accentColor: string;
  startNote?: number;  // MIDI note number
  endNote?: number;
  keyboardMidiEnabled?: boolean;
  onNoteOn?: (note: number) => void;
  onNoteOff?: (note: number) => void;
  whiteKeyColor?: string;
  blackKeyColor?: string;
  /** Show "C0", "C1", … labels on every C key */
  showOctaveLabels?: boolean;
  /** Gap between white keys in px (default 3) */
  gap?: number;
  /** Black key height as CSS value (default '60%') */
  blackKeyHeight?: string;
  /** Font size for octave labels in px (default 11) */
  octaveLabelFontSize?: number;
  /** Font size for QWERTY key labels in px (default 20) */
  qwertyLabelFontSize?: number;
}

// Note layout: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

// const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Position-based key labels: offset from startNote → keyboard key label.
// White keys: A S D F G H J K L ;   Black keys: W E T Y U O P
// Spans C through E above (17 semitones: one octave + major third).
const SEMITONE_TO_KEY: Record<number, string> = {
  0: 'A', 1: 'W', 2: 'S', 3: 'E', 4: 'D', 5: 'F', 6: 'T', 7: 'G', 8: 'Y', 9: 'H', 10: 'U', 11: 'J',
  12: 'K', 13: 'O', 14: 'L', 15: 'P', 16: ';',
};

/** Returns the octave label for a MIDI note if it's a C, e.g. "C3" for note 48. */
function octaveLabel(midiNote: number): string | null {
  if (midiNote % 12 !== 0) return null;
  // MIDI note 0 = C-1 in scientific pitch, so octave = floor(note/12) - 1
  const octave = Math.floor(midiNote / 12) - 1;
  return `C${octave}`;
}

export function PianoKeyboard({
  activeNotes,
  accentColor,
  startNote = 60,
  endNote = 76,
  keyboardMidiEnabled = false,
  onNoteOn,
  onNoteOff,
  whiteKeyColor = '#FAFAFA',
  blackKeyColor = '#000',
  showOctaveLabels = false,
  gap = 3,
  blackKeyHeight = '60%',
  octaveLabelFontSize = 11,
  qwertyLabelFontSize = 20,
}: PianoKeyboardProps) {
  const activeSet = new Set(activeNotes);
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Track which note is currently held by the pointer
  const heldNoteRef = useRef<number | null>(null);

  const getNoteFromPoint = useCallback((x: number, y: number): number | null => {
    const els = document.elementsFromPoint(x, y);
    for (const el of els) {
      const noteAttr = (el as HTMLElement).dataset?.note;
      if (noteAttr !== undefined) return parseInt(noteAttr, 10);
    }
    return null;
  }, []);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const note = getNoteFromPoint(e.clientX, e.clientY);
    if (note !== null) {
      heldNoteRef.current = note;
      onNoteOn?.(note);
    }
  }, [onNoteOn, getNoteFromPoint]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (heldNoteRef.current === null) return;
    const note = getNoteFromPoint(e.clientX, e.clientY);
    if (note !== null && note !== heldNoteRef.current) {
      // Glissando: release old note, press new one
      onNoteOff?.(heldNoteRef.current);
      heldNoteRef.current = note;
      onNoteOn?.(note);
    }
  }, [onNoteOn, onNoteOff, getNoteFromPoint]);

  const handlePointerUp = useCallback(() => {
    if (heldNoteRef.current !== null) {
      onNoteOff?.(heldNoteRef.current);
      heldNoteRef.current = null;
    }
  }, [onNoteOff]);

  // Generate white keys first
  const whiteKeys: { note: number; label?: string; whiteIdx: number }[] = [];
  let whiteIdx = 0;
  for (let n = startNote; n <= endNote; n++) {
    if (!BLACK_KEYS.has(n % 12)) {
      const offset = n - startNote;
      const keyLabel = SEMITONE_TO_KEY[offset];
      whiteKeys.push({ note: n, label: keyLabel, whiteIdx });
      whiteIdx++;
    }
  }
  const whiteKeyCount = whiteIdx;

  // Generate black keys centered over white key seams
  const blackKeys: { note: number; label?: string; leftExpr: string }[] = [];
  for (let n = startNote; n <= endNote; n++) {
    if (BLACK_KEYS.has(n % 12)) {
      const prevWhite = n - 1;
      const prevKey = whiteKeys.find(k => k.note === prevWhite);
      if (prevKey) {
        const wIdx = prevKey.whiteIdx;
        const whiteKeyWidthExpr = `((100% - ${(whiteKeyCount - 1) * gap}px) / ${whiteKeyCount})`;
        const leftExpr = `calc(${wIdx + 1} * ${whiteKeyWidthExpr} + ${wIdx} * ${gap}px + ${gap / 2}px - (${whiteKeyWidthExpr} * 0.7 / 2))`;
        const offset = n - startNote;
        const keyLabel = SEMITONE_TO_KEY[offset];
        blackKeys.push({ note: n, label: keyLabel, leftExpr });
      }
    }
  }

  const whiteKeyWidthExpr = `((100% - ${(whiteKeyCount - 1) * gap}px) / ${whiteKeyCount})`;
  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        display: 'flex',
        gap: `${gap}px`,
        userSelect: 'none',
        WebkitUserSelect: 'none',
        touchAction: 'none',
      }}
    >
      {/* White Keys */}
      {whiteKeys.map((key) => {
        const isActive = activeSet.has(key.note);
        const oLabel = showOctaveLabels ? octaveLabel(key.note) : null;
        return (
          <div
            key={key.note}
            className="white-key"
            data-note={key.note}
            style={{
              flex: 1,
              height: '100%',
              backgroundColor: isActive ? accentColor : whiteKeyColor,
              borderRadius: '8%',
              cursor: 'pointer',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '12%',
              backgroundColor: 'rgba(0, 0, 0, 0.54)',
              pointerEvents: 'none',
            }} />
            {/* QWERTY label */}
            {keyboardMidiEnabled && key.label && (
              <span
                style={{
                  fontFamily: "'Google Sans', system-ui, sans-serif",
                  fontSize: `${qwertyLabelFontSize}px`,
                  fontWeight: 400,
                  color: blackKeyColor,
                  userSelect: 'none',
                  pointerEvents: 'none',
                  position: 'absolute',
                  bottom: '35%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                }}
              >
                {key.label}
              </span>
            )}
            {/* Octave label (C0, C1, …) */}
            {!keyboardMidiEnabled && oLabel && (
              <span
                style={{
                  fontFamily: "'Google Sans', system-ui, sans-serif",
                  fontSize: `${octaveLabelFontSize}px`,
                  fontWeight: 500,
                  color: isActive ? 'rgba(0,0,0,0.6)' : 'rgba(0,0,0,0.35)',
                  userSelect: 'none',
                  pointerEvents: 'none',
                  position: 'absolute',
                  bottom: '12%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  letterSpacing: '0.3px',
                }}
              >
                {oLabel}
              </span>
            )}
          </div>
        );
      })}

      {/* Black Keys Overlay */}
      {blackKeys.map((key) => {
        const isActive = activeSet.has(key.note);
        return (
          <div
            key={key.note}
            className="black-key"
            data-note={key.note}
            style={{
              position: 'absolute',
              top: 0,
              left: key.leftExpr,
              width: `calc(${whiteKeyWidthExpr} * 0.7)`,
              height: blackKeyHeight,
              backgroundColor: isActive ? accentColor : blackKeyColor,
              borderBottomLeftRadius: '12%',
              borderBottomRightRadius: '12%',
              boxSizing: 'border-box',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
              zIndex: 2,
              cursor: 'pointer',
              overflow: 'hidden',
            }}
          >
            <div style={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '10%',
              backgroundColor: 'rgba(34, 126, 230, 0.26)',
              pointerEvents: 'none',
            }} />
            {keyboardMidiEnabled && key.label && (
              <span
                style={{
                  fontFamily: "'Google Sans', system-ui, sans-serif",
                  fontSize: `${qwertyLabelFontSize}px`,
                  fontWeight: 400,
                  color: isActive ? blackKeyColor : whiteKeyColor,
                  userSelect: 'none',
                  pointerEvents: 'none',
                  position: 'absolute',
                  bottom: '18%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                }}
              >
                {key.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
