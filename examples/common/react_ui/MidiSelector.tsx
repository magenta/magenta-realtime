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

import MenuItem from '@mui/material/MenuItem';
import { MagentaDropdown } from './MagentaDropdown';

export interface MidiSource {
  name: string;
  endpoint: number;
  connected: boolean;
}

interface MidiSelectorProps {
  /** List of available physical MIDI sources from the native host */
  midiSources: MidiSource[];
  /** Whether "Computer Keyboard" mode is active */
  keyboardMidiEnabled: boolean;
  /** Called with endpoint ID: 0 = Computer Keyboard, >0 = physical source */
  onSelectSource: (endpoint: number) => void;
  /** Show "Computer Keyboard" as a selectable option (default: true) */
  showComputerKeyboard?: boolean;
  /** Whether any MIDI note is currently active (lights up the activity LED) */
  midiActive?: boolean;
  /** Extra sx passed through to MagentaDropdown's trigger button */
  buttonSx?: Record<string, any>;
}

/**
 * Shared MIDI input selector dropdown.
 *
 * Shows available physical MIDI sources plus an optional
 * "Computer Keyboard" option. Selection is communicated back
 * to the native host via onSelectSource(endpoint).
 */
export function MidiSelector({
  midiSources,
  keyboardMidiEnabled,
  onSelectSource,
  showComputerKeyboard = true,
  midiActive = false,
  buttonSx,
}: MidiSelectorProps) {
  const connectedName = midiSources.find(s => s.connected)?.name;
  const displayLabel = keyboardMidiEnabled
    ? 'COMPUTER KEYBOARD'
    : (connectedName ?? 'NONE');

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', verticalAlign: 'baseline', gap: '4px' }}>
      <span style={{
        fontFamily: "'Google Sans', system-ui, sans-serif",
        fontSize: '12px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        opacity: 0.7,
        marginRight: '2px',
        userSelect: 'none',
      }}>
        MIDI INPUT
      </span>
      <MagentaDropdown
        id="midi-input-button"
        label={displayLabel}
        buttonSx={{
          fontSize: '12px',
          fontWeight: 700,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          color: '#FFF',
          '&:hover': { background: 'rgba(255, 255, 255, 0.12)' },
          ...buttonSx,
        }}
      >
        {showComputerKeyboard && (
          <MenuItem
            selected={keyboardMidiEnabled}
            onClick={() => onSelectSource(0)}
          >
            Computer Keyboard
          </MenuItem>
        )}
        {midiSources.map((src) => (
          <MenuItem
            key={src.endpoint}
            selected={!keyboardMidiEnabled && src.connected}
            onClick={() => onSelectSource(src.endpoint)}
          >
            {src.name}
          </MenuItem>
        ))}
        {midiSources.length === 0 && (
          <MenuItem disabled sx={{ opacity: 0.5 }}>
            No physical MIDI devices detected
          </MenuItem>
        )}
      </MagentaDropdown>

      {/* Activity LED */}
      <div
        style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: midiActive ? '#4CAF50' : 'rgba(255, 255, 255, 0.35)',
          transition: 'all 0.1s ease-out',
          flexShrink: 0,
        }}
      />
    </div>
  );
}
