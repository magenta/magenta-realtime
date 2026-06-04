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

import Tooltip from '@mui/material/Tooltip';
import { InfoOutlined } from '@mui/icons-material';

interface MagentaToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
  labelFontSize?: number;
}

const TRACK_W = 26;
const TRACK_H = 12;
const THUMB_SIZE = TRACK_H - 4;
const THUMB_PAD = 2;

export function MagentaToggle({ label, checked, onChange, tooltip, labelFontSize = 12 }: MagentaToggleProps) {
  const thumbLeft = checked ? TRACK_W - THUMB_SIZE - THUMB_PAD : THUMB_PAD;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: 'fit-content' }}>
      <div
        onClick={() => onChange(!checked)}
        className="magenta-toggle"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          cursor: 'pointer',
          width: 'fit-content',
        }}
      >
        {/* Toggle track */}
        <div
          style={{
            width: `${TRACK_W}px`,
            height: `${TRACK_H}px`,
            borderRadius: `${TRACK_H / 2}px`,
            outline: `2px solid ${checked ? '#71fade' : '#fff'}`,
            position: 'relative',
            transition: 'background 0.15s ease, outline 0.15s ease',
            flexShrink: 0,
          }}
        >
          {/* Thumb */}
          <div style={{
            position: 'absolute',
            top: `${THUMB_PAD}px`,
            left: `${thumbLeft}px`,
            width: `${THUMB_SIZE}px`,
            height: `${THUMB_SIZE}px`,
            borderRadius: '50%',
            background: checked ? '#71fade' : '#fff',
            transition: 'left 0.15s ease, background 0.15s ease',
          }} />
        </div>

        <span
          className="magenta-toggle-label"
          style={{
            color: '#FFF',
            opacity: 0.7,
            fontFamily: "'Google Sans', sans-serif",
            fontSize: `${labelFontSize}px`,
            fontWeight: 500,
            lineHeight: 'normal',
            letterSpacing: '0.56px',
          }}
        >
          {label}
        </span>
      </div>

      {tooltip && (
        <Tooltip title={tooltip} placement="top" arrow>
          <InfoOutlined
            style={{
              fontSize: `13px`,
              opacity: 0.3,
              cursor: 'help',
              color: '#FFF',
              flexShrink: 0,
            }}
          />
        </Tooltip>
      )}
    </div>
  );
}
