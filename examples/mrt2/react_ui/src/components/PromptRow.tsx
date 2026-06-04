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


import { useRef } from 'react';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';

interface PromptRowProps {
  text: string;
  color: string;
  weight: number;
  isEmpty?: boolean;
  isAudio?: boolean;
  onTextChange?: (text: string) => void;
  onWeightChange?: (weight: number) => void;
  onRemove?: () => void;
  onUpload?: () => void;
  onClearAudio?: () => void;
}

export function PromptRow({
  text,
  color,
  weight,
  isEmpty,
  isAudio,
  onTextChange,
  onWeightChange,
  onRemove,
  onClearAudio,
}: PromptRowProps) {
  // Snapshot text on focus so we can revert on blur if empty
  const textOnFocusRef = useRef<string>('');

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
    }}>

      <div
        className="prompt-box"
        style={{
          display: 'flex',
          width: '224px',
          height: '40px',
          padding: '0 16px 0 0',
          alignItems: 'center',
          gap: '4px',
          flexShrink: 0,
        }}
      >
        {isAudio ? (
          /* Audio prompt: show filename + clear button */
          <>
            <span
              className="material-icons"
              style={{
                fontSize: '16px',
                color: 'var(--color-muted)',
                flexShrink: 0,
                marginLeft: '16px',
              }}
            >audio_file</span>
            <span style={{
              flex: 1,
              minWidth: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontFamily: "'Google Sans', sans-serif",
              fontSize: '13px',
              fontWeight: 500,
              color: '#FFF',
            }}>
              {text || 'Loading...'}
            </span>
            <Tooltip title="Remove audio prompt">
              <IconButton
                onClick={onClearAudio}
                sx={{
                  width: '32px',
                  height: '32px',
                  color: 'var(--color-muted)',
                  backgroundColor: 'transparent',
                  border: 'none',
                  marginRight: '-8px',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    color: '#ffffff',
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                <span className="material-icons" style={{ fontSize: '18px' }}>close</span>
              </IconButton>
            </Tooltip>
          </>
        ) : (
          /* Text prompt: normal text input */
          <>
            <input
              type="text"
              value={isEmpty ? '' : text}
              onChange={(e) => onTextChange?.(e.target.value)}
              onFocus={() => { textOnFocusRef.current = text; }}
              onBlur={(e) => {
                if (!e.target.value.trim() && textOnFocusRef.current) {
                  onTextChange?.(textOnFocusRef.current);
                }
              }}
              placeholder="Add a prompt"
              style={{
                flex: 1,
                height: '100%',
                paddingLeft: '16px',
                background: 'none',
                border: 'none',
                outline: 'none',
                lineHeight: 'normal',
                minWidth: 0,
              }}
            />
            <IconButton
              onClick={onRemove}
              sx={{
                width: '32px',
                height: '32px',
                color: 'var(--color-muted)',
                backgroundColor: 'transparent',
                border: 'none',
                marginRight: '-8px',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  color: '#ffffff',
                },
                transition: 'all 0.2s ease-in-out',
              }}
            >
              <span className="material-icons" style={{ fontSize: '18px' }}>close</span>
            </IconButton>
          </>
        )}
      </div>

      {/* Weight slider */}
      <div style={{
        flex: '1 1 0',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
      }}>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={weight}
          onChange={(e) => onWeightChange?.(parseFloat(e.target.value))}
          className="prompt-slider"
          style={{
            '--slider-color': color,
            '--slider-pct': `calc(${weight * 100}% + ${(0.5 - weight) * 20}px)`,
          } as React.CSSProperties}
        />
      </div>

    </div>
  );
}
