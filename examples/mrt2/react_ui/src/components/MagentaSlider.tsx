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
import './MagentaSlider.css';

interface MagentaSliderProps {
  label: string;
  tooltip?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  color?: string;
  labelFontSize?: number;
  /** When true, label+tooltip sit above the slider instead of inline. */
  stacked?: boolean;
}

export function MagentaSlider({
  label,
  tooltip,
  value,
  min,
  max,
  step,
  onChange,
  color = '#71fade',
  labelFontSize = 12,
  stacked = false,
}: MagentaSliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div style={{
      display: 'flex',
      flexDirection: stacked ? 'column' : 'row',
      alignItems: stacked ? 'stretch' : 'center',
      gap: stacked ? '10px' : '8px',
      width: '100%',
    }}>
      {/* Label + info icon */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        flexShrink: 0,
        whiteSpace: 'nowrap',
      }}>
        <span style={{
          fontSize: `${labelFontSize}px`,
          fontFamily: "'Google Sans', sans-serif",
          fontWeight: 500,
          color: '#FFF',
          opacity: 0.7,
          letterSpacing: '0.56px',
        }}>
          {label}
        </span>
        {tooltip && (
          <Tooltip title={tooltip} arrow placement="top">
            <InfoOutlined style={{
              fontSize: '13px',
              opacity: 0.3,
              cursor: 'help',
              color: '#FFF',
            }} />
          </Tooltip>
        )}
      </div>

      {/* Slider */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="magenta-slider"
        style={{
          '--slider-color': color,
          '--slider-pct': `calc(${pct}% + ${(0.5 - pct / 100) * 20}px)`,
        } as React.CSSProperties}
      />
    </div>
  );
}
