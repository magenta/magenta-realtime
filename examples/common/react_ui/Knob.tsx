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

import { useState, useCallback, useRef, useEffect } from 'react';
import Tooltip from '@mui/material/Tooltip';
import { InfoOutlined } from '@mui/icons-material';

interface KnobProps {
  label: string;
  tooltip?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange?: (value: number) => void;
  size?: number;
  labelFontSize?: number;
  accentColor?: string;
}

// Arc rotated 90° CW: gap is on the right
const ARC_START = 225;
const ARC_SWEEP = 270;

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const sweep = endAngle - startAngle;
  const largeArc = sweep > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

function snap(val: number, step: number) {
  return Math.round(val / step) * step;
}

function formatValue(value: number, step: number): string {
  if (step >= 1) return Math.round(value).toString();
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  return value.toFixed(decimals);
}

export function Knob({ label, tooltip, value: controlledValue, min, max, step, onChange, size = 80, labelFontSize = 12, accentColor = '#71fade' }: KnobProps) {
  const [localValue, setLocalValue] = useState(controlledValue);
  const dragRef = useRef<{ startY: number; startVal: number } | null>(null);

  useEffect(() => {
    if (!dragRef.current) setLocalValue(controlledValue);
  }, [controlledValue]);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { startY: e.clientY, startVal: localValue };
    document.body.classList.add('dragging-vertical');
  }, [localValue]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dy = dragRef.current.startY - e.clientY;
    const sensitivity = 200;
    const range = max - min;
    const raw = dragRef.current.startVal + (dy / sensitivity) * range;
    const clamped = Math.max(min, Math.min(max, raw));
    const snapped = snap(clamped, step);
    setLocalValue(snapped);
    onChange?.(snapped);
  }, [min, max, step, onChange]);

  const handlePointerUp = useCallback(() => {
    dragRef.current = null;
    document.body.classList.remove('dragging-vertical');
  }, []);

  const normalized = (localValue - min) / (max - min);
  const VIEW = 80;            // fixed internal coordinate space
  const cx = VIEW / 2;
  const cy = VIEW / 2;
  const trackR = 31.5;       // arc diameter 63px
  const knobR = 19.5;        // knob circle radius
  const strokeW = 2;
  const knobStrokeW = 2;

  const endAngle = ARC_START + ARC_SWEEP * normalized;
  // Indicator line from center to edge of knob circle
  const lineEnd = polarToCartesian(cx, cy, knobR - 2, endAngle);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      gap: '8px',
    }}>
      {/* Knob SVG */}
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        style={{ touchAction: 'none', flexShrink: 0 }}
      >
        {/* Track arc (background) */}
        <path
          d={describeArc(cx, cy, trackR, ARC_START, ARC_START + ARC_SWEEP)}
          fill="none"
          stroke="#373737"
          strokeWidth={strokeW}
          strokeLinecap="round"
          style={{ pointerEvents: 'none' }}
        />

        {/* Value arc (fill) — accent color */}
        {normalized > 0.005 && (
          <path
            d={describeArc(cx, cy, trackR, ARC_START, endAngle)}
            fill="none"
            stroke={accentColor}
            strokeWidth={strokeW}
            strokeLinecap="round"
            style={{ pointerEvents: 'none' }}
          />
        )}

        {/* Knob body — stroked circle, matching the track color */}
        <circle
          cx={cx} cy={cy} r={knobR}
          fill="none"
          stroke="#373737"
          strokeWidth={knobStrokeW}
          style={{ pointerEvents: 'none' }}
        />

        {/* Indicator line from center to edge */}
        <line
          x1={cx} y1={cy}
          x2={lineEnd.x} y2={lineEnd.y}
          stroke="#FFF"
          strokeWidth={2}
          strokeLinecap="round"
          style={{ pointerEvents: 'none' }}
        />

        {/* Circular hit area overlay */}
        <circle
          cx={cx}
          cy={cy}
          r={trackR + 5}
          fill="transparent"
          stroke="transparent"
          strokeWidth={1}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={{ cursor: 'grab', pointerEvents: 'auto' }}
        />
      </svg>

      {/* Label + value to the right */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '2px',
        minWidth: 0,
      }}>
        {/* Label row with inline tooltip */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}>
          <span style={{
            color: '#FFF',
            opacity: 0.7,
            fontFamily: "'Google Sans', sans-serif",
            fontSize: `${labelFontSize}px`,
            fontWeight: 500,
            lineHeight: 'normal',
            letterSpacing: '0.56px',
            whiteSpace: 'nowrap',
          }}>
            {label}
          </span>
          {tooltip && (
            <Tooltip title={tooltip} arrow placement="top">
              <InfoOutlined sx={{ fontSize: '13px', opacity: 0.3, cursor: 'help', color: '#FFF' }} />
            </Tooltip>
          )}
        </div>

        {/* Current value — accent color */}
        <span style={{
          color: accentColor,
          fontFamily: "'Google Sans', sans-serif",
          fontSize: '14px',
          fontWeight: 600,
          lineHeight: 'normal',
        }}>
          {formatValue(localValue, step)}
        </span>
      </div>
    </div>
  );
}
