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

import React, { useRef, useState, useCallback, useEffect } from 'react';

const THUMB_SIZE = 24;
const TRACK_WIDTH = 24;

interface JamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (val: number) => void;
  showValueOnThumb?: boolean;
  valueFormatter?: (val: number) => string;
}

export function JamSlider({
  label,
  value,
  min,
  max,
  step = 0.1,
  onChange,
  showValueOnThumb = false,
  valueFormatter,
}: JamSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Map drag vertical pointer client coordinates dynamically using bounding rect height
  const updateValue = useCallback((clientY: number) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const insetTop = THUMB_SIZE / 2;
    const insetBottom = THUMB_SIZE / 2;
    const usable = rect.height - insetTop - insetBottom;
    const relativeY = clientY - rect.top - insetTop;
    const frac = Math.max(0, Math.min(1, 1 - relativeY / usable));
    const rawValue = min + frac * (max - min);
    const steppedValue = Math.round(rawValue / step) * step;
    const finalValue = parseFloat(Math.max(min, Math.min(max, steppedValue)).toFixed(2));
    onChange(finalValue);
  }, [min, max, step, onChange]);

  useEffect(() => {
    return () => {
      document.body.classList.remove('dragging-vertical', 'cursor-none', 'cursor-grabbing');
    };
  }, []);

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setIsDragging(true);
    const dragClass = showValueOnThumb ? 'cursor-none' : 'cursor-grabbing';
    document.body.classList.add(dragClass);
    updateValue(e.clientY);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging) return;
    updateValue(e.clientY);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    document.body.classList.remove('dragging-vertical', 'cursor-none', 'cursor-grabbing');
  };

  const percent = ((value - min) / (max - min)) * 100;
  const thumbBottomCss = `calc(${percent}% - (${percent / 100} * ${THUMB_SIZE}px))`;
  // Fill from bottom of track up to slightly above the center of the thumb to cover any rounding
  const fillHeightCss = `calc(${percent}% - (${percent / 100} * ${THUMB_SIZE}px) + ${THUMB_SIZE / 2 + 2}px)`;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      height: '100%',
      width: '64px',
      userSelect: 'none',
      WebkitUserSelect: 'none',
    }}>
      {/* Slider Track Container */}
      <div
        ref={trackRef}
        style={{
          position: 'relative',
          width: `${TRACK_WIDTH}px`,
          flex: '1 1 0px',
          minHeight: '30px',
        }}
      >
        {/* Track background — light semi-transparent */}
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(26, 26, 29, 0.10)',
          borderRadius: `${TRACK_WIDTH / 2}px`,
          pointerEvents: 'none',
        }} />

        {/* Track fill — dark gray from bottom up to thumb */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: fillHeightCss,
          background: '#1A1A1D',
          borderRadius: `0 0 ${TRACK_WIDTH / 2}px ${TRACK_WIDTH / 2}px`,
          pointerEvents: 'none',
        }} />

        {/* Thumb — white circle */}
        <div style={{
          position: 'absolute',
          bottom: thumbBottomCss,
          left: '50%',
          transform: 'translateX(-50%)',
          width: `${THUMB_SIZE}px`,
          height: `${THUMB_SIZE}px`,
          borderRadius: '50%',
          background: '#FFF',
          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.25)',
          pointerEvents: 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {showValueOnThumb && (
            <span style={{
              fontWeight: 800,
              textAlign: 'center',
              fontSize: '10px',
              color: '#000'
            }}>
              {valueFormatter ? valueFormatter(value) : value.toString()}
            </span>
          )}
        </div>

        {/* Hitbox overlay — generous touch/click target */}
        <div
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: '50%',
            transform: 'translateX(-50%)',
            width: `calc(100% + 40px)`,
            cursor: 'grab',
            touchAction: 'none',
          }}
        />
      </div>

      {/* Label */}
      <span style={{
        marginTop: '12px',
        flexShrink: 0,
        fontFamily: "'Google Sans Text', system-ui, sans-serif",
        fontSize: '11px',
        fontWeight: 500,
        color: '#1B1C17',
        userSelect: 'none',
        lineHeight: '1',
      }}>
        {label}
      </span>
    </div>
  );
}
