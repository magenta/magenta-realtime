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

import React, { useState, useRef, useEffect } from 'react';
import IconButton from '@mui/material/IconButton';
import VolumeUp from '@mui/icons-material/VolumeUp';
import VolumeDown from '@mui/icons-material/VolumeDown';
import VolumeMute from '@mui/icons-material/VolumeMute';
import VolumeOff from '@mui/icons-material/VolumeOff';

interface VolumeControlProps {
  volume: number; // 0.0 to 1.0
  onVolumeChange: (volume: number) => void;
  sliderPosition?: 'top' | 'bottom';
  buttonSize?: number;
}

// Converts decibels (-60.0 to 0.0+) to linear (0.0 to 1.0)
const dbToLinear = (db: number) => {
  if (db <= -60.0) return 0.0;
  return Math.min(1.0, Math.max(0.0, (db + 60.0) / 60.0));
};

// Converts linear (0.0 to 1.0) to decibels (-60.0 to 0.0)
const linearToDb = (linear: number) => {
  if (linear <= 0.0) return -60.0;
  return (linear * 60.0) - 60.0;
};

export function VolumeControl({
  volume,
  onVolumeChange,
  sliderPosition = 'top',
  buttonSize = 40,
}: VolumeControlProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const linearVol = dbToLinear(volume);

  // Close popup if clicked outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Get correct volume icon based on level
  const getVolumeIcon = () => {
    if (linearVol <= 0.0) {
      return <VolumeOff sx={{ fontSize: buttonSize * 0.5 }} />;
    }
    if (linearVol < 0.3) {
      return <VolumeMute sx={{ fontSize: buttonSize * 0.5 }} />;
    }
    if (linearVol < 0.7) {
      return <VolumeDown sx={{ fontSize: buttonSize * 0.5 }} />;
    }
    return <VolumeUp sx={{ fontSize: buttonSize * 0.5 }} />;
  };

  const toggleOpen = () => {
    setIsOpen(!isOpen);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const linearVal = parseFloat(e.target.value);
    onVolumeChange(linearToDb(linearVal));
  };

  // Absolute positioning styles for the slider capsule based on position property
  const sliderContainerStyle: React.CSSProperties = {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: 'var(--color-raised, #36373a)',
    borderRadius: '24px',
    padding: '16px 10px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
    zIndex: 100,
    height: '140px',
    width: '40px',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    opacity: isOpen ? 1 : 0,
    pointerEvents: isOpen ? 'auto' : 'none',
    ...(sliderPosition === 'top'
      ? { bottom: `${buttonSize + 8}px`, transformOrigin: 'bottom center' }
      : { top: `${buttonSize + 8}px`, transformOrigin: 'top center' }
    ),
  };

  return (
    <div ref={containerRef} style={{ position: 'relative', display: 'inline-block' }}>
      {/* Popout Slider Container */}
      <div style={sliderContainerStyle}>
        {/* Custom Styled Vertical Slider */}
        <div style={{
          position: 'relative',
          height: '100%',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={linearVol}
            onChange={handleSliderChange}
            onMouseDown={() => document.body.classList.add('is-dragging')}
            onMouseUp={() => document.body.classList.remove('is-dragging')}
            className="volume-slider-input"
            style={{
              WebkitAppearance: 'none',
              appearance: 'none',
              width: '100px',
              height: '6px',
              borderRadius: '3px',
              background: `linear-gradient(to right, #ffffff ${linearVol * 100}%, rgba(255, 255, 255, 0.2) ${linearVol * 100}%)`,
              outline: 'none',
              transform: 'rotate(-90deg)',
              transformOrigin: 'center',
              position: 'absolute',
            }}
          />

          {/* Custom slider styles scoped specifically to this input */}
          <style>{`
            .volume-slider-input[type="range"]::-webkit-slider-thumb {
              -webkit-appearance: none;
              appearance: none;
              width: 14px;
              height: 14px;
              border-radius: 50%;
              background: #ffffff;
              box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            }
          `}</style>
        </div>
      </div>

      {/* Icon Button */}
      <IconButton
        onClick={toggleOpen}
        sx={{
          width: buttonSize,
          height: buttonSize,
        }}
        title={linearVol === 0 ? "Muted" : `Volume: ${Math.round(linearVol * 100)}%`}
      >
        {getVolumeIcon()}
      </IconButton>
    </div>
  );
}
