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

import { useEffect, useRef } from 'react';
import Box from '@mui/material/Box';

interface AudioMeterProps {
  leftLevel: number; // target left level 0.0 to 1.0 (linear amplitude)
  rightLevel: number; // target right level 0.0 to 1.0 (linear amplitude)
  decayRate?: number; // decay rate per animation frame, default 0.95
  width?: string | number;
  height?: string | number;
  color?: string; // color of the meter bars, default is '#FFF'
  minDb?: number; // minimum decibel reference for log scaling, default is -40
  orientation?: 'horizontal' | 'vertical';
}

// Convert linear amplitude to logarithmic (decibel) scale, mapped to [0, 1]
const toLogScale = (linearVal: number, minDb: number): number => {
  if (linearVal <= 0.0001) return 0;
  const db = 20 * Math.log10(linearVal);
  if (db <= minDb) return 0;
  return (db - minDb) / (-minDb);
};

export function AudioMeter({
  leftLevel,
  rightLevel,
  decayRate = 0.95,
  width = '80px',
  height = '14px',
  color = '#FFF',
  minDb = -40, // 40dB dynamic range is excellent for compact visual meters
  orientation = 'horizontal',
}: AudioMeterProps) {
  const isVertical = orientation === 'vertical';
  const leftBarRef = useRef<HTMLDivElement>(null);
  const rightBarRef = useRef<HTMLDivElement>(null);

  // Keep track of current displaying values
  const curLeft = useRef(0);
  const curRight = useRef(0);

  // Keep track of the latest target levels (log-scaled)
  const targetLeft = useRef(0);
  const targetRight = useRef(0);

  useEffect(() => {
    targetLeft.current = toLogScale(leftLevel, minDb);
  }, [leftLevel, minDb]);

  useEffect(() => {
    targetRight.current = toLogScale(rightLevel, minDb);
  }, [rightLevel, minDb]);

  useEffect(() => {
    let animationId: number;

    const updateMeter = () => {
      // Rise quickly to target, decay slowly
      if (targetLeft.current > curLeft.current) {
        curLeft.current = targetLeft.current;
      } else {
        curLeft.current = Math.max(0, curLeft.current * decayRate);
      }

      if (targetRight.current > curRight.current) {
        curRight.current = targetRight.current;
      } else {
        curRight.current = Math.max(0, curRight.current * decayRate);
      }

      // Cap to 1.0 max for visual bounds
      const displayLeft = Math.min(1.0, curLeft.current);
      const displayRight = Math.min(1.0, curRight.current);

      if (leftBarRef.current) {
        if (isVertical) leftBarRef.current.style.height = `${displayLeft * 100}%`;
        else leftBarRef.current.style.width = `${displayLeft * 100}%`;
      }
      if (rightBarRef.current) {
        if (isVertical) rightBarRef.current.style.height = `${displayRight * 100}%`;
        else rightBarRef.current.style.width = `${displayRight * 100}%`;
      }

      animationId = requestAnimationFrame(updateMeter);
    };

    animationId = requestAnimationFrame(updateMeter);
    return () => cancelAnimationFrame(animationId);
  }, [decayRate, isVertical]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: isVertical ? 'row' : 'column',
        gap: '4px',
        width,
        height,
        justifyContent: 'center',
      }}
    >
      {/* Left Channel */}
      <Box
        sx={{
          ...(isVertical
            ? { width: '4px', height: '100%' }
            : { height: '4px', width: '100%' }),
          backgroundColor: 'rgba(128, 128, 128, 0.25)',
          borderRadius: '0px',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Box
          ref={leftBarRef}
          sx={{
            ...(isVertical
              ? { width: '100%', height: '0%', position: 'absolute', bottom: 0 }
              : { height: '100%', width: '0%' }),
            backgroundColor: color, // Solid single color matching theme
            borderRadius: '0px',
            transition: 'none', // Disable Transitions for real-time responsiveness
          }}
        />
      </Box>

      {/* Right Channel */}
      <Box
        sx={{
          ...(isVertical
            ? { width: '4px', height: '100%' }
            : { height: '4px', width: '100%' }),
          backgroundColor: 'rgba(128, 128, 128, 0.25)',
          borderRadius: '0px',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <Box
          ref={rightBarRef}
          sx={{
            ...(isVertical
              ? { width: '100%', height: '0%', position: 'absolute', bottom: 0 }
              : { height: '100%', width: '0%' }),
            backgroundColor: color, // Solid single color matching theme
            borderRadius: '0px',
            transition: 'none',
          }}
        />
      </Box>
    </Box>
  );
}
