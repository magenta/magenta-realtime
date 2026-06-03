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

const THUMB_HEIGHT = 30;
const THUMB_WIDTH = 64;
const TRACK_WIDTH = 40;
const TICK_SPACING = 24;
const TICK_WIDTH = 16;
const TICK_HEIGHT = 2;

const SPRING_EASE = 0.2; // fraction of remaining distance per frame
const SPRING_THRESHOLD = 0.002; // stop when this close to center

interface JamSliderElasticProps {
  label: string;
  // Param A range (e.g., temperature)
  minA: number;
  midA: number;
  maxA: number;
  // Param B range (e.g., topK)
  minB: number;
  midB: number;
  maxB: number;
  // Called with both mapped values during drag and spring-back
  onChange: (valueA: number, valueB: number) => void;
  accentColor?: string;
}

// Map a 0–1 fraction to a value using min/mid/max
// fraction=0 → min, fraction=0.5 → mid, fraction=1 → max
function mapFraction(frac: number, min: number, mid: number, max: number): number {
  if (frac <= 0.5) {
    return min + (mid - min) * (frac / 0.5);
  } else {
    return mid + (max - mid) * ((frac - 0.5) / 0.5);
  }
}

export function JamSliderElastic({
  label,
  minA, midA, maxA,
  minB, midB, maxB,
  onChange,
  accentColor = '#FFF',
}: JamSliderElasticProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [fraction, setFraction] = useState(0.5); // 0–1, 0.5 = rest
  const [isDragging, setIsDragging] = useState(false);
  const animRef = useRef<number | null>(null);
  const fractionRef = useRef(0.5);

  // Keep ref in sync for animation callback
  fractionRef.current = fraction;

  const emitValues = useCallback((frac: number) => {
    const a = mapFraction(frac, minA, midA, maxA);
    const b = mapFraction(frac, minB, midB, maxB);
    onChange(a, b);
  }, [minA, midA, maxA, minB, midB, maxB, onChange]);

  const updateFromPointer = useCallback((clientY: number) => {
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const insetTop = THUMB_HEIGHT / 2;
    const insetBottom = THUMB_HEIGHT / 2;
    const usable = rect.height - insetTop - insetBottom;
    const relativeY = clientY - rect.top - insetTop;
    const frac = Math.max(0, Math.min(1, 1 - relativeY / usable));
    setFraction(frac);
    emitValues(frac);
  }, [emitValues]);

  // Spring-back animation
  const startSpringBack = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);

    const animate = () => {
      const current = fractionRef.current;
      const delta = 0.5 - current;
      if (Math.abs(delta) < SPRING_THRESHOLD) {
        setFraction(0.5);
        emitValues(0.5);
        animRef.current = null;
        return;
      }
      const next = current + delta * SPRING_EASE;
      setFraction(next);
      fractionRef.current = next;
      emitValues(next);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
  }, [emitValues]);

  useEffect(() => {
    return () => {
      document.body.classList.remove('dragging-vertical');
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    if (animRef.current) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    setIsDragging(true);
    document.body.classList.add('dragging-vertical');
    updateFromPointer(e.clientY);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging) return;
    updateFromPointer(e.clientY);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    document.body.classList.remove('dragging-vertical');
    startSpringBack();
  };

  // Visual position — fraction maps directly to thumb position
  const percent = fraction * 100;
  const thumbBottomCss = `calc(${percent}% - (${percent / 100} * ${THUMB_HEIGHT}px))`;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      height: '100%',
      width: '100%',
      userSelect: 'none',
      WebkitUserSelect: 'none',
    }}>
      {/* Track container */}
      <div
        ref={trackRef}
        style={{
          position: 'relative',
          width: `${TRACK_WIDTH}px`,
          flex: '1 1 0px',
          minHeight: '30px',
        }}
      >
        {/* Track background */}
        <div style={{
          position: 'absolute',
          inset: 0,
          background: '#1A1A1D',
          borderRadius: '12px',
          pointerEvents: 'none',
          overflow: 'hidden',
        }}>
          {/* Tick marks — translate in lockstep with the thumb */}
          <div style={{
            position: 'absolute',
            top: '-50%',
            bottom: '-50%',
            left: 0,
            right: 0,
            transform: `translateY(calc(${0.5 - fraction} * (50% - ${THUMB_HEIGHT}px)))`,
            backgroundImage: `repeating-linear-gradient(
              to bottom,
              rgba(255, 255, 255, 0.12) 0px,
              rgba(255, 255, 255, 0.12) ${TICK_HEIGHT}px,
              transparent ${TICK_HEIGHT}px,
              transparent ${TICK_SPACING}px
            )`,
            backgroundSize: `${TICK_WIDTH}px ${TICK_SPACING}px`,
            backgroundPosition: 'center',
            backgroundRepeat: 'repeat-y',
          }} />
        </div>

        {/* Vignette overlay — darkens extremes, clear in the middle */}
        <div style={{
          position: 'absolute',
          inset: 0,
          borderRadius: '12px',
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 50%, rgba(0,0,0,0.7) 100%)',
          pointerEvents: 'none',
        }} />

        {/* Thumb */}
        <div style={{
          position: 'absolute',
          bottom: thumbBottomCss,
          left: '50%',
          transform: 'translateX(-50%)',
          width: `${THUMB_WIDTH}px`,
          height: `${THUMB_HEIGHT}px`,
          borderRadius: '10px',
          background: '#36373A',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none',
        }}>
          {/* Accent line */}
          <div style={{
            width: '32px',
            height: '3px',
            borderRadius: '1.5px',
            background: accentColor,
          }} />
        </div>

        {/* Hitbox overlay */}
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
            width: `calc(100% + 24px)`,
            cursor: 'grab',
            touchAction: 'none',
          }}
        />

        {/* Slender Wedge Triangle — floats right, doesn't affect layout */}
        <svg
          viewBox="0 0 10 100"
          preserveAspectRatio="none"
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: 'calc(100% + 12px)',
            width: '10px',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          <polygon points="0,0 10,0 5,100" fill="rgba(26, 26, 29, 0.25)" />
        </svg>
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
