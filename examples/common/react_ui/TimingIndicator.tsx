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
import { useState, useEffect, useRef } from 'react';
import { AlertCircle } from 'lucide-react';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import { styled, keyframes } from '@mui/material/styles';
import { MagentaDropdown } from './MagentaDropdown';

const pulse = keyframes`
  0% {
    opacity: 1;
    text-shadow: 0 0 4px rgba(239, 68, 68, 0.5);
  }
  50% {
    opacity: 0.4;
    text-shadow: none;
  }
  100% {
    opacity: 1;
    text-shadow: 0 0 4px rgba(239, 68, 68, 0.5);
  }
`;

const WarningIconWrapper = styled('span')({
  display: 'inline-flex',
  alignItems: 'center',
  marginLeft: '4px',
  cursor: 'pointer',
  verticalAlign: 'middle',
  color: '#ef4444',
  transform: 'translateY(-1px)',
});

const SAMPLE_RATE = 48000;

const BUFFER_OPTIONS = [
  { value: 0, samples: 2048 },
  { value: 1, samples: 4096 },
  { value: 2, samples: 8192 },
];

const samplesToMsString = (samples: number) => {
  const ms = Math.round((samples * 1000) / SAMPLE_RATE);
  return `${ms} ms`;
};

/** How long to keep the warning and tooltip visible after an inference spike (in playback milliseconds) */
const WARNING_HOLD_MS = 5000;

/** How long to suppress subsequent warnings after the user manually changes their buffer size (in playback milliseconds) */
const COOLDOWN_MS = 5000;

/** Accumulated high-frame time within a window needed to trigger the warning (in playback milliseconds) */
const SUSTAINED_WARNING_MS = 3000;

interface TimingIndicatorProps {
  frameMs: number;
  /** Cumulative dropped-frame (ring-buffer underrun) count from the engine */
  droppedFrames?: number;
  buffersize?: number;
  onBufferChange?: (value: number) => void;
  /** Extra sx passed through to MagentaDropdown's trigger button */
  buttonSx?: Record<string, any>;
  /** Whether the audio engine is currently playing */
  isPlaying?: boolean;
  /** Customizable label for the buffer option */
  bufferLabel?: string;
  /** Whether to stack frame and buffer readouts vertically (default: false) */
  stacked?: boolean;
}

export function TimingIndicator({
  frameMs,
  droppedFrames = 0,
  buffersize,
  onBufferChange,
  buttonSx = {},
  isPlaying = true,
  bufferLabel = 'buffer size',
  stacked = false,
}: TimingIndicatorProps) {
  const [warningVisible, setWarningVisible] = useState(false);
  const [isCooldownActive, setIsCooldownActive] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);

  const prevDroppedFramesRef = useRef(droppedFrames);
  const dropoutRunStartRef = useRef<number | null>(null);
  const lastDropoutTimeRef = useRef(0);
  const warningHoldTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const cooldownTimerRef = useRef<ReturnType<typeof setTimeout>>();

  // Native metrics update at ~5Hz. If no new dropout arrives within
  // this window, the current run of sustained dropouts is considered over.
  const DROPOUT_GRACE_MS = 400;

  // React to dropout count changes — no rAF needed
  useEffect(() => {
    if (droppedFrames <= prevDroppedFramesRef.current) return;
    prevDroppedFramesRef.current = droppedFrames;

    if (!isPlaying || isCooldownActive) return;

    const now = Date.now();

    // Start a new run if this is the first dropout, or if the gap since
    // the last one exceeded the grace period (i.e. dropouts had stopped)
    if (dropoutRunStartRef.current === null ||
        now - lastDropoutTimeRef.current > DROPOUT_GRACE_MS) {
      dropoutRunStartRef.current = now;
    }
    lastDropoutTimeRef.current = now;

    const runDuration = now - dropoutRunStartRef.current;
    console.log(`[TimingIndicator] Dropout! Total: ${droppedFrames}. Run: ${runDuration}ms / ${SUSTAINED_WARNING_MS}ms`);

    if (runDuration >= SUSTAINED_WARNING_MS && !warningVisible) {
      console.log(`[TimingIndicator] 🔥 WARNING TRIGGERED!`);
      setWarningVisible(true);
      dropoutRunStartRef.current = null;

      clearTimeout(warningHoldTimerRef.current);
      warningHoldTimerRef.current = setTimeout(() => {
        setWarningVisible(false);
        console.log(`[TimingIndicator] Warning hold expired.`);
      }, WARNING_HOLD_MS);
    }
  }, [droppedFrames, isPlaying, isCooldownActive, warningVisible]);

  // Reset state when playback stops
  useEffect(() => {
    if (!isPlaying) {
      dropoutRunStartRef.current = null;
      setWarningVisible(false);
      clearTimeout(warningHoldTimerRef.current);
    }
  }, [isPlaying]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      clearTimeout(warningHoldTimerRef.current);
      clearTimeout(cooldownTimerRef.current);
    };
  }, []);

  const showWarning = warningVisible && !isCooldownActive;
  const showTooltip = showWarning && !isDropdownOpen;

  useEffect(() => {
    console.log(`[TimingIndicator] showWarning transitioned to: ${showWarning} (warningVisible: ${warningVisible}, isCooldownActive: ${isCooldownActive})`);
  }, [showWarning, warningVisible, isCooldownActive]);

  const currentBufferOpt = BUFFER_OPTIONS.find(opt => opt.value === buffersize);
  const currentBufferLabel = currentBufferOpt ? samplesToMsString(currentBufferOpt.samples) : '43 ms';

  const handleBufferChange = (value: number) => {
    if (onBufferChange) {
      onBufferChange(value);
      setWarningVisible(false);
      setIsCooldownActive(true);
      clearTimeout(cooldownTimerRef.current);
      cooldownTimerRef.current = setTimeout(() => {
        setIsCooldownActive(false);
      }, COOLDOWN_MS);
    }
  };

  if (frameMs <= 0) return null;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: stacked ? 'column' : 'row',
        gap: stacked ? '2px' : '6px',
        alignItems: stacked ? 'flex-start' : 'baseline',
        userSelect: 'none',
      }}
    >
      {/* Left side: Latency timing */}
      <span
        style={{
          fontFamily: '"Google Sans Text", system-ui, sans-serif',
          fontSize: '11px',
          fontWeight: 500,
          letterSpacing: '0.5px',
          textTransform: 'uppercase',
          color: showWarning ? '#ef4444' : 'inherit',
          opacity: showWarning ? 1 : 0.70,
          animation: showWarning ? `${pulse} 1.5s infinite ease-in-out` : 'none',
          transition: 'color 0.3s ease',
          display: 'inline-block',
          verticalAlign: 'baseline',
          marginLeft: stacked ? '12px' : '0px',
          paddingRight: stacked ? '2.2ch' : '0px',
        }}
      >
        FRAME: <span style={{ display: 'inline-block',
          width: stacked ? '3.6ch' : '2.8ch',
          lineHeight: 1.3, verticalAlign: 'baseline', textAlign: 'right', paddingRight: '2px' }}>{Math.round((frameMs / 40.0) * 100)}</span>%
        <Tooltip
            open={showTooltip}
            title="Inference time is high. Increase the buffer size to prevent audio dropouts."
            arrow
            placement="top"
            enterTouchDelay={0}
            leaveTouchDelay={1500}
            slotProps={{
              tooltip: {
                sx: {
                  border: '1px solid #374151',
                  maxWidth: '220px',
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                },
              },
            }}
          >
            <WarningIconWrapper style={{ opacity: showWarning ? 1 : 0, pointerEvents: showWarning ? 'auto' : 'none' }}>
              <AlertCircle style={{ width: '13px', height: '13px' }} />
            </WarningIconWrapper>
          </Tooltip>
      </span>
      {/* Right side: Buffer size dropdown */}
      {onBufferChange && (
        <MagentaDropdown
          id="buffer-size-button"
          onOpenChange={setIsDropdownOpen}
          label={<>{bufferLabel}: <span style={{ display: 'inline-block', width: '6.5ch', textAlign: 'right' }}>{currentBufferLabel}</span></>}
          buttonSx={{
            px: 1.5,
            fontFamily: '"Google Sans Text", system-ui, sans-serif',
            fontSize: '11px',
            fontWeight: 500,
            letterSpacing: '0.5px',
            textTransform: 'uppercase',
            color: showWarning ? '#ef4444' : 'inherit',
            opacity: showWarning ? 1 : 0.70,
            animation: showWarning ? `${pulse} 1.5s infinite ease-in-out` : 'none',
            transition: 'color 0.3s ease',
            '& .MuiButton-endIcon': {
              marginLeft: '3px',
            },
            ...buttonSx,
          }}
          menuSx={{ minWidth: '100px' }}
        >
          {BUFFER_OPTIONS.map((opt) => (
            <MenuItem
              key={opt.value}
              selected={buffersize === opt.value}
              onClick={() => handleBufferChange(opt.value)}
            >
              {samplesToMsString(opt.samples)}
            </MenuItem>
          ))}
        </MagentaDropdown>
      )}
    </div>
  );
}
