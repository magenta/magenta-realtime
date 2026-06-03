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

import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import { VolumeControl } from './VolumeControl';
import { Replay } from '@mui/icons-material';

interface TransportControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  volume: number;
  onVolumeChange: (v: number) => void;
  onReset: () => void;
  onResetDown?: () => void;
  onResetUp?: () => void;
  volumeSliderPosition?: 'top' | 'bottom';
  model?: string;
  resetTooltip?: string;
  showPlay?: boolean;
  showVolume?: boolean;
  isDawPlaying?: boolean;
}

export function TransportControls({
  isPlaying,
  onTogglePlay,
  volume,
  onVolumeChange,
  onReset,
  onResetDown,
  onResetUp,
  volumeSliderPosition = 'top',
  model,
  resetTooltip = 'Reset model state',
  showPlay = true,
  showVolume = true,
  isDawPlaying = false,
}: TransportControlsProps) {
  const noModel = !model || model === 'No model loaded';
  const playButton = (
    <button
      onClick={noModel ? undefined : onTogglePlay}
      disabled={noModel}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        border: 'none',
        background: isDawPlaying ? '#FF7A00' : '#FFF',
        color: '#000',
        padding: 0,
        flexShrink: 0,
        opacity: noModel ? 0.4 : 1,
        animation: isDawPlaying ? 'magenta-pulse 2s infinite ease-in-out' : 'none',
      }}
    >
      <span className="material-icons" style={{ fontSize: '28px' }}>
        {isDawPlaying ? 'cable' : (isPlaying ? 'pause' : 'play_arrow')}
      </span>
    </button>
  );

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    }}>
      {/* Reset */}
      <Tooltip title={resetTooltip}>
        <IconButton
          onClick={onReset}
          onMouseDown={onResetDown}
          onMouseUp={onResetUp}
          onMouseLeave={onResetUp}
          sx={{
            width: 40,
            height: 40,
          }}
        >
          <Replay sx={{ fontSize: 20 }} />
        </IconButton>
      </Tooltip>

      {/* Play/Pause — large circle */}
      {showPlay && (noModel ? (
        <Tooltip title="No model selected" placement="top">
          <span>{playButton}</span>
        </Tooltip>
      ) : isDawPlaying ? (
        <Tooltip title="Linked to DAW" placement="top">
          <span>{playButton}</span>
        </Tooltip>
      ) : (
        playButton
      ))}


      {/* Volume */}
      {showVolume && (
        <VolumeControl
          volume={volume}
          onVolumeChange={onVolumeChange}
          sliderPosition={volumeSliderPosition}
        />
      )}
    </div>
  );
}
