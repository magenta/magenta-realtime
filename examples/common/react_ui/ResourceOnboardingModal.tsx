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
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import { useState, useEffect } from 'react';

const GREEN_ACCENT = '#5be8cc';
const GREEN_HOVER = '#71fade';

interface ResourceProgress {
  status: 'idle' | 'downloading' | 'success' | 'error';
  percent: number;
  text: string;
}

interface ResourceOnboardingModalProps {
  progress: ResourceProgress | null;
  remoteModels: string[];
  downloadPath: string;
  onStartDownload: (modelName: string) => void;
  onSelectFolder: () => void;
  isFetchingModels?: boolean;

}

export function ResourceOnboardingModal({
  progress,
  remoteModels = [],
  downloadPath,
  onStartDownload,
  onSelectFolder,
  isFetchingModels = false,

}: ResourceOnboardingModalProps) {
  const status = progress?.status || 'idle';
  const percent = Math.round((progress?.percent || 0) * 100);
  const text = progress?.text || "Select a model and destination folder to start playing";

  const isSuccess = status === 'success';
  const isError = status === 'error';

  // Default selection to the first available remote model if present.
  // Update selection dynamically when remoteModels finishes fetching.
  const [selectedModel, setSelectedModel] = useState<string>('');
  useEffect(() => {
    if (remoteModels.length > 0 && !selectedModel) {
      setSelectedModel(remoteModels[0]);
    }
  }, [remoteModels, selectedModel]);

  // Default destination folder path displayed in the modal (without the "models" suffix as requested)
  const cleanPath = downloadPath.endsWith('/models') ? downloadPath.slice(0, -7) : downloadPath;
  const friendlyPath = cleanPath.replace(/\/Users\/[^\/]+/, '~');

  const handleStartDownload = () => {
    if (selectedModel) {
      onStartDownload(selectedModel);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: 'var(--color-bg)',
      zIndex: 9999,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      padding: 'var(--app-padding)',
      boxSizing: 'border-box',
      color: 'var(--color-fg)',
      fontFamily: "'Google Sans Text', system-ui, sans-serif",
    }}>
      <div style={{
        background: 'var(--color-surface)',
        border: '1px solid var(--color-border)',
        borderRadius: '10px',
        width: '100%',
        maxWidth: '420px',
        padding: '32px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
      }}>
        {/* Title & Description */}
        <div style={{ textAlign: 'center' }}>
          <div style={{
            fontFamily: "'Google Sans', system-ui, sans-serif",
            fontSize: '18px',
            fontWeight: 700,
            letterSpacing: '0.02em',
            marginBottom: '8px',
            color: 'var(--color-fg)',
          }}>
            {isSuccess
              ? "Onboarding Completed!"
              : isError
              ? "Initialization Failed"
              : "Download Model"}
          </div>
          <div style={{
            fontSize: '13px',
            color: 'var(--color-muted)',
            lineHeight: 1.5,
          }}>
            {isSuccess ? "All base shared resources and your selected model are ready." : text}
          </div>
        </div>

        {status === 'idle' || isError ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>

            {/* Form Part 1: Model Selection Dropdown */}
            <div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '8px',
              }}>
                  <span className="panel-header" style={{ fontSize: '11px', letterSpacing: '0.8px' }}>
                    Select Model
                  </span>
                {isFetchingModels && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div className="magenta-spinner" style={{ color: GREEN_ACCENT }}></div>
                    <span style={{ fontSize: '10px', color: GREEN_ACCENT, fontWeight: 700, letterSpacing: '0.5px' }}>
                      FETCHING…
                    </span>
                  </div>
                )}
              </div>
              <FormControl fullWidth size="small">
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as string)}
                  sx={{
                    color: 'var(--color-fg)',
                    fontFamily: "'Google Sans Text', system-ui, sans-serif",
                    fontSize: '13px',
                    fontWeight: 500,
                    borderRadius: '8px',
                    background: 'var(--color-bg)',
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'var(--color-border)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(255, 255, 255, 0.25)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: GREEN_ACCENT,
                    },
                    '& .MuiSvgIcon-root': {
                      color: 'var(--color-muted)',
                    },
                  }}
                  MenuProps={{
                    disablePortal: true,
                    PaperProps: {
                      sx: {
                        background: 'var(--color-surface)',
                        border: '1px solid var(--color-border)',
                        color: 'var(--color-fg)',
                        borderRadius: '8px',
                        '& .MuiMenuItem-root': {
                          fontFamily: "'Google Sans Text', system-ui, sans-serif",
                          fontSize: '13px',
                          py: 1,
                          '&.Mui-selected': {
                            background: 'rgba(113, 250, 222, 0.12)',
                          },
                          '&:hover': {
                            background: 'rgba(255, 255, 255, 0.05)',
                          },
                        },
                      },
                    },
                  }}
                >
                  {remoteModels.length === 0 && (
                    <MenuItem disabled value="">Fetching available models...</MenuItem>
                  )}
                  {remoteModels.map((m) => (
                    <MenuItem key={m} value={m}>{m}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </div>

            {/* Form Part 2: Destination Folder Row */}
            <div>
              <span className="panel-header" style={{
                fontSize: '11px',
                letterSpacing: '0.8px',
                display: 'block',
                marginBottom: '8px',
              }}>
                Destination Folder
              </span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <div style={{
                  flex: 1,
                  background: 'var(--color-bg)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  fontSize: '13px',
                  fontWeight: 500,
                  color: 'var(--color-fg)',
                  fontFamily: "'Google Sans Text', system-ui, sans-serif",
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  display: 'flex',
                  alignItems: 'center',
                  lineHeight: '24px',
                }}>
                  {friendlyPath}
                </div>
                <button
                  onClick={onSelectFolder}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    color: 'rgba(255, 255, 255, 0.7)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '8px',
                    fontSize: '11px',
                    fontWeight: 500,
                    fontFamily: "'Google Sans', system-ui, sans-serif",
                    textTransform: 'uppercase',
                    letterSpacing: '0.48px',
                    padding: '8px 14px',
                    background: 'none',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    lineHeight: '24px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.25)';
                    e.currentTarget.style.color = 'var(--color-fg)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--color-border)';
                    e.currentTarget.style.color = 'rgba(255, 255, 255, 0.7)';
                  }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>folder_open</span>
                  Browse…
                </button>
              </div>
            </div>

            {/* Form Part 3: Start Download Button */}
            <button
              onClick={handleStartDownload}
              disabled={!selectedModel}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                background: selectedModel ? GREEN_ACCENT : 'var(--color-raised)',
                color: selectedModel ? '#121214' : 'var(--color-muted)',
                borderRadius: '8px',
                fontWeight: 700,
                fontSize: '13px',
                fontFamily: "'Google Sans', system-ui, sans-serif",
                letterSpacing: '0.3px',
                padding: '14px 20px',
                border: 'none',
                cursor: selectedModel ? 'pointer' : 'default',
                transition: 'background 0.15s ease, color 0.15s ease',
                marginTop: '4px',
              }}
              onMouseEnter={(e) => {
                if (selectedModel) {
                  e.currentTarget.style.background = GREEN_HOVER;
                }
              }}
              onMouseLeave={(e) => {
                if (selectedModel) {
                  e.currentTarget.style.background = GREEN_ACCENT;
                }
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>cloud_download</span>
              Start Download
            </button>

            {/* Link to select an existing folder */}
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontSize: '11px', color: 'var(--color-muted)' }}>
                – or –
              </span>
              <br />
              <button
                onClick={onSelectFolder}
                style={{
                  color: GREEN_ACCENT,
                  fontSize: '12px',
                  fontWeight: 500,
                  fontFamily: "'Google Sans Text', system-ui, sans-serif",
                  textDecoration: 'underline',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '4px 0',
                  marginTop: '2px',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.color = GREEN_HOVER; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = GREEN_ACCENT; }}
              >
                select an existing model folder
              </button>
            </div>
          </div>
        ) : (
          /* Downloading / Success state */
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="panel-header" style={{ fontSize: '11px', letterSpacing: '0.8px' }}>
                {isSuccess ? "Ready to generate" : "Installing components"}
              </span>
              {!isSuccess && (
                <span style={{
                  fontSize: '12px',
                  fontWeight: 700,
                  color: GREEN_ACCENT,
                  fontFamily: "'Google Sans Text', monospace",
                }}>
                  {percent}%
                </span>
              )}
            </div>
            {/* Progress bar */}
            <div style={{
              width: '100%',
              height: '6px',
              borderRadius: '3px',
              background: 'var(--color-raised)',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${isSuccess ? 100 : percent}%`,
                height: '100%',
                borderRadius: '3px',
                background: GREEN_ACCENT,
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
