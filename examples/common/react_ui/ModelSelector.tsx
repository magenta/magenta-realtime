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
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import { CloudDownload, Trash2, Check } from 'lucide-react';
import { MagentaDropdown } from './MagentaDropdown';
import './theme';

interface ModelSelectorProps {
  modelName: string;
  localModels: string[];
  onSelectModel?: (name: string) => void;
  onSelectFolder?: () => void;
  remoteModels?: string[];
  onDownloadModel?: (name: string) => void;
  onDeleteModel?: (name: string) => void;
  downloadProgress?: { status: string; percent: number; text: string; modelName?: string } | null;

  /** Extra sx passed through to MagentaDropdown's trigger button */
  buttonSx?: Record<string, any>;
}

const defaultPost = (msg: any) => {
  // @ts-ignore
  window.webkit?.messageHandlers?.auHost?.postMessage(msg);
};

const stripMlxfn = (s: string) => s.endsWith('.mlxfn') ? s.replace('.mlxfn', '') : s;

export function ModelSelector({
  modelName,
  localModels = [],
  onSelectModel,
  onSelectFolder,
  remoteModels = [],
  onDownloadModel,
  onDeleteModel,
  downloadProgress = null,

  buttonSx = {},
}: ModelSelectorProps) {
  const resolvedSelectModel = onSelectModel || ((name: string) => defaultPost({ type: 'selectModel', name }));
  const selectFolder = onSelectFolder || (() => defaultPost({ type: 'selectDownloadFolder' }));

  const selectVal = modelName.endsWith(".mlxfn") ? modelName.replace(".mlxfn", "") : modelName;
  const isDownloading = !!(downloadProgress && downloadProgress.status === 'downloading');

  // The native host now sends the model folder name directly in the progress payload
  const downloadingModelName = isDownloading ? (downloadProgress.modelName || "") : "";

  const isNoModel = !selectVal || selectVal === "No model loaded" || selectVal === "Choose Model...";

  // Auto-select a model when a download completes and no model is currently loaded
  const prevDownloadingRef = useRef<string | null>(null);
  useEffect(() => {
    if (isDownloading && downloadingModelName) {
      // Remember which model is being downloaded
      prevDownloadingRef.current = downloadingModelName;
    } else if (!isDownloading && prevDownloadingRef.current) {
      // Download just finished — auto-select if no model is loaded
      const justDownloaded = prevDownloadingRef.current;
      prevDownloadingRef.current = null;
      if (isNoModel && localModels.includes(justDownloaded)) {
        resolvedSelectModel(justDownloaded);
      }
    }
  }, [isDownloading]);
  const buttonText = isDownloading
    ? `Downloading... (${Math.round((downloadProgress?.percent || 0) * 100)}%)`
    : (isNoModel ? "Select model…" : selectVal);

  // Compile a unified list of unique model names (removing duplicate listings)
  const allModelNames = Array.from(new Set([...localModels, ...remoteModels]));

  const labelContent = (
    <span style={{
      maxWidth: '90px',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      display: 'block',
      color: (!isDownloading && isNoModel) ? 'red' : 'inherit'
    }}>
      {buttonText}
    </span>
  );

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', verticalAlign: 'baseline' }}>
      <span
        style={{
          fontFamily: "'Google Sans', system-ui, sans-serif",
          fontSize: '12px',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          opacity: 0.7,
          marginRight: '2px',
          userSelect: 'none',
        }}
      >
        MODEL
      </span>
      <MagentaDropdown
        id="model-selector-button"
        label={labelContent}
        endIcon={isDownloading ? <div className="magenta-spinner" /> : undefined}
        buttonSx={buttonSx}
      >
        {allModelNames.length === 0 && (
          <MenuItem disabled sx={{ opacity: 0.5 }}>
            No models found
          </MenuItem>
        )}

        {allModelNames.map((m) => {
          const isLocal = localModels.includes(m);
          const isSelected = stripMlxfn(m) === stripMlxfn(modelName);
          const isThisDownloading = isDownloading && downloadingModelName === m;

          return (
            <MenuItem
              key={m}
              selected={isSelected}
              onClick={(e) => {
                if (isLocal) {
                  resolvedSelectModel(m);
                } else {
                  e.stopPropagation(); // Keep menu open, block selects and closes
                }
              }}
              disabled={isThisDownloading}
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: isLocal ? 'pointer' : 'default',
                ...(!isLocal ? {
                  '&:hover': {
                    backgroundColor: 'transparent !important',
                  }
                } : {})
              }}
            >
              <span style={{
                fontWeight: isSelected ? 'bold' : 'normal',
                opacity: isLocal || isThisDownloading ? 1.0 : 0.6,
              }}>
                {m.endsWith(".mlxfn") ? m.replace(".mlxfn", "") : m}
              </span>

              {/* Context Action Icon Block — fixed width so the panel doesn't shift */}
              <div style={{ width: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginLeft: '12px', marginRight: '-8px', flexShrink: 0 }}>
                {isThisDownloading ? (
                  <div className="magenta-spinner" style={{ color: '#fff' }} />
                ) : isLocal ? (
                  isSelected ? (
                    <Check size={14} style={{ color: '#10b981' }} />
                  ) : (
                    <IconButton
                      title="Delete local model files"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation(); // prevent select action
                        if (onDeleteModel) {
                          onDeleteModel(m);
                        }
                      }}
                      sx={{
                        '&:hover': {
                          backgroundColor: 'rgba(239, 68, 68, 0.15)',
                          color: '#ef4444',
                        },
                      }}
                    >
                      <Trash2 size={16} />
                    </IconButton>
                  )
                ) : (
                  <IconButton
                    title="Download model"
                    disabled={isDownloading}
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation(); // prevent select action
                      if (onDownloadModel) {
                        onDownloadModel(m);
                      }
                    }}
                    sx={{
                      color: '#fff',
                      '&:hover': {
                        color: '#fff',
                        backgroundColor: 'rgba(255, 255, 255, 0.08)',
                      },
                    }}
                  >
                    <CloudDownload size={16} />
                  </IconButton>
                )}
              </div>
            </MenuItem>
          );
        })}

        <MenuItem
          onClick={() => {
            selectFolder();
          }}
          sx={{
            fontWeight: 'bold',
            borderTop: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          Select custom folder&hellip;
        </MenuItem>
      </MagentaDropdown>
    </div>
  );
}
