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

import { Settings } from './Settings';
import IconButton from '@mui/material/IconButton';

declare const __COMMIT_HASH__: string;

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  temperature: number;
  topk: number;
  cfgnotes: number;
  cfgmusiccoca: number;
  cfgdrums: number;
  unmaskwidth: number;
  onParamChange: (address: number, value: number) => void;
  onResetDefaults: () => void;
  showNoteCfg?: boolean;
  showPromptCfg?: boolean;
  showDrumsCfg?: boolean;
  showUnmaskWidth?: boolean;
  showOnsetMode?: boolean;
  onsetmode?: boolean;
  columns?: number;
  drumless?: boolean;
  showDrumless?: boolean;
}

export function SettingsPanel({
  open,
  onClose,
  columns = 2,
  ...settingsProps
}: SettingsPanelProps) {
  return (
    <>
      {/* Backdrop */}
      <div
        className={`settings-backdrop${open ? ' open' : ''}`}
        onClick={onClose}
      />
      {/* Panel */}
      <div
        className={`settings-panel${open ? ' open' : ''}`}
        style={{ display: 'flex', flexDirection: 'column' }}
      >
        {/* Header */}
        <div className="app-header-bar" style={{ justifyContent: 'space-between', flexShrink: 0 }}>
          <span style={{
            color: '#FFF',
            fontFamily: '"Google Sans"',
            fontSize: '16px',
            fontWeight: 500,
            letterSpacing: '0.96px',
            textTransform: 'uppercase' as const,
          }}>
            SETTINGS
          </span>
          <IconButton
            onClick={onClose}
            variant="ghost"
            sx={{
              width: 40,
              height: 40,
            }}
          >
            <span className="material-icons" style={{ fontSize: '20px' }}>close</span>
          </IconButton>
        </div>
        <div style={{ padding: '0 var(--app-padding) var(--app-padding)', flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Settings {...settingsProps} columns={columns} />
          {/* Footer showing build git commit hash */}
          <div style={{
            marginTop: 'auto',
            paddingTop: '20px',
            textAlign: 'center',
            fontSize: '10px',
            opacity: 0.25,
            fontFamily: "'Google Sans Text', system-ui, sans-serif",
            letterSpacing: '0.5px',
            color: '#FFF',
            userSelect: 'none',
          }}>
            Build: {__COMMIT_HASH__}
          </div>
        </div>
      </div>
    </>
  );
}
