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
import { Knob } from './Knob';
import { MagentaToggle } from './MagentaToggle';
import Button from '@mui/material/Button';

// CFG knobs: native range 0–5
const CFG_MIN = 0;
const CFG_MAX = 5;

interface SettingsProps {
  // Parameters values
  temperature: number;
  topk: number;
  cfgnotes: number;
  cfgmusiccoca: number;
  cfgdrums: number;
  unmaskwidth: number;
  // Callback when a parameter is changed
  onParamChange: (address: number, value: number) => void;
  // Callback to reset defaults
  onResetDefaults: () => void;
  // Number of knob columns (default: 3)
  columns?: number;
  showNoteCfg?: boolean;
  showPromptCfg?: boolean;
  showDrumsCfg?: boolean;
  showUnmaskWidth?: boolean;
  // Toggle controls (default: false — opt-in per host)
  showMute?: boolean;
  mute?: boolean;
  showBypass?: boolean;
  bypass?: boolean;
  showDelayComp?: boolean;
  latencycomp?: boolean;
  showMidiGate?: boolean;
  midigate?: boolean;
  showOnsetMode?: boolean;
  onsetmode?: boolean;
  drumless?: boolean;
  showDrumless?: boolean;
  labelFontSize?: number;
  knobGap?: number;
  knobSize?: number;
}

export function Settings({
  temperature,
  topk,
  cfgnotes,
  cfgmusiccoca,
  cfgdrums,
  unmaskwidth,
  onParamChange,
  onResetDefaults,
  columns = 3,
  showNoteCfg = true,
  showPromptCfg = true,
  showDrumsCfg = true,
  showUnmaskWidth = true,
  showMute = false,
  mute = false,
  showBypass = false,
  bypass = false,
  showDelayComp = false,
  latencycomp = false,
  showMidiGate = false,
  midigate = false,
  showOnsetMode = false,
  onsetmode = false,
  drumless = false,
  showDrumless = true,
  labelFontSize = 14,
  knobGap = 36,
  knobSize = 64,
}: SettingsProps) {
  const knobWidth = `calc((100% - ${(columns - 1) * knobGap}px) / ${columns})`;

  // Build left and right toggle columns directly
  type Toggle = { label: string; checked: boolean; addr: number; onValue: number; offValue: number; tooltip?: string };
  const leftToggles: Toggle[] = [];
  const rightToggles: Toggle[] = [];

  if (showUnmaskWidth) {
    leftToggles.push({
      label: 'Solo',
      checked: unmaskwidth === 127,
      addr: 7,
      onValue: 127,
      offValue: 4,
      tooltip: 'Encourages the model to only play the input notes, and not add accompaniment.',
    });
  }

  if (showDrumless) {
    leftToggles.push({
      label: 'No Drums',
      checked: drumless,
      addr: 39,
      onValue: 1,
      offValue: 0,
      tooltip: 'Encourages the model to not play drums.',
    });
  }

  if (showOnsetMode) {
    leftToggles.push({
      label: 'Auto-Strum',
      checked: !onsetmode,
      addr: 46,
      onValue: 0,
      offValue: 1,
      tooltip: 'Allows the model to continuously retrigger (e.g. strum, bow, or arpeggiate) when notes are held.',
    });
  }

  // Right column: MIDI Gate first, then Mute, Bypass, Delay Comp
  if (showMidiGate)  rightToggles.push({
    label: 'MIDI Gate',
    checked: midigate,
    addr: 45,
    onValue: 1,
    offValue: 0,
    tooltip: 'Gates the output so the model only makes sound when keys are pressed. When enabled, the plugin will mute when you release all notes.'
  });
  if (showMute)      rightToggles.push({ label: 'Mute',       checked: mute,        addr: 6,  onValue: 1, offValue: 0 });
  if (showBypass)    rightToggles.push({ label: 'Bypass',     checked: bypass,      addr: 32, onValue: 1, offValue: 0 });
  if (showDelayComp) rightToggles.push({
    label: 'Delay Comp',
    checked: latencycomp,
    addr: 9,
    onValue: 1,
    offValue: 0,
    tooltip: "Reports the plugin's internal buffering latency to your DAW. When enabled, your host DAW will automatically shift all other project tracks to keep the AI's generation in perfect sync with the grid."
  });

  const hasToggles = leftToggles.length > 0 || rightToggles.length > 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', color: '#FFF', height: '100%', flex: 1 }}>
      {/* Vertically centering container for the knobs grid */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {/* Knobs Grid */}
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: `${knobGap}px`,
          justifyContent: 'center',

        }}>
          <div style={{ width: knobWidth }}>
            <Knob
              label="Temperature"
              tooltip="Scales the unpredictability of the generated music. Lower values keep the output focused and conservative, while higher values make it more adventurous"
              value={temperature} min={0} max={3} step={0.01} onChange={(v) => onParamChange(0, v)}
              labelFontSize={labelFontSize}
              size={knobSize}
            />
          </div>
          <div style={{ width: knobWidth }}>
            <Knob
              label="Top-K Sampling"
              tooltip="Restricts the model to choosing from the 'K' most likely next audio tokens. Lower numbers keep the music safe and predictable; higher numbers allow for more unexpected, diverse choices."
              value={topk} min={1} max={1024} step={1} onChange={(v) => onParamChange(1, v)}
              labelFontSize={labelFontSize}
              size={knobSize}
            />
          </div>
          {showPromptCfg && (
            <div style={{ width: knobWidth }}>
              <Knob
                label="Prompt Strength"
                tooltip="Controls how strongly the model follows your style prompts. Higher values stick closely to the prompt but may reduce audio quality, while lower values prioritize musicality over strict accuracy."
                value={cfgmusiccoca} min={CFG_MIN} max={CFG_MAX} step={0.1} onChange={(v) => onParamChange(3, v)}
                labelFontSize={labelFontSize}
                size={knobSize}
              />
            </div>
          )}
          {showNoteCfg && (
            <div style={{ width: knobWidth }}>
              <Knob
                label="Note Strength"
                tooltip="Controls how strongly the model adheres to your input notes. Higher values force strict compliance, while lower values allow the model more creative drift."
                value={cfgnotes} min={CFG_MIN} max={CFG_MAX} step={0.1} onChange={(v) => onParamChange(4, v)}
                labelFontSize={labelFontSize}
                size={knobSize}
              />
            </div>
          )}
          {showDrumsCfg && (
            <div style={{ width: knobWidth }}>
              <Knob
                label="Drums Adherence"
                tooltip="Controls how strongly the model adheres to your drum guidelines. Higher values force strict compliance, while lower values allow the model more creative drift"
                value={cfgdrums} min={CFG_MIN} max={CFG_MAX} step={0.1} onChange={(v) => onParamChange(48, v)}
                labelFontSize={labelFontSize}
                size={knobSize}
              />
            </div>
          )}
        </div>
      </div>

      {hasToggles && (
        <div style={{
          marginTop: '18px',
          display: 'flex',
          padding: '18px',
          paddingBottom: '12px',
          gap: '24px',
        }}>
          {/* Left column */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {leftToggles.map((t) => (
              <MagentaToggle
                key={t.label}
                label={t.label}
                checked={t.checked}
                onChange={(v) => {
                  if (t.addr === -1) return;
                  onParamChange(t.addr, v ? t.onValue : t.offValue);
                }}
                tooltip={t.tooltip}
                labelFontSize={labelFontSize}
              />
            ))}
          </div>
          {/* Right column */}
          {rightToggles.length > 0 && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {rightToggles.map((t) => (
                <MagentaToggle
                  key={t.label}
                  label={t.label}
                  checked={t.checked}
                  onChange={(v) => {
                    if (t.addr === -1) return;
                    onParamChange(t.addr, v ? t.onValue : t.offValue);
                  }}
                  tooltip={t.tooltip}
                  labelFontSize={labelFontSize}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Restore Defaults Button */}
      <div style={{ marginTop: 'auto', paddingTop: '16px', display: 'flex', justifyContent: 'center', flexShrink: 0 }}>
        <Button
          onClick={onResetDefaults}
          sx={{
            px: 3,
            py: 0.6,
            borderRadius: '20px',
            fontSize: '11px',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.8px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            color: 'rgba(255, 255, 255, 0.7)',
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#FFF',
              borderColor: 'rgba(255, 255, 255, 0.18)',
            },
          }}
        >
          Restore Defaults
        </Button>
      </div>
    </div>
  );
}
