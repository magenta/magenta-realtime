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

import { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import { ModelSelector, MidiSelector, ResourceOnboardingModal, AudioMeter, Knob, MagentaToggle, ALL_COLORS, TransportControls, TimingIndicator, PromptSurface, calculateWeights, ALL_SUGGESTIONS, DEFAULT_TEMPERATURE, DEFAULT_TOPK, DEFAULT_CFG_MUSICCOCA, DEFAULT_CFG_NOTES, DEFAULT_CFG_DRUMS, DEFAULT_VOLUME, DEFAULT_UNMASK_WIDTH, DEFAULT_BUFFER_SIZE } from '@magenta-rt/common';
import { MagentaSlider } from './components/MagentaSlider';
import { PianoKeyboard } from './components/PianoKeyboard';
import type { PromptNode, ListenerNode, MidiSource } from '@magenta-rt/common';
import IconButton from '@mui/material/IconButton';

import { PromptRow } from './components/PromptRow';
import Tooltip from '@mui/material/Tooltip';
import { InfoOutlined } from '@mui/icons-material';

// ── Native bridge types ──
declare global {
  interface Window {
    __HOST_MODE__?: 'standalone' | 'auv3';
    updateState: (state: any) => void;
    webkit?: {
      messageHandlers?: {
        auHost?: {
          postMessage: (msg: any) => void;
        };
      };
    };
  }
}

const isAUv3 = window.__HOST_MODE__ === 'auv3';

const MAX_PROMPTS = 6;

/** Fisher-Yates shuffle (in place). */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Shuffled copy of ALL_SUGGESTIONS used as a deck. First 2 seed the default
 *  prompts; the lucky button deals from index 2 onward. */
const SHUFFLED_SUGGESTIONS = shuffle([...ALL_SUGGESTIONS]);

// When switching from Surface to Prompts (List) mode, should we sync the surface's IDW weights to the sliders?
const SYNC_SURFACE_WEIGHTS_TO_LIST_ON_SWITCH = false;

// Map AU parameter addresses to state keys
const paramKeyForAddress: Record<number, string> = {
  0: 'temperature', 1: 'topk', 3: 'cfgmusiccoca', 4: 'cfgnotes',
  5: 'volume', 6: 'mute', 7: 'unmaskwidth', 8: 'buffersize', 9: 'latencycomp',
  10: 'weight_0', 11: 'weight_1', 12: 'weight_2',
  13: 'weight_3', 14: 'weight_4', 15: 'weight_5',
  31: 'resetstate', 32: 'bypass',
  39: 'drumless',
  45: 'midigate',
  46: 'onsetmode',
  47: 'seedrotation',
  48: 'cfgdrums',
};

const boolParams = new Set([6, 9, 31, 32, 39, 45, 46]);

// ─── Computer keyboard → MIDI (Ableton Live layout) ──────────────────────────
// Base row (lower octave): A S D F G H J = C D E F G A B, with W E T Y U as
// black keys (C# D# F# G# A#). Upper octave continues on K O L P ; (C C# D D# E).
// Z / X shift the base octave down/up.
const KEY_TO_SEMITONE: Record<string, number> = {
  a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11,
  k: 12, o: 13, l: 14, p: 15, ';': 16,
};
const KEYBOARD_MIDI_BASE_DEFAULT = 48; // C3 in MIDI

const DEFAULT_PARAMS = {
  temperature: DEFAULT_TEMPERATURE,
  topk: DEFAULT_TOPK,
  cfgmusiccoca: DEFAULT_CFG_MUSICCOCA,
  cfgnotes: DEFAULT_CFG_NOTES,
  cfgdrums: DEFAULT_CFG_DRUMS,
  volume: DEFAULT_VOLUME,
  mute: false,
  unmaskwidth: DEFAULT_UNMASK_WIDTH,
  buffersize: DEFAULT_BUFFER_SIZE,
  latencycomp: false,
  weight_0: 0, weight_1: 0, weight_2: 0,
  weight_3: 0, weight_4: 0, weight_5: 0,
  resetstate: false,
  bypass: false,
  seedrotation: 0,
  drumless: false,
  midigate: false,
  onsetmode: false,
};

// Default surface positions (normalised 0–1) — purely frontend state
const DEFAULT_SURFACE_POSITIONS = [
  { x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.5, y: 0.8 },
  { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.5, y: 0.5 },
];
const DEFAULT_CURSOR = { x: 0.5, y: 0.5 };

declare const __COMMIT_HASH__: string;

export default function App() {
  // ── Helpers ──
  function resetToDefaults(addresses: number[]) {
    for (const addr of addresses) {
      const key = paramKeyForAddress[addr] as keyof typeof DEFAULT_PARAMS;
      const val = DEFAULT_PARAMS[key];
      sendParamChange(addr, typeof val === 'boolean' ? (val ? 1 : 0) : val as number);
    }
  }

  // ── State ──
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });

  const [metrics, setMetrics] = useState({
    frameMs: 0,
    bufferAvail: 0,
    bufferCap: 0,
    leftLevel: 0,
    rightLevel: 0,
    droppedFrames: 0,
    transportFlags: -1,
  });

  const [isPlaying, setIsPlaying] = useState(false);

  const [modelName, setModelName] = useState("No model loaded");
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [remoteModels, setRemoteModels] = useState<string[]>([]);
  const [downloadProgress, setDownloadProgress] = useState<any>(null);
  const [downloadPath, setDownloadPath] = useState("~/Documents/Magenta/magenta-rt-v2/models");

  // Onboarding States
  const [resourcesMissing, setResourcesMissing] = useState(false);
  const [resourcesProgress, setResourcesProgress] = useState<any>(null);
  const [isFetchingModels, setIsFetchingModels] = useState(true);


  const [bankStatus, setBankStatus] = useState([false, false, false]);
  const [lastRestoredBank, setLastRestoredBank] = useState<string>('factory');
  const [customPrefillLoaded, setCustomPrefillLoaded] = useState(false);
  const [remotePressedBank, setRemotePressedBank] = useState<string | null>(null);

  // MIDI sources (standalone only — AUv3 gets MIDI from the DAW)
  const [midiSources, setMidiSources] = useState<MidiSource[]>([]);
  const [keyboardMidiEnabled, setKeyboardMidiEnabled] = useState(false);
  const [activeNotes, setActiveNotes] = useState<number[]>([]);
  const keyboardBaseNote = useRef(KEYBOARD_MIDI_BASE_DEFAULT);
  const pressedKeys = useRef<Map<string, number>>(new Map());
  const [octaveOffset, setOctaveOffset] = useState(0);
  const [prompts, setPrompts] = useState<Array<{ text: string; weight: number; isAudio: boolean }>>(() => {
    return SHUFFLED_SUGGESTIONS.slice(0, 2).map(text => ({ text, weight: 1.0, isAudio: false }));
  });

  const promptSurfaceRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState({ w: 340, h: 275 });
  const [selectedBallId, setSelectedBallId] = useState<number | null>(null);
  const [mixMode, setMixMode] = useState<'surface' | 'list'>('list');
  const [newPromptText, setNewPromptText] = useState('');
  /** Index into SHUFFLED_SUGGESTIONS — starts at 2 (first 2 used for defaults). */
  const deckIndexRef = useRef(2);

  // Surface positions are purely local React state (not engine params)
  const [surfacePositions, setSurfacePositions] = useState(DEFAULT_SURFACE_POSITIONS);
  const [cursorPos, setCursorPos] = useState(DEFAULT_CURSOR);
  const initialLoadDone = useRef(false);
  const initialPromptsSynced = useRef(false);

  useLayoutEffect(() => {
    const el = promptSurfaceRef.current;
    if (!el) return;

    const measure = () => {
      const { width, height } = el.getBoundingClientRect();
      if (width > 0 && height > 0) {
        setStageSize({ w: Math.round(width), h: Math.round(height) });
      }
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [mixMode]);

  // Compute activeNodes and listenerNode in pixel coordinates from local state
  const activeNodes: PromptNode[] = prompts.map((p, i) => {
    const pos = surfacePositions[i] ?? { x: 0.5, y: 0.5 };
    return {
      id: i,
      x: pos.x * stageSize.w,
      y: (1.0 - pos.y) * stageSize.h,
      label: p.text,
      colorIndex: i,
      isAudio: p.isAudio,
    };
  });

  const listenerNode: ListenerNode = {
    x: cursorPos.x * stageSize.w,
    y: (1.0 - cursorPos.y) * stageSize.h,
  };

  // Helper: compute IDW weights from current surface positions and send to engine
  const sendSurfaceWeights = (nodes: PromptNode[], listener: ListenerNode) => {
    const weights = calculateWeights(listener, nodes);
    setPrompts(prev => {
      const next = prev.map((p, i) => ({ ...p, weight: weights[i] ?? 0 }));
      postNormalizedPrompts(next);
      return next;
    });
  };

  const handlePromptMove = (id: number, x_pixel: number, y_pixel: number) => {
    const x_norm = x_pixel / stageSize.w;
    const y_norm = 1.0 - (y_pixel / stageSize.h);
    setSurfacePositions(prev => {
      const next = [...prev];
      next[id] = { x: x_norm, y: y_norm };
      return next;
    });
    // Recompute IDW weights with the moved prompt
    const updatedNodes = activeNodes.map((n, i) =>
      i === id ? { ...n, x: x_pixel, y: y_pixel } : n
    );
    sendSurfaceWeights(updatedNodes, listenerNode);
  };

  const handleListenerMove = (x_pixel: number, y_pixel: number) => {
    const x_norm = x_pixel / stageSize.w;
    const y_norm = 1.0 - (y_pixel / stageSize.h);
    setCursorPos({ x: x_norm, y: y_norm });
    // Recompute IDW weights with the moved listener
    const updatedListener = { x: x_pixel, y: y_pixel };
    sendSurfaceWeights(activeNodes, updatedListener);
  };

  const handleNodeAdded = (x_pixel: number, y_pixel: number) => {
    if (prompts.length >= MAX_PROMPTS) return;
    const x_norm = x_pixel / stageSize.w;
    const y_norm = 1.0 - (y_pixel / stageSize.h);
    setSurfacePositions(prev => [...prev, { x: x_norm, y: y_norm }]);
    setPrompts(currentPrompts => {
      const next = [...currentPrompts, { text: "", weight: 1.0, isAudio: false }];
      postNormalizedPrompts(next);
      return next;
    });
  };

  const handleNodeDeleted = (id: number) => {
    setPrompts(currentPrompts => {
      const next = [...currentPrompts];
      next.splice(id, 1);
      postNormalizedPrompts(next);
      return next;
    });
  };

  // ── Helpers ──
  const postMessage = (msg: any) => {
    window.webkit?.messageHandlers?.auHost?.postMessage(msg);
  };

  const sendParamChange = (index: number, value: number) => {
    const key = paramKeyForAddress[index];
    if (key) {
      setParams(p => ({ ...p, [key]: boolParams.has(index) ? value > 0.5 : value }));
    }
    postMessage({ type: 'param', index, value });
  };

  // will need later
  // const handleResetDefaults = () => {
  //   sendParamChange(0, 1.3);   // temperature
  //   sendParamChange(1, 40);    // topk
  //   sendParamChange(3, 3.0);   // cfgmusiccoca
  //   sendParamChange(4, 1.0);   // cfgnotes
  //   sendParamChange(48, 1.0);  // cfgdrums
  //   sendParamChange(5, 0.0);   // volume
  //   sendParamChange(6, 0.0);   // mute
  //   sendParamChange(7, 0);     // unmaskwidth
  //   sendParamChange(8, 0.0);   // buffersize
  //   sendParamChange(9, 0.0);   // latencycomp
  //   sendParamChange(32, 0.0);  // bypass
  //   sendParamChange(39, 0.0);  // drumless
  // };

  const onResetModel = () => {
    sendParamChange(31, 1.0);
    // Edge-triggered: reset back to 0 after a short delay
    setTimeout(() => sendParamChange(31, 0.0), 100);
  };

  const getResetTooltip = () => {
    if (lastRestoredBank === 'factory') {
      return 'Reset to initial model state';
    }
    if (lastRestoredBank === 'silence') {
      return 'Reset to silence';
    }
    if (lastRestoredBank === 'custom') {
      return 'Reset from custom bank';
    }
    if (lastRestoredBank.startsWith('bank')) {
      const num = parseInt(lastRestoredBank.replace('bank', ''), 10) + 1;
      return `Reset from Bank ${num}`;
    }
    return 'Reset model state';
  };

  const postNormalizedPrompts = (promptsList: typeof prompts) => {
    // Send raw weights — the C++ engine normalises at inference time.
    const mapped = promptsList.map(p => ({
      text: p.text, weight: p.weight, isAudio: p.isAudio,
    }));
    postMessage({ type: 'textPrompts', value: mapped });
  };

  const handlePromptTextChange = (idx: number, text: string) => {
    setPrompts(currentPrompts => {
      const next = [...currentPrompts];
      next[idx] = { ...next[idx], text };
      postNormalizedPrompts(next);
      return next;
    });
  };

  const handlePromptWeightChange = (idx: number, weight: number) => {
    setPrompts(currentPrompts => {
      const next = [...currentPrompts];
      next[idx] = { ...next[idx], weight };
      postNormalizedPrompts(next);
      return next;
    });
  };

  const handlePromptRemove = (idx: number) => {
    setPrompts(currentPrompts => {
      const next = [...currentPrompts];
      next.splice(idx, 1);
      postNormalizedPrompts(next);
      return next;
    });
  };

  const handlePromptUpload = (idx: number) => {
    // Tell native host to open file picker and load audio for this prompt index
    postMessage({ type: 'loadAudioPrompt', index: idx });
  };

  const handleClearAudio = (idx: number) => {
    setPrompts(currentPrompts => {
      const next = [...currentPrompts];
      next[idx] = { ...next[idx], text: '', isAudio: false };
      postNormalizedPrompts(next);
      return next;
    });
    // Tell native to clear the audio prompt for this slot
    postMessage({ type: 'clearAudioPrompt', index: idx });
  };

  const handleAddPromptWithText = (text: string) => {
    if (prompts.length >= MAX_PROMPTS) return;
    const rx = 0.2 + Math.random() * 0.6;
    const ry = 0.2 + Math.random() * 0.6;
    setSurfacePositions(prev => [...prev, { x: rx, y: ry }]);
    setPrompts(currentPrompts => {
      const next = [...currentPrompts, { text, weight: 1.0, isAudio: false }];
      postNormalizedPrompts(next);
      return next;
    });
    setNewPromptText('');
  };

  const handleUploadNewPrompt = () => {
    if (prompts.length >= MAX_PROMPTS) return;
    // Add a placeholder entry, then trigger the upload for that new slot
    const nextIdx = prompts.length;
    setPrompts(currentPrompts => {
      const next = [...currentPrompts, { text: '', weight: 1.0, isAudio: false }];
      postNormalizedPrompts(next);
      return next;
    });
    // Small delay to let state propagate to native before opening file picker
    setTimeout(() => handlePromptUpload(nextIdx), 50);
  };

  const togglePlay = () => {
    console.log('togglePlay clicked, posting message');
    postMessage({ type: 'togglePlay' });
  };

  // ── Native bridge: receive state from host ──
  useEffect(() => {
    window.updateState = (state: any) => {
      if (state.params) {
        setParams(p => ({ ...p, ...state.params }));
      }
      // DAW automation changed a weight knob — switch to list mode and
      // update the prompt sliders with the raw automation values.
      if (state.weightAutomation) {
        setMixMode('list');
        const wa = state.weightAutomation;
        setPrompts(prev => prev.map((p, i) => {
          const wKey = `weight_${i}`;
          return wKey in wa ? { ...p, weight: wa[wKey] } : p;
        }));
      }
      if (state.metrics) {
        setMetrics(m => ({ ...m, ...state.metrics }));
      }
      if (state.audioLevels) {
        setMetrics(m => ({ ...m, leftLevel: state.audioLevels.left, rightLevel: state.audioLevels.right }));
      }
      if (state.modelName !== undefined) setModelName(state.modelName);
      if (state.isPlaying !== undefined) setIsPlaying(state.isPlaying);
      if (state.localModels !== undefined) setLocalModels(state.localModels);
      if (state.remoteModels !== undefined) {
        setRemoteModels(state.remoteModels);
        setIsFetchingModels(false);
      }
      if (state.remoteModelsError !== undefined) {
        setIsFetchingModels(false);
      }
      if (state.downloadProgress !== undefined) setDownloadProgress(state.downloadProgress);
      if (state.resourcesMissing !== undefined) setResourcesMissing(state.resourcesMissing);
      if (state.resourcesProgress !== undefined) setResourcesProgress(state.resourcesProgress);
      if (state.downloadPath !== undefined) setDownloadPath(state.downloadPath);

      if (state.bankStatus !== undefined) setBankStatus(state.bankStatus);
      if (state.midiSources !== undefined) setMidiSources(state.midiSources);
      if (state.computerKeyboardMidi !== undefined) setKeyboardMidiEnabled(!!state.computerKeyboardMidi);
      if (state.activeNotes !== undefined) setActiveNotes(state.activeNotes);
      if (state.audioPrefillStatus === 'Success') setCustomPrefillLoaded(true);
      if (state.textPrompts !== undefined) {
        // Filter to only active entries (non-empty text) — dynamic length array
        const active = state.textPrompts
          .map((p: any) => ({ text: p.text || '', weight: p.weight || 0, isAudio: p.isAudio || false }))
          .filter((p: any) => p.text.length > 0);
        setPrompts(active);
      }
      if (state.prompt_surface !== undefined && state.prompt_surface !== null) {
        const ps = state.prompt_surface;
        if (ps.surfacePositions) {
          setSurfacePositions(ps.surfacePositions);
        }
        if (ps.cursorPos) {
          setCursorPos(ps.cursorPos);
        }
      }
      // After the engine connects (signaled by receiving a params snapshot),
      // if the host didn't restore any saved prompts, push React's initial
      // defaults so the engine doesn't stay on its hardcoded "piano" fallback.
      if (!initialPromptsSynced.current && state.params) {
        initialPromptsSynced.current = true;
        if (state.textPrompts === undefined) {
          postNormalizedPrompts(prompts);
        }
      }
      initialLoadDone.current = true;
    };

    // Handle Cmd+A in text inputs (WebView doesn't handle this natively)
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'a') {
        const active = document.activeElement;
        if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement) {
          e.preventDefault();
          active.select();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    // Notify backend that React is ready to receive state
    postMessage({ type: 'uiReady' });
    postMessage({ type: 'listRemoteModels' });
    postMessage({ type: 'checkBanks' });

    return () => {
      // @ts-ignore
      delete window.updateState;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (!initialLoadDone.current) return;
    postMessage({
      type: 'promptSurfaceState',
      value: {
        surfacePositions,
        cursorPos,
      }
    });
  }, [surfacePositions, cursorPos]);

  // ─── Computer keyboard → MIDI ────────────────────────────────────────────

  const handleOctaveDown = useCallback(() => {
    setOctaveOffset(prev => {
      const next = Math.max(-2, prev - 1);
      keyboardBaseNote.current = KEYBOARD_MIDI_BASE_DEFAULT + next * 12;
      return next;
    });
  }, []);

  const handleOctaveUp = useCallback(() => {
    setOctaveOffset(prev => {
      const next = Math.min(4, prev + 1);
      keyboardBaseNote.current = KEYBOARD_MIDI_BASE_DEFAULT + next * 12;
      return next;
    });
  }, []);

  useEffect(() => {
    if (!keyboardMidiEnabled) {
      // Release any still-held notes
      pressedKeys.current.forEach((note) => {
        postMessage({ type: 'kbdNote', note, on: false });
      });
      pressedKeys.current.clear();
      return;
    }

    const handleDown = (e: KeyboardEvent) => {
      if (document.activeElement instanceof HTMLInputElement || document.activeElement instanceof HTMLTextAreaElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === 'z') {
        e.preventDefault();
        if (e.repeat) return;
        handleOctaveDown();
        return;
      }
      if (key === 'x') {
        e.preventDefault();
        if (e.repeat) return;
        handleOctaveUp();
        return;
      }
      const semi = KEY_TO_SEMITONE[key];
      if (semi === undefined) return;
      e.preventDefault();
      if (e.repeat) return;
      if (pressedKeys.current.has(key)) return;
      const note = keyboardBaseNote.current + semi;
      if (note < 0 || note > 127) return;
      pressedKeys.current.set(key, note);
      postMessage({ type: 'kbdNote', note, on: true });
    };

    const handleUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      const note = pressedKeys.current.get(key);
      if (note === undefined) return;
      pressedKeys.current.delete(key);
      postMessage({ type: 'kbdNote', note, on: false });
    };

    // Release held notes when window loses focus (otherwise stuck notes).
    const handleBlur = () => {
      pressedKeys.current.forEach((note) => {
        postMessage({ type: 'kbdNote', note, on: false });
      });
      pressedKeys.current.clear();
    };

    window.addEventListener('keydown', handleDown);
    window.addEventListener('keyup', handleUp);
    window.addEventListener('blur', handleBlur);
    return () => {
      window.removeEventListener('keydown', handleDown);
      window.removeEventListener('keyup', handleUp);
      window.removeEventListener('blur', handleBlur);
      handleBlur();
    };
  }, [keyboardMidiEnabled]);

  const isDawPlaying = metrics.transportFlags >= 0 && (metrics.transportFlags & 2) !== 0;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      width: '100%',
      color: 'var(--color-fg)',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* Tiny ghost bug button in absolute top-right corner of the window */}
      <Tooltip title={`Build: ${__COMMIT_HASH__}`} placement="bottom-end" arrow={false}>
        <div style={{
          position: 'absolute',
          top: '8px',
          right: '8px',
          opacity: 0.1,
          cursor: 'help',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: '13px', color: '#FFF' }}>bug_report</span>
        </div>
      </Tooltip>

      {/* ── Top section: Zone A (left) + Zone B (right) ── */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'row',
        overflow: 'hidden',
        minHeight: 0,
      }}>

        {/* ══════════════════════════════════════════════════════
            ZONE A — Left column
            ══════════════════════════════════════════════════════ */}
        <div style={{
          width: '455px',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}>

          {/* ── A: Header row — View Mode Switcher + ModelSelector ── */}
          <div style={{
            display: 'flex',
            padding: '9px 16px',
            alignItems: 'center',
            gap: '10px',
            flexShrink: 0,
          }}>
            {/* View Mode Switcher */}
            <div style={{
              display: 'flex',
              background: '#202124',
              borderRadius: '6px',
              padding: '3px',
              flexShrink: 0,
            }}>
              <button
                onClick={() => {
                  if (mixMode === 'surface') {
                    if (SYNC_SURFACE_WEIGHTS_TO_LIST_ON_SWITCH) {
                      // Sync IDW weights from surface into the list sliders and push to native
                      const weights = calculateWeights(listenerNode, activeNodes);
                      setPrompts(prev => {
                        const next = [...prev];
                        activeNodes.forEach((node, i) => {
                          next[node.id] = { ...next[node.id], weight: weights[i] };
                        });
                        postNormalizedPrompts(next);
                        return next;
                      });
                    } else {
                      // Push existing list weights to native (sliders stay as-is)
                      postNormalizedPrompts(prompts);
                    }
                  }
                  setMixMode('list');
                }}
                style={{
                  padding: '6px 14px',
                  borderRadius: '4px',
                  fontSize: '14px',
                  fontWeight: 600,
                  fontFamily: "'Google Sans Text', system-ui, sans-serif",
                  background: mixMode === 'list' ? '#e8eaed' : 'transparent',
                  color: mixMode === 'list' ? '#1a1a1d' : '#A3A3A5',
                  transition: 'all 0.15s ease',
                  border: 'none',
                  outline: 'none',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                Prompts
              </button>
              <button
                onClick={() => {
                  if (mixMode !== 'surface') {
                    // Compute IDW weights from current surface positions and push to native
                    const weights = calculateWeights(listenerNode, activeNodes);
                    setPrompts(prev => {
                      const next = prev.map((p, i) => ({ ...p, weight: weights[i] ?? 0 }));
                      postNormalizedPrompts(next);
                      return next;
                    });
                  }
                  setMixMode('surface');
                }}
                style={{
                  padding: '6px 14px',
                  borderRadius: '4px',
                  fontSize: '14px',
                  fontWeight: 600,
                  fontFamily: "'Google Sans Text', system-ui, sans-serif",
                  background: mixMode === 'surface' ? '#e8eaed' : 'transparent',
                  color: mixMode === 'surface' ? '#1a1a1d' : '#A3A3A5',
                  transition: 'all 0.15s ease',
                  border: 'none',
                  outline: 'none',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                Surface
              </button>
            </div>

            {/* ModelSelector */}
            <div style={{ flex: 1, minWidth: 0, display: 'flex', justifyContent: 'flex-end' }}>
              <ModelSelector
                modelName={modelName}
                localModels={localModels}
                remoteModels={remoteModels}
                downloadProgress={downloadProgress}

                onSelectModel={(m) => postMessage({ type: 'selectModel', name: m })}
                onDownloadModel={(m) => postMessage({ type: 'downloadModel', name: m })}
                onDeleteModel={(m) => postMessage({ type: 'deleteModel', name: m })}
                onSelectFolder={() => postMessage({ type: 'selectDownloadFolder' })}
              />
            </div>
          </div>

          {/* ── A: PromptSurface / PromptList area ── */}
          <div style={{
            flex: 1,
            position: 'relative',
            overflow: 'hidden',
            minHeight: 0,
          }}>
            <div style={{
              height: '100%',
              overflow: mixMode === 'list' ? 'auto' : 'hidden',
              position: 'relative',
            }}
            className={mixMode === 'list' ? 'thin-scrollbar' : ''}
            >
              <div style={{ height: mixMode === 'surface' ? '100%' : 'auto' }}>
                {/* ── PromptSurface (always mounted, toggle visibility) ── */}
                <div
                  ref={promptSurfaceRef}
                  style={{
                    width: '100%',
                    height: '100%',
                    background: 'transparent',
                    position: 'relative',
                    display: mixMode === 'surface' ? 'block' : 'none',
                  }}
                >
                  <PromptSurface
                    prompts={activeNodes}
                    listener={listenerNode}
                    selectedBallId={selectedBallId}
                    onPromptMove={handlePromptMove}
                    onListenerMove={handleListenerMove}
                    onBallSelect={setSelectedBallId}
                    onPromptAdd={handleNodeAdded}
                    onPromptTextChange={handlePromptTextChange}
                    onPromptDelete={handleNodeDeleted}
                    physicsSpeed={0}
                    onFirstThrow={() => {}}
                    isPlaying={isPlaying}
                    audioLevel={0}
                    physicsEnabled={false}
                    active={mixMode === 'surface'}
                  />
                </div>

                {/* ── Prompt list (always mounted, toggle visibility) ── */}
                <div style={{
                  padding: '16px 16px',
                  flexShrink: 0,
                  display: mixMode === 'surface' ? 'none' : 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                }}>
                  {prompts.map((p, idx) => (
                    <PromptRow
                      key={idx}
                      text={p.text}
                      color={ALL_COLORS[idx % ALL_COLORS.length]}
                      weight={p.weight}
                      isEmpty={!p.text && !p.isAudio}
                      isAudio={p.isAudio}
                      onTextChange={(newText) => handlePromptTextChange(idx, newText)}
                      onWeightChange={(newWeight) => handlePromptWeightChange(idx, newWeight)}
                      onRemove={() => handlePromptRemove(idx)}
                      onUpload={() => handlePromptUpload(idx)}
                      onClearAudio={() => handleClearAudio(idx)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ── A: Prompt Strength slider ── */}
          <div style={{
            padding: '10px 16px 0',
            flexShrink: 0,
          }}>
            <MagentaSlider
              label="Prompt Strength"
              tooltip="Controls how strongly the model follows your style prompts. Higher values stick closely to the prompt but may reduce audio quality, while lower values prioritize musicality over strict accuracy."
              value={params.cfgmusiccoca} min={0} max={5} step={0.1} onChange={(v) => sendParamChange(3, v)}
            />
          </div>

          {/* ── A: Prompt input bar ── */}
          {(() => {
            if (prompts.length >= MAX_PROMPTS) return null;
            return (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '10px 16px',
                flexShrink: 0,
              }}>
                {/* Input bar container */}
                <div
                  className="prompt-box"
                  style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    padding: '0 4px 0 0',
                  }}
                >
                  <input
                    type="text"
                    placeholder="Type a prompt or upload a sample"
                    value={newPromptText}
                    onChange={(e) => setNewPromptText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && newPromptText.trim()) {
                        handleAddPromptWithText(newPromptText.trim());
                      }
                    }}
                    style={{
                      flex: 1,
                      background: 'transparent',
                      border: 'none',
                      outline: 'none',
                      padding: '14px 16px',
                      fontSize: '13px',
                    }}
                  />
                  <Tooltip title="Random prompt" arrow placement="top">
                    <IconButton
                      onClick={() => {
                        if (deckIndexRef.current >= SHUFFLED_SUGGESTIONS.length) {
                          shuffle(SHUFFLED_SUGGESTIONS);
                          deckIndexRef.current = 0;
                        }
                        setNewPromptText(SHUFFLED_SUGGESTIONS[deckIndexRef.current++]);
                      }}
                      sx={{
                        width: '36px',
                        height: '36px',
                        color: 'var(--color-muted)',
                        backgroundColor: 'transparent',
                        border: 'none',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                          color: '#ffffff',
                        },
                        transition: 'all 0.2s ease-in-out',
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>casino</span>
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Upload audio prompt" arrow placement="top">
                    <IconButton
                      onClick={handleUploadNewPrompt}
                      sx={{
                        width: '36px',
                        height: '36px',
                        color: 'var(--color-muted)',
                        backgroundColor: 'transparent',
                        border: 'none',
                        marginRight: '4px',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                          color: '#ffffff',
                        },
                        transition: 'all 0.2s ease-in-out',
                      }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>upload</span>
                    </IconButton>
                  </Tooltip>
                </div>

                {/* Solid circle plus button */}
                <IconButton
                  onClick={() => {
                    if (newPromptText.trim()) {
                      handleAddPromptWithText(newPromptText.trim());
                    }
                  }}
                  disabled={!newPromptText.trim()}
                  sx={{
                    width: 36,
                    height: 36,
                    background: 'var(--color-raised)',
                    color: '#FFF',
                    flexShrink: 0,
                  }}
                  title="Add prompt"
                >
                  <span className="material-icons" style={{ fontSize: '18px' }}>add</span>
                </IconButton>
              </div>
            );
          })()}
        </div>

        {/* ══════════════════════════════════════════════════════
            ZONE B — Right column
            ══════════════════════════════════════════════════════ */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
          padding: '20px',
          gap: '28px',
        }}>

          {/* ── B: Global Controls ── */}
          <div className="section-box" style={{
            display: 'flex',
            alignItems: 'center',
            padding: '20px 16px',
            gap: '16px',
            flexShrink: 0,
            position: 'relative',
          }}>
            {/* Reset button — top right */}
            <IconButton
              onClick={() => resetToDefaults([0, 1, 39, 9])}
              variant="ghost"
              style={{
                position: 'absolute',
                top: '3px',
                right: '3px',
                padding: '4px',
                opacity: 0.35,
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '16px', color: '#FFF' }}>refresh</span>
            </IconButton>
            <Knob
              label="Temperature"
              tooltip="Scales the unpredictability of the generated music. Lower values keep the output focused and conservative, while higher values make it more adventurous"
              value={params.temperature} min={0} max={3} step={0.01} onChange={(v) => sendParamChange(0, v)}
              size={70}
            />
            <Knob
              label="Top-K Sampling"
              tooltip="Restricts the model to choosing from the 'K' most likely next audio tokens. Lower numbers keep the music safe and predictable; higher numbers allow for more unexpected, diverse choices."
              value={params.topk} min={1} max={1024} step={1} onChange={(v) => sendParamChange(1, v)}
              size={70}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginLeft: '8px' }}>
              <MagentaToggle
                label="No Drums"
                checked={params.drumless}
                onChange={(v) => sendParamChange(39, v ? 1 : 0)}
                tooltip="Encourages the model to not play drums."
              />
              {isAUv3 && (
                <MagentaToggle
                  label="Delay Comp"
                  checked={params.latencycomp}
                  onChange={(v) => sendParamChange(9, v ? 1 : 0)}
                  tooltip="Reports the plugin's internal buffering latency to your DAW. When enabled, your host DAW will automatically shift all other project tracks to keep the AI's generation in perfect sync with the grid."
                />
              )}
            </div>
          </div>

          {/* ── B: Note Controls + Memory Banks ── */}
          <div style={{
            flex: 1,
            display: 'flex',
            gap: '18px',
            minHeight: 0,
          }}>

            {/* ── B: Note Controls ── */}
            <div className="section-box" style={{
              width: '200px',
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'stretch',
              padding: '24px 16px',
              justifyContent: 'space-between',
              position: 'relative',
            }}>
              {/* Header — fieldset-legend style, out of flow */}
              <div style={{
                position: 'absolute',
                top: '-8px',
                left: '14px',
                display: 'flex',
                alignItems: 'center',
                background: 'var(--color-bg)',
                padding: '0 6px',
              }}>
              <span className="section-header" style={{ margin: 0, fontSize: '11px' }}>Note Controls</span>
              </div>
              {/* Reset button — top right */}
              <IconButton
                onClick={() => resetToDefaults([4, 7, 45, 46])}
                variant="ghost"
                style={{
                  position: 'absolute',
                  top: '3px',
                  right: '3px',
                  padding: '4px',
                  opacity: 0.35,
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '16px', color: '#FFF' }}>refresh</span>
              </IconButton>
              <MagentaSlider
                label="Note Strength"
                tooltip="Controls how strongly the model adheres to your input notes. Higher values force strict compliance, while lower values allow the model more creative drift."
                value={params.cfgnotes} min={0} max={5} step={0.1} onChange={(v) => sendParamChange(4, v)}
                stacked
              />
              <MagentaToggle
                label="Solo"
                checked={params.unmaskwidth === 127}
                onChange={(v) => sendParamChange(7, v ? 127 : 4)}
                tooltip="Encourages the model to only play the input notes, and not add accompaniment."
              />
              <MagentaToggle
                label="MIDI Gate"
                checked={params.midigate}
                onChange={(v) => sendParamChange(45, v ? 1 : 0)}
                tooltip="Gates the output so the model only makes sound when keys are pressed. When enabled, the plugin will mute when you release all notes."
              />
              <MagentaToggle
                label="Auto-Strum"
                checked={!params.onsetmode}
                onChange={(v) => sendParamChange(46, v ? 0 : 1)}
                tooltip="Allows the model to continuously retrigger (e.g. strum, bow, or arpeggiate) when notes are held."
              />
            </div>

            {/* ── B: Memory Banks ── */}
            <div className="section-box" style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              padding: '24px 16px',
              minWidth: 0,
              position: 'relative',
            }}>
              {/* Header — fieldset-legend style, out of flow */}
              <div style={{
                position: 'absolute',
                top: '-8px',
                left: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                background: 'var(--color-bg)',
                padding: '0 6px',
              }}>
                <span className="section-header" style={{ margin: 0, fontSize: '11px' }}>Memory Banks</span>
                <Tooltip title="Snapshots to save and restore the model's audio context (up to the last 20s of music). You can save the current context and restore it at will." arrow placement="top">
                  <InfoOutlined style={{ fontSize: '13px', opacity: 0.3, cursor: 'help', color: '#FFF' }} />
                </Tooltip>
              </div>

              {/* Banks grid */}
              <div style={{ display: 'flex', gap: '8px', flex: 1, minHeight: 0 }}>
                {/* Column 1: User Banks */}
                <div className="bank-column">
                  {[0, 1, 2].map((i) => {
                    const filled = bankStatus[i];
                    return (
                      <div
                        key={i}
                        className={`bank-cell-wrapper${remotePressedBank === `bank${i}` ? ' bank-pressed' : ''}`}
                      >
                        <div className={`bank-cell${lastRestoredBank === `bank${i}` ? ' active' : ''}`}>
                          {/* Left/Center: Label */}
                          <div className="bank-cell-label">
                            <span className={`bank-cell-dot${filled ? ' filled' : ''}`} />
                            Bank {i + 1}
                          </div>

                          {/* Right: Actions */}
                          <div className="bank-cell-actions">
                            <IconButton
                              onClick={() => { postMessage({ type: 'saveBank', index: i }); setLastRestoredBank(`bank${i}`); }}
                              variant="ghost"
                            >
                              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>save</span>
                            </IconButton>
                            <IconButton
                              onClick={() => { postMessage({ type: 'loadBank', index: i }); setLastRestoredBank(`bank${i}`); }}
                              disabled={!filled}
                              variant="ghost"
                            >
                              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>replay</span>
                            </IconButton>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Column 2: Factory, Silence, Custom */}
                <div className="bank-column">
                  {/* Factory row */}
                  <div
                    className={`bank-cell-wrapper${remotePressedBank === 'factory' ? ' bank-pressed' : ''}`}
                  >
                    <div className={`bank-cell${lastRestoredBank === 'factory' ? ' active' : ''}`}>
                      <div className="bank-cell-label">
                        <Tooltip title="Reset audio context to the model's initial state" arrow placement="top">
                          <InfoOutlined className="bank-info-icon" />
                        </Tooltip>
                        Empty
                      </div>

                      <div className="bank-cell-actions">
                        <IconButton
                          onClick={() => { postMessage({ type: 'resetToFactory' }); setLastRestoredBank('factory'); }}
                          variant="ghost"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>replay</span>
                        </IconButton>
                      </div>
                    </div>
                  </div>

                  {/* Custom bank row */}
                  <div
                    className={`bank-cell-wrapper${remotePressedBank === 'custom' ? ' bank-pressed' : ''}`}
                  >
                    <div className={`bank-cell${lastRestoredBank === 'custom' ? ' active' : ''}`}>
                      <div className="bank-cell-label">
                        <Tooltip title="Fill the model's audio context with an audio file" arrow placement="top">
                          <InfoOutlined className="bank-info-icon" />
                        </Tooltip>
                        Custom
                        <span className={`bank-cell-dot${customPrefillLoaded ? ' filled' : ''}`} style={{ marginLeft: '8px', marginRight: 0 }} />
                      </div>

                      <div className="bank-cell-actions">
                        <IconButton
                          onClick={() => postMessage({ type: 'audioPrefill' })}
                          variant="ghost"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>upload_file</span>
                        </IconButton>
                        <IconButton
                          onClick={() => { postMessage({ type: 'audioPrefill' }); setLastRestoredBank('custom'); }}
                          disabled={!customPrefillLoaded}
                          variant="ghost"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>replay</span>
                        </IconButton>
                      </div>
                    </div>
                  </div>

                  {/* Silence row */}
                  <div
                    className={`bank-cell-wrapper${remotePressedBank === 'silence' ? ' bank-pressed' : ''}`}
                    style={{ visibility: 'hidden' }}
                  >
                    <div className={`bank-cell${lastRestoredBank === 'silence' ? ' active' : ''}`}>
                      <div className="bank-cell-label">
                        <Tooltip title="Fill the model's audio context with silence" arrow placement="top">
                          <InfoOutlined className="bank-info-icon" />
                        </Tooltip>
                        Silence
                      </div>

                      <div className="bank-cell-actions">
                        <IconButton
                          onClick={() => { postMessage({ type: 'silentPrefill' }); setLastRestoredBank('silence'); }}
                          variant="ghost"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>replay</span>
                        </IconButton>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════
          ZONE C — Bottom bar (full width)
          ══════════════════════════════════════════════════════ */}
      <div style={{
        flexShrink: 0,
        display: 'flex',
        alignItems: 'stretch',
        height: '90px',
        boxSizing: 'border-box',
        padding: '8px 16px',
        background: '#202124',
        gap: '16px',
      }}>
        {/* Left: Transport Controls */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          flex: 1,
          justifyContent: 'flex-start',
          minWidth: 0,
        }}>
          <TransportControls
            isPlaying={isPlaying}
            onTogglePlay={togglePlay}
            volume={params.volume}
            onVolumeChange={(v) => sendParamChange(5, v)}
            onReset={onResetModel}
            onResetDown={() => setRemotePressedBank(lastRestoredBank)}
            onResetUp={() => setRemotePressedBank(null)}
            volumeSliderPosition="top"
            model={modelName}
            resetTooltip={getResetTooltip()}
            showPlay={true}
            showVolume={true}
            isDawPlaying={isDawPlaying}
          />
        </div>

        {/* Center: MIDI (top) + Keyboard (bottom) */}
        <div style={{
          width: '682px',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-start',
        }}>
          {/* Top row: MIDI selector (left) + octave rocker (center) */}
          <div style={{
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'relative',
          }}>
            {!isAUv3 && (
              <MidiSelector
                midiSources={midiSources}
                keyboardMidiEnabled={keyboardMidiEnabled}
                onSelectSource={(endpoint) => postMessage({ type: 'selectMidiSource', endpoint })}
                showComputerKeyboard={true}
                midiActive={activeNotes.length > 0}
              />
            )}
            {/* Octave rocker — QWERTY mode only, absolutely centered */}
            {keyboardMidiEnabled && (
              <div style={{
                position: 'absolute',
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                alignItems: 'center',
                gap: '2px',
              }}>
                <IconButton
                  variant="ghost"
                  onClick={handleOctaveDown}
                  disabled={octaveOffset <= -2}
                  sx={{ width: 24, height: 24 }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>chevron_left</span>
                </IconButton>
                <span style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  minWidth: '24px',
                  textAlign: 'center',
                  fontFamily: "'Google Sans', system-ui, sans-serif",
                  letterSpacing: '0.5px',
                  opacity: 0.7,
                }}>
                  C{Math.floor((KEYBOARD_MIDI_BASE_DEFAULT + octaveOffset * 12) / 12) - 1}
                </span>
                <IconButton
                  variant="ghost"
                  onClick={handleOctaveUp}
                  disabled={octaveOffset >= 4}
                  sx={{ width: 24, height: 24 }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>chevron_right</span>
                </IconButton>
              </div>
            )}
            <div style={{ flexShrink: 0 }} />
          </div>
          {/* Keyboard */}
          <div style={{ flex: 1, marginTop: '6px' }}>
            <PianoKeyboard
              activeNotes={activeNotes}
              accentColor="#71fade"
              startNote={24}
              endNote={96}
              keyboardMidiEnabled={keyboardMidiEnabled}
              onNoteOn={(note) => postMessage({ type: 'kbdNote', note, on: true })}
              onNoteOff={(note) => postMessage({ type: 'kbdNote', note, on: false })}
              showOctaveLabels
              gap={2}
              blackKeyHeight="55%"
              octaveLabelFontSize={9}
              keyboardBaseNote={KEYBOARD_MIDI_BASE_DEFAULT + octaveOffset * 12}
            />
          </div>
        </div>

        {/* Right: Full-height container for Timing Indicator + Audio Meter */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          flexShrink: 0,
          height: '100%',
          flex: 1,
          justifyContent: 'flex-end',
          minWidth: 0,
        }}>
          <div style={{ marginBottom: '-7px' }}>
            <TimingIndicator
              frameMs={metrics.frameMs}
              droppedFrames={metrics.droppedFrames}
              buffersize={params.buffersize}
              onBufferChange={(v) => sendParamChange(8, v)}
              isPlaying={isPlaying || isDawPlaying}
              bufferLabel="buffer"
              stacked={true}
            />
          </div>
          <AudioMeter
            leftLevel={metrics.leftLevel}
            rightLevel={metrics.rightLevel}
            orientation="vertical"
            width="14px"
            height="80%"
          />
        </div>
      </div>

      {resourcesMissing && (
        <ResourceOnboardingModal
          progress={resourcesProgress}
          remoteModels={remoteModels}
          downloadPath={downloadPath}
          isFetchingModels={isFetchingModels}

          onSelectFolder={() => postMessage({ type: 'selectDownloadFolder' })}
          onStartDownload={(modelName) => postMessage({ type: 'initResources', modelName })}
        />
      )}

    </div>
  );
}
