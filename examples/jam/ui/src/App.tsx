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

import { useState, useEffect, useRef, useCallback } from 'react';
import { PianoKeyboard } from './PianoKeyboard';
import { JamSlider } from './JamSlider';
import { JamSliderElastic } from './JamSliderElastic';
import { MagentaDropdown, MidiSelector, ModelSelector, ResourceOnboardingModal, PROMPT_SUGGESTIONS, INSTRUMENT_SUGGESTIONS, AudioMeter, TimingIndicator, SettingsPanel, GREY_900, ALL_COLORS, DEFAULT_TEMPERATURE, DEFAULT_TOPK, DEFAULT_CFG_NOTES, DEFAULT_CFG_MUSICCOCA, DEFAULT_CFG_DRUMS, DEFAULT_UNMASK_WIDTH, DEFAULT_BUFFER_SIZE, DEFAULT_VOLUME } from '@magenta-rt/common';
import {
  IconButton,
  MenuItem,
  CircularProgress,
  TextField,
  InputAdornment,
  Tooltip,
} from '@mui/material';
import {
  ArrowBack,
  ArrowForward,
  ChevronLeft,
  ChevronRight,
  Tune as TuneIcon,
  Close,
  UploadFile,
  Refresh,
  PlayArrow,
  Pause,
  Save,
} from '@mui/icons-material';

// ─── WebKit bridge ───────────────────────────────────────────────────────────

declare global {
  interface Window {
    updateState: (state: any) => void;
    webkit?: {
      messageHandlers?: {
        auHost?: { postMessage: (msg: any) => void };
      };
    };
  }
}

const post = (msg: any) => window.webkit?.messageHandlers?.auHost?.postMessage(msg);

// ─── Computer keyboard → MIDI (Ableton Live layout) ──────────────────────────
// Base row (lower octave): A S D F G H J = C D E F G A B, with W E T Y U as
// black keys (C# D# F# G# A#). Upper octave continues on K O L P ; (C C# D D# E).
// Z / X shift the base octave down/up.

const KEY_TO_SEMITONE: Record<string, number> = {
  a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11,
  k: 12, o: 13, l: 14, p: 15, ';': 16,
};
const KEYBOARD_MIDI_BASE_DEFAULT = 60; // C4 in MIDI (Middle C base)

// Inactivity timeout for SOLO mode playback auto-stop (in milliseconds)
const SOLO_INACTIVITY_TIMEOUT_MS = 30_000;




// CFG parameter bounds: range 0–5 (values are native, no remap needed).
const CFG_MIN = 0;
const CFG_MAX = 5;

// Column width constant for the 3-column layout
const CENTER_COL_WIDTH = '435px';

// Deterministically assign a color from ALL_COLORS to any prompt string
const getPromptColor = (prompt: string): string => {
  if (!prompt) return ALL_COLORS[0];
  const hash = prompt.charCodeAt(0);
  const index = Math.abs(hash) % ALL_COLORS.length;
  return ALL_COLORS[index];
};

// ─── App ─────────────────────────────────────────────────────────────────────

function App() {
  const [metrics, setMetrics] = useState({ frameMs: 0, bufferAvail: 0, bufferCap: 0, textEncoderStatus: 0, droppedFrames: 0 });
  const [audioLevels, setAudioLevels] = useState({ left: 0, right: 0 });
  const [modelName, setModelName] = useState("No model loaded");
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeNotes, setActiveNotes] = useState<number[]>([]);
  const [noteActivityCounter, setNoteActivityCounter] = useState(0);
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [remoteModels, setRemoteModels] = useState<string[]>([]);
  const [downloadProgress, setDownloadProgress] = useState<any>(null);
  const [downloadPath, setDownloadPath] = useState("~/Documents/Magenta/magenta-rt-v2/models");
  // Onboarding States
  const [resourcesMissing, setResourcesMissing] = useState(false);
  const [resourcesProgress, setResourcesProgress] = useState<any>(null);
  const [isFetchingModels, setIsFetchingModels] = useState(true);


  // Settings Drawer states
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [paramsState, setParamsState] = useState({
    temperature: DEFAULT_TEMPERATURE,
    topk: DEFAULT_TOPK,
    cfgnotes: DEFAULT_CFG_NOTES,
    cfgnotesuser: DEFAULT_CFG_NOTES,
    cfgmusiccoca: DEFAULT_CFG_MUSICCOCA,
    cfgdrums: DEFAULT_CFG_DRUMS,
    unmaskwidth: DEFAULT_UNMASK_WIDTH,
    buffersize: DEFAULT_BUFFER_SIZE,
    volume: DEFAULT_VOLUME,
    drumless: false,
    onsetmode: false,
  });


  // Prompt state
  const [promptText, setPromptText] = useState('');
  const [isPromptEdited, setIsPromptEdited] = useState(false);
  const [isAudioPrompt, setIsAudioPrompt] = useState(false);
  const lastSentText = useRef('');
  const promptInputRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);

  // Color state
  const [activeColor, setActiveColor] = useState(() => ALL_COLORS[Math.floor(Math.random() * ALL_COLORS.length)]);

  // Solo / Accompany state
  const [isSoloMode, setIsSoloMode] = useState(false);
  const lastSentSoloMode = useRef(false);

  // ─── User preset overrides ───────────────────────────────────────────────
  // Sparse overlay: only indices the user has explicitly saved get entries.
  // `null` means "use factory default" (reserved for future reset support).
  const [userPresetsSolo, setUserPresetsSolo] = useState<Record<number, string>>({});
  const [userPresetsJam, setUserPresetsJam] = useState<Record<number, string>>({});

  const getFactoryList = useCallback((solo: boolean): string[] => {
    return solo ? INSTRUMENT_SUGGESTIONS : PROMPT_SUGGESTIONS;
  }, []);

  const getUserOverrides = useCallback((solo: boolean): Record<number, string> => {
    return solo ? userPresetsSolo : userPresetsJam;
  }, [userPresetsSolo, userPresetsJam]);

  // ─── Prompt Rocker state ─────────────────────────────────────────────────
  const [rockerIndex, setRockerIndex] = useState(0);
  const rockerInitialized = useRef(false);



  // Persist rocker index to native whenever it changes (skip the initial 0)
  useEffect(() => {
    if (rockerInitialized.current) {
      post({ type: 'saveRockerIndex', value: rockerIndex });
    }
  }, [rockerIndex]);

  /** Returns the effective preset list: factory with user overrides applied. */
  const getActivePresetList = useCallback((solo: boolean) => {
    const factory = getFactoryList(solo);
    const overrides = getUserOverrides(solo);
    return factory.map((text, i) => (i in overrides ? overrides[i] : text));
  }, [getFactoryList, getUserOverrides]);

  const applyPresetAtIndex = useCallback((list: string[], index: number) => {
    const preset = list[index];
    if (preset) {
      if (isAudioPrompt) post({ type: 'clearAudioPrompt' });
      setPromptText(preset);
      setActiveColor(getPromptColor(preset));
      sendPrompt(preset, true);
      setIsPromptEdited(false);
    }
  }, [isAudioPrompt]);

  /** Navigate to the next/previous preset sequentially. */
  const navigatePreset = useCallback((direction: 1 | -1) => {
    const list = getActivePresetList(isSoloMode);
    if (list.length === 0) return;

    let nextIndex = rockerIndex + direction;
    if (nextIndex < 0) {
      nextIndex = list.length - 1;
    } else if (nextIndex >= list.length) {
      nextIndex = 0;
    }

    applyPresetAtIndex(list, nextIndex);
    setRockerIndex(nextIndex);
  }, [isSoloMode, rockerIndex, getActivePresetList, applyPresetAtIndex]);

  const handleRockerLeft = useCallback(() => navigatePreset(-1), [navigatePreset]);
  const handleRockerRight = useCallback(() => navigatePreset(1), [navigatePreset]);


  /** Persist the full user-overrides map to native side. */
  const persistUserPresets = useCallback((solo: Record<number, string>, jam: Record<number, string>) => {
    post({ type: 'saveUserPresets', solo, jam });
  }, []);

  /** Save the current prompt text as a user override for the active preset slot. */
  const handleSavePreset = useCallback(() => {
    const text = promptText.trim();
    if (!text) return;
    const setter = isSoloMode ? setUserPresetsSolo : setUserPresetsJam;
    setter(prev => {
      const next = { ...prev, [rockerIndex]: text };
      // Persist both maps — grab the latest of the "other" map from current state
      if (isSoloMode) {
        persistUserPresets(next, userPresetsJam);
      } else {
        persistUserPresets(userPresetsSolo, next);
      }
      setIsPromptEdited(false);
      return next;
    });
  }, [promptText, isSoloMode, rockerIndex, userPresetsSolo, userPresetsJam, persistUserPresets]);

  const handleModeChange = (solo: boolean) => {
    setIsSoloMode(solo);
    post({ type: 'setSoloMode', value: solo });
    sendParamChange(7, solo ? 127 : 0); // unmaskwidth
    setParamsState(p => ({ ...p, unmaskwidth: solo ? 127 : 0 }));

    // Pick the first preset from the new mode's preset list (top to bottom)
    const list = getActivePresetList(solo);
    if (list.length > 0) {
      const preset = list[0];
      setRockerIndex(0);
      setPromptText(preset);
      setActiveColor(getPromptColor(preset));
      if (isAudioPrompt) post({ type: 'clearAudioPrompt' });
      setIsAudioPrompt(false);
      sendPrompt(preset, true, solo);
    }
    setIsPromptEdited(false);
  };

  // MIDI sources list state
  const [midiSources, setMidiSources] = useState<{ name: string, endpoint: number, connected: boolean }[]>([]);



  // Octave shifting state and handlers
  const [octaveOffset, setOctaveOffset] = useState(0);

  const handleOctaveDown = useCallback(() => {
    setOctaveOffset(prev => {
      const next = Math.max(-4, prev - 1);
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

  // Computer keyboard → MIDI
  const [keyboardMidiEnabled, setKeyboardMidiEnabled] = useState(false);
  const keyboardBaseNote = useRef(KEYBOARD_MIDI_BASE_DEFAULT);
  const pressedKeys = useRef<Map<string, number>>(new Map()); // key → MIDI note currently held

  // Determine which MIDI option is selected (0 represents Computer Keyboard, otherwise exact endpoint ID)
  const selectedMidiValue = keyboardMidiEnabled ? 0 : (midiSources.find(s => s.connected)?.endpoint ?? 0xFFFFFFFF);

  // ─── Engine communication ───────────────────────────────────────────────

  const textUpdateTimeout = useRef<number | null>(null);
  const waitingForEncoder = useRef(false);
  const encoderTimeoutRef = useRef<number | null>(null);

  const startEncoderTimeout = () => {
    if (encoderTimeoutRef.current) {
      clearTimeout(encoderTimeoutRef.current);
    }
    encoderTimeoutRef.current = window.setTimeout(() => {
      if (waitingForEncoder.current) {
        waitingForEncoder.current = false;
        // Force-update metrics to trigger isProgressActive updates
        setMetrics(m => ({ ...m }));
      }
      encoderTimeoutRef.current = null;
    }, 2000);
  };

  const sendPrompt = (text: string, immediate = false, soloOverride?: boolean) => {
    const soloActive = soloOverride !== undefined ? soloOverride : isSoloMode;
    const textWithPrefix = soloActive ? `SOLO ${text}` : text;

    if (text === lastSentText.current && soloActive === lastSentSoloMode.current) return;

    const prompts = [
      { text: textWithPrefix, weight: 1.0 },
    ];
    if (textUpdateTimeout.current) {
      clearTimeout(textUpdateTimeout.current);
      textUpdateTimeout.current = null;
    }
    if (immediate) {
      lastSentText.current = text;
      lastSentSoloMode.current = soloActive;
      waitingForEncoder.current = true;
      startEncoderTimeout();
      post({ type: 'textPrompts', value: prompts });
    } else {
      textUpdateTimeout.current = window.setTimeout(() => {
        lastSentText.current = text;
        lastSentSoloMode.current = soloActive;
        waitingForEncoder.current = true;
        startEncoderTimeout();
        post({ type: 'textPrompts', value: prompts });
        textUpdateTimeout.current = null;
      }, 400);
    }
  };

  const sendParamChange = (index: number, value: number) => {
    post({ type: 'param', index, value });
  };
  const handleResetDefaults = () => {
    sendParamChange(0, DEFAULT_TEMPERATURE);       // temperature
    sendParamChange(1, DEFAULT_TOPK);              // topk
    sendParamChange(3, DEFAULT_CFG_MUSICCOCA);     // cfgmusiccoca
    sendParamChange(4, DEFAULT_CFG_NOTES);         // cfgnotes
    sendParamChange(48, DEFAULT_CFG_DRUMS);        // cfgdrums
    sendParamChange(7, DEFAULT_UNMASK_WIDTH);      // unmaskwidth
    sendParamChange(8, DEFAULT_BUFFER_SIZE);       // buffersize
    sendParamChange(39, 0);   // drumless
    sendParamChange(46, 0);   // onsetmode = false (Auto-Strum = true)
    setParamsState(p => ({ ...p, cfgnotesuser: DEFAULT_CFG_NOTES, cfgmusiccoca: DEFAULT_CFG_MUSICCOCA }));
  };

  const togglePlay = () => {
    const newPlaying = !isPlaying;
    setIsPlaying(newPlaying);
    post({ type: 'togglePlay', value: newPlaying });
  };

  const resetModel = () => {
    sendParamChange(31, 1.0);
    setTimeout(() => sendParamChange(31, 0.0), 100);
  };

  const loadAudioPrompt = () => {
    post({ type: 'loadAudioPrompt', index: 0 });
  };

  const clearAudioPrompt = () => {
    post({ type: 'clearAudioPrompt' });
  };



  const openSettings = () => {
    post({ type: 'openSettings' });
  };

  // ─── State updates from native ─────────────────────────────────────────

  // Track whether the user has received initial state yet. Before that,
  // `prompt` updates from native should populate the UI. After, we ignore
  // subsequent `prompt` echoes so they don't stomp in-progress typing.
  const promptInitialized = useRef(false);

  useEffect(() => {
    window.updateState = (state: any) => {
      if (state.metrics) {
        setMetrics(m => {
          const next = { ...m, ...state.metrics };
          if (next.textEncoderStatus === 1) {
            waitingForEncoder.current = false;
            if (encoderTimeoutRef.current) {
              clearTimeout(encoderTimeoutRef.current);
              encoderTimeoutRef.current = null;
            }
          }
          return next;
        });
      }
      if (state.audioLevels) setAudioLevels(state.audioLevels);
      if (state.modelName !== undefined) setModelName(state.modelName);
      if (state.isPlaying !== undefined) setIsPlaying(state.isPlaying);
      if (state.activeNotes) {
        setActiveNotes(state.activeNotes);
        if (state.activeNotes.length > 0) {
          setNoteActivityCounter(n => n + 1);
          setIsPlaying(prev => {
            if (!prev) {
              post({ type: 'togglePlay', value: true });
              return true;
            }
            return prev;
          });
        }
      }

      let solo = isSoloMode;
      if (state.solomode !== undefined) {
        solo = !!state.solomode;
        setIsSoloMode(solo);
      }

      if (state.params !== undefined) {
        setParamsState(p => {
          const next = { ...p };
          if (state.params.temperature !== undefined) next.temperature = state.params.temperature;
          if (state.params.topk !== undefined) next.topk = state.params.topk;
          if (state.params.cfgnotes !== undefined) next.cfgnotes = state.params.cfgnotes;
          if (state.params.cfgmusiccoca !== undefined) next.cfgmusiccoca = state.params.cfgmusiccoca;
          if (state.params.cfgnotesuser !== undefined) next.cfgnotesuser = state.params.cfgnotesuser;
          if (state.params.cfgdrums !== undefined) next.cfgdrums = state.params.cfgdrums;
          if (state.params.unmaskwidth !== undefined) next.unmaskwidth = state.params.unmaskwidth;
          if (state.params.buffersize !== undefined) next.buffersize = state.params.buffersize;
          if (state.params.volume !== undefined) next.volume = state.params.volume;
          if (state.params.drumless !== undefined) next.drumless = state.params.drumless;
          if (state.params.onsetmode !== undefined) next.onsetmode = !!state.params.onsetmode;
          return next;
        });
      }

      if (state.openSettings !== undefined) {
        setIsSettingsOpen(!!state.openSettings);
      }


      // Restore user preset overrides from native if present
      if (state.savedUserPresets !== undefined) {
        if (state.savedUserPresets.solo) setUserPresetsSolo(state.savedUserPresets.solo);
        if (state.savedUserPresets.jam) setUserPresetsJam(state.savedUserPresets.jam);
      }

      if (state.prompt !== undefined && !promptInitialized.current) {
        // Use saved rocker index if available, otherwise try to find a match
        let presetIdx = -1;
        if (state.savedRockerIndex !== undefined) {
          presetIdx = state.savedRockerIndex;
        }

        // Build effective preset list using user overrides that arrived in this same state update
        const userSolo = state.savedUserPresets?.solo ?? {};
        const userJam = state.savedUserPresets?.jam ?? {};
        const factoryList = solo ? INSTRUMENT_SUGGESTIONS : PROMPT_SUGGESTIONS;
        const effectiveList = factoryList.map((text, i) => {
          const overrides = solo ? userSolo : userJam;
          return (i in overrides) ? overrides[i] : text;
        });

        let promptToUse = state.prompt;

        if (!promptToUse) {
          // No saved prompt — use preset at the saved index (or first)
          const idx = presetIdx >= 0 ? presetIdx : 0;
          promptToUse = effectiveList[idx] || '';
          presetIdx = idx;
        } else if (presetIdx < 0) {
          // No saved rocker index — try to find the prompt in the effective list
          presetIdx = effectiveList.findIndex(p => p.toLowerCase() === promptToUse.toLowerCase());
        }

        setPromptText(promptToUse);
        setActiveColor(getPromptColor(promptToUse));
        setIsAudioPrompt(state.isAudioPrompt || false);
        setIsPromptEdited(false);
        if (presetIdx >= 0) {
          setRockerIndex(presetIdx);
        }
        rockerInitialized.current = true;
        // Frontend takes ownership — sync prompt to engine
        sendPrompt(promptToUse, true, solo);
        promptInitialized.current = true;
      } else if (state.isAudioPrompt !== undefined) {
        if (state.isAudioPrompt) {
          // Audio prompt loaded — honor native's value (user explicitly triggered upload)
          setPromptText(state.prompt);
          setActiveColor(getPromptColor(state.prompt));
          setIsPromptEdited(false);
          lastSentText.current = state.prompt;
        }
        setIsAudioPrompt(state.isAudioPrompt);
      }

      if (state.computerKeyboardMidi !== undefined) {
        setKeyboardMidiEnabled(!!state.computerKeyboardMidi);
      }
      if (state.localModels !== undefined) {
        setLocalModels(state.localModels);
      }
      if (state.remoteModels !== undefined) {
        setRemoteModels(state.remoteModels);
        setIsFetchingModels(false);
      }
      if (state.remoteModelsError !== undefined) {
        setIsFetchingModels(false);
      }
      if (state.downloadProgress !== undefined) {
        setDownloadProgress(state.downloadProgress);
      }
      if (state.resourcesMissing !== undefined) {
        setResourcesMissing(state.resourcesMissing);
      }
      if (state.resourcesProgress !== undefined) {
        setResourcesProgress(state.resourcesProgress);
      }
      if (state.downloadPath !== undefined) {
        setDownloadPath(state.downloadPath);
      }

      if (state.midiSources !== undefined) {
        setMidiSources(state.midiSources);
      }
      if (state.solomode !== undefined) {
        setIsSoloMode(!!state.solomode);
      }
    };

    post({ type: 'uiReady' });
    post({ type: 'listRemoteModels' });

    // Auto-focus prompt text input on app mount and place caret at the end
    if (promptInputRef.current) {
      const el = promptInputRef.current;
      el.focus();
      const len = el.value.length;
      el.setSelectionRange(len, len);
    }

    return () => {
      delete (window as any).updateState;
      if (encoderTimeoutRef.current) {
        clearTimeout(encoderTimeoutRef.current);
      }
    };
  }, []);

  // Transport keys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement instanceof HTMLInputElement || document.activeElement instanceof HTMLTextAreaElement) return;
      if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying]);

  // Solo mode: auto-stop after timeout of no incoming notes
  useEffect(() => {
    if (!isSoloMode || !isPlaying) return;
    const timer = window.setTimeout(() => {
      setIsPlaying(false);
      post({ type: 'togglePlay', value: false });
    }, SOLO_INACTIVITY_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [isSoloMode, isPlaying, noteActivityCounter]);

  // Automatically trigger a model reset when MusicCoCa embedding finishes
  // loading. This covers all prompt-change paths: typing, rocker arrows,
  // mode switch, audio file drop, and settings changes.
  //
  // IMPORTANT: We must wait for MusicCoCa to fully finish encoding
  // (textEncoderStatus goes 1→0 AND the debounced prompt has been sent)
  // before resetting. Resetting too early causes a race condition where the
  // state is cleared but the new prompt hasn't been applied yet, leading to
  // bleed from the previous prompt.
  const isProgressActive = metrics.textEncoderStatus === 1 || waitingForEncoder.current;
  const prevProgressActive = useRef(false);

  useEffect(() => {
    if (prevProgressActive.current && !isProgressActive) {
      resetModel();
    }
    prevProgressActive.current = isProgressActive;
  }, [isProgressActive]);



  // Computer keyboard → MIDI. Only intercept when enabled and when no input
  // is focused (so typing prompts still works).
  useEffect(() => {
    if (!keyboardMidiEnabled) {
      // Release any still-held notes
      pressedKeys.current.forEach((note) => {
        post({ type: 'kbdNote', note, on: false });
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
      post({ type: 'kbdNote', note, on: true });
    };

    const handleUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      const note = pressedKeys.current.get(key);
      if (note === undefined) return;
      pressedKeys.current.delete(key);
      post({ type: 'kbdNote', note, on: false });
    };

    // Release held notes when window loses focus (otherwise stuck notes).
    const handleBlur = () => {
      pressedKeys.current.forEach((note) => {
        post({ type: 'kbdNote', note, on: false });
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

  // ─── Render ─────────────────────────────────────────────────────────────

  const keyboardStartNote = keyboardMidiEnabled ? 60 : 48;
  const keyboardEndNote = keyboardMidiEnabled ? 76 : 72;
  const noModel = !modelName || modelName === 'No model loaded';

  // Current preset list for the rocker display
  const currentPresetList = getActivePresetList(isSoloMode);

  // Determine if the user has modified the prompt relative to the saved preset
  const savedPresetText = currentPresetList[rockerIndex] ?? '';
  const promptIsDirty = isPromptEdited && promptText.trim() !== '' && promptText !== savedPresetText;

  // Tab style helper for the Solo/Jam switcher
  const modeTabStyle = (active: boolean): React.CSSProperties => ({
    height: '100%',
    padding: '0 24px',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 400,
    fontFamily: "'Google Sans', system-ui, sans-serif",
    letterSpacing: '0.5px',
    textTransform: 'none',
    background: active ? '#36373A' : 'transparent',
    color: active ? activeColor : 'rgba(255, 255, 255, 0.45)',
    transition: 'all 0.15s ease',
    border: 'none',
    outline: 'none',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  });

  const playButton = (
    <IconButton
      onClick={noModel ? undefined : togglePlay}
      disabled={noModel}
      sx={{
        width: 63,
        height: 44,
        borderRadius: '8px',
        backgroundColor: '#FFF',
        color: '#000',
        borderBottom: '1.5px solid #ddd',
        transition: 'opacity 0.15s ease',
        '&:hover': {
          backgroundColor: '#FFF',
          color: '#000',
          opacity: 0.9,
        },
        '&.Mui-disabled': {
          backgroundColor: 'rgba(255, 255, 255, 0.3)',
          color: 'rgba(0, 0, 0, 0.3)',
        },
      }}
      title={isPlaying ? 'Pause' : 'Play'}
    >
      {isPlaying ? (
        <Pause sx={{ fontSize: 24 }} />
      ) : (
        <PlayArrow sx={{ fontSize: 24 }} />
      )}
    </IconButton>
  );

  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        boxSizing: 'border-box',
        color: '#FFF',
        fontFamily: "'Google Sans Text', system-ui, sans-serif",
      }}
    >
      {/* ══════════════════════════════════════════════════════════════════
          UPPER SECTION — colored background, contains rows 1 & 2, octave rocker
          ══════════════════════════════════════════════════════════════════ */}
      <div
        style={{
          flex: '1 1 auto',
          background: activeColor,
          transition: 'background-color 0.3s ease',
          display: 'flex',
          flexDirection: 'column',
          padding: '18px 24px',
          boxSizing: 'border-box',
          minHeight: 0,
        }}
      >
        {/* ── 3-column layout: Left controls | Prompt | Right controls ── */}
        <div style={{
          display: 'flex',
          gap: '16px',
          flex: '1 1 auto',
          minHeight: 0,
        }}>

          {/* LEFT COLUMN — Tabs on top, sliders centered below */}
          <div style={{
            flex: '1 1 0px',
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
          }}>
            {/* Solo / Jam Tab Switcher */}
            <div style={{ display: 'flex' }}>
              <div
                className="jam-box"
                style={{
                  display: 'inline-flex',
                  height: '44px',
                  boxSizing: 'border-box',
                  padding: '2px',
                  alignItems: 'center',
                }}
              >
                <button
                  onClick={() => handleModeChange(false)}
                  style={modeTabStyle(!isSoloMode)}
                >
                  Jam
                </button>
                <button
                  onClick={() => handleModeChange(true)}
                  style={modeTabStyle(isSoloMode)}
                >
                  Solo
                </button>
              </div>
            </div>

            {/* Sliders */}
            <div style={{
              flex: '1 1 auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '20px',
              paddingRight: '24px',
            }}>
              <JamSliderElastic
                label="Chaos"
                minA={0.5} midA={1.0} maxA={2.0}
                minB={10}  midB={100} maxB={500}
                accentColor={activeColor}
                onChange={(temperature, topk) => {
                  setParamsState(p => ({ ...p, temperature, topk }));
                  sendParamChange(0, temperature);
                  sendParamChange(1, topk);
                }}
              />
              <JamSlider
                label="Volume"
                value={Math.min(1, Math.max(0, (paramsState.volume + 60) / 60))}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => {
                  const db = parseFloat(((v * 60) - 60).toFixed(1));
                  setParamsState(p => ({ ...p, volume: db }));
                  sendParamChange(5, db);
                }}
              />
            </div>
          </div>

          {/* CENTER COLUMN — Prompt Box with semicircle rockers on each side */}
          <div style={{
            width: CENTER_COL_WIDTH,
            flexShrink: 0,
            position: 'relative',
            minWidth: 0,
          }}>
            {/* Rocker Left — semicircle, vertically centered */}
            <IconButton
              variant="jam"
              onClick={handleRockerLeft}
              sx={{
                position: 'absolute',
                left: 0,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 40,
                height: 56,
                borderRadius: '0 28px 28px 0',
                zIndex: 5,
              }}
              title="Previous preset"
            >
              <ArrowBack sx={{ fontSize: 20, color: '#FFF', transform: 'translateX(-3px)' }} />
            </IconButton>

            {/* Prompt Box */}
            <div
              className="jam-box"
              onClick={(e) => {
                if (e.target instanceof Element && e.target.closest('.upload-btn-container')) {
                  return;
                }
                if (promptInputRef.current) {
                  promptInputRef.current.focus();
                }
              }}
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-start',
                minWidth: 0,
                padding: '12px 64px',
                cursor: 'text',
              }}
            >
              {/* Loading Spinner */}
              {isProgressActive && (
                <CircularProgress
                  size={16}
                  sx={{
                    color: 'rgba(255, 255, 255, 0.6)',
                    position: 'absolute',
                    right: '12px',
                    top: '12px',
                    zIndex: 10,
                  }}
                />
              )}

              <TextField
                value={promptText}
                onChange={e => {
                  const val = e.target.value;
                  const oldVal = promptText;
                  setPromptText(val);
                  sendPrompt(val);
                  setIsPromptEdited(true);

                  const oldFirstChar = oldVal.charAt(0);
                  const newFirstChar = val.charAt(0);
                  const isBlank = val.trim() === "";

                  if (!isBlank && newFirstChar !== oldFirstChar) {
                    setActiveColor(getPromptColor(val));
                  }
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    sendPrompt(promptText, true);
                    promptInputRef.current?.blur();
                  }
                }}
                onBlur={() => {
                  sendPrompt(promptText, true);
                }}
                placeholder="Type a prompt or upload an audio file."
                disabled={isAudioPrompt}
                variant="standard"
                fullWidth
                multiline
                maxRows={6}
                inputRef={promptInputRef}
                inputProps={{
                  autoComplete: 'off',
                  autoCorrect: 'off',
                  autoCapitalize: 'off',
                  spellCheck: 'false',
                }}
                InputProps={{
                  disableUnderline: true,
                  sx: {
                    textWrap: 'pretty',
                    color: activeColor,
                    fontSize: '36px',
                    fontWeight: 400,
                    fontFamily: "'Google Sans', system-ui, sans-serif",
                    lineHeight: 1.25,
                    caretColor: activeColor,
                    '&::placeholder': {
                      color: 'rgba(255, 255, 255, 0.25)',
                      opacity: 1,
                    }
                  }
                }}
                sx={{
                  flex: '1 1 auto',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  '& .MuiInputBase-input': {
                    padding: '0',
                  },
                  '& .MuiInputBase-input.Mui-disabled': {
                    WebkitTextFillColor: activeColor,
                  },
                  '& .MuiInputBase-root': {
                    alignItems: 'center',
                    flex: '1 1 auto',
                    display: 'flex',
                  }
                }}
              />

              <div
                className="upload-btn-container"
                style={{
                  position: 'absolute',
                  right: '12px',
                  bottom: '12px',
                  zIndex: 10,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}>
                {/* Save preset button — visible only when text prompt is modified (dirty) */}
                {!isAudioPrompt && promptIsDirty && (
                  <Tooltip title="Save preset" placement="top">
                    <span>
                      <IconButton
                        variant="jam"
                        onClick={handleSavePreset}
                        sx={{
                          width: 36,
                          height: 36,
                        }}
                      >
                        <Save sx={{ fontSize: 18, color: activeColor }} />
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
                {isAudioPrompt ? (
                  <IconButton
                    variant="jam"
                    onClick={clearAudioPrompt}
                    sx={{ width: 36, height: 36 }}
                    title="Remove audio file"
                  >
                    <Close sx={{ fontSize: 18 }} />
                  </IconButton>
                ) : (
                  <Tooltip title="Upload audio prompt" placement="top">
                    <IconButton
                      variant="jam"
                      onClick={loadAudioPrompt}
                      sx={{ width: 36, height: 36 }}
                      title="Choose audio file"
                    >
                      <UploadFile sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                )}
              </div>
            </div>

            {/* Rocker Right — semicircle, vertically centered */}
            <IconButton
              variant="jam"
              onClick={handleRockerRight}
              sx={{
                position: 'absolute',
                right: 0,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 40,
                height: 56,
                borderRadius: '28px 0 0 28px',
                zIndex: 5,
              }}
              title="Next preset"
            >
              <ArrowForward sx={{ fontSize: 20, color: '#FFF', transform: 'translateX(3px)' }} />
            </IconButton>
          </div>

          {/* RIGHT COLUMN — Buttons on top, sliders centered below */}
          <div style={{
            flex: '1 1 0px',
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
          }}>
            {/* Reset / Play / Settings */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              gap: '8px',
            }}>
              <IconButton
                variant="jam"
                onClick={resetModel}
                sx={{ width: 44, height: 44 }}
                title="Reset model state"
              >
                <Refresh sx={{ fontSize: 20 }} />
              </IconButton>

              {noModel ? (
                <Tooltip title="No model selected" placement="top">
                  <span>{playButton}</span>
                </Tooltip>
              ) : (
                playButton
              )}

              <IconButton
                variant="jam"
                onClick={() => setIsSettingsOpen(true)}
                sx={{ width: 44, height: 44 }}
                title="Settings (Cmd+,)"
              >
                <TuneIcon sx={{ fontSize: 20 }} />
              </IconButton>
            </div>

            {/* Sliders */}
            <div style={{
              flex: '1 1 auto',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{
                fontFamily: "'Google Sans Text', system-ui, sans-serif",
                fontSize: '11px',
                fontWeight: 500,
                color: '#1B1C17',
                marginBottom: '8px',
                letterSpacing: '0.3px',
              }}>Strength</span>
              <div style={{
                flex: '1 1 auto',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
              }}>
              <JamSlider
                label="Notes"
                value={paramsState.cfgnotesuser}
                min={CFG_MIN}
                max={CFG_MAX}
                showValueOnThumb={true}
                valueFormatter={v => (v === 0 || v === 5 ? v.toString() : v.toFixed(1))}
                onChange={(v) => {
                  setParamsState(p => ({ ...p, cfgnotesuser: v }));
                  sendParamChange(4, v);
                }}
              />
              <JamSlider
                label="Style"
                value={paramsState.cfgmusiccoca}
                min={CFG_MIN}
                max={CFG_MAX}
                showValueOnThumb={true}
                valueFormatter={v => (v === 0 || v === 5 ? v.toString() : v.toFixed(1))}
                onChange={(v) => {
                  setParamsState(p => ({ ...p, cfgmusiccoca: v }));
                  sendParamChange(3, v);
                }}
              />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          PIANO KEYBOARD — full bleed, no padding
          ══════════════════════════════════════════════════════════════════ */}
      <div
        style={{
          flexShrink: 0,
          height: '240px',
          backgroundColor: '#000',
          paddingTop: '10px',
          position: 'relative',
        }}
      >
        {/* Octave Rocker — floats top-right over the keyboard */}
        <div style={{
          position: 'absolute',
          top: '0',
          right: '0',
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '4px 8px',
          borderRadius: '8px',
          backgroundColor: '#000',
          visibility: keyboardMidiEnabled ? 'visible' : 'hidden',
        }}>
          <IconButton
            variant="ghost"
            onClick={handleOctaveDown}
            disabled={octaveOffset <= -4}
            sx={{
              width: 32,
              height: 32,
              color: '#FFF',
              '&:hover': { backgroundColor: '#36373A' }
            }}
          >
            <ChevronLeft sx={{ fontSize: 18 }} />
          </IconButton>

          <span style={{
            fontSize: '13px',
            fontWeight: 600,
            minWidth: '36px',
            textAlign: 'center',
            color: '#FFF',
            fontFamily: "'Google Sans', system-ui, sans-serif",
            letterSpacing: '0.5px',
          }}>
            C{Math.floor((KEYBOARD_MIDI_BASE_DEFAULT + octaveOffset * 12) / 12) - 1}
          </span>

          <IconButton
            variant="ghost"
            onClick={handleOctaveUp}
            disabled={octaveOffset >= 4}
            sx={{
              width: 32,
              height: 32,
              color: '#FFF',
              '&:hover': { backgroundColor: '#36373A' }
            }}
          >
            <ChevronRight sx={{ fontSize: 18 }} />
          </IconButton>
        </div>

        <PianoKeyboard
          activeNotes={keyboardMidiEnabled
            ? activeNotes.map(n => 60 + (n - keyboardBaseNote.current))
                .filter(n => n >= 60 && n <= 76)
            : activeNotes
          }
          accentColor={activeColor}
          startNote={keyboardStartNote}
          endNote={keyboardEndNote}
          keyboardMidiEnabled={keyboardMidiEnabled}
          onNoteOn={(visualNote) => {
            // Remap visual piano note to actual MIDI note
            const note = keyboardMidiEnabled
              ? keyboardBaseNote.current + (visualNote - 60)
              : visualNote;
            if (note >= 0 && note <= 127) post({ type: 'kbdNote', note, on: true });
          }}
          onNoteOff={(visualNote) => {
            const note = keyboardMidiEnabled
              ? keyboardBaseNote.current + (visualNote - 60)
              : visualNote;
            if (note >= 0 && note <= 127) post({ type: 'kbdNote', note, on: false });
          }}
        />
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          BLACK FOOTER — ModelSelector, MIDI, spacing, TimingIndicator, AudioMeter
          ══════════════════════════════════════════════════════════════════ */}
      <div
        style={{
          flexShrink: 0,
          height: '76px',
          background: '#000',
          color: '#FFF',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 16px',
          boxSizing: 'border-box',
        }}
      >
        {/* Left cluster: ModelSelector + MIDI Input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ModelSelector
            modelName={modelName}
            localModels={localModels}
            remoteModels={remoteModels}
            downloadProgress={downloadProgress}

            onSelectModel={(m) => post({ type: 'selectModel', name: m })}
            onDownloadModel={(m) => post({ type: 'downloadModel', name: m })}
            onDeleteModel={(m) => post({ type: 'deleteModel', name: m })}
            onSelectFolder={() => post({ type: 'selectDownloadFolder' })}
            buttonSx={{
              color: '#FFF',
              '&:hover': { background: 'rgba(255, 255, 255, 0.12)' },
            }}
          />

          <MidiSelector
            midiSources={midiSources}
            keyboardMidiEnabled={keyboardMidiEnabled}
            onSelectSource={(endpoint) => post({ type: 'selectMidiSource', endpoint })}
            midiActive={activeNotes.length > 0}
          />
        </div>

        {/* Spacer */}
        <div style={{ flex: '1 1 auto' }} />

        {/* Right cluster: TimingIndicator + AudioMeter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <TimingIndicator
            frameMs={metrics.frameMs}
            droppedFrames={metrics.droppedFrames}
            buffersize={paramsState.buffersize}
            onBufferChange={(v) => sendParamChange(8, v)}
            buttonSx={{
              color: '#FFF',
              '&:hover': { background: 'rgba(255, 255, 255, 0.12)' },
            }}
            isPlaying={isPlaying}
          />

          <AudioMeter leftLevel={audioLevels.left} rightLevel={audioLevels.right} width="45px" height="14px" />
        </div>
      </div>

      {/* ── Settings Panel (drawer overlay) ── */}
      <SettingsPanel
        open={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        temperature={paramsState.temperature}
        topk={paramsState.topk}
        cfgnotes={paramsState.cfgnotesuser}
        cfgmusiccoca={paramsState.cfgmusiccoca}
        cfgdrums={paramsState.cfgdrums}
        unmaskwidth={paramsState.unmaskwidth}
        onParamChange={sendParamChange}
        onResetDefaults={handleResetDefaults}
        showNoteCfg={false}
        showPromptCfg={false}
        showDrumsCfg={false}
        showUnmaskWidth={false}
        showOnsetMode={true}
        onsetmode={paramsState.onsetmode}
        showDrumless={true}
        columns={1}
        drumless={paramsState.drumless}
      />

      {resourcesMissing && (
        <ResourceOnboardingModal
          progress={resourcesProgress}
          remoteModels={remoteModels}
          downloadPath={downloadPath}
          isFetchingModels={isFetchingModels}

          onSelectFolder={() => post({ type: 'selectDownloadFolder' })}
          onStartDownload={(modelName) => post({ type: 'initResources', modelName })}
        />
      )}
    </div>
  );
}

export default App;
