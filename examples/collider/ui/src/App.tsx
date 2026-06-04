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

import { useState, useRef, useEffect, useCallback } from 'react';
import { Turtle, Rabbit } from 'lucide-react';
import IconButton from '@mui/material/IconButton';
import TuneIcon from '@mui/icons-material/Tune';
import Tooltip from '@mui/material/Tooltip';
import { ModelSelector, SettingsPanel, TimingIndicator, AudioMeter, ResourceOnboardingModal, TransportControls, PromptSurface, calculateWeights, ALL_SUGGESTIONS, DEFAULT_TEMPERATURE, DEFAULT_TOPK, DEFAULT_CFG_MUSICCOCA, DEFAULT_CFG_DRUMS, DEFAULT_UNMASK_WIDTH, DEFAULT_BUFFER_SIZE, DEFAULT_VOLUME, COLLIDER_CFG_NOTES, COLLIDER_CFG_MUSICCOCA } from '@magenta-rt/common';
import type { PromptNode, ListenerNode } from '@magenta-rt/common';


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

const MAX_ENGINE_PROMPTS = 6;

// ─── Defaults ────────────────────────────────────────────────────────────────

const DEFAULT_PHYSICS_SPEED = 0.5;

/** Fisher-Yates shuffle (in place). */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** A shuffled copy of ALL_SUGGESTIONS used as a deck. The first 3 entries
 *  seed the initial prompts; subsequent entries are dealt on double-tap. */
const SHUFFLED_SUGGESTIONS = shuffle([...ALL_SUGGESTIONS]);
const INITIAL_PROMPT_LABELS = SHUFFLED_SUGGESTIONS.slice(0, 3);

const INITIAL_LISTENER: ListenerNode = { x: 0, y: 0 }; // recalculated on mount

// ─── Speed slider mapping ────────────────────────────────────────────────────
// Exponential curve so most of the slider is dedicated to slow speeds.
// slider 0–1 → speed 0–MAX  via  t^exp

const SPEED_CURVE_EXP = 2;

const sliderToSpeed = (t: number) => Math.pow(t, SPEED_CURVE_EXP) * DEFAULT_PHYSICS_SPEED;
const speedToSlider = (s: number) => Math.pow(s / DEFAULT_PHYSICS_SPEED, 1 / SPEED_CURVE_EXP);

/** Build an equilateral triangle of prompts centered in the canvas, with `pad` px above/below. */
function buildInitialLayout(w: number, h: number, pad = 60) {
  const cx = w / 2;
  // For an equilateral triangle: top vertex at pad, bottom vertices at h-pad
  // top = cy - R = pad, bottom = cy + R/2 = h - pad
  // Solving: R = 2*(h - 2*pad)/3, cy = pad + R
  const r = (2 * (h - 2 * pad)) / 3;
  const cy = pad + r;
  // 3 vertices at -90°, 30°, 150° (top, bottom-right, bottom-left)
  const angles = [-Math.PI / 2, Math.PI / 6, (5 * Math.PI) / 6];
  const prompts: PromptNode[] = INITIAL_PROMPT_LABELS.map((label, i) => ({
    id: i,
    x: cx + r * Math.cos(angles[i]),
    y: cy + r * Math.sin(angles[i]),
    label,
    colorIndex: i,
  }));
  const listener: ListenerNode = { x: cx, y: cy };
  return { prompts, listener };
}

// ─── App ─────────────────────────────────────────────────────────────────────

function App() {
  const [prompts, setPrompts] = useState<PromptNode[]>([]);
  const [listener, setListener] = useState<ListenerNode>(INITIAL_LISTENER);
  const layoutInitialized = useRef(false);
  const [selectedBallId, setSelectedBallId] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [sliderPos, setSliderPos] = useState(0.5);
  const physicsSpeed = sliderToSpeed(sliderPos);
  const [collisionsEnabled, setCollisionsEnabled] = useState(true);
  const [hasThrown, setHasThrown] = useState(false);
  const [debug, setDebug] = useState(false);
  const [modelName, setModelName] = useState('No model loaded');
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [remoteModels, setRemoteModels] = useState<string[]>([]);
  const [downloadProgress, setDownloadProgress] = useState<any>(null);
  const [downloadPath, setDownloadPath] = useState("~/Documents/Magenta/magenta-rt-v2");
  const [resourcesMissing, setResourcesMissing] = useState(false);
  const [resourcesProgress, setResourcesProgress] = useState<any>(null);
  const [isFetchingModels, setIsFetchingModels] = useState(true);


  // Metrics state
  const [metrics, setMetrics] = useState({ frameMs: 0, bufferAvail: 0, bufferCap: 0, droppedFrames: 0 });

  // Settings Drawer states
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [paramsState, setParamsState] = useState({
    temperature: DEFAULT_TEMPERATURE,
    topk: DEFAULT_TOPK,
    cfgnotes: COLLIDER_CFG_NOTES,
    cfgmusiccoca: COLLIDER_CFG_MUSICCOCA,
    cfgdrums: DEFAULT_CFG_DRUMS,
    unmaskwidth: DEFAULT_UNMASK_WIDTH,
    buffersize: DEFAULT_BUFFER_SIZE,
    volume: DEFAULT_VOLUME,
    drumless: false,
  });

  // ─── Measure prompt surface and build initial layout ─────────────────
  const promptSurfaceRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (layoutInitialized.current) return;
    // Defer to next frame so flex layout (header + bottom bar) has settled
    requestAnimationFrame(() => {
      const el = promptSurfaceRef.current;
      if (!el || layoutInitialized.current) return;
      const { width, height } = el.getBoundingClientRect();
      if (width > 0 && height > 0) {
        const layout = buildInitialLayout(width, height);
        setPrompts(layout.prompts);
        setListener(layout.listener);
        layoutInitialized.current = true;
      }
    });
  }, []);

  const sendParamChange = (index: number, value: number) => {
    post({ type: 'param', index, value });
  };
  const handleResetDefaults = () => {
    sendParamChange(0, DEFAULT_TEMPERATURE);       // temperature
    sendParamChange(1, DEFAULT_TOPK);              // topk
    sendParamChange(3, COLLIDER_CFG_MUSICCOCA);    // cfgmusiccoca (Collider override)
    sendParamChange(4, COLLIDER_CFG_NOTES);        // cfgnotes (Collider default)
    sendParamChange(48, DEFAULT_CFG_DRUMS);        // cfgdrums
    sendParamChange(7, DEFAULT_UNMASK_WIDTH);      // unmaskwidth
    sendParamChange(8, DEFAULT_BUFFER_SIZE);       // buffersize
    sendParamChange(39, 0);                        // drumless = false
  };

  const resetModel = () => {
    sendParamChange(31, 1.0);
    setTimeout(() => sendParamChange(31, 0.0), 100);
  };

  const nextIdRef = useRef(3);
  const nextColorRef = useRef(3);
  /** Index into SHUFFLED_SUGGESTIONS — starts at 3 because the first 3 are used for initial prompts. */
  const deckIndexRef = useRef(3);


  // Refs for current state (used by updateState callback)
  const promptsRef = useRef(prompts);
  promptsRef.current = prompts;
  const listenerRef = useRef(listener);
  listenerRef.current = listener;

  // ─── Bridge: send prompts + weights to native ──────────────────────

  const sendPrompts = useCallback(() => {
    const weights = calculateWeights(listenerRef.current, promptsRef.current);
    // Build engine payload — audio prompt must be at index 0 (native hardcodes it there)
    const data: { text: string; weight: number }[] = Array.from({ length: MAX_ENGINE_PROMPTS }, () => ({ text: '', weight: 0 }));
    const audioIdx = promptsRef.current.findIndex(p => p.isAudio);
    if (audioIdx !== -1) {
      data[0] = { text: promptsRef.current[audioIdx].label, weight: weights[audioIdx] ?? 0 };
      let dest = 1;
      promptsRef.current.forEach((p, i) => {
        if (i !== audioIdx && dest < MAX_ENGINE_PROMPTS) {
          data[dest++] = { text: p.label, weight: weights[i] ?? 0 };
        }
      });
    } else {
      promptsRef.current.forEach((p, i) => {
        if (i < MAX_ENGINE_PROMPTS) {
          data[i] = { text: p.label, weight: weights[i] ?? 0 };
        }
      });
    }
    post({ type: 'textPrompts', value: data });
  }, []);

  // ─── Throttled prompt sending ─────────────────────────────────────
  // Decouple engine IPC from the 60fps animation loop. Position changes
  // from physics update refs instantly (so the visual is smooth), but we
  // only push weight updates to the native engine at ~10Hz — fast enough
  // for perceptible audio blending, slow enough to avoid flooding the
  // TFLite quantizer with redundant invocations.
  const sendThrottleRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSendTimeRef = useRef(0);

  useEffect(() => {
    const THROTTLE_MS = 100; // ~10 Hz
    const now = Date.now();
    const elapsed = now - lastSendTimeRef.current;

    if (sendThrottleRef.current) {
      clearTimeout(sendThrottleRef.current);
      sendThrottleRef.current = null;
    }

    if (elapsed >= THROTTLE_MS) {
      sendPrompts();
      lastSendTimeRef.current = now;
    } else {
      // Trailing edge — guarantees the final position is always sent
      sendThrottleRef.current = setTimeout(() => {
        sendPrompts();
        lastSendTimeRef.current = Date.now();
        sendThrottleRef.current = null;
      }, THROTTLE_MS - elapsed);
    }
  }, [prompts, listener, sendPrompts]);

  // Clean up trailing-edge timer on unmount
  useEffect(() => () => {
    if (sendThrottleRef.current) clearTimeout(sendThrottleRef.current);
  }, []);

  // ─── Bridge: lifecycle ─────────────────────────────────────────────
  useEffect(() => {
    window.updateState = (state: any) => {
      // When model loads, re-send our prompts so the engine uses the prompt surface
      if (state.modelName) {
        setModelName(state.modelName);
        sendPrompts();
      }
      if (state.isPlaying !== undefined) {
        setIsPlaying(state.isPlaying);
      }
      if (state.audioLevel !== undefined) {
        setAudioLevel(state.audioLevel);
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
      if (state.downloadPath !== undefined) {
        setDownloadPath(state.downloadPath);
      }
      if (state.resourcesMissing !== undefined) {
        setResourcesMissing(state.resourcesMissing);
      }
      if (state.resourcesProgress !== undefined) {
        setResourcesProgress(state.resourcesProgress);
      }

      if (state.metrics !== undefined) {
        setMetrics(m => ({ ...m, ...state.metrics }));
      }
      if (state.params !== undefined) {
        setParamsState(p => {
          const next = { ...p };
          if (state.params.temperature !== undefined) next.temperature = state.params.temperature;
          if (state.params.topk !== undefined) next.topk = state.params.topk;
          if (state.params.cfgnotes !== undefined) next.cfgnotes = state.params.cfgnotes;
          if (state.params.cfgmusiccoca !== undefined) next.cfgmusiccoca = state.params.cfgmusiccoca;
          if (state.params.cfgdrums !== undefined) next.cfgdrums = state.params.cfgdrums;
          if (state.params.unmaskwidth !== undefined) next.unmaskwidth = state.params.unmaskwidth;
          if (state.params.buffersize !== undefined) next.buffersize = state.params.buffersize;
          if (state.params.volume !== undefined) next.volume = state.params.volume;
          if (state.params.drumless !== undefined) next.drumless = state.params.drumless;
          return next;
        });
      }
      if (state.openSettings !== undefined) {
        setIsSettingsOpen(!!state.openSettings);
      }
      // Audio prompt loaded from native file picker
      if (state.isAudioPrompt && state.prompt) {
        setPrompts(prev => {
          const existing = prev.findIndex(p => p.isAudio);
          if (existing !== -1) {
            return prev.map((p, i) => i === existing ? { ...p, label: state.prompt } : p);
          }
          const el = promptSurfaceRef.current;
          const w = el ? el.getBoundingClientRect().width : 800;
          const h = el ? el.getBoundingClientRect().height : 600;
          const pad = 60;
          return [...prev, {
            id: nextIdRef.current++,
            x: pad + Math.random() * (w - pad * 2),
            y: pad + Math.random() * (h - pad * 2),
            label: state.prompt,
            colorIndex: nextColorRef.current++,
            isAudio: true,
          }];
        });
      }
    };

    post({ type: 'uiReady' });
    post({ type: 'listRemoteModels' });

    return () => {
      delete (window as any).updateState;
    };
  }, [sendPrompts]);

  // ─── UI callbacks ─────────────────────────────────────────────────

  const openSettings = () => {
    post({ type: 'openSettings' });
  };

  const togglePlay = () => {
    post({ type: 'togglePlay' });
  };

  const handlePromptMove = useCallback((id: number, x: number, y: number) => {
    setPrompts(prev => prev.map(p => p.id === id ? { ...p, x, y } : p));
  }, []);

  const handleListenerMove = useCallback((x: number, y: number) => {
    setListener({ x, y });
  }, []);

  const handleBallSelect = useCallback((id: number | null) => {
    setSelectedBallId(id);
  }, []);

  const handlePromptAdd = useCallback((x: number, y: number) => {
    const id = nextIdRef.current++;
    const colorIndex = nextColorRef.current++;
    // When the deck runs out, reshuffle and reset the index
    if (deckIndexRef.current >= SHUFFLED_SUGGESTIONS.length) {
      shuffle(SHUFFLED_SUGGESTIONS);
      deckIndexRef.current = 0;
    }
    const label = SHUFFLED_SUGGESTIONS[deckIndexRef.current++];
    setPrompts(prev => [...prev, { id, x, y, label, colorIndex }]);
  }, []);

  const handleTextChange = useCallback((id: number, text: string) => {
    setPrompts(prev => prev.map(p => p.id === id ? { ...p, label: text } : p));
  }, []);

  const handlePromptDelete = useCallback((id: number) => {
    // If deleting an audio prompt, clear it in the engine
    const deleted = promptsRef.current.find(p => p.id === id);
    if (deleted?.isAudio) {
      post({ type: 'clearAudioPrompt' });
    }
    setPrompts(prev => prev.filter(p => p.id !== id));
    setSelectedBallId(prev => prev === id ? null : prev);
  }, []);

  const handleFirstThrow = useCallback(() => setHasThrown(true), []);

  // ─── Render ────────────────────────────────────────────────────────

  return (
    <div style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)' }}>
      {/* Transport — top left */}
      <div style={{
        position: 'fixed',
        top: 'var(--app-padding)',
        left: 'var(--app-padding)',
        zIndex: 10,
        color: '#FFF',
      }}>
        <TransportControls
          isPlaying={isPlaying}
          onTogglePlay={togglePlay}
          volume={paramsState.volume}
          onVolumeChange={(v) => sendParamChange(5, v)}
          onReset={resetModel}
          volumeSliderPosition="bottom"
          model={modelName}
        />
      </div>

      {/* Model selector + Settings — top right */}
      <div style={{
        position: 'fixed',
        top: 'var(--app-padding)',
        right: 'var(--app-padding)',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        color: '#FFF',
      }}>
        <ModelSelector
          modelName={modelName}
          localModels={localModels}
          remoteModels={remoteModels}
          downloadProgress={downloadProgress}

          onSelectModel={(name: string) => post({ type: 'selectModel', name })}
          onDownloadModel={(name: string) => post({ type: 'downloadModel', name })}
          onDeleteModel={(name: string) => post({ type: 'deleteModel', name })}
          onSelectFolder={() => post({ type: 'selectDownloadFolder' })}
        />
        <IconButton
          onClick={() => setIsSettingsOpen(true)}
          variant="ghost"
          sx={{
            width: 40,
            height: 40,
          }}
          title="Settings (Cmd+,)"
        >
          <TuneIcon sx={{ fontSize: 20 }} />
        </IconButton>
      </div>

      {/* Audio Meter — left edge, vertical, centered */}
      {/* <div style={{
        position: 'fixed',
        right: '34px',
        top: '50%',
        transform: 'translateY(-50%) rotate(-90deg) translateX(50%)',
        transformOrigin: 'top right',
        zIndex: 10,
        pointerEvents: 'none',
      }}>
        <AudioMeter leftLevel={audioLevel} rightLevel={audioLevel} width="120px" height="14px" />
      </div> */}

      {/* Top spacer — keeps prompt surface below fixed header elements */}
      <div style={{ height: 'calc(var(--app-padding) + 56px + var(--app-padding))', flexShrink: 0 }} />

      {/* PromptSurface */}
      <div ref={promptSurfaceRef} style={{ flex: 1, position: 'relative' }}>
        <PromptSurface
          prompts={prompts}
          listener={listener}
          selectedBallId={selectedBallId}
          onPromptMove={handlePromptMove}
          onListenerMove={handleListenerMove}
          onBallSelect={handleBallSelect}
          onPromptAdd={handlePromptAdd}
          onPromptTextChange={handleTextChange}
          onPromptDelete={handlePromptDelete}
          physicsSpeed={physicsSpeed}
          onFirstThrow={handleFirstThrow}
          isPlaying={isPlaying}
          audioLevel={audioLevel}
          debug={debug}
          collisions={collisionsEnabled}
        />
      </div>

      {/* TimingIndicator — fixed bottom-left */}
      <div style={{
        position: 'fixed',
        bottom: 'calc(var(--app-padding) + 3px)',
        left: 'var(--app-padding)',
        zIndex: 10,
        color: 'var(--color-muted)',
      }}>
        <TimingIndicator frameMs={metrics.frameMs} droppedFrames={metrics.droppedFrames} buffersize={paramsState.buffersize} onBufferChange={(v) => sendParamChange(8, v)} isPlaying={isPlaying} bufferLabel="buffer" />
      </div>

      {/* ── Bottom bar ── */}
      <div style={{ display: 'flex', alignItems: 'center', padding: 'var(--app-padding)', flexShrink: 0, gap: '12px', position: 'relative', justifyContent: 'flex-end' }}>

        {/* Upload Audio Prompt */}
        <Tooltip title="Upload audio prompt">
          <IconButton
            onClick={() => post({ type: 'loadAudioPrompt' })}
            sx={{
              width: 40,
              height: 40,
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>upload</span>
          </IconButton>
        </Tooltip>

        {/* Add Prompt */}
        <Tooltip title="Add prompt">
          <IconButton
            onClick={() => {
              const el = promptSurfaceRef.current;
              if (!el) return;
              const { width, height } = el.getBoundingClientRect();
              const pad = 60;
              const x = pad + Math.random() * (width - pad * 2);
              const y = pad + Math.random() * (height - pad * 2);
              handlePromptAdd(x, y);
            }}
            sx={{
              width: 40,
              height: 40,
            }}
          >
            <span className="material-icons" style={{ fontSize: '20px' }}>add</span>
          </IconButton>
        </Tooltip>

        {/* Speed slider — absolute, aligned to the right of the bar (left of Add Prompt) */}
        <div
          className={`speed-slider-dock${hasThrown ? ' visible' : ''}`}
          style={{
            position: 'absolute',
            bottom: 'calc(var(--app-padding) + 1px)',
            right: '140px',
            transform: hasThrown ? 'translateY(0)' : 'translateY(200%)',
            maxWidth: '260px',
            width: '100%',
            zIndex: 10,
            pointerEvents: hasThrown ? 'auto' : 'none',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Tooltip title={collisionsEnabled ? "Collisions enabled" : "Collisions disabled"}>
            <IconButton
              onClick={() => setCollisionsEnabled(prev => !prev)}
              sx={{
                width: 32,
                height: 32,
                color: collisionsEnabled ? '#71fade' : '#FFF',
                mr: '11px',
                flexShrink: 0,
              }}
            >
              {collisionsEnabled ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{overflow: 'visible'}}>
                  <circle cx="4.5" cy="12" r="7" />
                  <circle cx="19.5" cy="12" r="7" />
                </svg>
              ) : (
                <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>join</span>
              )}
            </IconButton>
          </Tooltip>
          <Turtle style={{ width: '20px', height: '20px', flexShrink: 0 }} color="white" strokeWidth={1.5} />
          <input
            type="range"
            min="0"
            max="1"
            step="0.005"
            value={sliderPos}
            onChange={(e) => setSliderPos(parseFloat(e.target.value))}
            onMouseDown={() => document.body.classList.add('is-dragging')}
            onMouseUp={() => document.body.classList.remove('is-dragging')}
            className={physicsSpeed === 0 ? 'speed-zero' : undefined}
            style={{ flex: 1 }}
          />
          <Rabbit style={{ width: '20px', height: '20px', flexShrink: 0 }} color="white" strokeWidth={1.5} />
        </div>
      </div>



      {/* <div
        className={`dev-badge${debug ? ' debug-on' : ''}`}
        onClick={() => setDebug(d => !d)}
      >DEV</div> */}
      <SettingsPanel
        open={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        temperature={paramsState.temperature}
        topk={paramsState.topk}
        cfgnotes={paramsState.cfgnotes}
        cfgmusiccoca={paramsState.cfgmusiccoca}
        cfgdrums={paramsState.cfgdrums}
        unmaskwidth={paramsState.unmaskwidth}
        onParamChange={sendParamChange}
        onResetDefaults={handleResetDefaults}
        showNoteCfg={false}
        showPromptCfg={false}
        showDrumsCfg={false}
        showUnmaskWidth={false}
        showOnsetMode={false}
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
