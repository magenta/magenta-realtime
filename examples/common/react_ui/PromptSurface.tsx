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

import { useRef, useState, useEffect, useCallback } from 'react';
import { flushSync } from 'react-dom';
import GUI from 'lil-gui';
import { ALL_COLORS } from './colors';
import './prompt-surface.css';

// ─── Constants ───────────────────────────────────────────────────────────────

const STROKE_WIDTH = 3;
const DRAG_THRESHOLD_PX = 3;
const THROW_FRESHNESS_MS = 40;

// Tunable via lil-gui in debug mode
const config = {
  promptRadius: 20,
  listenerRadius: 25,
  minThrowSpeed: 250,
  maxThrowSpeed: 3000,
  movingHitboxScale: 4,
  boostSettleRate: 4,
  // Lines
  lineDashMinSpeed: 20,
  lineDashMaxSpeed: 100,
  lineDashLen: 5,
  lineDashGap: 5,
  lineMaxWidth: 1,
  showDashes: true,

  // Volume rings (audio-reactive)
  showVolRings: true,
  showWeightPie: true,
  volRingMin: 20,
  volRingMax: 120,
  volRingAlpha: 0.9,
  volRingRelease: 12,
  volRingWeightScale: true,
  // Collisions
  collisions: false,
  // Debug
  outlines: false,
};

// ─── Types ───────────────────────────────────────────────────────────────────

export interface PromptNode {
  id: number;
  x: number;       // pixels
  y: number;       // pixels
  label: string;
  colorIndex: number;
  isAudio?: boolean;
}

export interface ListenerNode {
  x: number;       // pixels
  y: number;       // pixels
}

// ─── Drag state (ref-based, no re-renders during drag) ──────────────────────

interface DragInfo {
  type: 'prompt' | 'listener';
  id?: number;
  startX: number;
  startY: number;
  offsetX: number;   // cursor offset from element center (pixels)
  offsetY: number;
  didDrag: boolean;
}

type BallEntry = { type: 'prompt'; prompt: PromptNode } | { type: 'listener' };

type BallState = { x: number; y: number; vx: number; vy: number; boost: number };

const FALLOFF = 2.0;

export function calculateWeights(listener: ListenerNode, prompts: PromptNode[]): number[] {
  if (prompts.length === 0) return [];
  const distances = prompts.map(p =>
    Math.sqrt((p.x - listener.x) ** 2 + (p.y - listener.y) ** 2)
  );
  const zeroIdx = distances.findIndex(d => d < 1);
  if (zeroIdx !== -1) {
    return prompts.map((_, i) => i === zeroIdx ? 1.0 : 0.0);
  }
  const raw = distances.map(d => 1 / d ** FALLOFF);
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map(w => w / sum);
}

function hitboxRadius(movingRef: Map<string, { vx: number; vy: number }>, key: string, baseRadius: number): number {
  const ball = movingRef.get(key);
  const speed = ball ? Math.sqrt(ball.vx ** 2 + ball.vy ** 2) : 0;
  const t = Math.min(1, speed / config.maxThrowSpeed);
  const scale = 1 + (config.movingHitboxScale - 1) * t;
  return baseRadius * scale;
}

/** Advance a single ball: decay boost, clamp speed, integrate position, bounce off walls. */
function advanceBall(ball: BallState, key: string, rawDt: number, globalSpeed: number, w: number, h: number) {
  ball.boost += (globalSpeed - ball.boost) * Math.min(1, rawDt * config.boostSettleRate);
  const dt = rawDt * ball.boost;

  const speed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
  if (speed > config.maxThrowSpeed) {
    const s = config.maxThrowSpeed / speed;
    ball.vx *= s;
    ball.vy *= s;
  }

  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;

  const r = key === 'listener' ? config.listenerRadius : config.promptRadius;
  if (ball.x < r)     { ball.x = 2 * r - ball.x;     ball.vx = Math.abs(ball.vx); }
  if (ball.x > w - r) { ball.x = 2 * (w - r) - ball.x; ball.vx = -Math.abs(ball.vx); }
  if (ball.y < r)     { ball.y = 2 * r - ball.y;     ball.vy = Math.abs(ball.vy); }
  if (ball.y > h - r) { ball.y = 2 * (h - r) - ball.y; ball.vy = -Math.abs(ball.vy); }
}

/** Resolve elastic collisions between all balls. Activates stationary balls that get hit. */
function resolveCollisions(
  moving: Map<string, BallState>,
  prompts: PromptNode[],
  listener: ListenerNode,
  drag: DragInfo | null,
  globalSpeed: number,
) {
  type Snap = { key: string; x: number; y: number; vx: number; vy: number; r: number; boost: number };
  const snaps: Snap[] = [];

  // Listener
  const lm = moving.get('listener');
  const isDraggingListener = drag?.type === 'listener';
  snaps.push({
    key: 'listener',
    x: lm?.x ?? listener.x, y: lm?.y ?? listener.y,
    vx: isDraggingListener ? 0 : (lm?.vx ?? 0),
    vy: isDraggingListener ? 0 : (lm?.vy ?? 0),
    r: config.listenerRadius,
    boost: lm?.boost ?? globalSpeed,
  });

  // Prompts
  for (const p of prompts) {
    const k = String(p.id);
    const m = moving.get(k);
    const isDragging = drag?.type === 'prompt' && drag.id === p.id;
    snaps.push({
      key: k,
      x: m?.x ?? p.x, y: m?.y ?? p.y,
      vx: isDragging ? 0 : (m?.vx ?? 0),
      vy: isDragging ? 0 : (m?.vy ?? 0),
      r: config.promptRadius,
      boost: m?.boost ?? globalSpeed,
    });
  }

  // Pairwise elastic collision
  for (let i = 0; i < snaps.length; i++) {
    for (let j = i + 1; j < snaps.length; j++) {
      const a = snaps[i], b = snaps[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const minDist = a.r + b.r;
      if (dist >= minDist || dist < 0.001) continue;

      const nx = dx / dist, ny = dy / dist;
      const vRel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
      if (vRel <= 0) continue;

      const dragA = drag && ((drag.type === 'listener' && a.key === 'listener') || (drag.type === 'prompt' && String(drag.id) === a.key));
      const dragB = drag && ((drag.type === 'listener' && b.key === 'listener') || (drag.type === 'prompt' && String(drag.id) === b.key));

      // Velocity exchange
      if (dragA) {
        b.vx += vRel * nx; b.vy += vRel * ny;
      } else if (dragB) {
        a.vx -= vRel * nx; a.vy -= vRel * ny;
      } else {
        a.vx -= vRel * nx; a.vy -= vRel * ny;
        b.vx += vRel * nx; b.vy += vRel * ny;
      }

      // Separate overlap
      const overlap = minDist - dist;
      if (!dragA && !dragB) {
        a.x -= nx * overlap * 0.5; a.y -= ny * overlap * 0.5;
        b.x += nx * overlap * 0.5; b.y += ny * overlap * 0.5;
      } else if (dragA) {
        b.x += nx * overlap; b.y += ny * overlap;
      } else {
        a.x -= nx * overlap; a.y -= ny * overlap;
      }
    }
  }

  // Write back
  for (const snap of snaps) {
    const isDragging = drag && ((drag.type === 'listener' && snap.key === 'listener') || (drag.type === 'prompt' && String(drag.id) === snap.key));
    if (isDragging) continue;
    const existing = moving.get(snap.key);
    if (existing) {
      existing.x = snap.x; existing.y = snap.y;
      existing.vx = snap.vx; existing.vy = snap.vy;
    } else if (snap.vx !== 0 || snap.vy !== 0) {
      moving.set(snap.key, { x: snap.x, y: snap.y, vx: snap.vx, vy: snap.vy, boost: snap.boost });
    }
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

export function PromptSurface({
  prompts,
  listener,
  selectedBallId,
  onPromptMove,
  onListenerMove,
  onBallSelect,
  onPromptAdd,
  onPromptTextChange,
  onPromptDelete,
  physicsSpeed,
  onFirstThrow,
  isPlaying,
  audioLevel,
  debug,
  physicsEnabled = true,
  active = true,
  collisions = false,
}: {
  prompts: PromptNode[];
  listener: ListenerNode;
  selectedBallId: number | null;
  onPromptMove: (id: number, x: number, y: number) => void;
  onListenerMove: (x: number, y: number) => void;
  onBallSelect: (id: number | null) => void;
  onPromptAdd: (x: number, y: number) => void;
  onPromptTextChange: (id: number, text: string) => void;
  onPromptDelete: (id: number) => void;
  physicsSpeed: number;
  onFirstThrow: () => void;
  isPlaying: boolean;
  audioLevel: number;
  debug?: boolean;
  physicsEnabled?: boolean;
  active?: boolean;
  collisions?: boolean;
}) {
  config.collisions = collisions;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<DragInfo | null>(null);
  const justCreatedIdRef = useRef<number | null>(null);
  const knownIdsRef = useRef<Set<number> | null>(null);
  const animatedIdsRef = useRef<Set<number>>(new Set());

  // Detect newly added prompts → mark for auto-focus
  // Don't seed until prompts is non-empty (initial layout arrives via rAF)
  if (knownIdsRef.current === null) {
    if (prompts.length > 0) {
      knownIdsRef.current = new Set(prompts.map(p => p.id));
      // Mark existing prompts as already animated — only new ones should scale in
      prompts.forEach(p => animatedIdsRef.current.add(p.id));
    }
  } else {
    for (const p of prompts) {
      if (!knownIdsRef.current.has(p.id)) {
        justCreatedIdRef.current = p.id;
      }
    }
    knownIdsRef.current = new Set(prompts.map(p => p.id));
  }

  const preEditRef = useRef('');
  const trashRef = useRef<HTMLDivElement | null>(null);
  const trashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isDraggingPrompt, setIsDraggingPrompt] = useState(false);
  const [isOverTrash, setIsOverTrash] = useState(false);
  // Sorted ball keys (farthest-first) — only recomputed on mouse move
  const [sortOrder, setSortOrder] = useState<string[]>([]);

  const dragSamplesRef = useRef<{ x: number; y: number; t: number }[]>([]);
  const rafRef = useRef<number>(0);
  const dashOffsetsRef = useRef<Map<string, number>>(new Map());


  // ─── Refs mirroring props (for ResizeObserver callback) ─────────────
  const promptsRef = useRef(prompts);
  promptsRef.current = prompts;
  const listenerRef = useRef(listener);
  listenerRef.current = listener;

  // Physics speed (prop-driven, ref for rAF access)
  const physicsSpeedRef = useRef(physicsSpeed);
  physicsSpeedRef.current = physicsSpeed;
  const isPlayingRef = useRef(isPlaying);
  isPlayingRef.current = isPlaying;
  const audioLevelRef = useRef(audioLevel);
  audioLevelRef.current = audioLevel;
  const smoothedLevelRef = useRef(0);


  // ─── Physics (billiard ball throw) ──────────────────────────────────
  const movingRef = useRef<Map<string, BallState>>(new Map());

  // ─── lil-gui (debug only) ─────────────────────────────────────────
  // Force re-render when lil-gui changes config
  const [, forceRender] = useState(0);
  useEffect(() => {
    if (!debug) return;
    const gui = new GUI({ closeFolders: true });
    // Show toggles at root level
    gui.add(config, 'showDashes');
    gui.add(config, 'showVolRings');
    gui.add(config, 'showWeightPie');
    const appearance = gui.addFolder('Appearance');
    appearance.add(config, 'promptRadius');
    appearance.add(config, 'listenerRadius');
    appearance.add(config, 'outlines');
    const physics = gui.addFolder('Physics');
    physics.add(config, 'minThrowSpeed');
    physics.add(config, 'maxThrowSpeed');
    physics.add(config, 'movingHitboxScale');
    physics.add(config, 'boostSettleRate');
    physics.add(config, 'collisions');
    const lines = gui.addFolder('Lines');
    lines.add(config, 'lineDashMinSpeed');
    lines.add(config, 'lineDashMaxSpeed');
    lines.add(config, 'lineDashLen');
    lines.add(config, 'lineDashGap');
    lines.add(config, 'lineMaxWidth');

    const vol = gui.addFolder('Volume Rings');
    vol.add(config, 'volRingMin');
    vol.add(config, 'volRingMax');
    vol.add(config, 'volRingAlpha');
    vol.add(config, 'volRingRelease');
    vol.add(config, 'volRingWeightScale');
    gui.onChange(() => forceRender(n => n + 1));

    return () => gui.destroy();
  }, [debug]);

  // ─── Stage size (pixels) ──────────────────────────────────────────
  const [stageW, setStageW] = useState(800);
  const [stageH, setStageH] = useState(600);
  const stageSizeRef = useRef({ w: 800, h: 600 });

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const measure = () => {
      const { width, height } = svg.getBoundingClientRect();
      const w = Math.round(width);
      const h = Math.round(height);
      if (w <= 0 || h <= 0) return;  // Skip when hidden (display:none)
      if (w !== stageSizeRef.current.w || h !== stageSizeRef.current.h) {
        stageSizeRef.current = { w, h };

        // flushSync forces React to commit the viewBox resize AND
        // position clamps in one synchronous render — no intermediate
        // frame where the stage shrank but balls haven't moved yet.
        flushSync(() => {
          setStageW(w);
          setStageH(h);

          const lRef = listenerRef.current;
          const lx = Math.max(config.listenerRadius, Math.min(w - config.listenerRadius, lRef.x));
          const ly = Math.max(config.listenerRadius, Math.min(h - config.listenerRadius, lRef.y));
          if (lx !== lRef.x || ly !== lRef.y) onListenerMove(lx, ly);

          promptsRef.current.forEach(p => {
            const px = Math.max(config.promptRadius, Math.min(w - config.promptRadius, p.x));
            const py = Math.max(config.promptRadius, Math.min(h - config.promptRadius, p.y));
            if (px !== p.x || py !== p.y) onPromptMove(p.id, px, py);
          });
        });

        // Clamp physics state (mutable refs, no render needed)
        const movingListener = movingRef.current.get('listener');
        if (movingListener) {
          movingListener.x = Math.max(config.listenerRadius, Math.min(w - config.listenerRadius, movingListener.x));
          movingListener.y = Math.max(config.listenerRadius, Math.min(h - config.listenerRadius, movingListener.y));
        }
        movingRef.current.forEach((ball, key) => {
          if (key === 'listener') return;
          const r = config.promptRadius;
          ball.x = Math.max(r, Math.min(w - r, ball.x));
          ball.y = Math.max(r, Math.min(h - r, ball.y));
        });
      }
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(svg);
    return () => ro.disconnect();
  }, [onPromptMove, onListenerMove]);

  useEffect(() => {
    if (!active) return;  // Don't run rAF loop when inactive

    let lastTime = performance.now();

    const tick = (now: number) => {
      const rawDt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;

      const { w, h } = stageSizeRef.current;
      const globalSpeed = physicsSpeedRef.current;

      // Advance each moving ball (only when physics is enabled)
      if (physicsEnabled) {
        movingRef.current.forEach((ball, key) => advanceBall(ball, key, rawDt, globalSpeed, w, h));

      // Resolve ball-to-ball collisions
        if (config.collisions) {
          resolveCollisions(movingRef.current, promptsRef.current, listenerRef.current, dragRef.current, globalSpeed);
        }

      // Commit positions to React
        movingRef.current.forEach((ball, key) => {
          if (key === 'listener') {
            onListenerMove(ball.x, ball.y);
          } else {
            onPromptMove(parseInt(key), ball.x, ball.y);
          }
        });
      }

      // Advance per-line dash offsets based on IDW weight (only while playing)
      if (isPlayingRef.current) {
        const weights = calculateWeights(listenerRef.current, promptsRef.current);
        promptsRef.current.forEach((p, i) => {
          const key = String(p.id);
          const prev = dashOffsetsRef.current.get(key) ?? 0;
          const speed = config.lineDashMinSpeed + (config.lineDashMaxSpeed - config.lineDashMinSpeed) * weights[i];
          dashOffsetsRef.current.set(key, prev + rawDt * speed);
        });
      }

      // Smooth audio level: instant attack, exponential release
      const target = audioLevelRef.current;
      const current = smoothedLevelRef.current;
      if (target >= current) {
        smoothedLevelRef.current = target;
      } else {
        smoothedLevelRef.current += (target - current) * Math.min(1, rawDt * config.volRingRelease);
      }

      // Force re-render for animations
      forceRender(n => n + 1);

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [onPromptMove, onListenerMove, physicsEnabled, active]);

  // Keep a ref to selectedBallId for the keydown handler
  const selectedBallIdRef = useRef(selectedBallId);
  selectedBallIdRef.current = selectedBallId;

  // Delete/Backspace deletes the selected ball (only when no input is focused)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        // Don't intercept if an input/textarea is focused
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;
        if (selectedBallIdRef.current !== null) {
          e.preventDefault();
          onPromptDelete(selectedBallIdRef.current);
          movingRef.current.delete(String(selectedBallIdRef.current));
          onBallSelect(null);
        }
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onPromptDelete, onBallSelect]);

  // ─── Coordinate helpers ──────────────────────────────────────────────

  const clientToStage = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const r = svg.getBoundingClientRect();
    return {
      x: clientX - r.left,
      y: clientY - r.top,
    };
  }, []);

  // ─── Pointer handlers ───────────────────────────────────────────────

  const handlePointerMove = useCallback((e: PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;

    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    if (!drag.didDrag && Math.sqrt(dx * dx + dy * dy) < DRAG_THRESHOLD_PX) return;
    drag.didDrag = true;

    // Hit-test trash zone during prompt drag
    if (drag.type === 'prompt' && trashRef.current) {
      const rect = trashRef.current.getBoundingClientRect();
      setIsOverTrash(
        e.clientX >= rect.left && e.clientX <= rect.right &&
        e.clientY >= rect.top && e.clientY <= rect.bottom
      );
    }

    const pos = clientToStage(e.clientX, e.clientY);
    const { w, h } = stageSizeRef.current;
    const r = drag.type === 'listener' ? config.listenerRadius : config.promptRadius;
    const x = Math.max(r, Math.min(w - r, pos.x - drag.offsetX));
    const y = Math.max(r, Math.min(h - r, pos.y - drag.offsetY));

    // Record velocity samples for throw
    dragSamplesRef.current.push({ x, y, t: performance.now() });
    if (dragSamplesRef.current.length > 5) dragSamplesRef.current.shift();

    if (drag.type === 'prompt' && drag.id !== undefined) {
      onPromptMove(drag.id, x, y);
    } else if (drag.type === 'listener') {
      onListenerMove(x, y);
    }
  }, [clientToStage, onPromptMove, onListenerMove]);

  const handlePointerUp = useCallback((e: PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;

    document.body.classList.remove('is-dragging');
    document.removeEventListener('pointermove', handlePointerMove);
    document.removeEventListener('pointerup', handlePointerUp);
    if (trashTimerRef.current) {
      clearTimeout(trashTimerRef.current);
      trashTimerRef.current = null;
    }
    setIsDraggingPrompt(false);
    setIsOverTrash(false);

    // Drop on trash zone → delete prompt
    if (drag.didDrag && drag.type === 'prompt' && drag.id !== undefined && trashRef.current) {
      const rect = trashRef.current.getBoundingClientRect();
      if (
        e.clientX >= rect.left && e.clientX <= rect.right &&
        e.clientY >= rect.top && e.clientY <= rect.bottom
      ) {
        onPromptDelete(drag.id);
        dragSamplesRef.current = [];
        movingRef.current.delete(String(drag.id));
        dragRef.current = null;
        return;
      }
    }

    // Click (no drag) on a prompt ball → select the ball
    if (!drag.didDrag && drag.type === 'prompt' && drag.id !== undefined) {
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      onBallSelect(drag.id);
    }

    // Compute release velocity and start physics (only when physics is enabled)
    if (drag.didDrag && physicsEnabled) {
      const samples = dragSamplesRef.current;
      const now = performance.now();
      if (samples.length >= 2 && (now - samples[samples.length - 1].t) < THROW_FRESHNESS_MS) {
        const last = samples[samples.length - 1];
        const prev = samples[Math.max(0, samples.length - 3)];
        const dt = (last.t - prev.t) / 1000;
        if (dt > 0.005) {
          let vx = (last.x - prev.x) / dt;
          let vy = (last.y - prev.y) / dt;
          const speed = Math.sqrt(vx * vx + vy * vy);
          if (speed > config.minThrowSpeed) {
            // Cap to max speed
            if (speed > config.maxThrowSpeed) {
              const scale = config.maxThrowSpeed / speed;
              vx *= scale;
              vy *= scale;
            }
            const ballKey = drag.type === 'listener' ? 'listener' : String(drag.id);
            // boost=1 → launch at full speed, decays toward physicsSpeed
            movingRef.current.set(ballKey, { x: last.x, y: last.y, vx, vy, boost: 1.0 });
            onFirstThrow();
          }
        }
      }
    }
    dragSamplesRef.current = [];

    dragRef.current = null;
  }, [onBallSelect, handlePointerMove, physicsEnabled, onPromptDelete, onFirstThrow]);

  const handleElementPointerDown = useCallback((
    e: React.PointerEvent<SVGCircleElement>,
    type: 'prompt' | 'listener',
    id?: number,
  ) => {
    e.stopPropagation();
    e.preventDefault();
    document.body.classList.add('is-dragging');

    // Grabbing anything other than the selected prompt clears ball selection
    if (type === 'listener' || id !== selectedBallId) {
      onBallSelect(null);
    }

    // Compute grab offset so element doesn't snap to cursor center
    const grabPos = clientToStage(e.clientX, e.clientY);
    const el = e.target as SVGCircleElement;
    const elX = parseFloat(el.getAttribute('cx') || '0');
    const elY = parseFloat(el.getAttribute('cy') || '0');

    dragRef.current = {
      type, id,
      startX: e.clientX, startY: e.clientY,
      offsetX: grabPos.x - elX,
      offsetY: grabPos.y - elY,
      didDrag: false,
    };

    // Stop the ball if it's currently moving
    const ballKey = type === 'listener' ? 'listener' : String(id);
    movingRef.current.delete(ballKey);
    dragSamplesRef.current = [];

    if (type === 'prompt') {
      trashTimerRef.current = setTimeout(() => setIsDraggingPrompt(true), 200);
    }

    document.addEventListener('pointermove', handlePointerMove);
    document.addEventListener('pointerup', handlePointerUp);
  }, [clientToStage, handlePointerMove, handlePointerUp, selectedBallId, onBallSelect]);

  const handleCanvasPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    const target = e.target as Element;
    if (target === svgRef.current || target.getAttribute('data-bg') === 'true') {
      onBallSelect(null);
    }
  }, [onBallSelect]);

  const handleDoubleClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const target = e.target as Element;
    if (target !== svgRef.current && target.getAttribute('data-bg') !== 'true') return;

    const pos = clientToStage(e.clientX, e.clientY);
    onPromptAdd(pos.x, pos.y);
  }, [clientToStage, onPromptAdd]);

  // ─── Render ──────────────────────────────────────────────────────────

  // Compute weights once for the whole render pass
  const weights = calculateWeights(listener, prompts);
  const weightMap = new Map(prompts.map((p, i) => [p.id, weights[i]]));

  // Build ball entries and apply stored sort order
  const ballMap = new Map<string, BallEntry>();
  prompts.forEach(p => ballMap.set(String(p.id), { type: 'prompt', prompt: p }));
  ballMap.set('listener', { type: 'listener' });
  const allBalls: BallEntry[] = [];
  // Add balls in sort order first (farthest → nearest)
  for (const key of sortOrder) {
    const entry = ballMap.get(key);
    if (entry) {
      allBalls.push(entry);
      ballMap.delete(key);
    }
  }
  // Append any new balls not yet in the sort order
  ballMap.forEach(entry => allBalls.push(entry));

  return (
    <div
      className={config.outlines ? 'debug' : undefined}
      style={{ position: 'absolute', inset: 0 }}
      onPointerMove={(e) => {
        if (dragRef.current) return; // Freeze layer order while dragging
        const pos = clientToStage(e.clientX, e.clientY);
        // Sort all balls by distance to cursor
        const entries: { key: string; d2: number }[] = promptsRef.current.map(p => ({
          key: String(p.id),
          d2: (p.x - pos.x) ** 2 + (p.y - pos.y) ** 2,
        }));
        const l = listenerRef.current;
        entries.push({ key: 'listener', d2: (l.x - pos.x) ** 2 + (l.y - pos.y) ** 2 });
        entries.sort((a, b) => b.d2 - a.d2); // farthest first
        const newOrder = entries.map(e => e.key);
        setSortOrder(prev => {
          // Only re-render if order actually changed
          if (prev.length === newOrder.length && prev.every((k, i) => k === newOrder[i])) return prev;
          return newOrder;
        });
      }}
    >
      {/* Trash drop zone — right edge, slides in from right */}
      <div
        ref={trashRef}
        className={`trash-zone${isDraggingPrompt ? ' visible' : ''}`}
        style={{ position: 'absolute', left: '16px', top: '50%', pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', zIndex: 0 }}
      >
        <div className={`trash-zone-inner${isOverTrash ? ' over' : ''}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '48px', height: '48px', borderRadius: '9999px', background: isOverTrash ? '#dc2626' : 'var(--color-raised)' }}>
          <span className="material-icons" style={{ fontSize: '20px', color: 'var(--color-muted)' }}>delete_outline</span>
        </div>
      </div>

      {/* SVG layer — geometry + interaction */}
      <svg
        ref={svgRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
        viewBox={`0 0 ${stageW} ${stageH}`}
        onPointerDown={handleCanvasPointerDown}
        onDoubleClick={handleDoubleClick}
      >
        {/* Background rect for event capture */}
        <rect data-bg="true" x="0" y="0" width={stageW} height={stageH} fill="transparent" />
        {/* Debug: physics boundary (shown via .debug CSS) */}
        <rect className="debug-bounds" x="0" y="0" width={stageW} height={stageH}
          fill="none" stroke="cyan" strokeWidth={2} vectorEffect="non-scaling-stroke" />


        {/* Influence lines + volume rings — opacity tracks IDW weight */}
        {(() => {
          const dash = `${config.lineDashLen} ${config.lineDashGap}`;
          return (
            <>
              {prompts.map((p, i) => {
                const w = weights[i];
                const promptColor = ALL_COLORS[p.colorIndex % ALL_COLORS.length];
                return (
                  <line
                    key={`line-${p.id}`}
                    x1={listener.x}
                    y1={listener.y}
                    x2={p.x}
                    y2={p.y}
                    stroke={promptColor}
                    strokeWidth={1 + (config.lineMaxWidth - 1) * w}
                    strokeDasharray={config.showDashes ? dash : undefined}
                    strokeDashoffset={config.showDashes ? (dashOffsetsRef.current.get(String(p.id)) ?? 0) : undefined}
                    vectorEffect="non-scaling-stroke"
                    opacity={w}
                  />
                );
              })}
              {/* Volume rings — radius tracks audio level, opacity tracks IDW weight */}
              {config.showVolRings && prompts.map((p, i) => {
                const base = config.volRingMin + (config.volRingMax - config.volRingMin) * smoothedLevelRef.current;
                const r = config.volRingWeightScale ? config.volRingMin + (base - config.volRingMin) * weights[i] : base;
                const promptColor = ALL_COLORS[p.colorIndex % ALL_COLORS.length];
                return (
                  <circle
                    key={`vol-${p.id}`}
                    cx={p.x}
                    cy={p.y}
                    r={r}
                    fill={promptColor}
                    opacity={weights[i] * config.volRingAlpha}
                  />
                );
              })}
            </>
          );
        })()}

        {/* Balls sorted by distance to cursor — nearest on top */}
        {allBalls.map((entry) => {
          if (entry.type === 'listener') {
            return (
              <g key="listener">
                <circle
                  className="draggable"
                  cx={listener.x}
                  cy={listener.y}
                  r={config.listenerRadius}
                  fill="white"
                  onPointerDown={(e) => handleElementPointerDown(e, 'listener')}
                />
                {movingRef.current.has('listener') && (
                  <circle
                    className="hitbox draggable"
                    cx={listener.x}
                    cy={listener.y}
                    r={hitboxRadius(movingRef.current, 'listener', config.listenerRadius)}
                    fill="transparent"
                    onPointerDown={(e) => handleElementPointerDown(e, 'listener')}
                  />
                )}
              </g>
            );
          }
          const p = entry.prompt;
          const w = weightMap.get(p.id) ?? 0;
          const r = config.promptRadius - 3;
          // Pie slice arc path (12 o'clock start, clockwise)
          let piePath = '';
          if (config.showWeightPie && w > 0.001) {
            const angle = w * Math.PI * 2;
            const sx = p.x + r * Math.sin(0);       // start at top
            const sy = p.y - r;
            const ex = p.x + r * Math.sin(angle);
            const ey = p.y - r * Math.cos(angle);
            const large = angle > Math.PI ? 1 : 0;
            piePath = w >= 0.999
              ? `M ${p.x - r} ${p.y} A ${r} ${r} 0 1 1 ${p.x + r} ${p.y} A ${r} ${r} 0 1 1 ${p.x - r} ${p.y} Z`
              : `M ${p.x} ${p.y} L ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey} Z`;
          }
          const promptColor = ALL_COLORS[p.colorIndex % ALL_COLORS.length];
          const activeColor = promptColor;
          const isBallSelected = p.id === selectedBallId;
          const gClass = animatedIdsRef.current.has(p.id) ? undefined : 'prompt-node';
          return (
            <g
              key={p.id}
              className={gClass}
              onAnimationEnd={() => animatedIdsRef.current.add(p.id)}
            >
              <circle
                className={`ball-selected${isBallSelected ? ' active' : ''}`}
                cx={p.x}
                cy={p.y}
                r={config.promptRadius + 4}
                fill="none"
                stroke={activeColor}
                strokeWidth={1}
                pointerEvents="none"
              />
              <circle
                className="draggable"
                cx={p.x}
                cy={p.y}
                r={config.promptRadius}
                fill="var(--color-bg, #202124)"
                stroke={activeColor}
                strokeWidth={STROKE_WIDTH}
                onPointerDown={(e) => handleElementPointerDown(e, 'prompt', p.id)}
              />
              {piePath && (
                <path
                  d={piePath}
                  fill={activeColor}
                  pointerEvents="none"
                />
              )}
              {movingRef.current.has(String(p.id)) && (
                <circle
                  className="hitbox draggable"
                  cx={p.x}
                  cy={p.y}
                  r={hitboxRadius(movingRef.current, String(p.id), config.promptRadius)}
                  fill="transparent"
                  onPointerDown={(e) => handleElementPointerDown(e, 'prompt', p.id)}
                />
              )}
            </g>
          );
        })}
      </svg>

      {/* DOM overlay — text labels + inline editing */}
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {prompts.map((p) => {
          const isJustCreated = p.id === justCreatedIdRef.current;
          return (
            <div
              key={p.id}
              className="prompt-label"
              style={{
                position: 'absolute',
                left: p.x,
                top: p.y - config.promptRadius - 6,
                transform: 'translate(-50%, -100%)',
              }}
            >
              <input
                ref={!p.isAudio && isJustCreated ? (el) => {
                  if (el) {
                    el.focus();
                    el.select();
                    justCreatedIdRef.current = null;
                  }
                } : undefined}
                readOnly={p.isAudio}
                style={{
                  padding: '2px 12px',
                  borderRadius: '9999px',
                  fontSize: '13px',
                  fontWeight: 500,
                  textAlign: 'center',
                  outline: 'none',
                  margin: 0,
                  border: 'none',
                  whiteSpace: 'nowrap',
                  pointerEvents: p.isAudio ? 'none': 'auto',
                  cursor: p.isAudio ? 'default' : 'text',
                  color: 'white',
                  background: 'rgba(0, 0, 0, 0.5)',
                  fontFamily: "'Google Sans Text', system-ui, sans-serif",
                  fieldSizing: 'content',
                } as React.CSSProperties}
                value={p.isAudio ? `♪ ${p.label}` : p.label}
                spellCheck={false}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                onFocus={() => {
                  if (p.isAudio) return;
                  preEditRef.current = p.label;
                  onBallSelect(null);
                }}
                onBlur={() => {
                  if (p.isAudio) return;
                  if (!p.label.trim()) {
                    onPromptTextChange(p.id, preEditRef.current);
                  }
                }}
                onChange={(e) => {
                  if (p.isAudio) return;
                  onPromptTextChange(p.id, e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    (e.target as HTMLInputElement).blur();
                  }
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
