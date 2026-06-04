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

/**
 * Shared default parameter values for all example apps (MRT2, Jam, Collider).
 *
 * All values are native engine values — no display↔native remapping.
 * CFG params (notes, style/musiccoca, drums) range from 0 to 5.
 */

// ─── Default values (native engine units) ────────────────────────────────────

export const DEFAULT_TEMPERATURE = 1.1;
export const DEFAULT_TOPK = 50;

/** Style / Prompt Strength (0–5 range). */
export const DEFAULT_CFG_MUSICCOCA = 1.6;

/** Note Strength (0–5 range). */
export const DEFAULT_CFG_NOTES = 2.4;

export const DEFAULT_CFG_DRUMS = 4.0;
export const DEFAULT_UNMASK_WIDTH = 0;
export const DEFAULT_BUFFER_SIZE = 0;
export const DEFAULT_VOLUME = 0.0;

// ─── Per-app overrides ───────────────────────────────────────────────────────

/** Collider uses cfgnotes = 0.0. */
export const COLLIDER_CFG_NOTES = 0.0;

/** Collider style */
export const COLLIDER_CFG_MUSICCOCA = 5.0;
