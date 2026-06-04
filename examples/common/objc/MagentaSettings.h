// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>
#include <magentart/realtime_runner.h>

// ─── Shared default parameter values ─────────────────────────────────────────
// All values are native engine values — no display↔native remapping.
// CFG params (notes, style/musiccoca, drums) range from 0 to 5.

static const float kMagentaDefaultTemperature    = 1.1f;
static const float kMagentaDefaultTopK           = 50.0f;
static const float kMagentaDefaultCfgMusicCoCa   = 1.6f;    // Style / Prompt Strength
static const float kMagentaDefaultCfgNotes       = 2.4f;    // Note Strength
static const float kMagentaDefaultCfgDrums       = 4.0f;
static const float kMagentaDefaultVolume         = 0.0f;
static const float kMagentaDefaultUnmaskWidth    = 0.0f;
static const float kMagentaDefaultBufferSize     = 0.0f;

// Per-app overrides
static const float kColliderDefaultCfgNotes      = 0.0f;
static const float kColliderDefaultCfgMusicCoCa  = 5.0f;

@interface MagentaSettings : NSObject

+ (NSString*)paramKeyForAddress:(int)address;
+ (BOOL)paramIsBool:(int)address;
+ (BOOL)shouldPersistParam:(int)address;

+ (void)applyParamToEngine:(magentart::core::RealtimeRunner*)engine
                   address:(int)address
                     value:(float)value
              prefixString:(NSString*)prefixString;

+ (float)readParamFromEngine:(magentart::core::RealtimeRunner*)engine
                     address:(int)address;

+ (void)restoreSavedParams:(magentart::core::RealtimeRunner*)engine
              prefixString:(NSString*)prefixString;

+ (void)restoreSavedParams:(magentart::core::RealtimeRunner*)engine
              prefixString:(NSString*)prefixString
                  cfgNotes:(float)cfgNotes
              cfgMusicCoCa:(float)cfgMusicCoCa;

/// Reset all generation params to shared defaults.
/// Pass per-app override values for cfgNotes and cfgMusicCoCa if they
/// differ from the shared defaults (e.g. Collider uses cfgNotes=0).
+ (void)resetDefaultsOnEngine:(magentart::core::RealtimeRunner*)engine
                 prefixString:(NSString*)prefixString
                     cfgNotes:(float)cfgNotes
                 cfgMusicCoCa:(float)cfgMusicCoCa;

/// Convenience: reset with shared defaults (no per-app overrides).
+ (void)resetDefaultsOnEngine:(magentart::core::RealtimeRunner*)engine
                 prefixString:(NSString*)prefixString;

@end
