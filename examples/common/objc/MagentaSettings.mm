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

#import "MagentaSettings.h"

using magentart::core::RealtimeRunner;

@implementation MagentaSettings

+ (NSString*)paramKeyForAddress:(int)address {
    switch (address) {
        case 0: return @"temperature";
        case 1: return @"topk";
        case 3: return @"cfgmusiccoca";
        case 4: return @"cfgnotes";
        case 5: return @"volume";
        case 6: return @"mute";
        case 7: return @"unmaskwidth";
        case 8: return @"buffersize";
        case 9: return @"latencycomp";
        case 10: return @"cursor_x";
        case 11: return @"cursor_y";
        case 12: return @"falloff";
        case 31: return @"resetstate";
        case 32: return @"bypass";
        case 39: return @"drumless";
        case 45: return @"midigate";
        case 46: return @"onsetmode";
        case 47: return @"seedrotation";
        case 48: return @"cfgdrums";
        default:
            if (address >= 13 && address <= 30) {
                int idx = (address - 13) / 3;
                int type = (address - 13) % 3;
                if (type == 0) return [NSString stringWithFormat:@"prompt%d_x", idx];
                else if (type == 1) return [NSString stringWithFormat:@"prompt%d_y", idx];
                else return [NSString stringWithFormat:@"prompt%d_exists", idx];
            }
            if (address >= 33 && address <= 38) {
                return [NSString stringWithFormat:@"pca_coeff_%d", address - 33];
            }
            return nil;
    }
}

+ (BOOL)paramIsBool:(int)address {
    if (address == 6 || address == 9 || address == 31 || address == 32 || address == 39 || address == 45 || address == 46) return YES;
    if (address >= 13 && address <= 30 && ((address - 13) % 3 == 2)) return YES;
    return NO;
}

+ (BOOL)shouldPersistParam:(int)address {
    return (address >= 0 && address <= 9 && address != 2) || address == 39 || address == 48;
}

+ (void)applyParamToEngine:(RealtimeRunner*)engine
                   address:(int)address
                     value:(float)value
              prefixString:(NSString*)prefixString {
    if (!engine) return;

    if (address == 0) engine->set_temperature(value);
    else if (address == 1) engine->set_top_k((int)value);
    else if (address == 3) engine->set_cfg_musiccoca(value);
    else if (address == 4) engine->set_cfg_notes(value);
    else if (address == 5) engine->set_volume_db(value);
    else if (address == 6) engine->set_mute(value > 0.5f);
    else if (address == 7) engine->set_unmask_width((int)value);
    else if (address == 8) {
        size_t cap = 8192;
        if (value < 0.5f) cap = 2048;
        else if (value < 1.5f) cap = 4096;
        engine->set_buffer_size(cap);
    }
    else if (address == 9) engine->set_latency_comp(value > 0.5f);
    else if (address >= 10 && address <= 15) engine->set_blend_weight(address - 10, value);
    else if (address == 31) { if (value > 0.5f) engine->trigger_reset(); }
    else if (address == 32) engine->set_bypass(value > 0.5f);
    else if (address >= 33 && address <= 38) {
        engine->set_pca_coeff(address - 33, value);
    }
    else if (address == 39) engine->set_drumless(value > 0.5f);
    else if (address == 45) engine->set_midi_gate_enabled(value > 0.5f);
    else if (address == 46) engine->set_onset_mode(value > 0.5f);
    else if (address == 48) engine->set_cfg_drums(value);

    if ([self shouldPersistParam:address]) {
        NSString* key = [self paramKeyForAddress:address];
        if (key && prefixString) {
            NSString* defaultsKey = [NSString stringWithFormat:@"%@_Param_%@", prefixString, key];
            [[NSUserDefaults standardUserDefaults] setFloat:value forKey:defaultsKey];
        }
    }
}

+ (float)readParamFromEngine:(RealtimeRunner*)engine
                     address:(int)address {
    if (!engine) return 0;

    if (address == 0) return engine->get_temperature();
    else if (address == 1) return (float)engine->get_top_k();
    else if (address == 3) return engine->get_cfg_musiccoca();
    else if (address == 4) return engine->get_cfg_notes();
    else if (address == 5) return engine->get_volume_db();
    else if (address == 6) return engine->get_mute() ? 1.0f : 0.0f;
    else if (address == 7) return (float)engine->get_unmask_width();
    else if (address == 8) {
        size_t cap = engine->get_buffer_size();
        if (cap <= 2048) return 0.0f;
        if (cap <= 4096) return 1.0f;
        return 2.0f;
    }
    else if (address == 9) return engine->get_latency_comp() ? 1.0f : 0.0f;
    else if (address >= 10 && address <= 15) return engine->get_blend_weight(address - 10);
    else if (address == 31) return 0.0f;
    else if (address == 32) return engine->get_bypass() ? 1.0f : 0.0f;
    else if (address >= 33 && address <= 38) {
        return engine->get_pca_coeff(address - 33);
    }
    else if (address == 39) return engine->get_drumless() ? 1.0f : 0.0f;
    else if (address == 45) return engine->get_midi_gate_enabled() ? 1.0f : 0.0f;
    else if (address == 46) return engine->get_onset_mode() ? 1.0f : 0.0f;
    else if (address == 48) return engine->get_cfg_drums();
    return 0.0f;
}

+ (void)restoreSavedParams:(RealtimeRunner*)engine
              prefixString:(NSString*)prefixString {
    float cfgNotes = [prefixString isEqualToString:@"Collider"] ? kColliderDefaultCfgNotes : kMagentaDefaultCfgNotes;
    float cfgMusicCoCa = [prefixString isEqualToString:@"Collider"] ? kColliderDefaultCfgMusicCoCa : kMagentaDefaultCfgMusicCoCa;
    [self restoreSavedParams:engine
                prefixString:prefixString
                    cfgNotes:cfgNotes
                cfgMusicCoCa:cfgMusicCoCa];
}

+ (void)restoreSavedParams:(RealtimeRunner*)engine
              prefixString:(NSString*)prefixString
                  cfgNotes:(float)cfgNotes
              cfgMusicCoCa:(float)cfgMusicCoCa {
    NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
    for (int i = 0; i <= 100; i++) {
        if (![self shouldPersistParam:i]) continue;
        NSString* key = [self paramKeyForAddress:i];
        if (!key) continue;
        NSString* defaultsKey = [NSString stringWithFormat:@"%@_Param_%@", prefixString, key];
        if ([defaults objectForKey:defaultsKey]) {
            float value = [defaults floatForKey:defaultsKey];
            [self applyParamToEngine:engine address:i value:value prefixString:prefixString];
        } else {
            float value = 0.0f;
            if (i == 0) value = kMagentaDefaultTemperature;
            else if (i == 1) value = kMagentaDefaultTopK;
            else if (i == 3) value = cfgMusicCoCa;
            else if (i == 4) value = cfgNotes;
            else if (i == 5) value = kMagentaDefaultVolume;
            else if (i == 7) value = kMagentaDefaultUnmaskWidth;
            else if (i == 8) value = kMagentaDefaultBufferSize;
            else if (i == 48) value = kMagentaDefaultCfgDrums;

            [self applyParamToEngine:engine address:i value:value prefixString:prefixString];
        }
    }
}

+ (void)resetDefaultsOnEngine:(RealtimeRunner*)engine
                 prefixString:(NSString*)prefixString
                     cfgNotes:(float)cfgNotes
                 cfgMusicCoCa:(float)cfgMusicCoCa {
    [self applyParamToEngine:engine address:0  value:kMagentaDefaultTemperature prefixString:prefixString];
    [self applyParamToEngine:engine address:1  value:kMagentaDefaultTopK        prefixString:prefixString];
    [self applyParamToEngine:engine address:3  value:cfgMusicCoCa               prefixString:prefixString];
    [self applyParamToEngine:engine address:4  value:cfgNotes                   prefixString:prefixString];
    [self applyParamToEngine:engine address:5  value:kMagentaDefaultVolume      prefixString:prefixString];
    [self applyParamToEngine:engine address:6  value:0.0f                       prefixString:prefixString]; // mute off
    [self applyParamToEngine:engine address:7  value:kMagentaDefaultUnmaskWidth prefixString:prefixString];
    [self applyParamToEngine:engine address:8  value:kMagentaDefaultBufferSize  prefixString:prefixString];
    [self applyParamToEngine:engine address:39 value:0.0f                       prefixString:prefixString]; // drumless off
    [self applyParamToEngine:engine address:48 value:kMagentaDefaultCfgDrums    prefixString:prefixString];
}

+ (void)resetDefaultsOnEngine:(RealtimeRunner*)engine
                 prefixString:(NSString*)prefixString {
    [self resetDefaultsOnEngine:engine
                   prefixString:prefixString
                       cfgNotes:kMagentaDefaultCfgNotes
                   cfgMusicCoCa:kMagentaDefaultCfgMusicCoCa];
}

@end
