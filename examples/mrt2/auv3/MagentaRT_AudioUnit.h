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

#pragma once
#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudioKit/CoreAudioKit.h>
#include <magentart/realtime_runner.h>

using magentart::core::RealtimeRunner;

@interface MagentaRTAudioUnit : AUAudioUnit

@property (nonatomic, strong) NSArray<NSDictionary*>* prompts;
@property (nonatomic, copy) NSString* modelName;
@property (nonatomic, strong) NSData* modelBookmark;
@property (nonatomic, strong) NSDictionary* promptSurfaceState;
@property (nonatomic, copy) NSString* musicCocaModelName;
@property (nonatomic, copy) NSString* statePrefix;
@property (nonatomic, assign) BOOL uiPlaying;

- (RealtimeRunner*)engine;
- (void)pollOfflineState;
- (void)setNoteOn:(uint8_t)note on:(BOOL)on;
- (NSArray<NSNumber*>*)activeNotes;
- (void)readAudioLevels:(float*)outLeft right:(float*)outRight;
@end

@interface MagentaRTViewController : AUViewController <AUAudioUnitFactory>
@end
