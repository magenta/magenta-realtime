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

@interface MagentaModelDownloader : NSObject

/**
 Lists available remote models from the HuggingFace repository.
 */
+ (void)listRemoteModelsWithCompletion:(void (^)(NSArray<NSString *> *models, NSError *error))completion;

/**
 Downloads a specified model directory structure from HuggingFace into the local models directory.
 Spawns sequential task downloads with a progress block and final completion handler.
 */
+ (void)downloadModel:(NSString *)modelName
             progress:(void (^)(double progress, NSString *status))progressBlock
           completion:(void (^)(BOOL success, NSError *error))completion;

/**
 Fetches shared base resources (musiccoca, spectrostream) to initialize the home directory layout.
 */
+ (void)initializeSharedResourcesWithProgress:(void (^)(double progress, NSString *status))progressBlock
                                   completion:(void (^)(BOOL success, NSError *error))completion;

/**
 Validates if the base shared resources are fully initialized and present locally.
 */
+ (BOOL)areSharedResourcesValid;

@end
