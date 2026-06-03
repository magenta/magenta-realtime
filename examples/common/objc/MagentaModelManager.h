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

#import <Cocoa/Cocoa.h>

@interface MagentaModelManager : NSObject

/**
 Scans the provided directory path URL, filters valid subdirectories containing a `.mlxfn` file/folder
 or direct `.mlxfn` assets, and returns an array of found model folder/file names.
 */
+ (NSArray<NSString *> *)listLocalModelsInDirectory:(NSURL *)modelsDir;

+ (NSString *)defaultModelsDirectory;

+ (void)selectDownloadFolderWithParentWindow:(NSWindow *)parentWindow
                                  completion:(void (^)(NSString *selectedPath, NSData *bookmarkData, NSError *error))completion;

@end
