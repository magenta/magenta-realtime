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

#import "MagentaModelManager.h"
#include "magenta_paths.h"

@implementation MagentaModelManager

+ (NSArray<NSString *> *)listLocalModelsInDirectory:(NSURL *)modelsDir {
    if (!modelsDir) return @[];

    NSError* error = nil;
    NSArray* contents = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:modelsDir
                                                 includingPropertiesForKeys:@[NSURLIsDirectoryKey]
                                                                    options:0
                                                                      error:&error];
    if (error || !contents) {
        NSLog(@"MagentaModelManager: failed to list directory %@: %@", modelsDir.path, error.localizedDescription);
        return @[];
    }

    NSMutableArray* modelFiles = [NSMutableArray array];
    for (NSURL* url in contents) {
        NSNumber *isDirectory = nil;
        [url getResourceValue:&isDirectory forKey:NSURLIsDirectoryKey error:nil];
        if (isDirectory && [isDirectory boolValue]) {
            std::string dirPathStr = url.path.UTF8String;
            if (magentart::paths::is_valid_model_dir(dirPathStr)) {
                [modelFiles addObject:url.lastPathComponent];
            }
        } else if ([url.pathExtension isEqualToString:@"mlxfn"]) {
            [modelFiles addObject:url.lastPathComponent];
        }
    }

    return [modelFiles sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
}

+ (NSString *)defaultModelsDirectory {
    const char* env = std::getenv("MAGENTA_HOME");
    if (env && env[0] != '\0') {
        return [[NSString stringWithUTF8String:env] stringByAppendingPathComponent:@"models"];
    }
    NSString* userName = NSUserName();
    NSString* realHome = NSHomeDirectoryForUser(userName);
    return [realHome stringByAppendingPathComponent:@"Documents/Magenta/magenta-rt-v2/models"];
}

// NSUserDefaults preference keys for the user-selected models directory.
//
// 1. MagentaRT_ModelFolderBookmark (NSData):
//    The authoritative security-scoped bookmark token. This is used by the sandboxed
//    process on startup to resolve permissions for paths outside the container.
//
// 2. MagentaRT_ModelFolderPath (NSString):
//    The raw file path string (informative only). Sandboxed extensions have no permission
//    to access files via raw paths; this is used solely to display the path in the React UI settings.
static NSString* const kMagentaRT_ModelFolderBookmarkKey = @"MagentaRT_ModelFolderBookmark";
static NSString* const kMagentaRT_ModelFolderPathKey = @"MagentaRT_ModelFolderPath";

+ (void)selectDownloadFolderWithParentWindow:(NSWindow *)parentWindow
                                  completion:(void (^)(NSString *selectedPath, NSData *bookmarkData, NSError *error))completion {
    NSData* bookmark = [[NSUserDefaults standardUserDefaults] objectForKey:kMagentaRT_ModelFolderBookmarkKey];
    BOOL stale = NO;
    NSURL* defaultURL = bookmark ? [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&stale error:nil] : nil;
    BOOL accessGranted = [defaultURL startAccessingSecurityScopedResource];

    if (!defaultURL) {
        defaultURL = [NSURL fileURLWithPath:[self defaultModelsDirectory]];
    }

    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:NO];
    [panel setCanChooseDirectories:YES];
    [panel setShowsHiddenFiles:YES];
    [panel setMessage:@"Select a folder for model search path."];
    [panel setDirectoryURL:defaultURL];

    void (^completionBlock)(NSModalResponse) = ^(NSModalResponse result) {
        if (accessGranted) {
            [defaultURL stopAccessingSecurityScopedResource];
        }

        if (result != NSModalResponseOK || !panel.URL) {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, nil, nil);
            });
            return;
        }

        NSError* err = nil;
        NSData* bookmarkData = [panel.URL bookmarkDataWithOptions:NSURLBookmarkCreationWithSecurityScope
                                   includingResourceValuesForKeys:nil
                                                    relativeToURL:nil
                                                            error:&err];

        if (bookmarkData) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [[NSUserDefaults standardUserDefaults] setObject:bookmarkData forKey:kMagentaRT_ModelFolderBookmarkKey];
                [[NSUserDefaults standardUserDefaults] setObject:panel.URL.path forKey:kMagentaRT_ModelFolderPathKey];
                completion(panel.URL.path, bookmarkData, nil);
            });
        } else {
            dispatch_async(dispatch_get_main_queue(), ^{
                completion(nil, nil, err);
            });
        }
    };

    if (parentWindow) {
        [panel beginSheetModalForWindow:parentWindow completionHandler:completionBlock];
    } else {
        [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
        [panel setLevel:CGWindowLevelForKey(kCGFloatingWindowLevelKey)];
        [panel beginWithCompletionHandler:completionBlock];
    }
}

@end
