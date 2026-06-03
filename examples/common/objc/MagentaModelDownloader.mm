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

#import "MagentaModelDownloader.h"
#include "magenta_paths.h"
#include <glob.h>
#include <sys/types.h>
#include <pwd.h>
#include <unistd.h>

// Add utility category for safe string abbreviation (must be declared before first use)
@interface NSString (Abbreviation)
- (NSString *)stringByAbbreviatingWithMaxLength:(NSUInteger)maxLength;
@end

// ---------------------------------------------------------------------------
// HuggingFace configuration
// ---------------------------------------------------------------------------
static NSString *const kHfRepoId = @"google/magenta-realtime-2";
static NSString *const kHfModelsSubdir = @"models";
static NSString *const kHfResourcesSubdir = @"resources";

@implementation MagentaModelDownloader

// ============================================================================
// HuggingFace helpers
// ============================================================================

/// Build HF API tree URL: https://huggingface.co/api/models/<repo>/tree/main/<path>
+ (NSURL *)hfTreeURLForPath:(NSString *)path {
    NSString *urlString = [NSString stringWithFormat:@"https://huggingface.co/api/models/%@/tree/main/%@", kHfRepoId, path];
    return [NSURL URLWithString:urlString];
}

/// Build HF file download URL: https://huggingface.co/<repo>/resolve/main/<path>
+ (NSURL *)hfResolveURLForPath:(NSString *)path {
    NSString *encoded = [path stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLPathAllowedCharacterSet]];
    NSString *urlString = [NSString stringWithFormat:@"https://huggingface.co/%@/resolve/main/%@", kHfRepoId, encoded];
    return [NSURL URLWithString:urlString];
}

// ============================================================================
// listRemoteModelsWithCompletion
// ============================================================================

+ (void)listRemoteModelsWithCompletion:(void (^)(NSArray<NSString *> *models, NSError *error))completion {
    NSURL *url = [self hfTreeURLForPath:kHfModelsSubdir];
    NSURLRequest *request = [NSURLRequest requestWithURL:url];

    NSLog(@"MagentaModelDownloader: Listing remote models from HuggingFace: %@", url.absoluteString);

    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            NSLog(@"MagentaModelDownloader: HF network error: %@", error.localizedDescription);
            completion(nil, error);
            return;
        }

        NSHTTPURLResponse *httpResp = (NSHTTPURLResponse *)response;
        if (httpResp.statusCode != 200) {
            NSString *body = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"(empty)";
            NSString *msg = [NSString stringWithFormat:@"HuggingFace API returned HTTP %ld: %@", (long)httpResp.statusCode, body];
            NSLog(@"MagentaModelDownloader: %@", msg);
            completion(nil, [NSError errorWithDomain:@"com.magentart.downloader" code:104 userInfo:@{NSLocalizedDescriptionKey: msg}]);
            return;
        }

        NSError *jsonError = nil;
        NSArray *items = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError || ![items isKindOfClass:[NSArray class]]) {
            completion(nil, jsonError ?: [NSError errorWithDomain:@"com.magentart.downloader" code:104 userInfo:@{NSLocalizedDescriptionKey: @"Failed to parse HF tree JSON."}]);
            return;
        }

        NSMutableArray *modelNames = [NSMutableArray array];
        for (NSDictionary *item in items) {
            if ([item[@"type"] isEqualToString:@"directory"]) {
                NSString *path = item[@"path"];  // e.g. "models/my_model"
                NSString *name = [path lastPathComponent];
                if (name.length > 0) {
                    [modelNames addObject:name];
                }
            }
        }

        completion([modelNames sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)], nil);
    }] resume];
}

// ============================================================================
// downloadModel
// ============================================================================

+ (void)downloadModel:(NSString *)modelName
             progress:(void (^)(double progress, NSString *status))progressBlock
           completion:(void (^)(BOOL success, NSError *error))completion {

    NSString *treePath = [NSString stringWithFormat:@"%@/%@", kHfModelsSubdir, modelName];
    NSURL *url = [self hfTreeURLForPath:treePath];
    NSURLRequest *request = [NSURLRequest requestWithURL:url];

    progressBlock(0.01, @"Fetching model metadata…");

    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) { completion(NO, error); return; }

        NSHTTPURLResponse *httpResp = (NSHTTPURLResponse *)response;
        if (httpResp.statusCode != 200) {
            NSString *body = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:105 userInfo:@{
                NSLocalizedDescriptionKey: [NSString stringWithFormat:@"HF tree API HTTP %ld: %@", (long)httpResp.statusCode, body]
            }]);
            return;
        }

        NSArray *items = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (![items isKindOfClass:[NSArray class]]) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:105 userInfo:@{NSLocalizedDescriptionKey: @"Invalid HF tree response."}]);
            return;
        }

        // Collect files (skip directories)
        NSMutableArray *filesToDownload = [NSMutableArray array];
        for (NSDictionary *item in items) {
            if ([item[@"type"] isEqualToString:@"file"]) {
                NSString *path = item[@"path"];  // e.g. "models/my_model/file.safetensors"
                NSNumber *size = item[@"size"] ?: @0;
                [filesToDownload addObject:@{ @"path": path, @"size": size }];
            }
        }

        if (filesToDownload.count == 0) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:106 userInfo:@{NSLocalizedDescriptionKey: @"No files found for this model."}]);
            return;
        }

        [self downloadFiles:filesToDownload
                targetIndex:0
                  modelName:modelName
               isSharedInit:NO
                   progress:progressBlock
                 completion:completion];
    }] resume];
}

// ============================================================================
// initializeSharedResourcesWithProgress
// ============================================================================

+ (void)initializeSharedResourcesWithProgress:(void (^)(double progress, NSString *status))progressBlock
                                   completion:(void (^)(BOOL success, NSError *error))completion {
    // The tree API with ?recursive=true lists all files recursively
    NSString *urlString = [NSString stringWithFormat:@"https://huggingface.co/api/models/%@/tree/main/%@?recursive=true", kHfRepoId, kHfResourcesSubdir];
    NSURL *url = [NSURL URLWithString:urlString];
    NSURLRequest *request = [NSURLRequest requestWithURL:url];

    progressBlock(0.01, @"Fetching resource metadata…");

    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) { completion(NO, error); return; }

        NSHTTPURLResponse *httpResp = (NSHTTPURLResponse *)response;
        if (httpResp.statusCode != 200) {
            NSString *body = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:107 userInfo:@{
                NSLocalizedDescriptionKey: [NSString stringWithFormat:@"HF tree API HTTP %ld: %@", (long)httpResp.statusCode, body]
            }]);
            return;
        }

        NSArray *items = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (![items isKindOfClass:[NSArray class]]) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:107 userInfo:@{NSLocalizedDescriptionKey: @"Invalid HF tree response for resources."}]);
            return;
        }

        NSMutableArray *filesToDownload = [NSMutableArray array];
        for (NSDictionary *item in items) {
            if ([item[@"type"] isEqualToString:@"file"]) {
                NSString *path = item[@"path"];
                NSNumber *size = item[@"size"] ?: @0;
                [filesToDownload addObject:@{ @"path": path, @"size": size }];
            }
        }

        if (filesToDownload.count == 0) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:108 userInfo:@{NSLocalizedDescriptionKey: @"No base shared resources found on HuggingFace."}]);
            return;
        }

        [self downloadFiles:filesToDownload
                targetIndex:0
                  modelName:@""
               isSharedInit:YES
                   progress:progressBlock
                 completion:completion];
    }] resume];
}

// ============================================================================
// Sequential file downloader
// ============================================================================

+ (void)downloadFiles:(NSArray<NSDictionary *> *)files
          targetIndex:(NSUInteger)index
            modelName:(NSString *)modelName
         isSharedInit:(BOOL)isSharedInit
             progress:(void (^)(double progress, NSString *status))progressBlock
           completion:(void (^)(BOOL success, NSError *error))completion {

    if (index >= files.count) {
        progressBlock(1.0, @"Finished!");
        completion(YES, nil);
        return;
    }

    NSDictionary *fileInfo = files[index];
    NSString *repoPath =
        fileInfo[@"path"]; // e.g. "models/my_model/file.safetensors" or
                           // "resources/musiccoca/file.bin"
    NSNumber *fileSize = fileInfo[@"size"];
    NSString *fileName = [repoPath lastPathComponent];

    // Determine local destination path
    NSString *homePath = [NSString stringWithUTF8String:magentart::paths::get_magenta_home().c_str()];
    NSString *localDestPath = [homePath stringByAppendingPathComponent:repoPath];

    // Create parent directories
    NSString *parentDir = [localDestPath stringByDeletingLastPathComponent];
    [[NSFileManager defaultManager] createDirectoryAtPath:parentDir withIntermediateDirectories:YES attributes:nil error:nil];

    // Progress label
    NSString *shortFileName = [fileName stringByAbbreviatingWithMaxLength:25];
    NSString *friendlySize = [NSByteCountFormatter stringFromByteCount:fileSize.longLongValue countStyle:NSByteCountFormatterCountStyleFile];
    NSString *label = nil;
    if (isSharedInit) {
        label = [NSString stringWithFormat:@"Installing Resources: (%lu/%lu) %@ (%@)", index + 1, files.count, shortFileName, friendlySize];
    } else {
        label = [NSString stringWithFormat:@"Installing Model: (%lu/%lu) %@ (%@)", index + 1, files.count, shortFileName, friendlySize];
    }
    progressBlock((double)index / (double)files.count, label);

    // Download via resolve URL
    NSURL *downloadURL = [self hfResolveURLForPath:repoPath];
    NSURLRequest *request = [NSURLRequest requestWithURL:downloadURL];

    NSURLSessionDownloadTask *downloadTask = [[NSURLSession sharedSession] downloadTaskWithRequest:request completionHandler:^(NSURL *location, NSURLResponse *response, NSError *netError) {
        if (netError) {
            completion(NO, netError);
            return;
        }

        if (!location) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:109 userInfo:@{NSLocalizedDescriptionKey: @"Download source path not found."}]);
            return;
        }

        // Check HTTP status
        NSHTTPURLResponse *httpResp = (NSHTTPURLResponse *)response;
        if (httpResp.statusCode != 200) {
            completion(NO, [NSError errorWithDomain:@"com.magentart.downloader" code:109 userInfo:@{
                NSLocalizedDescriptionKey: [NSString stringWithFormat:@"HF download failed HTTP %ld for %@", (long)httpResp.statusCode, fileName]
            }]);
            return;
        }

        NSError *moveError = nil;
        if ([[NSFileManager defaultManager] fileExistsAtPath:localDestPath]) {
            [[NSFileManager defaultManager] removeItemAtPath:localDestPath error:nil];
        }
        [[NSFileManager defaultManager] moveItemAtURL:location toURL:[NSURL fileURLWithPath:localDestPath] error:&moveError];

        if (moveError) {
            completion(NO, moveError);
            return;
        }

        // Recurse to next file
        dispatch_async(dispatch_get_main_queue(), ^{
            [self downloadFiles:files
                    targetIndex:index + 1
                      modelName:modelName
                   isSharedInit:isSharedInit
                       progress:progressBlock
                     completion:completion];
        });
    }];
    [downloadTask resume];
}

// ============================================================================
// areSharedResourcesValid
// ============================================================================

+ (BOOL)hasFileMatchingPattern:(NSString *)pattern {
    glob_t *glob_results = (glob_t *)calloc(1, sizeof(glob_t));
    if (!glob_results) return NO;

    BOOL found = NO;
    int result = glob([pattern UTF8String], 0, NULL, glob_results);
    if (result == 0) {
        if (glob_results->gl_pathc > 0) {
            found = YES;
        }
        globfree(glob_results);
    }
    free(glob_results);
    return found;
}

+ (BOOL)areSharedResourcesValid {
    NSString *customPath = [[NSUserDefaults standardUserDefaults] objectForKey:@"MagentaRT_CustomResourcesPath"];
    NSString *resourcesDir = customPath ?: [NSString stringWithUTF8String:magentart::paths::get_resources_dir().c_str()];

    NSString *cocaPattern = [resourcesDir stringByAppendingPathComponent:@"musiccoca*/*.tflite"];
    NSString *streamPattern = [resourcesDir stringByAppendingPathComponent:@"spectrostream/*.mlxfn"];

    return [self hasFileMatchingPattern:cocaPattern] && [self hasFileMatchingPattern:streamPattern];
}

@end

@implementation NSString (Abbreviation)
- (NSString *)stringByAbbreviatingWithMaxLength:(NSUInteger)maxLength {
    if (self.length <= maxLength) return self;
    NSUInteger half = maxLength / 2 - 2;
    return [NSString stringWithFormat:@"%@...%@", [self substringToIndex:half], [self substringFromIndex:self.length - half]];
}
@end
