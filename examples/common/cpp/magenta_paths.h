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

// magenta_paths.h — Shared path resolution for Magenta RT examples.
//
// Provides centralized access to the ~/Documents/Magenta directory tree so that all
// examples (standalone, jam, auv3, hello_mrt2, pd, etc.) resolve
// resources and models from the same location.
//
// The root directory defaults to ~/Documents/Magenta but can be overridden via the
// MAGENTA_HOME environment variable.

#ifndef MAGENTART_MAGENTA_PATHS_H_
#define MAGENTART_MAGENTA_PATHS_H_

#include <string>
#include <vector>

namespace magentart {
namespace paths {

/// Returns ~/Documents/Magenta/magenta-rt-v2/banks.
std::string get_banks_dir();

/// Returns the magenta home directory (default: ~/Documents/Magenta/magenta-rt-v2).
/// Honors the MAGENTA_HOME environment variable if set.
std::string get_magenta_home();

/// Returns ~/Documents/Magenta/magenta-rt-v2/resources.
std::string get_resources_dir();

/// Returns ~/Documents/Magenta/magenta-rt-v2/resources/musiccoca.
std::string get_musiccoca_dir();

/// Returns ~/Documents/Magenta/magenta-rt-v2/resources/spectrostream.
std::string get_spectrostream_dir();

/// Returns ~/Documents/Magenta/magenta-rt-v2/models.
std::string get_models_dir();

/// Default model directory name.
constexpr const char* DEFAULT_MODEL_NAME = "mrt2_base";

/// Returns ~/Documents/Magenta/magenta-rt-v2/models/mrt2_base.
std::string get_default_model_dir();

/// Information about a discovered model.
struct ModelInfo {
    std::string name;       // Directory name (e.g. "mrt2_base").
    std::string path;       // Full path to the model directory.
    std::string mlxfn_path; // Full path to the .mlxfn file inside.
};

/// Lists all valid model directories under ~/Documents/Magenta/magenta-rt-v2/models/.
/// A directory is considered a valid model if it contains at least one .mlxfn
/// file.
std::vector<ModelInfo> list_available_models();

/// Checks if a directory contains a valid .mlxfn model.
bool is_valid_model_dir(const std::string& dir_path);

/// Finds the .mlxfn file inside a model directory.
/// Returns empty string if not found.
std::string find_mlxfn_in_dir(const std::string& dir_path);

}  // namespace paths
}  // namespace magentart

#endif  // MAGENTART_MAGENTA_PATHS_H_
