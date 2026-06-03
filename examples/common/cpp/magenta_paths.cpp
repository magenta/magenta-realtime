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

// magenta_paths.cpp — Implementation of shared path resolution.

#include "magenta_paths.h"

#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

#include <algorithm>
#include <string>
#include <vector>

namespace magentart {
namespace paths {

namespace {

std::string get_home_dir() {
    std::string home_str;
    const char* home = std::getenv("HOME");
    if (home) {
        home_str = home;
    } else {
        struct passwd* pw = getpwuid(getuid());
        if (pw) {
            home_str = pw->pw_dir;
        } else {
            home_str = "/tmp";
        }
    }
    // Strip sandbox container suffix if running in a sandboxed environment
    size_t container_pos = home_str.find("/Library/Containers/");
    if (container_pos != std::string::npos) {
        home_str = home_str.substr(0, container_pos);
    }
    return home_str;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool dir_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::vector<std::string> list_subdirs(const std::string& dir_path) {
    std::vector<std::string> result;
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return result;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;  // Skip hidden + . / ..
        std::string full_path = dir_path + "/" + entry->d_name;
        if (dir_exists(full_path)) {
            result.push_back(entry->d_name);
        }
    }
    closedir(dir);
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

std::string get_magenta_home() {
    const char* env = std::getenv("MAGENTA_HOME");
    std::string base = (env && env[0] != '\0') ? env : (get_home_dir() + "/Documents/Magenta");
    return base + "/magenta-rt-v2";
}

std::string get_banks_dir() {
    return get_magenta_home() + "/banks";
}

std::string get_resources_dir() {
    return get_magenta_home() + "/resources";
}

std::string get_musiccoca_dir() {
    return get_resources_dir() + "/musiccoca";
}

std::string get_spectrostream_dir() {
    return get_resources_dir() + "/spectrostream";
}

std::string get_models_dir() {
    return get_magenta_home() + "/models";
}

std::string get_default_model_dir() {
    return get_models_dir() + "/" + DEFAULT_MODEL_NAME;
}

std::string find_mlxfn_in_dir(const std::string& dir_path) {
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return "";
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (ends_with(name, ".mlxfn")) {
            closedir(dir);
            return dir_path + "/" + name;
        }
    }
    closedir(dir);
    return "";
}

bool is_valid_model_dir(const std::string& dir_path) {
    return !find_mlxfn_in_dir(dir_path).empty();
}

std::vector<ModelInfo> list_available_models() {
    std::vector<ModelInfo> models;
    std::string models_root = get_models_dir();
    for (const auto& name : list_subdirs(models_root)) {
        std::string model_path = models_root + "/" + name;
        std::string mlxfn = find_mlxfn_in_dir(model_path);
        if (!mlxfn.empty()) {
            models.push_back({name, model_path, mlxfn});
        }
    }
    return models;
}

}  // namespace paths
}  // namespace magentart
