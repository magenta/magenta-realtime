# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# examples/sc/CMakeLists.txt — builds the `MRT2` SuperCollider
# UGen plugin (.scx).
#
# Mirrors the standalone/jam/hello_mrt2 pattern: link against
# magentart::core (which transitively brings in mlx, sentencepiece, tflite,
# and the Apple frameworks the engine needs) and add only the SC-specific
# headers + linker flags here.
#
# Output layout (after build + optional `make deploy_mrt2_sc`):
#   ~/Library/Application Support/SuperCollider/Extensions/MRT2/
#     MRT2.scx       (the UGen plugin binary)
#     mlx.metallib        (Metal shaders, colocated for MLX runtime resolution)
#     MRT2.sc        (sclang wrapper class)
#     example.scd         (example session)

include(FetchContent)

# ─── SuperCollider headers (SC_PlugIn.h + SC_Unit.h + SC_InterfaceTable.h) ───
# We only need headers; we do NOT want to build scsynth/sclang. The SC repo
# has a top-level CMakeLists.txt though, so plain FetchContent_MakeAvailable
# would try to configure the entire project. Use FetchContent_Populate to
# fetch only — it's deprecated as of CMake 3.30 but still functional, and is
# the documented escape hatch for "download but don't add_subdirectory".
#
# Users with their own SC source tree can override via -DSC_PATH=<path>.

set(SC_PATH "" CACHE PATH "Path to a SuperCollider source tree (overrides FetchContent)")

if(NOT SC_PATH)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)  # keep FetchContent_Populate available
    endif()
    FetchContent_Declare(
        supercollider
        GIT_REPOSITORY https://github.com/supercollider/supercollider.git
        GIT_TAG        Version-3.13.0
        GIT_SHALLOW    ON
    )
    message(STATUS "Fetching supercollider headers...")
    FetchContent_GetProperties(supercollider)
    if(NOT supercollider_POPULATED)
        FetchContent_Populate(supercollider)
    endif()
    set(SC_PATH "${supercollider_SOURCE_DIR}")
endif()

# ─── Target ──────────────────────────────────────────────────────────────────
add_library(mrt2_sc MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/MRT2.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/magenta_paths.cpp
)

target_compile_features(mrt2_sc PRIVATE cxx_std_20)
# .mm because the realtime runner uses Objective-C autorelease pools internally
# and ARC keeps lifetime management consistent with the AU/standalone hosts.
target_compile_options(mrt2_sc PRIVATE -fobjc-arc)

target_include_directories(mrt2_sc PRIVATE
    "${SC_PATH}/include/plugin_interface"        # SC_PlugIn.h
    "${SC_PATH}/include/common"                  # SC_Endian.h, plugin_interface deps
    "${SC_PATH}/common"                           # SC_fftlib.h, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp     # magenta_paths.h
)

target_link_libraries(mrt2_sc PRIVATE
    magentart::core
    "-framework AudioToolbox"
)

# ─── SC plugin linker behaviour ──────────────────────────────────────────────
# scsynth loads .scx plugins via dlopen into the running scsynth process; the
# server provides the InterfaceTable symbols at load time. -undefined
# dynamic_lookup defers unresolved symbols to runtime — same approach as PD,
# and the documented pattern used by sc3-plugins, FluCoMa, and every external
# SC plugin we've seen.
target_link_options(mrt2_sc PRIVATE -undefined dynamic_lookup)

# ─── Output naming ───────────────────────────────────────────────────────────
# scsynth's plugin scanner (Scrobjects.cpp) only loads files with the platform-
# specific UGen extension (.scx on macOS, .so on linux). No `lib` prefix.
set_target_properties(mrt2_sc PROPERTIES
    PREFIX ""
    SUFFIX ".scx"
    OUTPUT_NAME "MRT2"
)

# ─── Post-build: mlx.metallib + ad-hoc codesign ──────────────────────────────
add_custom_command(TARGET mrt2_sc POST_BUILD
    # MLX expects mlx.metallib next to the module that loaded it; copy it in.
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_FILE_DIR:mrt2_sc>/mlx.metallib"
    # Apple Silicon refuses to load unsigned shared modules into scsynth's process.
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_FILE_DIR:mrt2_sc>/mlx.metallib"
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_FILE:mrt2_sc>"
    COMMENT "Finalizing MRT2.scx"
    VERBATIM
)

# ─── Optional deploy target ──────────────────────────────────────────────────
# `make deploy_mrt2_sc` copies the .scx + metallib + sclang class + example
# into the user's SuperCollider Extensions directory. SC scans this path
# recursively at boot, so a single MRT2/ subdirectory is enough.
set(SC_EXTENSIONS_DIR "$ENV{HOME}/Library/Application Support/SuperCollider/Extensions" CACHE PATH
    "Where `make deploy_mrt2_sc` will install the MRT2/ extension directory")
add_custom_target(deploy_mrt2_sc
    DEPENDS mrt2_sc
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SC_EXTENSIONS_DIR}/MRT2"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:mrt2_sc>"
        "${SC_EXTENSIONS_DIR}/MRT2/MRT2.scx"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE_DIR:mrt2_sc>/mlx.metallib"
        "${SC_EXTENSIONS_DIR}/MRT2/mlx.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/MRT2.sc"
        "${SC_EXTENSIONS_DIR}/MRT2/MRT2.sc"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/example.scd"
        "${SC_EXTENSIONS_DIR}/MRT2/example.scd"
    COMMENT "Deploying MRT2 to ${SC_EXTENSIONS_DIR}/MRT2"
    VERBATIM
)
