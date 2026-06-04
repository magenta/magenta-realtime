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

cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum OS X deployment version" FORCE)

project(MagentaRT LANGUAGES C CXX OBJCXX)

# macOS-only today: the inference stack depends on MLX (Metal), the
# hosts use AppKit/AudioToolbox/CoreMIDI/WebKit, and we link `-framework
# Metal / Accelerate / Foundation`. Fail fast rather than let the build
# die deep inside a framework link step on other platforms.
if(NOT APPLE)
  message(FATAL_ERROR
    "magenta-rt-v2's C++ build is macOS-only (MLX/Metal + Apple frameworks).\n"
    "For the Python JAX backend, skip CMake and use `pip install -e \".[jax]\"`.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(ABSL_INTERNAL_AT_LEAST_CXX20 ON CACHE BOOL "Force Abseil C++20 support")
set(ABSL_PROPAGATE_CXX_STD ON)

set(MAGENTART_VERSION "0.0.1")
add_definitions(-DMAGENTART_VERSION="${MAGENTART_VERSION}")

# Fetch Git SHA
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE MAGENTART_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT MAGENTART_GIT_SHA)
    set(MAGENTART_GIT_SHA "unknown")
endif()
add_definitions(-DMAGENTART_GIT_SHA="${MAGENTART_GIT_SHA}")

# --- Dependencies via FetchContent --------------------------------------------
include(FetchContent)

# --- MLX (static, from source) ------------------------------------------------
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG        v0.31.1
    GIT_SHALLOW    ON
)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
message(STATUS "Fetching MLX... (this may take a moment)")
FetchContent_MakeAvailable(mlx)

# Patch MLX header inclusion to support spaces in path.
# TODO: Remove once https://github.com/ml-explore/mlx/pull/3607 is merged.
file(READ "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" MLX_CONTENT)
string(REPLACE
  "declare -a HDRS_LIST=($HDRS)"
  "declare -a HDRS_LIST=(); while read -r dots path; do [ -n \"$dots\" ] && HDRS_LIST+=(\"$dots\" \"$path\"); done <<< \"$HDRS\""
  MLX_CONTENT
  "${MLX_CONTENT}"
)
file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" "${MLX_CONTENT}")

# ─── TFLite and SentencePiece ─────────────────────────────────────────────────

FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.2.0
  GIT_SHALLOW    ON
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
message(STATUS "Fetching sentencepiece... (this may take a moment)")
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        v2.21.0
  SOURCE_SUBDIR  tensorflow/lite
  GIT_SHALLOW    ON
)
# Disable TFLite features that cause build errors on MacOS 14.0+ with MLX
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "Disable XNNPACK delegate" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "Disable GPU delegate" FORCE)
set(TFLITE_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)

# IMPORTANT: Tell TF Lite's internal CMakeLists.txt where TensorFlow source
# already lives. Without this, TF Lite fetches a SECOND copy of TF at an older
# version (v2.19.0 hardcoded), causing FlatBuffers version mismatches.
set(FETCHCONTENT_SOURCE_DIR_TENSORFLOW "${CMAKE_BINARY_DIR}/_deps/tensorflow-lite-src" CACHE PATH
    "Point TF Lite's internal TF fetch at our already-cloned v2.21.0 source" FORCE)

message(STATUS "Fetching tensorflow-lite... (this may take a long time due to repo size)")
FetchContent_MakeAvailable(tensorflow-lite)



# --- Host-shared settings -----------------------------------------------------
# Code signing identity resolution, highest priority first:
#   1. -DCODESIGN_IDENTITY="..." on the cmake command line (cached, pins value)
#   2. MAGENTART_DEVELOPER_ID env var (re-read each configure)
#   3. "-" (ad-hoc, default)
# So `export MAGENTART_DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"`
# is enough — no need to pass -D every time.
set(CODESIGN_IDENTITY "" CACHE STRING
    "Override code signing identity (else uses \$MAGENTART_DEVELOPER_ID, else ad-hoc '-')")
# Treat empty or "-" as "unset" so the env var can still drive the value
# without `cmake --fresh`. To force ad-hoc when the env var is set, unset it.
if(NOT CODESIGN_IDENTITY OR CODESIGN_IDENTITY STREQUAL "-")
    if(DEFINED ENV{MAGENTART_DEVELOPER_ID} AND NOT "$ENV{MAGENTART_DEVELOPER_ID}" STREQUAL "")
        set(CODESIGN_IDENTITY "$ENV{MAGENTART_DEVELOPER_ID}")
        message(STATUS "CODESIGN_IDENTITY from \$MAGENTART_DEVELOPER_ID: ${CODESIGN_IDENTITY}")
    else()
        set(CODESIGN_IDENTITY "-")
    endif()
endif()
set(NOTARYTOOL_KEYCHAIN_PROFILE "notarytool-creds" CACHE STRING
    "Keychain profile name for `xcrun notarytool` (created via `xcrun notarytool store-credentials`)")

# Flags used everywhere we call `codesign`. When a real Developer ID is in
# use, also opt into hardened runtime + a secure timestamp so the resulting
# bundle is eligible for notarization without a second signing pass.
if(CODESIGN_IDENTITY STREQUAL "-")
    set(CODESIGN_FLAGS --force --sign "${CODESIGN_IDENTITY}")
else()
    set(CODESIGN_FLAGS --force --sign "${CODESIGN_IDENTITY}"
        --options=runtime --timestamp)
endif()

# magentart_add_notarize_target(<target_name>
#     APP <path-to-deployed-.app>
#     DEPENDS <deploy-target>
#     [ZIP_NAME <name>.zip])
#
# Zips the deployed bundle, submits it to Apple's notary service, and
# staples the ticket. Requires CODESIGN_IDENTITY to be a real Developer ID
# (not ad-hoc "-") so that the bundle is signed for hardened runtime, and a
# keychain profile named ${NOTARYTOOL_KEYCHAIN_PROFILE} (see
# `xcrun notarytool store-credentials`).
function(magentart_add_notarize_target target_name)
    cmake_parse_arguments(N "" "APP;DEPENDS;ZIP_NAME" "EXTRA_FILES" ${ARGN})
    if(NOT N_APP OR NOT N_DEPENDS)
        message(FATAL_ERROR
            "magentart_add_notarize_target: APP and DEPENDS are required")
    endif()
    if(NOT N_ZIP_NAME)
        set(N_ZIP_NAME "${target_name}.zip")
    endif()

    if(CODESIGN_IDENTITY STREQUAL "-")
        add_custom_target(${target_name}
            COMMAND ${CMAKE_COMMAND} -E echo
                "ERROR: ${target_name} requires a Developer ID. Reconfigure with -DCODESIGN_IDENTITY=\"Developer ID Application: Your Name (TEAMID)\""
            COMMAND ${CMAKE_COMMAND} -E false
            COMMENT "${target_name} disabled (CODESIGN_IDENTITY is ad-hoc)"
        )
        return()
    endif()

    set(zip_path "${CMAKE_BINARY_DIR}/${N_ZIP_NAME}")
    get_filename_component(app_name "${N_APP}" NAME)
    set(staging_dir "${CMAKE_BINARY_DIR}/staging_${target_name}")

    set(prep_commands)
    list(APPEND prep_commands COMMAND ${CMAKE_COMMAND} -E rm -rf "${staging_dir}")
    list(APPEND prep_commands COMMAND ${CMAKE_COMMAND} -E make_directory "${staging_dir}")
    list(APPEND prep_commands COMMAND ditto "${N_APP}" "${staging_dir}/${app_name}")
    foreach(extra_file IN LISTS N_EXTRA_FILES)
        list(APPEND prep_commands COMMAND ${CMAKE_COMMAND} -E copy "${extra_file}" "${staging_dir}/")
    endforeach()

    add_custom_target(${target_name}
        DEPENDS ${N_DEPENDS}
        ${prep_commands}
        COMMAND ditto -c -k "${staging_dir}" "${zip_path}"
        COMMAND xcrun notarytool submit "${zip_path}"
            --keychain-profile "${NOTARYTOOL_KEYCHAIN_PROFILE}" --wait
        COMMAND xcrun stapler staple "${N_APP}"
        COMMAND xcrun stapler validate "${N_APP}"
        # Re-zip the stapled bundle for distribution.
        ${prep_commands}
        COMMAND ditto -c -k "${staging_dir}" "${zip_path}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${staging_dir}"
        COMMENT "Notarizing ${N_APP} -> ${zip_path}"
        USES_TERMINAL
        VERBATIM
    )
endfunction()

# Enable the AUv3 host's developer-only debug overlay (NSTextField in the
# plugin UI) and mrt_debug.log file writer. Off by default for release builds.
option(MAGENTART_DEBUG_LOG "Enable AUv3 debug overlay + mrt_debug.log file writer" OFF)
if(MAGENTART_DEBUG_LOG)
    add_compile_definitions(MAGENTART_DEBUG_LOG=1)
else()
    add_compile_definitions(MAGENTART_DEBUG_LOG=0)
endif()

# Run a unified npm workspaces install on the examples folder
add_custom_target(npm_install_root
    COMMAND npm install
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples
    COMMENT "Running NPM workspaces install inside examples..."
)

# React UI for the main MRT2 plugin.
add_custom_target(build_mrt2_ui ALL
    COMMAND npm run build
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/mrt2/react_ui
    COMMENT "Building MRT2 UI..."
)
add_dependencies(build_mrt2_ui npm_install_root)

# --- Project targets ----------------------------------------------------------
add_subdirectory(core)
add_subdirectory(examples/mrt2/auv3)
add_subdirectory(examples/mrt2/standalone)
add_subdirectory(examples/jam)
add_subdirectory(examples/collider)
add_subdirectory(examples/hello_mrt2)
add_subdirectory(examples/max)
add_subdirectory(examples/pd)
add_subdirectory(examples/sc)
