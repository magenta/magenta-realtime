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

# examples/pd/CMakeLists.txt — builds the `mrt2~` Pure Data signal external (.pd_darwin).
#
# Mirrors the standalone/jam/hello_mrt2 pattern: link against
# magentart::core (which transitively brings in mlx, sentencepiece, tflite,
# and the Apple frameworks the engine needs) and add only the PD-specific
# headers + linker flags here.
#
# Output layout (after build + optional `make deploy_mrt2_pd`):
#   ~/Documents/Pd/externals/mrt2~/
#     mrt2~.pd_darwin   (the loadable module)
#     mlx.metallib     (Metal shaders, colocated for MLX runtime resolution)
#     mrt2~-help.pd     (PD help patcher)

include(FetchContent)

# ─── Pure Data headers (m_pd.h) ──────────────────────────────────────────────
# We only need the public header. The pure-data repo has no top-level
# CMakeLists.txt, so FetchContent_MakeAvailable just populates the source dir
# without invoking any build.
FetchContent_Declare(
    pure-data
    GIT_REPOSITORY https://github.com/pure-data/pure-data.git
    GIT_TAG        0.55-2
    GIT_SHALLOW    ON
)
message(STATUS "Fetching pure-data...")
FetchContent_MakeAvailable(pure-data)
set(PD_INCLUDE_DIR "${pure-data_SOURCE_DIR}/src")

# ─── Target ──────────────────────────────────────────────────────────────────
# CMake disallows `~` in target names, so the internal name is `mrt2_pd`; the
# loadable module is renamed to `mrt2~.pd_darwin` below.
add_library(mrt2_pd MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/mrt_tilde.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/magenta_paths.cpp
)

target_compile_features(mrt2_pd PRIVATE cxx_std_20)
# .mm because magentart::core's realtime_runner.h is C++ and the engine relies
# on Objective-C autorelease pools internally; building this TU as
# Objective-C++ keeps the module ABI consistent with the rest of the hosts.
target_compile_options(mrt2_pd PRIVATE -fobjc-arc)

target_include_directories(mrt2_pd PRIVATE
    ${PD_INCLUDE_DIR}                           # m_pd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp     # magenta_paths.h
)

target_link_libraries(mrt2_pd PRIVATE
    magentart::core
    "-framework AudioToolbox"
)

# ─── PD-specific linker behaviour ────────────────────────────────────────────
# PD externals on macOS are loaded via dlopen into the running Pd process,
# which provides the m_pd.h symbols at load time. `-undefined dynamic_lookup`
# defers all unresolved symbols to runtime resolution. Every PD external
# build system (pd.cmake, pd.build, pd-lib-builder, deken) does this. If
# Apple ever fully removes the flag, switch to `-bundle_loader <path-to-pd>`.
target_link_options(mrt2_pd PRIVATE -undefined dynamic_lookup)

# ─── Output naming ───────────────────────────────────────────────────────────
# Pd loads externals with the `.pd_darwin` extension on macOS. No `lib` prefix
# (Pd looks up the file by exact name). Tilde in OUTPUT_NAME yields
# `mrt2~.pd_darwin`.
set_target_properties(mrt2_pd PROPERTIES
    PREFIX ""
    SUFFIX ".pd_darwin"
    OUTPUT_NAME "mrt2~"
)

# ─── Post-build: mlx.metallib + ad-hoc codesign ──────────────────────────────
add_custom_command(TARGET mrt2_pd POST_BUILD
    # MLX expects mlx.metallib next to the module that loaded it; copy it in.
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_FILE_DIR:mrt2_pd>/mlx.metallib"
    # Apple Silicon refuses to load unsigned shared modules into Pd's process.
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_FILE_DIR:mrt2_pd>/mlx.metallib"
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_FILE:mrt2_pd>"
    COMMENT "Finalizing mrt2~.pd_darwin"
    VERBATIM
)

# ─── Optional deploy target ──────────────────────────────────────────────────
# `make deploy_mrt2_pd` copies the external + its metallib + help patch into
# the user's Pd externals directory. Disabled by default — Pd users may have
# multiple Pd installations or prefer different search-path layouts.
set(PD_LIBRARY_DIR "$ENV{HOME}/Documents/Pd/externals" CACHE PATH
    "Where `make deploy_mrt2_pd` will install the mrt2~/ library directory")
add_custom_target(deploy_mrt2_pd
    DEPENDS mrt2_pd
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PD_LIBRARY_DIR}/mrt2~"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:mrt2_pd>"
        "${PD_LIBRARY_DIR}/mrt2~/mrt2~.pd_darwin"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE_DIR:mrt2_pd>/mlx.metallib"
        "${PD_LIBRARY_DIR}/mrt2~/mlx.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/mrt2~-help.pd"
        "${PD_LIBRARY_DIR}/mrt2~/mrt2~-help.pd"
    COMMENT "Deploying mrt2~ to ${PD_LIBRARY_DIR}/mrt2~"
    VERBATIM
)
