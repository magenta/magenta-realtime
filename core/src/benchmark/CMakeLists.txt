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

project(MagentaRT_Benchmark LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- MLX (fetched only if not already available, e.g. when built standalone) ---
if(NOT TARGET mlx)
    include(FetchContent)
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
endif()

# ─── Standalone latency benchmark ─────────────────────────────────────────────

add_executable(benchmark_mlxfn
    benchmark_mlxfn.cpp
)

target_link_libraries(benchmark_mlxfn PRIVATE
    mlx
    "-framework Metal"
    "-framework Accelerate"
    "-framework Foundation"
)

set_target_properties(benchmark_mlxfn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_custom_command(TARGET benchmark_mlxfn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "${CMAKE_BINARY_DIR}/mlx.metallib"
    COMMENT "Copying mlx.metallib next to benchmark_mlxfn binary"
)
