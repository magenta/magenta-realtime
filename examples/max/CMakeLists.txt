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

# examples/max/CMakeLists.txt — builds the `mrt2~` MaxMSP external (.mxo bundle).
#
# Mirrors the standalone/jam/hello_mrt2 pattern: link against
# magentart::core (which transitively brings in mlx, sentencepiece, tflite,
# and the Apple frameworks the engine needs) and add only the Max-specific
# headers + linker flags here.
#
# Resulting bundle layout:
#   mrt2~.mxo/
#     Contents/
#       Info.plist
#       PkgInfo
#       MacOS/
#         mrt2~                  (the loadable module)
#         mlx.metallib          (Metal shaders, colocated for MLX runtime resolution)

include(FetchContent)

# ─── Max SDK base (headers + framework stubs) ────────────────────────────────
FetchContent_Declare(
    max-sdk-base
    GIT_REPOSITORY https://github.com/Cycling74/max-sdk-base.git
    GIT_TAG        main
    GIT_SHALLOW    ON
)
message(STATUS "Fetching max-sdk-base...")
FetchContent_MakeAvailable(max-sdk-base)

set(MAX_SDK_C74SUPPORT  "${max-sdk-base_SOURCE_DIR}/c74support")
set(MAX_SDK_INCLUDES    "${MAX_SDK_C74SUPPORT}/max-includes")
set(MAX_SDK_MSP_INCLUDES "${MAX_SDK_C74SUPPORT}/msp-includes")
set(MAX_SDK_JIT_INCLUDES "${MAX_SDK_C74SUPPORT}/jit-includes")
set(MAX_SDK_SCRIPTS     "${max-sdk-base_SOURCE_DIR}/script")

# ─── Target ──────────────────────────────────────────────────────────────────
# CMake doesn't allow `~` in target names, so the internal name is
# `mrt2_max`; the bundle output is renamed to `mrt2~.mxo` below.
add_library(mrt2_max MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/mrt_tilde.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/magenta_paths.cpp
)

target_compile_features(mrt2_max PRIVATE cxx_std_20)
target_compile_options(mrt2_max PRIVATE -fobjc-arc)

target_include_directories(mrt2_max PRIVATE
    ${MAX_SDK_C74SUPPORT}
    ${MAX_SDK_INCLUDES}
    ${MAX_SDK_MSP_INCLUDES}
    ${MAX_SDK_JIT_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp     # magenta_paths.h
)

target_link_libraries(mrt2_max PRIVATE
    magentart::core
    "-framework AudioToolbox"
)

# ─── Max-style undefined-symbol linker flags ─────────────────────────────────
# Max externals link with -Wl,-U,_<sym> for every Max API symbol; the symbols
# are resolved at load time by the host (Max.app provides them). The full list
# lives in max-linker-flags.txt — we pass it through verbatim.
file(READ "${MAX_SDK_SCRIPTS}/max-linker-flags.txt" MRT_MAX_LINKER_FLAGS_RAW)
separate_arguments(MRT_MAX_LINKER_FLAGS UNIX_COMMAND "${MRT_MAX_LINKER_FLAGS_RAW}")
target_link_options(mrt2_max PRIVATE ${MRT_MAX_LINKER_FLAGS})

# Link the MaxAudioAPI framework (provides MSP symbols like dsp_setup, class_dspinit, etc.)
find_library(
    MRT_MSP_FRAMEWORK MaxAudioAPI
    REQUIRED
    PATHS "${MAX_SDK_MSP_INCLUDES}"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
target_link_libraries(mrt2_max PRIVATE ${MRT_MSP_FRAMEWORK})

# ─── Bundle configuration ────────────────────────────────────────────────────
set_target_properties(mrt2_max PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION "mxo"
    XCODE_ATTRIBUTE_WRAPPER_EXTENSION "mxo"
    PREFIX ""                      # no `lib` prefix
    OUTPUT_NAME "mrt2~"             # bundle becomes `mrt2~.mxo`
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
    MACOSX_BUNDLE_BUNDLE_NAME "mrt2~"
    MACOSX_BUNDLE_EXECUTABLE_NAME "mrt2~"
    MACOSX_BUNDLE_BUNDLE_VERSION "${MAGENTART_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${MAGENTART_VERSION}"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.magentart.mrt2"
)

# Substitution variables expected by the SDK's Info.plist.in template.
set(AUTHOR_DOMAIN "com.magentart" CACHE INTERNAL "")
set(BUNDLE_IDENTIFIER "mrt2" CACHE INTERNAL "")
set(EXCLUDE_FROM_COLLECTIVES "no" CACHE INTERNAL "")
set(PACKAGE_VERSION "${MAGENTART_VERSION}" CACHE INTERNAL "")
set(COPYRIGHT_STRING "" CACHE INTERNAL "")

# ─── Post-build: PkgInfo, mlx.metallib, codesign ─────────────────────────────
add_custom_command(TARGET mrt2_max POST_BUILD
    # PkgInfo (Max expects this in the bundle)
    COMMAND ${CMAKE_COMMAND} -E copy
        "${MAX_SDK_SCRIPTS}/PkgInfo"
        "$<TARGET_BUNDLE_DIR:mrt2_max>/Contents/PkgInfo"
    # MLX Metal shaders next to the binary (same trick used by AU + standalone)
    COMMAND ${CMAKE_COMMAND} -E copy
        "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
        "$<TARGET_BUNDLE_DIR:mrt2_max>/Contents/MacOS/mlx.metallib"
    # Ad-hoc codesign (Apple Silicon refuses to load otherwise). Inside-out:
    # sign nested mlx.metallib first, then the outer bundle (no --deep, which
    # Apple has deprecated for signing).
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:mrt2_max>/Contents/MacOS/mlx.metallib"
    COMMAND codesign ${CODESIGN_FLAGS}
        "$<TARGET_BUNDLE_DIR:mrt2_max>"
    COMMENT "Finalizing mrt2~.mxo bundle"
    VERBATIM
)

# ─── Optional deploy target ──────────────────────────────────────────────────
# `make deploy_mrt2_max` copies the .mxo into the user's Max library so it shows
# up in Max's externals search path. Disabled by default — users may have Max 8
# vs Max 9, may prefer ~/Documents vs ~/Library, or may want to vendor it into
# a project package directory of their own.
set(MAX_LIBRARY_DIR "$ENV{HOME}/Documents/Max 9/Library" CACHE PATH
    "Where `make deploy_mrt2_max` will copy mrt2~.mxo")
add_custom_target(deploy_mrt2_max
    DEPENDS mrt2_max
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MAX_LIBRARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${MAX_LIBRARY_DIR}/mrt2~.mxo"
    COMMAND ditto
        "$<TARGET_BUNDLE_DIR:mrt2_max>"
        "${MAX_LIBRARY_DIR}/mrt2~.mxo"
    COMMENT "Deploying mrt2~.mxo to ${MAX_LIBRARY_DIR}"
    VERBATIM
)
