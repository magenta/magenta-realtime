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

# magentart::core — portable C++ inference library.
#
# Provides the MLXEngine (MLX + TFLite + SentencePiece inference pipeline) and
# RealtimeRunner (audio-thread-safe wrapper with ring buffers and MIDI gate).
# Link this target from any host (AUv3, standalone, CLI) that wants to run
# the Magenta RealTime 2 model.

add_library(magentart_core STATIC
    src/mlx_engine.cpp
    src/realtime_runner.cpp
    src/autorelease_pool.cpp
    src/midi_note_tracker.cpp
)
add_library(magentart::core ALIAS magentart_core)

target_include_directories(magentart_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(magentart_core PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
)

target_compile_features(magentart_core PUBLIC cxx_std_20)

target_link_libraries(magentart_core PUBLIC
    mlx
    sentencepiece
    tensorflow-lite
    "-framework Metal"
    "-framework Accelerate"
    "-framework Foundation"
)



# Install rules (optional — enables `cmake --install` for downstream packaging).
install(TARGETS magentart_core EXPORT magentartTargets)
install(DIRECTORY include/magentart DESTINATION include)

# Standalone unit test for NumpyRandomState bit-exactness
add_executable(numpy_random_state_test
    src/numpy_random_state_test.cpp
)
target_include_directories(numpy_random_state_test PRIVATE src)
