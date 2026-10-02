cmake_minimum_required(VERSION 3.14)
project(magenta-rt-ggml CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ggml comes from a llama.cpp (or ggml) checkout; point this at the directory
# containing ggml's own CMakeLists.txt.
set(GGML_SOURCE_DIR "$ENV{HOME}/code/personal_projects/llama.cpp/ggml"
    CACHE PATH "Path to the ggml source directory")

if(NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml sources not found at '${GGML_SOURCE_DIR}'.\n"
        "Re-run with -DGGML_SOURCE_DIR=/path/to/llama.cpp/ggml")
endif()

add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_library(mrt_ggml STATIC mrt_ggml.cpp)
target_link_libraries(mrt_ggml PUBLIC ggml)
target_include_directories(mrt_ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(mrt_parity parity_main.cpp)
target_link_libraries(mrt_parity PRIVATE mrt_ggml)

add_executable(mrt_bench benchmark_main.cpp)
target_link_libraries(mrt_bench PRIVATE mrt_ggml)

add_executable(mrt_generate generate_main.cpp)
target_link_libraries(mrt_generate PRIVATE mrt_ggml)
