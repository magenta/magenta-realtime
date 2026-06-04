# magentart::core

Portable C++ inference library for the Magenta RealTime 2 model. Static
library, zero Objective-C coupling, ~180 KB of code that pulls in MLX, TFLite,
and SentencePiece at link time.

This is the starting point for building a new application on top of the model
— any macOS AU/AUv3/VST3/JUCE/CLI host can consume this target.

## Contents

```
core/
  include/magentart/
    detail/
      autorelease_pool.h — portable RAII wrapper around an Objective-C autorelease pool
    mlx_engine.h         — MLX + TFLite + SentencePiece inference pipeline
    realtime_runner.h    — audio-thread-safe wrapper (ring buffers, MIDI gate,
                           volume smoothing, prompt surface prompt blending)
    ring_buffer.h        — lock-free SPSC sample buffer
  src/
    mlx_engine.cpp
    realtime_runner.cpp
    autorelease_pool.cpp
  CMakeLists.txt         — builds `magentart::core`
```

## Consume from your CMake project

Either pull the whole repo in and `add_subdirectory`:

```cmake
add_subdirectory(path/to/magenta-rt-v2)   # configures MLX/TFLite/SentencePiece
target_link_libraries(my_app PRIVATE magentart::core)
```

or fetch with FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(magenta_rt_v2
    GIT_REPOSITORY https://github.com/magenta/magenta-realtime.git
    GIT_TAG main)
FetchContent_MakeAvailable(magenta_rt_v2)
target_link_libraries(my_app PRIVATE magentart::core)
```

Once linked, the public headers are reachable as
`#include <magentart/mlx_engine.h>` etc.

See [`examples/hello_mrt2/`](../examples/hello_mrt2/) for the
shortest working consumer — ~100 lines of `main.cpp`.

## Performance Gotchas

* **GPU Idle Downclocking**: macOS will automatically downclock the GPU if there are gaps in workload. When this happens, the inference cost per frame permanently increases by several millseconds. To prevent this, `RealtimeRunner`'s inference loop executes tiny dummy GPU operations (`mx::array({0.0f}) + mx::array({0.0f})`) while waiting for space in the ring buffer, forcing the GPU to maintain its active power state.

## Objective-C Autorelease Pool

If you write custom long-running inference loops using `MLXEngine`, you must drain the Objective-C autorelease pool. MLX generates temporary Objective-C objects on Metal during evaluation (`mlx::core::eval`), which can cause unbound memory growth.

To help you write standard C++ code without renaming your source files to `.mm`, `magentart::core` provides a portable RAII wrapper:

```cpp
#include <magentart/detail/autorelease_pool.h>

for (int step = 0; step < num_steps; ++step) {
    magentart::detail::AutoreleasePool pool;
    engine.generate_frame(L, R);
}
```

## Requirements

- macOS 14.0 or later ([MLX](https://ml-explore.github.io/mlx/build/html/install.html) has Linux support, but we haven't tested it; support not expected for Windows).
- Apple Silicon (arm64).
- Xcode command-line tools and CMake ≥ 3.27.

## Threading model

Consumers usually pick one of two entry points:

- **`MLXEngine`** (`mlx_engine.h`) — low-level, direct access to prompts and
  `generate_frame`. Lifecycle methods (`init_assets`, `load_model`, …) are
  controller-thread only. `generate_frame` is the inference-thread call.
  Atomic setters (sampling params, MIDI, drum mode) are safe from any thread.
- **`RealtimeRunner`** (`realtime_runner.h`) — wraps `MLXEngine`, adds stereo
  ring buffers + inference thread + volume/mute smoothing + MIDI-gate envelope
  + prompt surface blending. `read_audio_stereo` is lock-free and safe from an
  audio callback.

Each header documents its per-method threading expectations. When in doubt,
default to `RealtimeRunner` — it's what the bundled AUv3 and standalone hosts
use.

## Model checkpoints

The library expects:

1. A `.mlxfn` transformer model (exported by `mrt mlx export`, see the
   top-level [README](../README.md#export-mlxfn)).
2. A `musiccoca/` folder with TFLite assets: `text_encoder.tflite`,
   `pretrained_vector_quantizer.tflite`, `audio_preprocessor.tflite`,
   `music_encoder.tflite`, `spm.model`. See
   [README → Download MusicCoCa models](../README.md#download-musiccoca-models).

The bundled hosts embed both inside their `.app/Contents/Resources/`. Your
own host can either do the same or point `init_assets` at any on-disk path.
