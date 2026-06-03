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

// benchmark_mlxfn.cpp — Standalone latency benchmark for the exported MLX
// model.
//
// Loads <model>.mlxfn and <model>_state.safetensors, then runs warmup + timed
// inference steps to measure per-step latency.
//
// Build:
//   cmake benchmark -B benchmark_build && cmake --build benchmark_build
//   --target benchmark_mlxfn -j10
//
// Run:
//   ./benchmark_build/benchmark_mlxfn [model_dir] [num_steps]

#include <mlx/mlx.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace mx = mlx::core;

// ─── Configuration matching run_model.py export ─────────────────────────────

static constexpr int kCondTokenSize =
    141; // 12 (musiccoca) + 128 (notes) + 1 (drums)
static constexpr int kNumReservedTokens =
    7; // system.NUM_RESERVED_TOKENS(6) + 1 for dropout

// Default parameter values (matching run_model.py)
static constexpr float kDefaultTemperature = 1.3f;
static constexpr int kDefaultTopK = 40;
static constexpr float kDefaultCfgMusicCoCa = -1.0f;
static constexpr float kDefaultCfgNotes = -1.0f;
static constexpr float kDefaultCfgDrums = -1.0f;

// Audio
static constexpr int kSampleRate = 48000;
static constexpr int kFrameSamples = 1920; // 48000 Hz / 25 Hz
static constexpr int kNumChannels = 2;

// Warmup & benchmark
static constexpr int kWarmupSteps = 5;
static constexpr int kDefaultNumSteps = 100;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build the conditioning input array.
/// cond_tokens = concat([musiccoca, notes]) + NUM_RESERVED_TOKENS
static mx::array make_cond_input() {
  int cond_size = kCondTokenSize;

  int musiccoca[] = {679, 132, 480, 389, 160, 1010,
                     254, 533, 156, 85,  874, 981}; // disco funk

  int notes[128];
  for (int i = 0; i < 127; ++i)
    notes[i] = 0;
  notes[127] = 1;

  std::vector<int> cond(cond_size, -1 + kNumReservedTokens);
  for (int i = 0; i < 12; ++i)
    cond[i] = musiccoca[i] + kNumReservedTokens;
  for (int i = 0; i < 128; ++i)
    cond[12 + i] = notes[i] + kNumReservedTokens;

  return mx::array(cond.data(), {1, 1, cond_size}, mx::int32);
}

/// Build negative conditioning arrays.
/// Returns {neg_musiccoca, neg_notes}
static std::vector<mx::array> make_neg_inputs() {
  int cond_size = kCondTokenSize;

  int musiccoca[] = {679, 132, 480, 389, 160, 1010,
                     254, 533, 156, 85,  874, 981}; // disco funk

  int notes[128];
  for (int i = 0; i < 127; ++i)
    notes[i] = 0;
  notes[127] = 1;

  // Initialize all negative arrays to -1 + reserved
  std::vector<int> neg_musiccoca(cond_size, -1 + kNumReservedTokens);
  std::vector<int> neg_notes(cond_size, -1 + kNumReservedTokens);

  // neg_musiccoca keeps notes, neg_notes keeps musiccoca
  for (int i = 0; i < 12; ++i)
    neg_notes[i] = musiccoca[i] + kNumReservedTokens;
  for (int i = 0; i < 128; ++i)
    neg_musiccoca[12 + i] = notes[i] + kNumReservedTokens;

  std::vector<mx::array> result;

  result.push_back(
      mx::array(neg_musiccoca.data(), {1, 1, cond_size}, mx::int32));
  result.push_back(mx::array(neg_notes.data(), {1, 1, cond_size}, mx::int32));

  return result;
}

/// Load ordered state arrays from safetensors: state_0, state_1, ...
static std::vector<mx::array> load_state(const std::string &path) {
  auto [arrays, metadata] = mx::load_safetensors(path);
  std::vector<mx::array> state;
  for (int i = 0;; ++i) {
    auto it = arrays.find("state_" + std::to_string(i));
    if (it == arrays.end())
      break;
    state.push_back(it->second);
  }
  mx::eval(state);
  return state;
}

/// Write interleaved stereo float32 WAV file (IEEE float format, tag 3).
/// This matches scipy.io.wavfile.write with float32 data exactly.
static bool write_wav_file(const std::string &path,
                           const std::vector<float> &interleaved,
                           int sample_rate, int num_channels) {
  uint32_t num_frames =
      static_cast<uint32_t>(interleaved.size()) / num_channels;
  uint16_t bits_per_sample = 32;
  uint16_t block_align = num_channels * (bits_per_sample / 8); // 2 * 4 = 8
  uint32_t byte_rate = sample_rate * block_align;
  uint32_t data_size = num_frames * block_align;
  uint32_t chunk_size = 36 + data_size;

  std::ofstream f(path, std::ios::binary);
  if (!f)
    return false;

  // RIFF header
  f.write("RIFF", 4);
  f.write(reinterpret_cast<const char *>(&chunk_size), 4);
  f.write("WAVE", 4);

  // fmt sub-chunk
  f.write("fmt ", 4);
  uint32_t subchunk1_size = 16;
  f.write(reinterpret_cast<const char *>(&subchunk1_size), 4);
  uint16_t audio_format = 3; // IEEE float (matches scipy float32 output)
  uint16_t nc = static_cast<uint16_t>(num_channels);
  uint32_t sr = static_cast<uint32_t>(sample_rate);
  f.write(reinterpret_cast<const char *>(&audio_format), 2);
  f.write(reinterpret_cast<const char *>(&nc), 2);
  f.write(reinterpret_cast<const char *>(&sr), 4);
  f.write(reinterpret_cast<const char *>(&byte_rate), 4);
  f.write(reinterpret_cast<const char *>(&block_align), 2);
  f.write(reinterpret_cast<const char *>(&bits_per_sample), 2);

  // data sub-chunk — write raw float32 samples (no clipping/conversion)
  f.write("data", 4);
  f.write(reinterpret_cast<const char *>(&data_size), 4);
  f.write(reinterpret_cast<const char *>(interleaved.data()),
          interleaved.size() * sizeof(float));

  return f.good();
}

/// Pack function arguments and run one step. Returns (audio, new_state).
static std::pair<mx::array, std::vector<mx::array>> run_step(
    const std::function<std::vector<mx::array>(const std::vector<mx::array> &)>
        &fn,
    const mx::array &cond, float temperature, int top_k, float cfg_musiccoca,
    float cfg_notes, float cfg_drums, const std::vector<mx::array> &neg_arrays,
    const std::vector<mx::array> &state, int dynamic_rvq_depth) {
  std::vector<mx::array> args;
  args.push_back(cond);
  args.push_back(mx::array({temperature}));
  args.push_back(mx::array({top_k}, mx::int32));
  args.push_back(mx::array({cfg_musiccoca}));
  args.push_back(mx::array({cfg_notes}));
  args.push_back(mx::array({cfg_drums}));
  args.insert(args.end(), neg_arrays.begin(), neg_arrays.end());
  args.push_back(
      mx::zeros({1, 0, dynamic_rvq_depth}, mx::int32)); // forced_tokens
  args.insert(args.end(), state.begin(), state.end());

  auto outputs = fn(args);
  auto audio = outputs[0];
  std::vector<mx::array> new_state(outputs.begin() + 1, outputs.end());
  return {audio, new_state};
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  // Parse arguments: benchmark_mlxfn [model_dir] [num_steps]
  std::string model_dir = "resources";
  int num_steps = kDefaultNumSteps;

  // Scan for positional args
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    positional.push_back(argv[i]);
  }

  if (positional.size() >= 1)
    model_dir = positional[0];
  if (positional.size() >= 2)
    num_steps = std::atoi(positional[1].c_str());

  // Strip trailing slashes
  while (!model_dir.empty() && model_dir.back() == '/')
    model_dir.pop_back();

  // Derive model name from directory basename
  std::string basename = model_dir;
  auto slash_pos = basename.rfind('/');
  if (slash_pos != std::string::npos)
    basename = basename.substr(slash_pos + 1);

  std::string mlxfn_path = model_dir + "/" + basename + ".mlxfn";
  std::string state_path = model_dir + "/" + basename + "_state.safetensors";

  printf("═══════════════════════════════════════════════════════════════\n");
  printf("  MagentaRT MLX C++ Latency Benchmark\n");
  printf("═══════════════════════════════════════════════════════════════\n");
  printf("  Model:       %s\n", mlxfn_path.c_str());
  printf("  State:       %s\n", state_path.c_str());
  printf("  Num steps:   %d\n", num_steps);
  printf("───────────────────────────────────────────────────────────────\n\n");

  // ── Load model ──────────────────────────────────────────────────────
  printf("[1/4] Loading exported function...\n");
  auto imported_fn = mx::import_function(mlxfn_path);
  printf("       ✓ Loaded .mlxfn\n");

  // Optionally compile for performance (mirrors mlx_engine.cpp)
  auto compiled_fn =
      mx::compile([&imported_fn](const std::vector<mx::array> &inputs) {
        return imported_fn(inputs);
      });
  printf("       ✓ Compiled function\n");

  // ── Load state ──────────────────────────────────────────────────────
  printf("[2/4] Loading initial state...\n");
  auto initial_state = load_state(state_path);
  printf("       ✓ Loaded %zu state arrays\n", initial_state.size());

  // todo: Find faster way of inferring RVQ depth.
  int dynamic_rvq_depth = 16; // Default
  for (const auto &arr : initial_state) {
    const auto &shape = arr.shape();
    if (shape.size() == 4 && shape[2] == 1) {
      dynamic_rvq_depth = shape[3];
    } else if (shape.size() == 3 && shape[1] == 1) {
      dynamic_rvq_depth = shape[2];
    }
  }
  printf("       ✓ Deduced dynamic RVQ depth: %d\n\n", dynamic_rvq_depth);

  // ── Build conditioning input ────────────────────────────────────────
  auto cond = make_cond_input();
  auto neg_arrays = make_neg_inputs();

  // ── Diagnostic: run one step from initial state and print output ────
  {
    printf("[diag] Running diagnostic step from initial state...\n");
    auto [diag_audio, diag_state] =
        run_step(compiled_fn, cond, kDefaultTemperature, kDefaultTopK,
                 kDefaultCfgMusicCoCa, kDefaultCfgNotes, kDefaultCfgDrums,
                 neg_arrays, initial_state, dynamic_rvq_depth);
    mx::eval(diag_audio);

    // Print shape and dtype
    auto shape = diag_audio.shape();
    printf("       Audio shape: [");
    for (size_t d = 0; d < shape.size(); ++d)
      printf("%d%s", shape[d], d + 1 < shape.size() ? ", " : "");
    printf("]\n");
    printf("       Audio dtype: %s\n",
           diag_audio.dtype() == mx::float32    ? "float32"
           : diag_audio.dtype() == mx::float16  ? "float16"
           : diag_audio.dtype() == mx::bfloat16 ? "bfloat16"
           : diag_audio.dtype() == mx::int16    ? "int16"
                                                : "other");

    // Convert to float32 for inspection
    auto diag_f32 = mx::astype(diag_audio, mx::float32);
    mx::eval(diag_f32);
    const float *ptr = diag_f32.data<float>();
    int total = diag_f32.size();

    // Compute min/max/mean
    float vmin = ptr[0], vmax = ptr[0];
    double vsum = 0;
    for (int i = 0; i < total; ++i) {
      if (ptr[i] < vmin)
        vmin = ptr[i];
      if (ptr[i] > vmax)
        vmax = ptr[i];
      vsum += ptr[i];
    }
    printf("       Audio min:   %.8f\n", vmin);
    printf("       Audio max:   %.8f\n", vmax);
    printf("       Audio mean:  %.8f\n", vsum / total);

    printf("\n");
  }

  // ── Warmup ──────────────────────────────────────────────────────────
  printf("[3/4] Warming up (%d steps)...\n", kWarmupSteps);
  auto state = initial_state;
  for (int i = 0; i < kWarmupSteps; ++i) {
    auto [audio, new_state] =
        run_step(compiled_fn, cond, kDefaultTemperature, kDefaultTopK,
                 kDefaultCfgMusicCoCa, kDefaultCfgNotes, kDefaultCfgDrums,
                 neg_arrays, state, dynamic_rvq_depth);
    mx::eval(audio);
    mx::eval(new_state);
    state = std::move(new_state);
    printf("       warmup step %d/%d done\n", i + 1, kWarmupSteps);
  }
  printf("       ✓ Warmed up\n\n");

  // ── Benchmark: per-step latency ─────────────────────────────────────
  printf("[4/4] Running benchmark (%d steps)...\n", num_steps);
  state = initial_state; // reset to initial state

  std::vector<double> step_times_ms;
  step_times_ms.reserve(num_steps);

  std::vector<mx::array> audio_frames; // collect for WAV export
  audio_frames.reserve(num_steps);

  using clock = std::chrono::steady_clock;
  auto total_start = clock::now();

  for (int i = 0; i < num_steps; ++i) {
    auto step_start = clock::now();

    auto [audio, new_state] =
        run_step(compiled_fn, cond, kDefaultTemperature, kDefaultTopK,
                 kDefaultCfgMusicCoCa, kDefaultCfgNotes, kDefaultCfgDrums,
                 neg_arrays, state, dynamic_rvq_depth);
    mx::eval(audio);
    mx::eval(new_state);

    auto step_end = clock::now();
    double ms = std::chrono::duration<double, std::milli>(step_end - step_start)
                    .count();
    step_times_ms.push_back(ms);
    audio_frames.push_back(audio);
    state = std::move(new_state);

    printf("       step %d: %.2f ms\n", i, ms);
  }

  auto total_end = clock::now();
  double total_elapsed_s =
      std::chrono::duration<double>(total_end - total_start).count();
  double avg_ms = (total_elapsed_s / num_steps) * 1000.0;
  double steps_per_sec = num_steps / total_elapsed_s;

  // Compute percentiles (sort a copy to preserve original order for CSV)
  auto sorted_times = step_times_ms;
  std::sort(sorted_times.begin(), sorted_times.end());
  double min_ms = sorted_times.front();
  double max_ms = sorted_times.back();
  double median_ms = sorted_times[num_steps / 2];
  double p95_ms = sorted_times[static_cast<int>(num_steps * 0.95)];
  double p99_ms = sorted_times[static_cast<int>(num_steps * 0.99)];

  // ── Results ─────────────────────────────────────────────────────────
  printf("\n");
  printf("═══════════════════════════════════════════════════════════════\n");
  printf("  Results\n");
  printf("═══════════════════════════════════════════════════════════════\n");
  printf("  Total:       %d steps in %.1fs\n", num_steps, total_elapsed_s);
  printf("  Throughput:  %.1f steps/s\n", steps_per_sec);
  printf("  Avg:         %.1f ms/step\n", avg_ms);
  printf("  Median:      %.1f ms/step\n", median_ms);
  printf("  Min:         %.1f ms/step\n", min_ms);
  printf("  Max:         %.1f ms/step\n", max_ms);
  printf("  P95:         %.1f ms/step\n", p95_ms);
  printf("  P99:         %.1f ms/step\n", p99_ms);
  printf("───────────────────────────────────────────────────────────────\n");
  printf("  Target:      25 steps/s, 40 ms/step for real-time\n");
  printf("  Status:      %s\n",
         avg_ms <= 40.0 ? "✅ WITHIN BUDGET" : "❌ EXCEEDS BUDGET");
  printf("═══════════════════════════════════════════════════════════════\n");

  // ── Save audio to WAV ───────────────────────────────────────────────
  // Each audio frame is [1, 2, 1920] (batch, channels, samples).
  // Concatenate along time → non-interleaved [L0..LN, R0..RN] → interleave
  // for WAV.
  size_t total_samples = static_cast<size_t>(num_steps) * kFrameSamples;
  std::vector<float> interleaved(total_samples * kNumChannels);

  for (int i = 0; i < num_steps; ++i) {
    bool is_int16 = (audio_frames[i].dtype() == mx::int16);
    auto frame = mx::astype(audio_frames[i], mx::float32);
    mx::eval(frame);
    const float *ptr = frame.data<float>();
    // frame layout: [1, 2, 1920] → ptr[0..1919] = L, ptr[1920..3839] = R
    size_t base = static_cast<size_t>(i) * kFrameSamples;
    float scale = is_int16 ? (1.0f / 32768.0f) : 1.0f;
    for (int s = 0; s < kFrameSamples; ++s) {
      interleaved[(base + s) * 2 + 0] = ptr[s] * scale;                 // L
      interleaved[(base + s) * 2 + 1] = ptr[kFrameSamples + s] * scale; // R
    }
  }

  // Print last 10 L and R channel samples
  printf("\n  Last 10 samples of generated audio:\n");
  printf("       L channel:\n");
  for (size_t i = total_samples - 10; i < total_samples; ++i)
    printf("         L[%5zu] = %.8f\n", i, interleaved[i * 2 + 0]);
  printf("       R channel:\n");
  for (size_t i = total_samples - 10; i < total_samples; ++i)
    printf("         R[%5zu] = %.8f\n", i, interleaved[i * 2 + 1]);

  std::string wav_path = "outputs/output_audio_cpp_" + basename + ".wav";
  if (write_wav_file(wav_path, interleaved, kSampleRate, kNumChannels)) {
    double duration_s = static_cast<double>(total_samples) / kSampleRate;
    printf("\n  Saved %.1fs of audio to %s\n", duration_s, wav_path.c_str());
  } else {
    printf("\n  ⚠ Failed to save WAV file\n");
  }

  // ── Save per-step latency CSV ───────────────────────────────────────
  std::string csv_path = "outputs/latency_log_" + basename + ".csv";
  std::ofstream csv(csv_path);
  if (csv.is_open()) {
    csv << "step,step_ms\n";
    for (int i = 0; i < num_steps; ++i)
      csv << i << "," << step_times_ms[i] << "\n";
    csv.close();
    printf("\n  Saved per-step latency log to %s\n", csv_path.c_str());
  }

  return 0;
}
