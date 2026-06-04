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

#pragma once

/// @file mlx_engine.h
/// @brief Public C++ interface to the Magenta RealTime 2 inference pipeline.
///
/// `magentart::core::MLXEngine` drives a two-stage pipeline:
///   1. MusicCoCa text/audio → embedding → quantized tokens (TFLite, CPU)
///   2. Transformer step: tokens → stereo audio [1, 2, 1920]          (MLX,
///   Metal GPU)
///
/// Target frame rate is 25 Hz (40 ms per frame of 1920 samples @ 48 kHz).
///
/// Threading model
///   - The engine is **not** thread-safe as a whole; see per-method notes.
///   - Lifecycle methods (`init_assets`, `load_model`, `unload`, state I/O)
///     must be called from a single controller thread, typically the UI.
///   - `generate_frame` is the inference-thread call.
///   - Atomic setters/getters (sampling params, MIDI, drum mode) are lock-free
///     and safe to call from any thread, including the audio thread.
///   - Text prompts (`set_text_prompt(s)`) spawn background MusicCoCa encoding
///   on
///     an internal thread; poll `get_text_encoder_status()` /
///     `get_quantizer_status()` to observe completion.
///
/// Most consumers should use `RealtimeRunner` instead of driving `MLXEngine`
/// directly — it adds ring-buffered audio output, a MIDI-gate envelope, and
/// a realtime-safe `read_audio_stereo()` suitable for the audio callback.
///
/// This header deliberately avoids pulling in MLX, TFLite, or SentencePiece:
/// all of that is hidden behind a pimpl (`struct Impl`) in the .cpp, so
/// consumers only need a C++20 toolchain and this header on their include
/// path.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace magentart {
namespace core {

// --- Model-shape constants -------------------------------------------------
//
// Frame rate is 25 Hz: one output frame = `kFrameSamples` = 1920 samples
// at 48 kHz = 40 ms.

inline constexpr std::size_t kVocabSize = 1024;
inline constexpr std::size_t kEmbeddingDim = 256;
inline constexpr std::size_t kNumRVQLevels = 12;   ///< RVQ depth.
inline constexpr std::size_t kFrameSamples = 1920; ///< 48 kHz / 25 Hz.
inline constexpr std::size_t kNumChannels = 2;
inline constexpr std::size_t kMaxPrompts = 6;
inline constexpr int kMusicCoCaEmbeddingDim = 768;
inline constexpr std::size_t kMaxPCAComponents = 6;
inline constexpr std::size_t kMaxCentroids = 6;
inline constexpr std::size_t kMusicCoCaRVQLevels = 12;

/// Number of trailing (finer) MusicCoCa RVQ levels masked out of the live
/// conditioning. Only the coarsest `kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels`
/// levels steer generation; the finer tail is pinned to the model's mask id
/// (`kNumReservedTokens - 1`). Set to 0 to use all levels. The stored tokens
/// remain full-width — masking happens only where they're written into the
/// conditioning array, so re-blend/PCA still operate on all 12 levels.
inline constexpr std::size_t kMusicCoCaMaskedTailLevels = 6;

/// Default MusicCoCa RVQ tokens used until the first prompt encode completes.
/// These encode the single word "piano" against the musiccoca model and are
/// baked in so the engine has a valid conditioning signal from `init_assets`
/// onwards.

inline constexpr int kDefaultMusicCoCaTokensPiano[kMusicCoCaRVQLevels] = {
    325, 98, 979, 657, 802, 453, 475, 622, 729, 934, 567, 782};

/// Per-frame timings populated by `generate_frame`.
struct FrameMetrics {
  float transformer_ms = 0; ///< GPU time for the transformer step.
  float total_ms = 0;       ///< End-to-end wall time including bookkeeping.
};

class MLXEngine {
public:
  static constexpr int kNumReservedTokens = 7;

  MLXEngine();
  ~MLXEngine();
  MLXEngine(const MLXEngine &) = delete;
  MLXEngine &operator=(const MLXEngine &) = delete;
  MLXEngine(MLXEngine &&) noexcept;
  MLXEngine &operator=(MLXEngine &&) noexcept;

  /// @name Lifecycle
  /// Controller-thread only. Call these from a single thread (UI or main).
  /// @{

  /// Load TFLite models from `resource_dir/<model_subfolder>`. Required
  /// before any prompt or audio-embedding call.
  bool init_assets(const char *resource_dir,
                   const char *model_subfolder = "musiccoca");
  /// Load a `.mlxfn` model.
  bool load_model(const char *mlxfn_path);
  /// Load a separate SpectroStream encoder + prefill model for audio prefill.
  bool load_prefill_model(const char *spectrostream_mlxfn_path,
                          const char *prefill_mlxfn_path);
  /// Seed transformer state from an audio clip. Blocks the caller.
  ///
  /// The clip is encoded to RVQ tokens by the SpectroStream encoder, then
  /// the *trimmed* token sequence — `[trim_front_frames,
  /// num_audio_frames - trim_back_frames)` — is fed through the
  /// transformer one frame at a time to populate its KV caches. SpectroStream
  /// has a non-causal STFT front end, so tokens at the very start (encoder
  /// not yet warmed up) and the very end (window padded with zeros) are
  /// unreliable; trimming ~1 s (25 frames) at each end is a sensible
  /// default. Pass 0 for either trim to disable.
  ///

  /// **Checkpointing:** on success, the post-prefill state is copied
  /// into `transformer_initial_state_` — `reset_state()` will return
  /// the model to *this* state rather than the factory initial state.
  /// This lets callers prefill once and `reset_state()` repeatedly
  /// (e.g. while trying different prompts on the same musical
  /// context). Reload the model to recover the factory initial.
  ///
  /// If `out_audio_L` / `out_audio_R` are non-null, every frame of the
  /// model's audio output during the prefill loop
  /// is appended to them — useful for stitching prefill audio in front
  /// of the continuation in offline tools. Note that this appended audio
  /// is a reconstruction of the input raw audio (going through raw audio
  /// => spectrostream tokens => reconstructed audio.
  /// For long prefill durations, these
  /// may still carry reconstruction artifacts from the lossy codec.
  bool
  prefill_state(const float *audio_samples, int num_samples,
                int trim_front_frames, int trim_back_frames,
                std::function<void(const std::string &)> log_callback = nullptr,
                std::vector<float> *out_audio_L = nullptr,
                std::vector<float> *out_audio_R = nullptr,
                bool mask_musiccoca_during_prefill = true);
  /// Like `prefill_state` but feeds raw RVQ tokens directly into the
  /// transformer, skipping the SpectroStream encoder. Tokens are expected
  /// in *raw* per-codebook space (`0..kVocabSize-1`) and laid out as a
  /// flat array of `num_frames * get_rvq_depth()` int32 values,
  /// frame-major: `tokens[frame * rvq_depth + codebook]`. Use this with
  /// tokens captured from `generate_frame`'s `tokens_out` to restore a
  /// state or branch generation without the lossy cycle of decoding to
  /// audio and re-encoding.
  ///
  /// Like `prefill_state`, this checkpoints on success: the post-prefill
  /// `transformer_state_` is copied into `transformer_initial_state_`
  /// so `reset_state()` returns to here.
  bool prefill_state_from_tokens(
      const int32_t *tokens, int num_frames,
      std::function<void(const std::string &)> log_callback = nullptr,
      std::vector<float> *out_audio_L = nullptr,
      std::vector<float> *out_audio_R = nullptr,
      bool mask_musiccoca_during_prefill = true);
  /// Convenience wrapper for "prefill with silence". The first call
  /// encodes a buffer of silence through the SpectroStream encoder once
  /// to learn the steady-state silent RVQ token (the per-codebook codes
  /// SpectroStream produces for a fully-warmed-up silent input); the
  /// result is cached. Subsequent calls broadcast that cached token to
  /// `duration_frames` frames and feed them through
  /// `prefill_state_from_tokens` — no encoder pass at runtime.
  ///
  /// `duration_frames` defaults to 550 (22 s @ 25 Hz), enough to push
  /// any prior content out of every layer's local-attention window
  /// and fully saturate the KV cache with silence. The models are trained
  /// with 20 sec chunks of audio (i.e., 500 frames receptive field).
  ///
  /// `reset_first=true` (default) calls `reset_state` before prefill so
  /// silent prefill yields a fully fresh "model has only ever heard
  /// silence" state, regardless of what was generated before.
  ///
  /// The current MusicCoCa tokens are temporarily replaced with the masked
  /// sentinel during the prefill so the conditioning signal doesn't
  /// steer the silent KV cache; they are restored on return.
  ///
  /// Like the other prefill paths, this checkpoints on success: the
  /// post-silent-prefill state becomes the new `reset_state()` target.
  bool prefill_silence(
      int duration_frames = 550, bool reset_first = true,
      std::function<void(const std::string &)> log_callback = nullptr,
      std::vector<float> *out_audio_L = nullptr,
      std::vector<float> *out_audio_R = nullptr);
  /// Release all model resources.
  void unload();
  /// Copy the engine's current reset target (`transformer_initial_state_`)
  /// into the live state (`transformer_state_`). The reset target is
  /// the model's factory initial state on load, but it can be moved
  /// by `prefill_state*()` (which checkpoints on success) and by
  /// `load_state()` (which loads from disk into the reset target).
  void reset_state();
  /// Persist the **live** `transformer_state_` to `path` (.safetensors).
  /// Pure dump; doesn't modify any state.
  bool save_state(const char *path);
  /// Load a state file from `path` (.safetensors) into the **reset
  /// target** (`transformer_initial_state_`). Validates that the loaded
  /// array count and shapes match the live model and refuses mismatches
  /// with a clear error to stderr.
  ///
  /// **Asymmetric with `save_state`:** this does NOT modify the live
  /// `transformer_state_`. To actually apply the loaded data, call
  /// `reset_state()` afterwards. The two-step UX is intentional — it
  /// lets the host present "pick a file" and "apply" as separate
  /// buttons.
  bool load_state(const char *path);
  /// Restore the **reset target** (`transformer_initial_state_`) to the
  /// model's factory initial state — the state arrays loaded from
  /// `<model>_state.safetensors` at `load_model` time. Also resets the
  /// live `transformer_state_` so the model immediately resumes from the
  /// factory state. Undoes the checkpointing side-effect of
  /// `prefill_state*()` and any prior `load_state()` without requiring
  /// a full model reload.
  void reset_to_factory();
  /// `true` once a `.mlxfn` model has been successfully loaded.
  bool is_loaded() const;
  /// @}

  /// @name Generation
  /// Inference-thread only. @{

  /// Produce the next 1920-sample stereo frame (48 kHz). Thread-safe with
  /// respect to UI setters (sampling params, MIDI, prompts), but must not
  /// be called concurrently with lifecycle methods above. If `tokens_out`
  /// is non-null, the just-sampled raw RVQ codes (`kNumRVQLevels` int32
  /// values, each in `0..kVocabSize-1`) are written there — feed them
  /// back later via `prefill_state_from_tokens` for a lossless prefill.
  bool generate_frame(float *audio_L, float *audio_R,
                      std::int32_t *tokens_out = nullptr);
  /// Timings from the most recent `generate_frame` call.
  const FrameMetrics &last_metrics() const;
  /// @}

  /// @name Prompts — text
  /// UI thread. Kicks an async MusicCoCa encode on an internal thread.
  /// Poll `get_text_encoder_status()` / `get_quantizer_status()` for progress.
  /// @{

  void set_text_prompt(const std::string &text);
  void set_text_prompts(const std::vector<std::string> &texts,
                        const std::vector<float> &weights);
  /// Replace the current MusicCoCa tokens with the "fully masked" sentinel
  /// (`-1`s, which `generate_frame` offsets to the model's mask id).
  /// Skips the TFLite text encoder entirely — the caller does not need
  /// to poll `get_text_encoder_status()` afterwards. Useful when you want
  /// to drive the model from prefill audio alone, with no text prompt.
  void set_musiccoca_tokens_masked();
  /// Encoder status code. 0 = idle, 1 = fetching, 2 = success, 3 = error.
  int get_text_encoder_status() const;
  /// Prompt status code for a specific index.
  int get_prompt_status(int index) const;
  /// Quantizer status code (same values as text-encoder status).
  int get_quantizer_status() const;
  /// Get the deduced RVQ depth of the loaded model.
  int get_rvq_depth() const;
  /// UI-consumable log lines accumulated by the async MusicCoCa worker.
  void add_log(const std::string &msg);
  std::vector<std::string> get_logs();
  /// Re-blend cached per-prompt embeddings with new weights and requantize.
  /// Skips if text encoding is in progress or embeddings aren't cached.
  /// `pca_coeffs` (optional): signed coefficients used to recompute any slot
  /// whose text is "pca" before blending.
  bool reblend_musiccoca_tokens(const float *weights, int count,
                                const float *pca_coeffs = nullptr,
                                int pca_count = 0);
  int get_active_prompt_count() const;
  std::string get_cached_text(int index);
  /// @}

  /// @name Prompts — PCA corpus
  /// UI thread. Optional PCA-based prompt interpolation over a corpus.
  /// @{

  bool load_pca_data(const float *mean, const float *components,
                     int num_components);
  bool load_pca_file(const char *path);
  bool is_pca_loaded() const;
  int pca_component_count() const;
  int pca_centroid_count() const;
  /// @}

  /// @name Sampling parameters
  /// Atomic setters/getters — safe to call from any thread.
  /// @{

  void set_temperature(float t);
  float get_temperature() const;
  void set_top_k(int k);
  int get_top_k() const;

  void set_cfg_musiccoca(float v);
  float get_cfg_musiccoca() const;
  void set_cfg_notes(float v);
  float get_cfg_notes() const;
  void set_cfg_drums(float v);
  float get_cfg_drums() const;
  void set_unmask_width(int w);
  int get_unmask_width() const;
  void set_seed_rotation(int r);
  int get_seed_rotation() const;
  /// @}

  /// @name MIDI notes
  /// Atomic — safe to call from the audio or MIDI thread.
  /// @{

  /// Mark MIDI note `n` as pressed. 0 ≤ n < 132.
  void set_note_on(int n);
  /// Mark MIDI note `n` as released.
  void set_note_off(int n);

  /// Onset mode for pianoroll tokens. It affects note-on token values.
  ///   0 = Mask onsets  (default) — off=0, on=3
  ///   1 = Unmask  onsets — off=0, onset=2, continuation=1
  void set_onset_mode(int mode);
  int get_onset_mode() const;
  /// @}

  /// @name Drumless
  /// Atomic — safe from any thread. @{

  void set_drumless(bool on);
  bool get_drumless() const;
  /// @}

  /// @name Prompts — audio
  /// UI thread. Inject raw audio as a prompt slot. @{

  /// Directly set the MusicCoCa embedding for prompt slot `index`
  /// (float[kMusicCoCaEmbeddingDim]).
  void set_audio_embedding(int index, const float *embedding);
  /// Set prompt slot `index` from audio PCM samples. `samples` is queued
  /// and encoded on the MusicCoCa worker thread.
  void set_audio_prompt_samples(int index, const std::string &filename,
                                const float *samples, std::size_t count);
  /// Copy the embedding currently stored in slot `index` into `out`
  /// (float[kMusicCoCaEmbeddingDim]).
  bool get_audio_embedding(int index, float *out) const;
  /// @}

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace core
} // namespace magentart
