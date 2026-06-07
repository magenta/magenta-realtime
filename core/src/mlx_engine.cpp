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

// MLX inference engine — two-stage pipeline:
//   1. MusicCoCa text/audio → embedding → quantized tokens (TFLite, CPU)
//   2. Transformer step: tokens → stereo audio [1, 2, 1920]  (MLX/Metal GPU)
//
// `MLXEngine` is a thin pimpl facade: its public methods forward to
// `MLXEngine::Impl`, which owns all the MLX / TFLite / SentencePiece state.
// Hiding those dependencies behind the pimpl lets the public header stay
// free of third-party includes.

#include "midi_note_tracker.h"
#include <magentart/mlx_engine.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>


#include "sentencepiece_processor.h"
#include "tensorflow/lite/c/c_api.h"

#include "numpy_random_state.h"

namespace magentart {
namespace core {

namespace mx = mlx::core;

namespace {

using detail::NumpyRandomState;

} // namespace

// NoteState tracks the lifecycle of a MIDI note to handle the "latch" behavior
// required for onsets. This ensures that quick note-on/note-off sequences are
// not missed by the inference thread, and avoids race conditions without
// requiring heavy locks in the audio critical path.

struct MLXEngine::Impl {
  ~Impl();

  // --- Lifecycle ---
  bool init_assets(const char *resource_dir, const char *model_subfolder);
  bool load_model(const char *mlxfn_path);
  bool load_prefill_model(const char *spectrostream_mlxfn_path,
                          const char *prefill_mlxfn_path);
  bool prefill_state(const float *audio_samples, int num_samples,
                     int trim_front_frames, int trim_back_frames,
                     std::function<void(const std::string &)> log_callback,
                     std::vector<float> *out_audio_L,
                     std::vector<float> *out_audio_R,
                     bool mask_musiccoca_during_prefill);
  bool prefill_state_from_tokens(
      const int32_t *tokens, int num_frames,
      std::function<void(const std::string &)> log_callback,
      std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
      bool mask_musiccoca_during_prefill);
  bool prefill_silence(int duration_frames, bool reset_first,
                       std::function<void(const std::string &)> log_callback,
                       std::vector<float> *out_audio_L,
                       std::vector<float> *out_audio_R);
  // Lazily compute and cache the steady-state SpectroStream RVQ tokens
  // for a silent input. Returns true on success and fills out_tokens
  // (size kNumRVQLevels) with the cached value.
  bool
  ensure_silent_tokens(std::vector<int32_t> &out_tokens,
                       std::function<void(const std::string &)> log_callback);
  // Shared helper: takes a [1, N, kNumRVQLevels] int32 tokens array (raw
  // codebook space) and runs the seeding loop.
  bool prefill_with_token_array(
      mx::array tokens, std::function<void(const std::string &)> log_callback,
      std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
      bool mask_musiccoca_during_prefill);
  void unload();
  void reset_state();
  bool save_state(const char *path);
  bool load_state(const char *path);
  void reset_to_factory();

  // --- Generation ---
  bool generate_frame(float *audio_L, float *audio_R,
                      int32_t *tokens_out = nullptr);

  // --- Prompts ---
  void set_text_prompt(const std::string &text);
  void set_text_prompts(const std::vector<std::string> &texts,
                        const std::vector<float> &weights);
  void set_musiccoca_tokens_masked();
  void add_log(const std::string &msg);
  std::vector<std::string> get_logs();
  bool reblend_musiccoca_tokens(const float *weights, int count,
                                const float *pca_coeffs, int pca_count);
  std::string get_cached_text(int index);

  // --- Prompts: PCA corpus ---
  bool load_pca_data(const float *mean, const float *components,
                     int num_components);
  bool load_pca_file(const char *path);

  // --- Audio prompts ---
  void set_audio_embedding(int index, const float *embedding);
  void set_audio_prompt_samples(int index, const std::string &filename,
                                const float *samples, std::size_t count);
  bool get_audio_embedding(int index, float *out) const;

  // --- Private helpers ---
  bool encode_single_prompt(const std::string &text, float *out_embedding);
  int encode_audio_prompt(const std::vector<float> &samples,
                          float *out_embedding);
  // Refines a text embedding in-place via the MusicCoCa mapper (maps text
  // embeddings toward audio space), then L2-normalizes. No-op returning false
  // if the mapper isn't loaded. Mirrors musiccoca.py's mapper path.
  bool apply_mapper(float *embedding);
  bool quantize_embedding(const float *embedding, std::vector<int> &out_tokens);
  void fetch_musiccoca_tokens(const std::vector<std::string> &texts,
                              const std::vector<float> &weights);
  void start_inference_thread_if_needed();

  int calculate_token(NoteState state, int onset_mode);
  void populate_condition_tokens(int32_t *cond_ptr);

  // --- Transformer (MLX) ---
  std::optional<mx::ImportedFunction> transformer_fn_;
  std::optional<mx::ImportedFunction> spectrostream_encoder_fn_;
  std::optional<mx::ImportedFunction> prefill_fn_;
  std::string model_path_;
  std::function<std::vector<mx::array>(const std::vector<mx::array> &)>
      compiled_fn_;
  std::vector<mx::array> transformer_state_;
  std::vector<mx::array> transformer_initial_state_;
  // Pristine copy of the state arrays loaded from `<model>_state.safetensors`
  // at `load_model` time. Never modified after model load (until the next
  // `load_model` call). `reset_to_factory()` restores
  // `transformer_initial_state_` from this, undoing any checkpointing done by
  // `prefill_state*()` or `load_state()`.
  std::vector<mx::array> transformer_factory_state_;

  std::vector<mx::array> persistent_args_;
  bool persistent_args_initialized_ = false;
  // Index of the decoder-internal previous_frame slot in transformer_state_.
  // Cached at load_model time. Used by generate_frame to read just-sampled
  // tokens (when caller passes tokens_out) and by the prefill helper to
  // seed the next frame's input. -1 if not yet found.
  int previous_frame_state_idx_ = -1;

  // Cached steady-state RVQ tokens for silent input (per-codebook raw
  // codes, length kNumRVQLevels). Computed lazily on first prefill_silence
  // call by encoding a silent buffer through the SpectroStream encoder and
  // taking a frame from the middle of the trimmed range. Cached so
  // subsequent silent prefills don't re-run the encoder.
  std::vector<int32_t> silent_tokens_cache_;
  bool silent_tokens_cached_ = false;

  // --- Sampling (atomic; safe from any thread) ---
  std::atomic<float> temperature_{1.0f};
  std::atomic<int> seed_rotation_{0};
  std::atomic<int> top_k_{100};
  std::atomic<float> cfg_musiccoca_{3.0f};
  std::atomic<float> cfg_notes_{5.0f};
  std::atomic<float> cfg_drums_{1.0f};
  std::atomic<int> unmask_width_{0};
  std::unique_ptr<MidiNoteTracker> note_tracker_ =
      std::make_unique<MidiNoteTracker>();
  std::atomic<bool> drumless_{false};
  std::atomic<int> onset_mode_{0}; // 0=MaskOnsets, 1=UnmaskOnsets

  // --- Model configuration ---
  int seed_tensor_idx_ = -1;

  // --- MusicCoCa async token state ---
  mutable std::mutex musiccoca_mutex_;
  std::atomic<bool> is_musiccoca_fetching_{false};
  std::atomic<int> text_encoder_status_{
      0}; // 0=idle 1=fetching 2=success 3=error
  std::atomic<int> prompt_statuses_[kMaxPrompts]{};
  std::atomic<int> quantizer_status_{0};
  std::vector<int> musiccoca_tokens_{std::begin(kDefaultMusicCoCaTokensPiano),
                                     std::end(kDefaultMusicCoCaTokensPiano)};

  bool has_pending_musiccoca_ = false;
  std::vector<std::string> pending_texts_;
  std::vector<float> pending_weights_;

  std::mutex log_mutex_;
  std::vector<std::string> log_lines_;

  // --- Cached per-prompt embeddings (for efficient re-blending) ---
  float cached_embeddings_[kMaxPrompts][kMusicCoCaEmbeddingDim] = {};
  std::string cached_texts_[kMaxPrompts];
  bool embedding_valid_[kMaxPrompts] = {};
  bool slot_is_audio_[kMaxPrompts] = {};
  std::vector<float> pending_audio_samples_[kMaxPrompts];
  int active_prompt_count_ = 0;

  // --- TFLite interpreters ---
  TfLiteModel *text_encoder_model_ = nullptr;
  TfLiteInterpreterOptions *text_encoder_options_ = nullptr;
  TfLiteInterpreter *text_encoder_interpreter_ = nullptr;

  TfLiteModel *quantizer_model_ = nullptr;
  TfLiteInterpreterOptions *quantizer_options_ = nullptr;
  TfLiteInterpreter *quantizer_interpreter_ = nullptr;

  TfLiteModel *mapper_model_ = nullptr;
  TfLiteInterpreterOptions *mapper_options_ = nullptr;
  TfLiteInterpreter *mapper_interpreter_ = nullptr;

  TfLiteModel *audio_preprocessor_model_ = nullptr;
  TfLiteInterpreterOptions *audio_preprocessor_options_ = nullptr;
  TfLiteInterpreter *audio_preprocessor_interpreter_ = nullptr;

  TfLiteModel *music_encoder_model_ = nullptr;
  TfLiteInterpreterOptions *music_encoder_options_ = nullptr;
  TfLiteInterpreter *music_encoder_interpreter_ = nullptr;

  sentencepiece::SentencePieceProcessor *tokenizer_ = nullptr;

  std::unordered_map<std::string, std::vector<float>> text_embedding_cache_;

  // --- PCA corpus ---
  float pca_components_[kMaxPCAComponents][kMusicCoCaEmbeddingDim] = {};
  std::unordered_map<std::string, std::vector<float>> custom_centroids_;
  int pca_component_count_ = 0;
  bool pca_loaded_ = false;
  std::atomic<float> current_pca_coeffs_[kMaxPCAComponents]{};

  FrameMetrics last_metrics_;
};

MLXEngine::Impl::~Impl() { unload(); }

bool MLXEngine::Impl::init_assets(const char *resource_dir,
                                  const char *model_subfolder) {
  try {
    std::string dir(resource_dir);
    std::string sub(model_subfolder);

    // Delete existing tokenizer and TFLite models before reloading
    if (tokenizer_)
      delete tokenizer_;
    tokenizer_ = nullptr;

    if (text_encoder_interpreter_)
      TfLiteInterpreterDelete(text_encoder_interpreter_);
    if (text_encoder_options_)
      TfLiteInterpreterOptionsDelete(text_encoder_options_);
    if (text_encoder_model_)
      TfLiteModelDelete(text_encoder_model_);
    text_encoder_interpreter_ = nullptr;
    text_encoder_options_ = nullptr;
    text_encoder_model_ = nullptr;

    if (quantizer_interpreter_)
      TfLiteInterpreterDelete(quantizer_interpreter_);
    if (quantizer_options_)
      TfLiteInterpreterOptionsDelete(quantizer_options_);
    if (quantizer_model_)
      TfLiteModelDelete(quantizer_model_);
    quantizer_interpreter_ = nullptr;
    quantizer_options_ = nullptr;
    quantizer_model_ = nullptr;

    if (mapper_interpreter_)
      TfLiteInterpreterDelete(mapper_interpreter_);
    if (mapper_options_)
      TfLiteInterpreterOptionsDelete(mapper_options_);
    if (mapper_model_)
      TfLiteModelDelete(mapper_model_);
    mapper_interpreter_ = nullptr;
    mapper_options_ = nullptr;
    mapper_model_ = nullptr;

    add_log("[MLXEngine] init_assets called. dir=" + dir + ", sub=" + sub);
    // Load tokenizer
    tokenizer_ = new sentencepiece::SentencePieceProcessor();
    std::string spm_path = dir + "/" + sub + "/spm.model";
    add_log("[MLXEngine] Loading tokenizer from: " + spm_path);
    auto sp_status = tokenizer_->Load(spm_path);
    if (!sp_status.ok()) {
      std::string err = "[MLXEngine] Failed to load SPM tokenizer from " +
                        spm_path + ": " + sp_status.ToString();
      std::cerr << err << std::endl;
      add_log(err);
      return false;
    }

    // Load TFLite models
    std::string text_path = dir + "/" + sub + "/text_encoder.tflite";
    add_log("[MLXEngine] Loading text encoder from: " + text_path);
    text_encoder_model_ = TfLiteModelCreateFromFile(text_path.c_str());
    if (!text_encoder_model_) {
      std::string err =
          "[MLXEngine] Failed to load text encoder model from " + text_path;
      std::cerr << err << std::endl;
      add_log(err);
      return false;
    }
    text_encoder_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(text_encoder_options_, 1);
    text_encoder_interpreter_ =
        TfLiteInterpreterCreate(text_encoder_model_, text_encoder_options_);
    TfLiteInterpreterAllocateTensors(text_encoder_interpreter_);

    std::string q_path =
        dir + "/" + sub + "/pretrained_vector_quantizer.tflite";
    add_log("[MLXEngine] Loading quantizer from: " + q_path);
    quantizer_model_ = TfLiteModelCreateFromFile(q_path.c_str());
    if (!quantizer_model_) {
      std::string err =
          "[MLXEngine] Failed to load quantizer model from " + q_path;
      std::cerr << err << std::endl;
      add_log(err);
      return false;
    }
    quantizer_options_ = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(quantizer_options_, 1);
    quantizer_interpreter_ =
        TfLiteInterpreterCreate(quantizer_model_, quantizer_options_);
    TfLiteInterpreterAllocateTensors(quantizer_interpreter_);

    // Mapper (optional): refines text embeddings toward audio space. Older
    // model bundles don't ship mapper.tflite, so a missing file is not fatal —
    // the engine simply falls back to unmapped text embeddings.
    std::string mapper_path = dir + "/" + sub + "/mapper.tflite";
    add_log("[MLXEngine] Loading mapper from: " + mapper_path);
    mapper_model_ = TfLiteModelCreateFromFile(mapper_path.c_str());
    if (mapper_model_) {
      mapper_options_ = TfLiteInterpreterOptionsCreate();
      TfLiteInterpreterOptionsSetNumThreads(mapper_options_, 1);
      mapper_interpreter_ =
          TfLiteInterpreterCreate(mapper_model_, mapper_options_);
      TfLiteInterpreterAllocateTensors(mapper_interpreter_);
      add_log("[MLXEngine] Loaded mapper.tflite");
    } else {
      add_log("[MLXEngine] mapper.tflite not found at " + mapper_path +
              " — text embedding refinement disabled.");
    }

    std::string preprocessor_path =
        dir + "/" + sub + "/audio_preprocessor.tflite";
    audio_preprocessor_model_ =
        TfLiteModelCreateFromFile(preprocessor_path.c_str());
    if (audio_preprocessor_model_) {
      audio_preprocessor_options_ = TfLiteInterpreterOptionsCreate();
      TfLiteInterpreterOptionsSetNumThreads(audio_preprocessor_options_, 1);
      audio_preprocessor_interpreter_ = TfLiteInterpreterCreate(
          audio_preprocessor_model_, audio_preprocessor_options_);
      TfLiteInterpreterAllocateTensors(audio_preprocessor_interpreter_);

      add_log("[MLXEngine] Loaded audio_preprocessor.tflite");
      int num_inputs =
          TfLiteInterpreterGetInputTensorCount(audio_preprocessor_interpreter_);
      for (int i = 0; i < num_inputs; ++i) {
        const TfLiteTensor *t =
            TfLiteInterpreterGetInputTensor(audio_preprocessor_interpreter_, i);
        std::stringstream ss;
        ss << "  Input " << i << ": dims=" << TfLiteTensorNumDims(t) << " [";
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
          ss << TfLiteTensorDim(t, d) << " ";
        ss << "], type=" << TfLiteTensorType(t);
        add_log(ss.str());
      }
      int num_outputs = TfLiteInterpreterGetOutputTensorCount(
          audio_preprocessor_interpreter_);
      for (int i = 0; i < num_outputs; ++i) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(
            audio_preprocessor_interpreter_, i);
        std::stringstream ss;
        ss << "  Output " << i << ": dims=" << TfLiteTensorNumDims(t) << " [";
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
          ss << TfLiteTensorDim(t, d) << " ";
        ss << "], type=" << TfLiteTensorType(t);
        add_log(ss.str());
      }
    } else {
      std::cerr << "Failed to load audio_preprocessor.tflite from "
                << preprocessor_path << std::endl;
    }

    std::string encoder_path = dir + "/" + sub + "/music_encoder.tflite";
    music_encoder_model_ = TfLiteModelCreateFromFile(encoder_path.c_str());
    if (music_encoder_model_) {
      music_encoder_options_ = TfLiteInterpreterOptionsCreate();
      TfLiteInterpreterOptionsSetNumThreads(music_encoder_options_, 1);
      music_encoder_interpreter_ =
          TfLiteInterpreterCreate(music_encoder_model_, music_encoder_options_);
      TfLiteInterpreterAllocateTensors(music_encoder_interpreter_);

      add_log("[MLXEngine] Loaded music_encoder.tflite");
      int num_inputs =
          TfLiteInterpreterGetInputTensorCount(music_encoder_interpreter_);
      for (int i = 0; i < num_inputs; ++i) {
        const TfLiteTensor *t =
            TfLiteInterpreterGetInputTensor(music_encoder_interpreter_, i);
        std::stringstream ss;
        ss << "  Input " << i << ": dims=" << TfLiteTensorNumDims(t) << " [";
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
          ss << TfLiteTensorDim(t, d) << " ";
        ss << "], type=" << TfLiteTensorType(t);
        add_log(ss.str());
      }
      int num_outputs =
          TfLiteInterpreterGetOutputTensorCount(music_encoder_interpreter_);
      for (int i = 0; i < num_outputs; ++i) {
        const TfLiteTensor *t =
            TfLiteInterpreterGetOutputTensor(music_encoder_interpreter_, i);
        std::stringstream ss;
        ss << "  Output " << i << ": dims=" << TfLiteTensorNumDims(t) << " [";
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d)
          ss << TfLiteTensorDim(t, d) << " ";
        ss << "], type=" << TfLiteTensorType(t);
        add_log(ss.str());
      }
    } else {
      std::cerr << "Failed to load music_encoder.tflite from " << encoder_path
                << std::endl;
    }

    musiccoca_tokens_.assign(std::begin(kDefaultMusicCoCaTokensPiano),
                             std::end(kDefaultMusicCoCaTokensPiano));

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MLXEngine] Failed to init assets: " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "[MLXEngine] Failed to init assets: unknown exception"
              << std::endl;
    return false;
  }
}

bool MLXEngine::Impl::load_model(const char *mlxfn_path) {
  try {
    std::string path(mlxfn_path);
    model_path_ = path;

    transformer_fn_.reset();
    compiled_fn_ = nullptr;
    transformer_state_.clear();
    transformer_initial_state_.clear();
    transformer_factory_state_.clear();

    // Ensure we are working with an absolute or reachable path.
    std::cout << "[MLXEngine] Loading mlxfn from: " << path << std::endl;

    transformer_fn_ = mx::import_function(path);

    if (!transformer_fn_) {
      std::cerr << "[MLXEngine] mx::import_function returned null" << std::endl;
      return false;
    }

    compiled_fn_ = mx::compile([this](const std::vector<mx::array> &inputs) {
      return (*transformer_fn_)(inputs);
    });

    std::string state_path = path;
    const std::string ext = ".mlxfn";
    if (state_path.length() >= ext.length() &&
        state_path.compare(state_path.length() - ext.length(), ext.length(),
                           ext) == 0) {
      state_path.replace(state_path.length() - ext.length(), ext.length(),
                         "_state.safetensors");
    } else {
      state_path += "_state.safetensors";
    }

    std::cout << "[MLXEngine] Loading state from: " << state_path << std::endl;

    auto [state_arrays, state_meta] = mx::load_safetensors(state_path);
    for (int i = 0;; ++i) {
      auto it = state_arrays.find("state_" + std::to_string(i));
      if (it == state_arrays.end())
        break;
      transformer_initial_state_.push_back(it->second);
    }

    if (transformer_initial_state_.empty()) {
      std::cerr << "[MLXEngine] Failed to load any state arrays" << std::endl;
      return false;
    }

    mx::eval(transformer_initial_state_);
    // Snapshot the freshly-loaded arrays for `reset_to_factory()`. mx::array
    // is reference-counted, so this is a cheap shallow copy; the underlying
    // GPU buffers are shared. `transformer_initial_state_` may later be
    // overwritten (prefill, load_state); `transformer_factory_state_` is not.
    transformer_factory_state_ = transformer_initial_state_;



    // Locate the seed/RNG key tensor. It is the only uint32 array in the model
    // state whose last dimension is 2.
    seed_tensor_idx_ = -1;
    int match_count = 0;
    for (size_t i = 0; i < transformer_initial_state_.size(); ++i) {
      const auto &arr = transformer_initial_state_[i];
      if (arr.dtype() == mx::uint32 && !arr.shape().empty() &&
          arr.shape().back() == 2) {
        seed_tensor_idx_ = static_cast<int>(i);
        match_count++;
      }
    }
    if (match_count != 1) {
      std::string err = "[MLXEngine] WARNING: Expected exactly one uint32 "
                        "array of shape [..., 2] in model state (RNG key), "
                        "but found " +
                        std::to_string(match_count) +
                        ". Seed rotation will be disabled.";
      std::cerr << err << std::endl;
      add_log(err);
      seed_tensor_idx_ = -1;
    } else {
      std::string msg =
          "[MLXEngine] Seed tensor index identified dynamically: " +
          std::to_string(seed_tensor_idx_);
      std::cout << msg << std::endl;
      add_log(msg);
    }

    // Locate the decoder-internal previous_frame slot once. Multiple
    // state arrays may match the previous_frame shape (the wrapper's
    // outer sampler_previous_output and the MultivariateDecoder's
    // inner previous_frame); per `magenta_rt/jax/depthformer.py:591`
    // the wrapper's slot is unused, so we want the *last* match.
    previous_frame_state_idx_ = -1;
    for (size_t i = 0; i < transformer_initial_state_.size(); ++i) {
      const auto &shape = transformer_initial_state_[i].shape();
      const bool match_rank3 = shape.size() == 3 && shape[0] == 1 &&
                               shape[1] == 1 && shape[2] == kNumRVQLevels;
      const bool match_rank4 = shape.size() == 4 && shape[0] == 1 &&
                               shape[2] == 1 && shape[3] == kNumRVQLevels;
      if (match_rank3 || match_rank4) {
        previous_frame_state_idx_ = static_cast<int>(i);
      }
    }
    if (previous_frame_state_idx_ < 0) {
      std::cerr << "[MLXEngine] WARNING: could not locate a previous_frame "
                   "state slot; prefill and tokens_out will be unavailable."
                << std::endl;
    }

    reset_state();

    // Pre-allocate persistent_args_ for zero-copy evaluate
    persistent_args_.clear();

    const int num_pitches = 128;
    const int cond_len = kMusicCoCaRVQLevels + num_pitches + 1;

    std::vector<int32_t> int_zeros(cond_len, 0);
    std::vector<float> float_zeros(1, 0.0f);

    persistent_args_.push_back(
        mx::array(int_zeros.data(), {1, 1, cond_len}, mx::int32));
    persistent_args_.push_back(mx::array(float_zeros.data(), {1}, mx::float32)); // temperature
    persistent_args_.push_back(
        mx::array(int_zeros.data(), {1}, mx::int32)); // top_k

    persistent_args_.push_back(
        mx::array(float_zeros.data(), {1}, mx::float32)); // cfg_musiccoca
    persistent_args_.push_back(
        mx::array(float_zeros.data(), {1}, mx::float32)); // cfg_notes
    persistent_args_.push_back(
        mx::array(float_zeros.data(), {1}, mx::float32)); // cfg_drums

    persistent_args_.push_back(mx::array(int_zeros.data(), {1, 1, cond_len},
                                         mx::int32)); // neg_musiccoca
    persistent_args_.push_back(
        mx::array(int_zeros.data(), {1, 1, cond_len}, mx::int32)); // neg_notes

    // Insert empty forced_tokens at index 8.
    // Shape should be [1, 0, kNumRVQLevels] to match the empty trace in
    // Python.
    std::vector<int32_t> empty_tokens(0);
    persistent_args_.push_back(
        mx::array(empty_tokens.data(), {1, 0, static_cast<int>(kNumRVQLevels)}, mx::int32));

    persistent_args_.insert(persistent_args_.end(),
                            transformer_initial_state_.begin(),
                            transformer_initial_state_.end());

    mx::eval(persistent_args_); // Materialize all persistent inputs
    persistent_args_initialized_ = true;

    std::cout << "[MLXEngine] Successfully loaded model and pre-allocated "
                 "persistent args."
              << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MLXEngine] Failed to load model: " << e.what() << std::endl;
    transformer_fn_.reset();
    compiled_fn_ = nullptr;
    return false;
  } catch (...) {
    std::cerr << "[MLXEngine] Failed to load model: unknown exception"
              << std::endl;
    transformer_fn_.reset();
    compiled_fn_ = nullptr;
    return false;
  }
}

bool MLXEngine::Impl::load_prefill_model(const char *spectrostream_mlxfn_path,
                                         const char *prefill_mlxfn_path) {
  try {
    std::string load_msg = "[MLXEngine] Loading spectrostream encoder from: " + std::string(spectrostream_mlxfn_path);
    std::cout << load_msg << std::endl;
    add_log(load_msg);
    spectrostream_encoder_fn_ = mx::import_function(spectrostream_mlxfn_path);
    if (!spectrostream_encoder_fn_) {
      std::string err_msg = "[MLXEngine] Failed to import spectrostream encoder";
      std::cerr << err_msg << std::endl;
      add_log(err_msg);
      return false;
    }
    add_log("[MLXEngine] Successfully loaded spectrostream encoder.");
    return true;
  } catch (const std::exception &e) {
    std::string err_msg = "[MLXEngine] Failed to load prefill models: " + std::string(e.what());
    std::cerr << err_msg << std::endl;
    add_log(err_msg);
    return false;
  }
}

// Seed the transformer's KV caches from a SpectroStream-encoded clip.
//
// Pipeline:
//   1. Pad/truncate caller audio to the encoder's traced fixed shape
//      (kEncoderInputSamples = 1344000 = 28 s).
//   2. Run the SpectroStream encoder → [1, 700, 16] RVQ tokens.
//   3. Trim `trim_front_frames` from the start and `trim_back_frames`
//      from the end of the token sequence; the kept range feeds the
//      transformer one frame at a time, building the KV cache.
//   4. Re-seed the decoder-internal previous_frame slot with the very
//      last fed token so generate_frame continues from the same context.
//
// Why trim?
//   The SpectroStream STFT front-end uses reverse-causal padding
//   (`magenta_rt/jax/spectrostream.py:1197`), so head tokens see a
//   not-yet-warmed-up encoder body and tail tokens include zero-padded
//   STFT lookahead. Both yield tokens that don't reflect real audio.
//
// Why 25 frames (1 s) on each side specifically?
//   Empirically, on one of our models:
//     - 0–10 frames: continuation is glitchy (artifacts not removed).
//     - 25 frames: clean, plausible continuations.
//     - 50–100 frames: continuation degrades again — useful tail context
//       gets cut off, and the transformer's most-recent ~20 s window
//       (its receptive field; see mlx_engine.h) loses fresh material.
//   So 25/25 sits at the sweet spot: just enough to clean encoder
//   transients without eating into the model's predictive context.
//   This default may need re-validation if the SpectroStream encoder is
//   re-exported with different STFT/encoder hyperparameters.
//
// Constraint: the encoder's fixed shape limits us to 28 s of input,
// which after 25/25 trim leaves 26 s of usable prefill — comfortably
// above the model's ~19.7 s effective receptive field. To prefill on
// longer clips, re-export spectrostream_encoder.mlxfn with a larger
// fixed input shape.
bool MLXEngine::Impl::prefill_state(
    const float *audio_samples, int num_samples, int trim_front_frames,
    int trim_back_frames, std::function<void(const std::string &)> log_callback,
    std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
    bool mask_musiccoca_during_prefill) {
  if (!spectrostream_encoder_fn_ || !compiled_fn_) {
    const std::string err =
        "[MLXEngine] Prefill models or transformer not loaded";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }

  auto shape_to_string = [](const auto &shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      ss << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    ss << "]";
    return ss.str();
  };

  if (trim_front_frames < 0 || trim_back_frames < 0) {
    const std::string err = "[MLXEngine] trim_front_frames and "
                            "trim_back_frames must be non-negative";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }

  try {
    // The exported SpectroStream encoder is traced with a fixed input
    // shape of `(1, kEncoderInputSamples, 2)` (28 s at 48 kHz),
    // so we must always pass exactly that many samples. Audio shorter
    // than that is zero-padded; longer audio is truncated. Tokens
    // corresponding to padded samples are excluded by the trim below.
    constexpr int kEncoderInputSamples = 1344000;
    const int caller_audio_frames =
        num_samples / static_cast<int>(kFrameSamples);
    const int caller_audio_samples =
        caller_audio_frames * static_cast<int>(kFrameSamples);
    const int usable_input_samples =
        std::min(caller_audio_samples, kEncoderInputSamples);
    const int usable_audio_frames =
        usable_input_samples / static_cast<int>(kFrameSamples);
    if (usable_audio_frames <= trim_front_frames + trim_back_frames) {
      const std::string err =
          "[MLXEngine] Prefill audio (" + std::to_string(usable_audio_frames) +
          " usable frames after the encoder's " +
          std::to_string(kEncoderInputSamples /
                         static_cast<int>(kFrameSamples)) +
          "-frame cap) is too short for the requested trim (" +
          std::to_string(trim_front_frames) + " front + " +
          std::to_string(trim_back_frames) + " back)";
      std::cerr << err << std::endl;
      if (log_callback)
        log_callback(err);
      return false;
    }
    if (caller_audio_samples > kEncoderInputSamples && log_callback) {
      log_callback("[MLXEngine] WARNING: input audio is " +
                   std::to_string(caller_audio_samples / 48000.0) +
                   " s; truncating to the encoder's " +
                   std::to_string(kEncoderInputSamples / 48000.0) +
                   " s capacity.");
    }

    // Pack caller audio into the fixed-size buffer the encoder was
    // traced with. Anything past usable_input_samples is zeros.
    std::vector<float> encoder_input(kEncoderInputSamples * 2, 0.0f);
    std::memcpy(encoder_input.data(), audio_samples,
                static_cast<size_t>(usable_input_samples) * 2 * sizeof(float));
    mx::array waveform = mx::array(encoder_input.data(),
                                   {1, kEncoderInputSamples, 2}, mx::float32);

    const std::string ss_start_msg =
        "[MLXEngine] Invoking SpectroStream encoder (" +
        std::to_string(usable_audio_frames) + " usable audio frames)...";
    std::cout << ss_start_msg << std::endl;
    if (log_callback)
      log_callback(ss_start_msg);

    auto tokens = (*spectrostream_encoder_fn_)({waveform})[0];
    // RVQ Truncation Stage:
    // The SpectroStream encoder typically produces tokens with a full depth
    // (e.g., 16 codebooks). However, the transformer model might expect fewer
    // codebooks (e.g., 12). Here we truncate the RVQ depth to
    // match what the transformer expects (`kNumRVQLevels`). [1, T,
    // num_quantizers] -> [1, T, kNumRVQLevels]
    tokens =
        mx::slice(tokens, {0, 0, 0},
                  {1, static_cast<int>(tokens.shape(1)), static_cast<int>(kNumRVQLevels)});

    // Trim away the unreliable head and tail of the encoded sequence.
    // SpectroStream's STFT front-end uses reverse-causal padding (see
    // magenta_rt/jax/spectrostream.py: time_padding='reverse_causal'), so
    // tokens near the very start (encoder not yet warmed up) and the
    // very end (window padded with zeros) don't reflect real audio.
    // Additionally, drop tokens past `usable_audio_frames` since they
    // correspond to the zero-padding in `encoder_input`.
    const int total_token_frames = static_cast<int>(tokens.shape(1));
    const int real_token_frames =
        std::min(total_token_frames, usable_audio_frames);
    const int trim_back_total =
        (total_token_frames - real_token_frames) + trim_back_frames;
    const int trim_back_aligned = std::min(trim_back_total, total_token_frames);
    const int trim_end = total_token_frames - trim_back_aligned;
    const int frames_to_process = trim_end - trim_front_frames;
    if (frames_to_process <= 0) {
      const std::string err =
          "[MLXEngine] After trimming, no prefill frames remain";
      std::cerr << err << std::endl;
      if (log_callback)
        log_callback(err);
      return false;
    }
    tokens = mx::slice(tokens, {0, trim_front_frames, 0},
                       {1, trim_end, static_cast<int>(kNumRVQLevels)});
    mx::eval(tokens);

    const std::string ss_end_msg =
        "[MLXEngine] SpectroStream encoder done. Trimmed tokens shape: " +
        shape_to_string(tokens.shape());
    std::cout << ss_end_msg << std::endl;
    if (log_callback)
      log_callback(ss_end_msg);

    return prefill_with_token_array(tokens, log_callback, out_audio_L,
                                    out_audio_R, mask_musiccoca_during_prefill);
  } catch (const std::exception &e) {
    const std::string err =
        "[MLXEngine] Failed to prefill state: " + std::string(e.what());
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
}

bool MLXEngine::Impl::prefill_state_from_tokens(
    const int32_t *tokens_data, int num_frames,
    std::function<void(const std::string &)> log_callback,
    std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
    bool mask_musiccoca_during_prefill) {
  if (!compiled_fn_) {
    const std::string err = "[MLXEngine] Transformer not loaded";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
  if (num_frames <= 0) {
    const std::string err = "[MLXEngine] num_frames must be positive";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
  try {
    // Tokens are in raw codebook space [0, kVocabSize); the helper
    // adds the per-codebook unique-codes offset internally. The
    // caller is responsible for providing `num_frames * get_rvq_depth()`
    // tokens; mismatched depths will manifest as a shape error in MLX.
    mx::array tokens =
        mx::array(tokens_data, {1, num_frames, static_cast<int>(kNumRVQLevels)}, mx::int32);
    return prefill_with_token_array(tokens, log_callback, out_audio_L,
                                    out_audio_R, mask_musiccoca_during_prefill);
  } catch (const std::exception &e) {
    const std::string err =
        "[MLXEngine] Failed to prefill from tokens: " + std::string(e.what());
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
}

bool MLXEngine::Impl::ensure_silent_tokens(
    std::vector<int32_t> &out_tokens,
    std::function<void(const std::string &)> log_callback) {
  if (silent_tokens_cached_) {
    out_tokens = silent_tokens_cache_;
    return true;
  }
  if (!spectrostream_encoder_fn_) {
    const std::string err =
        "[MLXEngine] Cannot compute silent tokens: "
        "SpectroStream encoder not loaded (call load_prefill_model first).";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
  try {
    // Encode a fully-silent 28 s buffer (the encoder's traced shape).
    constexpr int kEncoderInputSamples = 1344000;
    std::vector<float> silent_input(kEncoderInputSamples * 2, 0.0f);
    mx::array waveform = mx::array(silent_input.data(),
                                   {1, kEncoderInputSamples, 2}, mx::float32);
    auto tokens_arr = (*spectrostream_encoder_fn_)({waveform})[0];
    tokens_arr = mx::slice(
        tokens_arr, {0, 0, 0},
        {1, static_cast<int>(tokens_arr.shape(1)), static_cast<int>(kNumRVQLevels)});
    // Take a frame from the middle of the trimmed range so it's well
    // past the encoder's warm-up and well before any STFT zero-padding
    // pollution at the tail. The middle of [25, total - 25) is robust.
    const int total = static_cast<int>(tokens_arr.shape(1));
    const int middle = std::max(25, total / 2);
    mx::array middle_frame = mx::slice(tokens_arr, {0, middle, 0},
                                       {1, middle + 1, static_cast<int>(kNumRVQLevels)});
    auto middle_i32 = mx::astype(middle_frame, mx::int32);
    mx::eval(middle_i32);
    const int32_t *p = middle_i32.data<int32_t>();
    silent_tokens_cache_.assign(p, p + kNumRVQLevels);
    silent_tokens_cached_ = true;
    out_tokens = silent_tokens_cache_;
    if (log_callback) {
      std::stringstream ss;
      ss << "[MLXEngine] Cached silent tokens: [";
      for (size_t k = 0; k < silent_tokens_cache_.size(); ++k) {
        ss << silent_tokens_cache_[k]
           << (k + 1 < silent_tokens_cache_.size() ? ", " : "");
      }
      ss << "]";
      log_callback(ss.str());
    }
    return true;
  } catch (const std::exception &e) {
    const std::string err =
        "[MLXEngine] Failed to compute silent tokens: " + std::string(e.what());
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
}

bool MLXEngine::Impl::prefill_silence(
    int duration_frames, bool reset_first,
    std::function<void(const std::string &)> log_callback,
    std::vector<float> *out_audio_L, std::vector<float> *out_audio_R) {
  if (!compiled_fn_) {
    const std::string err = "[MLXEngine] Transformer not loaded";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }
  if (duration_frames <= 0) {
    const std::string err = "[MLXEngine] duration_frames must be positive";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }

  // Snapshot the current MusicCoCa tokens so we can restore them; mask
  // them during prefill so the conditioning doesn't steer the silent
  // KV cache.
  std::vector<int> saved_musiccoca;
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    saved_musiccoca = musiccoca_tokens_;
    musiccoca_tokens_.assign(kMusicCoCaRVQLevels, -1);
  }

  if (reset_first) {
    reset_state();
  }

  // Get the cached silent token (compute once on first call).
  std::vector<int32_t> silent_token;
  if (!ensure_silent_tokens(silent_token, log_callback)) {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    musiccoca_tokens_ = saved_musiccoca;
    return false;
  }

  // Broadcast the silent token across `duration_frames` frames.
  std::vector<int32_t> tokens(static_cast<size_t>(duration_frames) *
                              kNumRVQLevels);
  for (int f = 0; f < duration_frames; ++f) {
    std::memcpy(tokens.data() + f * kNumRVQLevels, silent_token.data(),
                kNumRVQLevels * sizeof(int32_t));
  }

  const std::string msg = "[MLXEngine] Silent prefill: feeding " +
                          std::to_string(duration_frames) +
                          " frames of cached silent tokens.";
  std::cout << msg << std::endl;
  if (log_callback)
    log_callback(msg);

  const bool ok = prefill_state_from_tokens(
      tokens.data(), duration_frames, log_callback, out_audio_L, out_audio_R,
      /*mask_musiccoca_during_prefill=*/true);

  // Restore the caller's MusicCoCa tokens.
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    musiccoca_tokens_ = saved_musiccoca;
  }
  return ok;
}

// Shared prefill core: takes a [1, N, kNumRVQLevels] int32 array of raw
// RVQ tokens (per-codebook 0..kVocabSize-1) and seeds the model's KV
// caches with their embeddings. Both
// `prefill_state` (audio path) and `prefill_state_from_tokens` (token
// path) reduce to a call here.
bool MLXEngine::Impl::prefill_with_token_array(
    mx::array tokens, std::function<void(const std::string &)> log_callback,
    std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
    bool mask_musiccoca_during_prefill) {
  if (previous_frame_state_idx_ < 0) {
    const std::string err = "[MLXEngine] previous_frame state slot unknown; "
                            "load_model probably did not find it.";
    std::cerr << err << std::endl;
    if (log_callback)
      log_callback(err);
    return false;
  }

  const int frames_to_process = static_cast<int>(tokens.shape(1));
  if (frames_to_process <= 0)
    return false;

  // Build the conditioning array we'll pass each step. We use the
  // caller's current MusicCoCa tokens (so `set_musiccoca_tokens_masked()` works
  // here too) and an all-masked notes block. The user's sampling knobs
  // (temperature, top_k, all CFG scales) and pre-built negative
  // conditioning blocks are left untouched in persistent_args_.
  const bool drumless = drumless_.load(std::memory_order_relaxed);
  const int num_pitches = 128;
  const int cond_len = static_cast<int>(kMusicCoCaRVQLevels) + num_pitches + 1;

  std::vector<int32_t> prefill_cond(cond_len, kNumReservedTokens - 1);
  if (!mask_musiccoca_during_prefill) {
    // Keep only the coarsest levels; mask the finer tail (see
    // kMusicCoCaMaskedTailLevels). prefill_cond is already filled with the
    // mask id, so we only need to write the kept levels.
    const int kept_mc_levels =
        static_cast<int>(kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels);
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    for (int i = 0; i < kept_mc_levels; ++i) {
      prefill_cond[i] = musiccoca_tokens_[i] + kNumReservedTokens;
    }
  }
  // Notes are intentionally pinned to the mask sentinel
  // (kNumReservedTokens - 1) for every prefill iteration, regardless
  // of whatever MIDI state the engine currently holds.
  //
  // Rationale: by default, the prefill loop is for populating the KV cache with
  // the *temporal* context of the supplied tokens. Mixing in a MusicCoCa
  // prompt during prefill would steer the cross-attention layers
  // toward the prompt at the same time the temporal layers are being
  // teacher-forced with the audio's own tokens — a confusing dual
  // signal that the model wasn't trained on. After prefill returns,
  // the next `generate_frame` call uses the user's actual MusicCoCa +
  // notes conditioning normally, so the prompt only steers the *new*
  // generation, not the prefilled history.
  //
  // If `mask_musiccoca_during_prefill` is false, we use the active MusicCoCa
  // tokens (e.g. extracted from audio) to steer the cross-attention layers
  // during prefill. WARNING: Listening tests showed this produced poor results
  // (style jumps, unstable continuations). Use with caution. Drum conditioning
  // slot: sits right after the 128 piano pitches.
  prefill_cond[kMusicCoCaRVQLevels + num_pitches] =
      drumless ? kNumReservedTokens + 0 : kNumReservedTokens - 1;

  mx::array cond_array =
      mx::array(prefill_cond.data(), {1, 1, cond_len}, mx::int32);

  // Refresh sampling parameters and CFG scales from the atomics into
  // persistent_args_ (generate_frame normally does this every call,
  // but prefill may run before any generate_frame and would otherwise
  // inherit the zeros allocated by load_model — sampling greedily
  // from temperature=0/top_k=0 produces glitchy output).
  persistent_args_[1].data<float>()[0] =
      temperature_.load(std::memory_order_relaxed);
  persistent_args_[2].data<int32_t>()[0] =
      top_k_.load(std::memory_order_relaxed);
  {
    persistent_args_[3].data<float>()[0] =
        cfg_musiccoca_.load(std::memory_order_relaxed);
    persistent_args_[4].data<float>()[0] =
        cfg_notes_.load(std::memory_order_relaxed);
    persistent_args_[5].data<float>()[0] =
        cfg_drums_.load(std::memory_order_relaxed);
  }

  // State start index in persistent_args_. Layout matches
  // generate_frame's persistent_args_ ordering:
  //   [cond, temperature, top_k,
  //    cfg_musiccoca, cfg_notes, cfg_drums,
  //    neg_musiccoca, neg_notes,
  //    forced_tokens,
  //    *state]
  const int kStateStart = 3    /*cond,temp,top_k*/
                          + 3  /*cfg scales*/
                          + 2  /*neg conds*/
                          + 1; /* forced_tokens */

  const std::string start_msg = "[MLXEngine] Starting prefill loop for " +
                                std::to_string(frames_to_process) +
                                " token frames";
  std::cout << start_msg << std::endl;
  if (log_callback)
    log_callback(start_msg);

  // Per-codebook token offset for the unique-codes scheme that the
  // depthformer's previous_frame slot uses:
  //   unique = c + k * kVocabSize + (kNumReservedTokens - 1)
  // The decoder's `num_reserved_tokens` is `system.NUM_RESERVED_TOKENS
  // = 6` (model.py:107: "Doesn't include dropout token"). The
  // conditioning vocab uses kNumReservedTokens = 7 because it adds the
  // dropout token (export.py:176), but the previous_frame embedding
  // table does not — so we offset by 6 here.
  mx::array token_offsets =
      mx::arange(static_cast<int>(kNumRVQLevels)) * static_cast<int>(kVocabSize) +
      (kNumReservedTokens - 1);
  token_offsets = mx::astype(token_offsets, mx::int32);
  token_offsets = mx::reshape(token_offsets, {1, 1, static_cast<int>(kNumRVQLevels)});

  const bool capture_audio = (out_audio_L && out_audio_R);

  // Helper: take a [1, 1, kNumRVQLevels] token slice and write it
  // into the previous_frame slot, broadcasting across CFG when present.
  auto seed_previous_frame = [&](const mx::array &prev_token_offset) {
    mx::array expanded4d =
        mx::expand_dims(prev_token_offset, 2); // [1,1,1,depth]
    auto &slot = transformer_state_[previous_frame_state_idx_];
    const auto &shape = slot.shape();
    const mx::Dtype original_type = slot.dtype();
    if (shape.size() == 3) {
      // [1, 1, kNumRVQLevels] — direct assignment.
      slot = mx::astype(prev_token_offset, original_type);
    } else {
      // [1, CFG, 1, kNumRVQLevels] — broadcast over CFG.
      const int cfg_dim = shape[1];
      mx::array broadcasted =
          mx::broadcast_to(expanded4d, {1, cfg_dim, 1, static_cast<int>(kNumRVQLevels)});
      slot = mx::astype(broadcasted, original_type);
    }
  };

  // Loop runs `frames_to_process - 1` iterations, feeding tokens[0..N-2]
  // through the model. Each iteration extends the temporal_body's KV
  // cache with the embedding of the seeded previous_frame. After the
  // loop we seed previous_frame = tokens[N-1] *without* calling the
  // model — the next generate_frame call will embed tokens[N-1] for
  // the first time, completing the prefill context. This matches the
  // model's natural state at "about to predict frame N", and avoids
  // the duplicate-token-in-KV-cache that comes from running N
  // iterations and then re-seeding.
  //
  // Test escape hatch: setting `MAGENTART_PREFILL_OPTION_A=1` reverts
  // to the buggy "Option A" semantics (N iterations + post-loop reseed)
  // for the duplicate-token-in-KV-cache empirical comparison. Saved
  // state from Option A vs Option B should differ in the trailing
  // entry of the temporal_body's per-layer KV caches.
  const bool use_option_a =
      std::getenv("MAGENTART_PREFILL_OPTION_A") != nullptr;
  const int prefill_loop_steps =
      use_option_a ? frames_to_process : std::max(0, frames_to_process - 1);
  for (int step = 0; step < prefill_loop_steps; ++step) {
    mx::array prev_token =
        mx::slice(tokens, {0, step, 0}, {1, step + 1, static_cast<int>(kNumRVQLevels)});
    prev_token = prev_token + token_offsets;

    seed_previous_frame(prev_token);

    persistent_args_[0] = cond_array;
    for (size_t i = 0; i < transformer_state_.size(); ++i) {
      persistent_args_[kStateStart + i] = transformer_state_[i];
    }

    // Pass the current token as forced_tokens to use the new multi-signature
    // function that skips the depth decoder during prefill.
    // In the python trace, forced_tokens is placed at index 7 (before
    // state_flat).
    std::vector<mx::array> call_args = persistent_args_;
    call_args[kStateStart - 1] = mx::astype(prev_token, mx::int32);
    auto outputs = compiled_fn_(call_args);
    mx::eval(outputs);

    if (capture_audio) {
      auto audio = outputs[0];
      if (audio.dtype() == mx::float32) {
        const float *p = audio.data<float>();
        out_audio_L->insert(out_audio_L->end(), p, p + kFrameSamples);
        out_audio_R->insert(out_audio_R->end(), p + kFrameSamples,
                            p + 2 * kFrameSamples);
      } else if (audio.dtype() == mx::int16) {
        const int16_t *p = audio.data<int16_t>();
        for (size_t i = 0; i < kFrameSamples; ++i) {
          out_audio_L->push_back(p[i] / 32768.0f);
          out_audio_R->push_back(p[kFrameSamples + i] / 32768.0f);
        }
      }
    }

    if (log_callback && (step % 10 == 0 || step == prefill_loop_steps - 1)) {
      int percent = ((step + 1) * 100) / std::max(1, prefill_loop_steps);
      log_callback("[PROGRESS] " + std::to_string(percent));
    }

    transformer_state_.assign(outputs.begin() + 1, outputs.end());
  }

  // Seed previous_frame with the *last* token (tokens[N-1]) without
  // calling the model. The next generate_frame call will embed it and
  // append it to the KV cache for the first time — matching the
  // state the model would have during natural autoregressive generation
  // right after producing tokens[0..N-1].
  {
    mx::array final_token =
        mx::slice(tokens, {0, frames_to_process - 1, 0},
                  {1, frames_to_process, static_cast<int>(kNumRVQLevels)});
    final_token = final_token + token_offsets;
    seed_previous_frame(final_token);
    for (size_t i = 0; i < transformer_state_.size(); ++i) {
      persistent_args_[kStateStart + i] = transformer_state_[i];
    }
  }
  mx::eval(persistent_args_);

  // Checkpoint the post-prefill state into transformer_initial_state_.
  // Subsequent calls to reset_state() now land here instead of the
  // model's factory initial state — letting the user prefill once and
  // then click Reset Model repeatedly to return to this same musical
  // context (typically while trying different prompts on top of it).
  // To recover the factory initial state, reload the model.
  transformer_initial_state_ = transformer_state_;
  mx::eval(transformer_initial_state_);

  const std::string done_msg =
      "[MLXEngine] Prefill complete (checkpointed). Processed " +
      std::to_string(frames_to_process) + " token frames.";
  std::cout << done_msg << std::endl;
  if (log_callback)
    log_callback(done_msg);
  return true;
}

void MLXEngine::Impl::unload() {
  if (text_encoder_interpreter_)
    TfLiteInterpreterDelete(text_encoder_interpreter_);
  if (text_encoder_options_)
    TfLiteInterpreterOptionsDelete(text_encoder_options_);
  if (text_encoder_model_)
    TfLiteModelDelete(text_encoder_model_);

  text_encoder_interpreter_ = nullptr;
  text_encoder_options_ = nullptr;
  text_encoder_model_ = nullptr;

  if (quantizer_interpreter_)
    TfLiteInterpreterDelete(quantizer_interpreter_);
  if (quantizer_options_)
    TfLiteInterpreterOptionsDelete(quantizer_options_);
  if (quantizer_model_)
    TfLiteModelDelete(quantizer_model_);

  quantizer_interpreter_ = nullptr;
  quantizer_options_ = nullptr;
  quantizer_model_ = nullptr;

  if (mapper_interpreter_)
    TfLiteInterpreterDelete(mapper_interpreter_);
  if (mapper_options_)
    TfLiteInterpreterOptionsDelete(mapper_options_);
  if (mapper_model_)
    TfLiteModelDelete(mapper_model_);

  mapper_interpreter_ = nullptr;
  mapper_options_ = nullptr;
  mapper_model_ = nullptr;

  if (tokenizer_)
    delete tokenizer_;
  tokenizer_ = nullptr;

  transformer_fn_.reset();
  compiled_fn_ = nullptr;
  transformer_state_.clear();
  transformer_initial_state_.clear();
  transformer_factory_state_.clear();
}

void MLXEngine::Impl::reset_state() {
  if (!compiled_fn_ || transformer_initial_state_.empty()) {
    return;
  }
  transformer_state_ = transformer_initial_state_;
  // Regenerate random keys to replace the ones created in the python
  // `MultivariateDecoder`'s `get_initial_state`.
  // This allows the user to generate a different musical variation from the
  // same starting prompt/state.
  int seed_rot = seed_rotation_.load(std::memory_order_relaxed);
  if (seed_rot != 0) {
    if (seed_tensor_idx_ >= 0 &&
        seed_tensor_idx_ < (int)transformer_state_.size()) {
      auto seed_tensor = transformer_state_[seed_tensor_idx_];
      int batch_size = seed_tensor.size() / 2;
      std::vector<mx::array> keys;
      keys.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        keys.push_back(mx::random::key(42 + i + seed_rot));
      }
      transformer_state_[seed_tensor_idx_] =
          mx::reshape(mx::stack(keys), seed_tensor.shape());
      std::cout << "[MLXEngine] Applied seed rotation offset: " << seed_rot
                << std::endl;
    } else {
      std::string err =
          "[MLXEngine] ERROR: Seed rotation requested mx.random.key(" +
          std::to_string(seed_rot) +
          "), but no unique RNG key tensor was identified in the model state. "
          "Rotation was not applied.";
      std::cerr << err << std::endl;
      add_log(err);
    }
  }
  is_musiccoca_fetching_ = false;

  // Check if we have active prompts still loaded
  bool has_active_prompts = false;
  for (int i = 0; i < (int)kMaxPrompts; ++i) {
    if (embedding_valid_[i] || !cached_texts_[i].empty()) {
      has_active_prompts = true;
      break;
    }
  }

  if (!has_active_prompts) {
    text_encoder_status_ = 0; // idle
    quantizer_status_ = 0;    // idle
  } else {
    // Keep them as success if we still have prompts active
    text_encoder_status_ = 2;
    quantizer_status_ = 2;
  }
}

void MLXEngine::Impl::reset_to_factory() {
  if (transformer_factory_state_.empty()) {
    std::cerr
        << "[MLXEngine] reset_to_factory called before load_model; ignoring."
        << std::endl;
    return;
  }
  transformer_initial_state_ = transformer_factory_state_;
  reset_state();
}

bool MLXEngine::Impl::save_state(const char *path) {
  std::unordered_map<std::string, mx::array> state_map;
  for (size_t i = 0; i < transformer_state_.size(); ++i) {
    state_map.insert({"state_" + std::to_string(i), transformer_state_[i]});
  }
  try {
    mx::save_safetensors(path, state_map);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MLXEngine] Failed to save state: " << e.what() << std::endl;
    return false;
  }
}

bool MLXEngine::Impl::load_state(const char *path) {
  try {
    auto [state_arrays, state_meta] = mx::load_safetensors(path);
    std::vector<mx::array> new_state;
    for (int i = 0;; ++i) {
      auto it = state_arrays.find("state_" + std::to_string(i));
      if (it == state_arrays.end())
        break;
      new_state.push_back(it->second);
    }
    if (new_state.empty()) {
      std::cerr << "[MLXEngine] Failed to load any state arrays from " << path
                << std::endl;
      return false;
    }

    // Validate that the loaded state matches the live model's shapes.
    // A mismatch usually means the file was saved from a different model
    // variant (different num_layers / num_heads / etc.) — silently
    // accepting it would corrupt persistent_args_ and break the next
    // inference call.
    if (new_state.size() != transformer_state_.size()) {
      std::cerr << "[MLXEngine] State count mismatch in " << path
                << ": file has " << new_state.size()
                << " arrays, model expects " << transformer_state_.size() << "."
                << std::endl;
      return false;
    }
    for (size_t i = 0; i < new_state.size(); ++i) {
      const auto &got = new_state[i].shape();
      const auto &want = transformer_state_[i].shape();
      if (got != want) {
        std::cerr << "[MLXEngine] State[" << i << "] shape mismatch in " << path
                  << ": file has [";
        for (size_t j = 0; j < got.size(); ++j) {
          std::cerr << got[j] << (j + 1 < got.size() ? ", " : "");
        }
        std::cerr << "], model expects [";
        for (size_t j = 0; j < want.size(); ++j) {
          std::cerr << want[j] << (j + 1 < want.size() ? ", " : "");
        }
        std::cerr << "]." << std::endl;
        return false;
      }
    }

    mx::eval(new_state);
    transformer_initial_state_ = new_state;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MLXEngine] Failed to load state: " << e.what() << std::endl;
    return false;
  }
}

// Calculates the model token for a given note `state` and `onset_mode`.
// Here's the logic table:
//
// | state                 | onset_mode==0 ("Mask") | onset_mode==1 ("Unmask") |
// |-----------------------|------------------------|--------------------------|
// | `NOTE_ONSET`          | `3` (active)           | `2` (onset)              |
// | `NOTE_ONSET_RELEASED` | `3` (active)           | `2` (onset)              |
// | `NOTE_SUSTAIN`        | `3` (active)           | `1` (continuation)       |
// | `NOTE_IDLE`           | `0` (off)              | `0` (off)                |
int MLXEngine::Impl::calculate_token(NoteState state, int onset_mode) {
  if (state == NOTE_IDLE)
    return 0;

  if (onset_mode == 1) {
    bool is_onset = (state == NOTE_ONSET || state == NOTE_ONSET_RELEASED);
    return is_onset ? 2 : 1;
  }
  return 3;
}

// Populates the condition tokens for the current frame.
// This is called exactly once per generate frame and fulfills the requirement
// to call note_tracker_->evaluateAndUpdate exactly once for every pitch.
void MLXEngine::Impl::populate_condition_tokens(int32_t *cond_ptr) {
  const int onset_mode = onset_mode_.load(std::memory_order_relaxed);
  const int u_width = unmask_width_.load(std::memory_order_relaxed);
  const bool drumless = drumless_.load(std::memory_order_relaxed);
  const int num_pitches = 128;

  if (u_width >= 127) {
    for (int i = 0; i < 128; ++i) {
      NoteState state = note_tracker_->evaluateAndUpdate(i);
      int token = calculate_token(state, onset_mode);
      cond_ptr[kMusicCoCaRVQLevels + i] = kNumReservedTokens + token;
    }
  } else {
    int active_indices[128];
    NoteState states[128];
    int num_active = 0;

    for (int i = 0; i < kNumStandardMidiNotes; ++i) {
      cond_ptr[kMusicCoCaRVQLevels + i] = kNumReservedTokens - 1;
      NoteState state = note_tracker_->evaluateAndUpdate(i);
      states[i] = state;
      if (state != NOTE_IDLE) {
        active_indices[num_active++] = i;
      }
    }

    for (int k = 0; k < num_active; ++k) {
      const int i = active_indices[k];
      const int start_idx = std::max(0, i - u_width);
      const int end_idx = std::min(kNumStandardMidiNotes, i + u_width + 1);
      for (int j = start_idx; j < end_idx; ++j) {
        cond_ptr[kMusicCoCaRVQLevels + j] = kNumReservedTokens;
      }
    }

    for (int k = 0; k < num_active; ++k) {
      int pitch = active_indices[k];
      int token = calculate_token(states[pitch], onset_mode);
      cond_ptr[kMusicCoCaRVQLevels + pitch] = kNumReservedTokens + token;
    }
  }

  // Drum conditioning slot: sits right after the 128 piano pitches.
  cond_ptr[kMusicCoCaRVQLevels + num_pitches] =
      drumless ? kNumReservedTokens + 0 : kNumReservedTokens - 1;


}

bool MLXEngine::Impl::generate_frame(float *audio_L, float *audio_R,
                                     int32_t *tokens_out) {
  if (!transformer_fn_ || !compiled_fn_)
    return false;

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  if (!persistent_args_initialized_ || persistent_args_.empty())
    return false;

  int num_pitches = 128;
  int cond_len = kMusicCoCaRVQLevels + num_pitches + 1;

  int32_t *cond_ptr = persistent_args_[0].data<int32_t>();

  // Prefix tokens
  std::vector<int> current_musiccoca;
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    current_musiccoca = musiccoca_tokens_;
  }
  // Keep only the coarsest levels; mask the finer tail with the model's
  // mask id (see kMusicCoCaMaskedTailLevels).
  const int kept_mc_levels =
      static_cast<int>(kMusicCoCaRVQLevels - kMusicCoCaMaskedTailLevels);
  for (int i = 0; i < kMusicCoCaRVQLevels; ++i) {
    cond_ptr[i] = (i < kept_mc_levels)
                      ? current_musiccoca[i] + kNumReservedTokens
                      : kNumReservedTokens - 1;
  }

  populate_condition_tokens(cond_ptr);

  // Update parameters in-place
  float *temp_ptr = persistent_args_[1].data<float>();
  temp_ptr[0] = temperature_.load(std::memory_order_relaxed);

  int32_t *top_k_ptr = persistent_args_[2].data<int32_t>();
  top_k_ptr[0] = top_k_.load(std::memory_order_relaxed);

  float *cfg_musiccoca_ptr = persistent_args_[3].data<float>();
  cfg_musiccoca_ptr[0] = cfg_musiccoca_.load(std::memory_order_relaxed);

  float *cfg_notes_ptr = persistent_args_[4].data<float>();
  cfg_notes_ptr[0] = cfg_notes_.load(std::memory_order_relaxed);

  float *cfg_drums_ptr = persistent_args_[5].data<float>();
  cfg_drums_ptr[0] = cfg_drums_.load(std::memory_order_relaxed);

  int32_t *neg_musiccoca_ptr = persistent_args_[6].data<int32_t>();
  int32_t *neg_notes_ptr = persistent_args_[7].data<int32_t>();

  std::memcpy(neg_musiccoca_ptr, cond_ptr, cond_len * sizeof(int32_t));
  for (int i = 0; i < kMusicCoCaRVQLevels; ++i) {
    neg_musiccoca_ptr[i] = kNumReservedTokens - 1;
  }

  std::memcpy(neg_notes_ptr, cond_ptr, cond_len * sizeof(int32_t));
  for (int i = 0; i < num_pitches; ++i) {
    neg_notes_ptr[kMusicCoCaRVQLevels + i] = kNumReservedTokens - 1;
  }

  // Update state in persistent_args_
  // cond(0), temp(1), top_k(2), cfg_mc(3), cfg_n(4), cfg_d(5),
  // neg_mc(6), neg_n(7), forced(8), state(9+)
  const int kStateStart = 9;
  for (size_t i = 0; i < transformer_state_.size(); ++i) {
    persistent_args_[kStateStart + i] = transformer_state_[i];
  }

  mx::eval(persistent_args_);
  auto outputs = compiled_fn_(persistent_args_);
  mx::eval(outputs);

  // Output 0 is audio: [1, 2, 1920]
  auto audio = outputs[0];

  // Convert to float from non-interleaved format
  if (audio.dtype() == mx::float32) {
    const float *audio_ptr = audio.data<float>();
    std::memcpy(audio_L, audio_ptr, kFrameSamples * sizeof(float));
    std::memcpy(audio_R, audio_ptr + kFrameSamples,
                kFrameSamples * sizeof(float));
  } else if (audio.dtype() == mx::int16) {
    const int16_t *audio_ptr = audio.data<int16_t>();
    for (size_t i = 0; i < kFrameSamples; ++i) {
      audio_L[i] = audio_ptr[i] / 32768.0f;
      audio_R[i] = audio_ptr[kFrameSamples + i] / 32768.0f;
    }
  } else {
    // Unsupported dtype
    memset(audio_L, 0, kFrameSamples * sizeof(float));
    memset(audio_R, 0, kFrameSamples * sizeof(float));
  }
  transformer_state_.assign(outputs.begin() + 1, outputs.end());

  // Optionally extract the just-sampled raw RVQ tokens. The
  // decoder-internal previous_frame slot in transformer_state_ holds
  // them in *unique-codes* space (`c + k*kVocabSize + (kNumReservedTokens-1)`);
  // convert back to per-codebook raw codes here so the caller gets a
  // tidy `[0..kVocabSize)` value per codebook.
  if (tokens_out && previous_frame_state_idx_ >= 0 &&
      previous_frame_state_idx_ < static_cast<int>(transformer_state_.size())) {
    auto slot_i32 =
        mx::astype(transformer_state_[previous_frame_state_idx_], mx::int32);
    mx::eval(slot_i32);
    const int32_t *p = slot_i32.data<int32_t>();
    const auto &shape = slot_i32.shape();
    // For both rank-3 [1, 1, 16] and rank-4 [1, CFG, 1, 16] the model
    // samples once per CFG group and broadcasts; the first 16 ints
    // hold the per-codebook tokens for the first batch element.
    //
    // Validate the broadcast assumption for rank-4 slots: every CFG
    // slice should hold the same tokens (the depthformer samples
    // once per group and replicates). If this ever fails, our
    // tokens_out value is for the first CFG branch only and the
    // caller may be reading a slice that doesn't represent the
    // model's actual sample.
    if (shape.size() == 4) {
      const int cfg = shape[1];
      for (int c = 1; c < cfg; ++c) {
        for (int k = 0; k < static_cast<int>(kNumRVQLevels); ++k) {
          if (p[c * kNumRVQLevels + k] != p[k]) {
            std::cerr << "[MLXEngine] WARNING: CFG-broadcast assumption "
                      << "violated in tokens_out — CFG slice " << c
                      << " codebook " << k << " differs from slice 0 (got "
                      << p[c * kNumRVQLevels + k] << " vs " << p[k] << ")."
                      << std::endl;
            c = cfg; // break outer loop too
            break;
          }
        }
      }
    }
    for (int k = 0; k < static_cast<int>(kNumRVQLevels); ++k) {
      tokens_out[k] = p[k] - k * static_cast<int32_t>(kVocabSize) -
                      (kNumReservedTokens - 1);
    }
  }

  auto t1 = clock::now();
  last_metrics_.transformer_ms =
      std::chrono::duration<float, std::milli>(t1 - t0).count();
  last_metrics_.total_ms = last_metrics_.transformer_ms;

  return true;
}

void MLXEngine::Impl::set_text_prompt(const std::string &text) {
  set_text_prompts({text}, {1.0f});
}

void MLXEngine::Impl::set_musiccoca_tokens_masked() {
  // -1 in any MusicCoCa slot becomes kNumReservedTokens - 1 (the model's mask
  // id) once generate_frame applies the +kNumReservedTokens offset. This
  // matches the JAX reference's `masked_musiccoca = [-1] * kMusicCoCaRVQLevels`
  // convention (12 levels).
  std::lock_guard<std::mutex> lock(musiccoca_mutex_);
  musiccoca_tokens_.assign(kMusicCoCaRVQLevels, -1);
  has_pending_musiccoca_ = false;
  pending_texts_.clear();
  pending_weights_.clear();
  text_encoder_status_.store(2, std::memory_order_relaxed);
  quantizer_status_.store(2, std::memory_order_relaxed);
}

void MLXEngine::Impl::set_text_prompts(const std::vector<std::string> &texts,
                                       const std::vector<float> &weights) {
  if (texts.empty())
    return;

  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    has_pending_musiccoca_ = true;
    pending_texts_ = texts;
    pending_weights_ = weights;
  }

  start_inference_thread_if_needed();
}

void MLXEngine::Impl::start_inference_thread_if_needed() {
  add_log("[MagentaRT] start_inference_thread_if_needed called.");
  if (is_musiccoca_fetching_.exchange(true)) {
    add_log("[MagentaRT] Thread already running, returning.");
    return;
  }

  add_log("[MagentaRT] Starting new inference thread.");
  text_encoder_status_ = 1; // fetching
  quantizer_status_ = 0;    // idle

  std::thread([this]() {
    struct Guard {
      std::atomic<bool> &flag;
      Impl *impl;
      ~Guard() {
        flag.store(false);
        if (impl->text_encoder_status_ == 1)
          impl->text_encoder_status_ = 3;
        if (impl->quantizer_status_ == 1)
          impl->quantizer_status_ = 3;
      }
    } guard{is_musiccoca_fetching_, this};

    try {
      while (true) {
        std::vector<std::string> texts_copy;
        std::vector<float> weights_copy;

        {
          std::lock_guard<std::mutex> lock(musiccoca_mutex_);
          if (!has_pending_musiccoca_) {
            break;
          }
          texts_copy = pending_texts_;
          weights_copy = pending_weights_;
          has_pending_musiccoca_ = false;
        }

        fetch_musiccoca_tokens(texts_copy, weights_copy);
      }
    } catch (const std::exception &e) {
      add_log(
          std::string("[MagentaRT] Inference thread died with exception: ") +
          e.what());
      text_encoder_status_ = 3;
      quantizer_status_ = 3;
    } catch (...) {
      add_log("[MagentaRT] Inference thread died with unknown exception");
      text_encoder_status_ = 3;
      quantizer_status_ = 3;
    }
  }).detach();
}

void MLXEngine::Impl::set_audio_embedding(int index, const float *embedding) {
  if (index < 0 || index >= (int)kMaxPrompts)
    return;

  std::lock_guard<std::mutex> lock(musiccoca_mutex_);
  if (embedding) {
    memcpy(cached_embeddings_[index], embedding,
           kMusicCoCaEmbeddingDim * sizeof(float));
    embedding_valid_[index] = true;
    slot_is_audio_[index] = true;
    cached_texts_[index] = "audio";
  } else {
    slot_is_audio_[index] = false;
    embedding_valid_[index] = false;
    cached_texts_[index].clear();
  }
}

void MLXEngine::Impl::set_audio_prompt_samples(int index,
                                               const std::string &filename,
                                               const float *samples,
                                               size_t count) {
  if (index < 0 || index >= (int)kMaxPrompts)
    return;

  bool should_trigger = false;
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    if (samples && count > 0) {
      pending_audio_samples_[index].assign(samples, samples + count);
      slot_is_audio_[index] = true;
      cached_texts_[index] = filename;
      embedding_valid_[index] = false; // Need to re-encode
      has_pending_musiccoca_ = true;
      should_trigger = true;
    } else {
      pending_audio_samples_[index].clear();
      slot_is_audio_[index] = false;
      embedding_valid_[index] = false;
      cached_texts_[index].clear();
    }
  }

  if (should_trigger) {
    start_inference_thread_if_needed();
  }
}

bool MLXEngine::Impl::get_audio_embedding(int index, float *out) const {
  if (index < 0 || index >= (int)kMaxPrompts)
    return false;
  std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(musiccoca_mutex_));
  if (!slot_is_audio_[index])
    return false;
  if (out) {
    memcpy(out, cached_embeddings_[index], 768 * sizeof(float));
  }
  return true;
}

bool MLXEngine::Impl::encode_single_prompt(const std::string &text,
                                           float *out_embedding) {
  if (!tokenizer_ || !text_encoder_interpreter_)
    return false;

  std::string lower_text = text;
  for (auto &c : lower_text)
    c = std::tolower(c);

  std::vector<int> ids;
  tokenizer_->Encode(lower_text, &ids);
  ids.insert(ids.begin(), 1); // bos token

  constexpr int kMaxSeqLen = 128;
  std::vector<int32_t> input_ids(kMaxSeqLen, 0);
  std::vector<float> input_paddings(kMaxSeqLen, 1.0f);

  for (size_t i = 0; i < ids.size() && i < kMaxSeqLen; ++i) {
    input_ids[i] = ids[i];
    input_paddings[i] = 0.0f;
  }

  int32_t *ids_ptr = nullptr;
  float *pads_ptr = nullptr;

  for (int i = 0;
       i < TfLiteInterpreterGetInputTensorCount(text_encoder_interpreter_);
       ++i) {
    TfLiteTensor *t =
        TfLiteInterpreterGetInputTensor(text_encoder_interpreter_, i);
    if (TfLiteTensorType(t) == kTfLiteInt32)
      ids_ptr = (int32_t *)TfLiteTensorData(t);
    if (TfLiteTensorType(t) == kTfLiteFloat32)
      pads_ptr = (float *)TfLiteTensorData(t);
  }

  if (ids_ptr)
    memcpy(ids_ptr, input_ids.data(), kMaxSeqLen * sizeof(int32_t));
  if (pads_ptr)
    memcpy(pads_ptr, input_paddings.data(), kMaxSeqLen * sizeof(float));

  if (TfLiteInterpreterInvoke(text_encoder_interpreter_) != kTfLiteOk)
    return false;

  const TfLiteTensor *output_embed =
      TfLiteInterpreterGetOutputTensor(text_encoder_interpreter_, 0);
  const float *embed_data = (const float *)TfLiteTensorData(output_embed);
  memcpy(out_embedding, embed_data, kMusicCoCaEmbeddingDim * sizeof(float));

  // Refine via the mapper when available; on failure (or older bundles that
  // ship no mapper.tflite) keep the unmapped embedding.
  apply_mapper(out_embedding);

  return true;
}

bool MLXEngine::Impl::apply_mapper(float *embedding) {
  if (!mapper_interpreter_)
    return false;

  if (TfLiteInterpreterGetInputTensorCount(mapper_interpreter_) < 2) {
    add_log("[MLXEngine] Mapper expects 2 inputs (embedding, noise); "
            "skipping mapper.");
    return false;
  }

  TfLiteTensor *emb_in = TfLiteInterpreterGetInputTensor(mapper_interpreter_, 0);
  TfLiteTensor *noise_in =
      TfLiteInterpreterGetInputTensor(mapper_interpreter_, 1);
  if (!emb_in || !noise_in)
    return false;

  constexpr size_t kEmbBytes = kMusicCoCaEmbeddingDim * sizeof(float);
  if (TfLiteTensorByteSize(emb_in) != kEmbBytes ||
      TfLiteTensorByteSize(noise_in) != kEmbBytes) {
    add_log("[MLXEngine] Unexpected mapper input size; skipping mapper.");
    return false;
  }

  // Input 0 = text embedding, input 1 = Gaussian noise. Matches the
  // input_details[0]/[1] ordering used in musiccoca.py. Seed 0 reproduces the
  // Python reference's default (`use_mapper=True, seed=0`).
  std::vector<float> noise(kMusicCoCaEmbeddingDim);
  NumpyRandomState rng(/*seed=*/0);
  rng.randn(noise.data(), kMusicCoCaEmbeddingDim);

  memcpy(TfLiteTensorData(emb_in), embedding, kEmbBytes);
  memcpy(TfLiteTensorData(noise_in), noise.data(), kEmbBytes);

  if (TfLiteInterpreterInvoke(mapper_interpreter_) != kTfLiteOk) {
    add_log("[MLXEngine] Mapper invoke failed; skipping mapper.");
    return false;
  }

  const TfLiteTensor *out =
      TfLiteInterpreterGetOutputTensor(mapper_interpreter_, 0);
  if (!out || TfLiteTensorByteSize(out) != kEmbBytes) {
    add_log("[MLXEngine] Unexpected mapper output size; skipping mapper.");
    return false;
  }
  const float *out_data = (const float *)TfLiteTensorData(out);

  // L2-normalize (matches `emb = emb / np.linalg.norm(emb)`). Accumulate in
  // float32 to mirror the numpy float32 norm.
  float norm_sq = 0.0f;
  for (int i = 0; i < kMusicCoCaEmbeddingDim; ++i)
    norm_sq += out_data[i] * out_data[i];
  float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 1.0f;
  for (int i = 0; i < kMusicCoCaEmbeddingDim; ++i)
    embedding[i] = out_data[i] * inv_norm;

  return true;
}

int MLXEngine::Impl::encode_audio_prompt(const std::vector<float> &samples,
                                         float *out_embedding) {
  if (!audio_preprocessor_interpreter_ || !music_encoder_interpreter_)
    return -1;
  if (samples.empty())
    return -5;

  // 1. Run Preprocessor
  TfLiteTensor *prep_input =
      TfLiteInterpreterGetInputTensor(audio_preprocessor_interpreter_, 0);
  if (!prep_input)
    return -6;
  size_t prep_input_size = TfLiteTensorByteSize(prep_input);
  size_t samples_bytes = samples.size() * sizeof(float);

  // Copy samples, pad if needed
  memset(TfLiteTensorData(prep_input), 0, prep_input_size);
  memcpy(TfLiteTensorData(prep_input), samples.data(),
         std::min(samples_bytes, prep_input_size));

  add_log("[MagentaRT] Invoking audio preprocessor...");
  if (TfLiteInterpreterInvoke(audio_preprocessor_interpreter_) != kTfLiteOk)
    return -2;
  add_log("[MagentaRT] Audio preprocessor done.");

  const TfLiteTensor *prep_output =
      TfLiteInterpreterGetOutputTensor(audio_preprocessor_interpreter_, 0);
  if (!prep_output)
    return -2;
  size_t prep_output_size = TfLiteTensorByteSize(prep_output);

  // 2. Run Music Encoder
  TfLiteTensor *enc_input =
      TfLiteInterpreterGetInputTensor(music_encoder_interpreter_, 0);
  if (!enc_input)
    return -3;
  size_t enc_input_size = TfLiteTensorByteSize(enc_input);

  // Verify sizes match or copy what fits
  memset(TfLiteTensorData(enc_input), 0, enc_input_size);
  memcpy(TfLiteTensorData(enc_input), TfLiteTensorData(prep_output),
         std::min(prep_output_size, enc_input_size));

  add_log("[MagentaRT] Invoking music encoder...");
  if (TfLiteInterpreterInvoke(music_encoder_interpreter_) != kTfLiteOk)
    return -3;
  add_log("[MagentaRT] Music encoder done.");

  const TfLiteTensor *enc_output =
      TfLiteInterpreterGetOutputTensor(music_encoder_interpreter_, 0);
  if (!enc_output)
    return -3;
  size_t enc_output_size = TfLiteTensorByteSize(enc_output);

  if (enc_output_size != kMusicCoCaEmbeddingDim * sizeof(float)) {
    std::cerr << "[MLXEngine] Unexpected music_encoder output size: "
              << enc_output_size << " bytes, expected "
              << kMusicCoCaEmbeddingDim * sizeof(float) << std::endl;
    return -4;
  }

  memcpy(out_embedding, TfLiteTensorData(enc_output), enc_output_size);
  return 0; // Success
}

bool MLXEngine::Impl::quantize_embedding(const float *embedding,
                                         std::vector<int> &out_tokens) {
  if (!quantizer_interpreter_)
    return false;

  TfLiteTensor *q_input =
      TfLiteInterpreterGetInputTensor(quantizer_interpreter_, 0);
  memcpy(TfLiteTensorData(q_input), embedding,
         kMusicCoCaEmbeddingDim * sizeof(float));

  if (TfLiteInterpreterInvoke(quantizer_interpreter_) != kTfLiteOk)
    return false;

  const TfLiteTensor *q_output =
      TfLiteInterpreterGetOutputTensor(quantizer_interpreter_, 0);
  const int32_t *token_data = (const int32_t *)TfLiteTensorData(q_output);

  out_tokens.resize(kMusicCoCaRVQLevels);
  for (int i = 0; i < kMusicCoCaRVQLevels; ++i) {
    out_tokens[i] = token_data[i];
  }
  return true;
}

void MLXEngine::Impl::fetch_musiccoca_tokens(
    const std::vector<std::string> &texts, const std::vector<float> &weights) {
  add_log("[MagentaRT] fetch_musiccoca_tokens started.");
  bool has_audio = false;
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
      if (slot_is_audio_[i] && !pending_audio_samples_[i].empty()) {
        has_audio = true;
        break;
      }
    }
  }

  if (!tokenizer_ || !text_encoder_interpreter_ || !quantizer_interpreter_ ||
      (texts.empty() && !has_audio)) {
    add_log("[MagentaRT] fetch_musiccoca_tokens failed: no inputs or "
            "interpreters missing.");
    text_encoder_status_ = 3;
    quantizer_status_ = 3;
    return;
  }

  // Check for special "musiccoca:" override (only if there is exactly 1
  // non-empty prompt)
  int non_empty_count = 0;
  std::string candidate_text;
  for (const auto &txt : texts) {
    if (!txt.empty()) {
      non_empty_count++;
      candidate_text = txt;
    }
  }

  static constexpr char kMusicCoCaOverridePrefix[] = "musiccoca:";
  static constexpr std::size_t kMusicCoCaOverridePrefixLen =
      sizeof(kMusicCoCaOverridePrefix) - 1; // strlen, excludes NUL
  if (non_empty_count == 1 &&
      candidate_text.rfind(kMusicCoCaOverridePrefix, 0) == 0) {
    std::string t = candidate_text.substr(kMusicCoCaOverridePrefixLen);
    std::stringstream ss(t);
    std::string item;
    std::vector<int> tokens;
    bool parse_success = true;

    while (std::getline(ss, item, ',')) {
      try {
        // Trim whitespace
        size_t first = item.find_first_not_of(" \t\r\n");
        size_t last = item.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && last != std::string::npos) {
          item = item.substr(first, (last - first + 1));
        }
        if (!item.empty()) {
          tokens.push_back(std::stoi(item));
        }
      } catch (...) {
        parse_success = false;
        break;
      }
    }

    if (parse_success && tokens.size() == kMusicCoCaRVQLevels) {
      std::cout << "[MagentaRT] Using raw MusicCoCa overrides: ";
      for (int tk : tokens)
        std::cout << tk << " ";
      std::cout << std::endl;

      {
        std::lock_guard<std::mutex> lock(musiccoca_mutex_);
        musiccoca_tokens_ = tokens;
      }
      text_encoder_status_ = 2; // success
      quantizer_status_ = 2;    // success
      return;                   // Skip standard pipeline
    }
    std::cout << "[MagentaRT] Failed to parse musiccoca override (found "
              << tokens.size() << " tokens), falling back to text."
              << std::endl;
  }

  quantizer_status_ = 0;

  // --- Snapshot audio slots under lock ---
  bool local_is_audio[kMaxPrompts] = {};
  std::vector<float> snapshot_samples[kMaxPrompts];
  std::string snapshot_texts[kMaxPrompts];
  // Snapshot whether the embedding is already valid to avoid unnecessary re-encoding.
  bool local_embedding_valid[kMaxPrompts] = {};
  float local_cached_embeddings[kMaxPrompts][kMusicCoCaEmbeddingDim] = {};
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
      local_is_audio[i] = slot_is_audio_[i];
      local_embedding_valid[i] = embedding_valid_[i];
      if (local_is_audio[i]) {
        snapshot_samples[i] = pending_audio_samples_[i];
        snapshot_texts[i] = cached_texts_[i];
        if (embedding_valid_[i]) {
          // Copy the cached embedding so we can use it outside the lock.
          memcpy(local_cached_embeddings[i], cached_embeddings_[i],
                 kMusicCoCaEmbeddingDim * sizeof(float));
        }
      }
    }
  }

  // --- Encode phase: work entirely on local buffers (no lock held) ---
  int local_count = std::min(texts.size(), kMaxPrompts);
  float local_embeddings[kMaxPrompts][kMusicCoCaEmbeddingDim] = {};
  std::string local_texts[kMaxPrompts];
  bool local_valid[kMaxPrompts] = {};

  bool needs_encoding = false;
  for (int i = 0; i < (int)kMaxPrompts; ++i) {
    if (local_is_audio[i]) {
      if (local_embedding_valid[i]) {
        // Use cached embedding to avoid re-encoding audio that hasn't changed.
        memcpy(local_embeddings[i], local_cached_embeddings[i],
               kMusicCoCaEmbeddingDim * sizeof(float));
        local_texts[i] = snapshot_texts[i];
        local_valid[i] = true;
        prompt_statuses_[i] = 2; // success
      } else {
        prompt_statuses_[i] = 1; // processing
        int err = encode_audio_prompt(snapshot_samples[i], local_embeddings[i]);
        if (err == 0) {
          local_texts[i] = snapshot_texts[i];
          local_valid[i] = true;
          prompt_statuses_[i] = 2; // success
        } else {
          local_texts[i] = "Err: " + std::to_string(err);
          local_valid[i] = false;
          prompt_statuses_[i] = 3; // error
        }
      }
      continue;
    }
    if (i < local_count && !texts[i].empty()) {
      std::string lower_text = texts[i];
      for (auto &c : lower_text)
        c = std::tolower(c);

      // Special syntax: ":anyNamedCentroid" — load pre-computed custom centroid
      // embedding
      if (lower_text.substr(0, 1) == ":" && pca_loaded_) {
        std::string key = lower_text.substr(1);
        auto it = custom_centroids_.find(key);
        if (it != custom_centroids_.end()) {
          memcpy(local_embeddings[i], it->second.data(),
                 kMusicCoCaEmbeddingDim * sizeof(float));
          local_texts[i] = lower_text;
          local_valid[i] = true;
          prompt_statuses_[i] = 2; // success
          std::cout << "[MagentaRT] Loaded custom named centroid '" << key
                    << "' into slot " << i << std::endl;
          continue;
        } else {
          std::cout << "[MagentaRT] Custom named centroid '" << key
                    << "' not found in corpus" << std::endl;
        }
        // Fall through to text encoding if not found
      }

      // Check text_embedding_cache_ (global hashmap). encode_single_prompt
      // applies the mapper deterministically, so the cached embedding is a
      // pure function of the (lowercased) text — no mapper state in the key.
      auto it = text_embedding_cache_.find(lower_text);
      if (it != text_embedding_cache_.end()) {
        memcpy(local_embeddings[i], it->second.data(),
               kMusicCoCaEmbeddingDim * sizeof(float));
        prompt_statuses_[i] = 2; // success
      } else {
        // Cache miss — encode
        needs_encoding = true;
        prompt_statuses_[i] = 1; // processing
        if (!encode_single_prompt(texts[i], local_embeddings[i])) {
          add_log(std::string("[MagentaRT] Failed to encode prompt: ") +
                  texts[i]);
          text_encoder_status_ = 3;
          prompt_statuses_[i] = 3; // error
          quantizer_status_ = 3;
          return;
        }
        // Store in global cache
        std::vector<float> emb_vec(
            local_embeddings[i], local_embeddings[i] + kMusicCoCaEmbeddingDim);
        text_embedding_cache_[lower_text] = emb_vec;
        prompt_statuses_[i] = 2; // success
      }
      local_texts[i] = texts[i];
      local_valid[i] = true;
    } else {
      local_valid[i] = false;
      local_texts[i].clear();
      prompt_statuses_[i] = 0; // idle
    }
  }

  text_encoder_status_ = needs_encoding ? 1 : 2;

  // Collect active weights
  std::vector<float> act_weights;
  std::vector<int> act_indices;
  for (int i = 0; i < (int)kMaxPrompts; ++i) {
    float w = (i < (int)weights.size()) ? weights[i] : 0.0f;
    if (local_valid[i] && w > 0.0f) {
      act_weights.push_back(w);
      act_indices.push_back(i);
    }
  }

  // Fallback: if no active prompts, use first cached embedding
  if (act_indices.empty()) {
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
      if (local_valid[i]) {
        act_weights.push_back(1.0f);
        act_indices.push_back(i);
        break;
      }
    }
  }
  if (act_indices.empty()) {
    encode_single_prompt("", local_embeddings[0]);
    local_texts[0] = "";
    local_valid[0] = true;
    act_weights.push_back(1.0f);
    act_indices.push_back(0);
  }

  // Normalize and blend
  float weight_sum = 0.0f;
  for (float w : act_weights)
    weight_sum += w;

  text_encoder_status_ = 2; // success
  quantizer_status_ = 1;    // fetching

  float acc_embedding[kMusicCoCaEmbeddingDim] = {};
  for (size_t p = 0; p < act_indices.size(); ++p) {
    float norm_w = act_weights[p] / weight_sum;
    const float *emb = local_embeddings[act_indices[p]];
    for (int i = 0; i < kMusicCoCaEmbeddingDim; ++i) {
      acc_embedding[i] += norm_w * emb[i];
    }
  }

  // Apply PCA components at the very end before quantization
  if (pca_loaded_) {
    for (int c = 0; c < pca_component_count_; ++c) {
      float coeff = current_pca_coeffs_[c].load(std::memory_order_relaxed);
      if (coeff == 0.0f)
        continue;
      for (int j = 0; j < kMusicCoCaEmbeddingDim; ++j) {
        acc_embedding[j] += coeff * pca_components_[c][j];
      }
    }
  }

  // Quantize
  std::vector<int> tokens_result;
  if (!quantize_embedding(acc_embedding, tokens_result)) {
    add_log("[MagentaRT] fetch_musiccoca_tokens failed quantization.");
    quantizer_status_ = 3;
    return;
  }

  std::cout << "[MagentaRT] Combined Prompt (" << texts.size() << ") tokens: ";
  for (int i = 0; i < static_cast<int>(kMusicCoCaRVQLevels); ++i) {
    std::cout << tokens_result[i] << " ";
  }
  std::cout << std::endl;

  // --- Commit phase: copy local results into shared state under the lock ---
  {
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
      memcpy(cached_embeddings_[i], local_embeddings[i],
             kMusicCoCaEmbeddingDim * sizeof(float));
      cached_texts_[i] = local_texts[i];
      embedding_valid_[i] = local_valid[i];
    }
    active_prompt_count_ = local_count;
    musiccoca_tokens_ = tokens_result;
  }
  quantizer_status_ = 2; // success
  add_log("[MagentaRT] fetch_musiccoca_tokens success.");
}

bool MLXEngine::Impl::load_pca_data(const float *mean, const float *components,
                                    int num_components) {
  std::lock_guard<std::mutex> lock(musiccoca_mutex_);

  pca_component_count_ = std::min(num_components, (int)kMaxPCAComponents);
  for (int i = 0; i < pca_component_count_; ++i) {
    memcpy(pca_components_[i], components + i * kMusicCoCaEmbeddingDim,
           kMusicCoCaEmbeddingDim * sizeof(float));
  }

  pca_loaded_ = true;
  std::cout << "[MagentaRT] Loaded PCA corpus mean and principal components"
            << std::endl;
  return true;
}

bool MLXEngine::Impl::load_pca_file(const char *path) {
  try {
    auto [arrays, meta] = mx::load_safetensors(std::string(path));

    // Load components (up to 6, require at least one)
    std::vector<mx::array> comp_arrays;
    for (int i = 0; i < (int)kMaxPCAComponents; ++i) {
      auto key = "component_" + std::to_string(i);
      auto it = arrays.find(key);
      if (it == arrays.end())
        break; // Stop at first missing component
      comp_arrays.push_back(mx::astype(it->second, mx::float32));
    }
    if (comp_arrays.empty()) {
      std::cerr << "[MagentaRT] PCA file missing required 'component_0' key "
                   "(need at least one component)"
                << std::endl;
      return false;
    }
    mx::eval(comp_arrays);

    // Collect all other keys as custom centroids (excluding components, mean,
    // and explained_variance)
    std::lock_guard<std::mutex> lock(musiccoca_mutex_);
    custom_centroids_.clear();

    for (const auto &[name, arr] : arrays) {
      if (name == "explained_variance")
        continue;
      if (name.substr(0, 10) == "component_")
        continue;

      auto arr_float = mx::astype(arr, mx::float32);
      mx::eval(arr_float);

      if (arr_float.size() != kMusicCoCaEmbeddingDim) {
        std::cerr << "[MagentaRT] Custom centroid '" << name
                  << "' has wrong size: " << arr_float.size() << " (expected "
                  << kMusicCoCaEmbeddingDim << ")" << std::endl;
        continue;
      }

      std::string lower_name = name;
      std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      std::vector<float> vec(kMusicCoCaEmbeddingDim);
      memcpy(vec.data(), arr_float.data<float>(),
             kMusicCoCaEmbeddingDim * sizeof(float));
      custom_centroids_[lower_name] = vec;
      std::cout << "[MagentaRT] Loaded custom named centroid: " << lower_name
                << std::endl;
    }

    // Build flat component buffer
    std::vector<float> flat_components(comp_arrays.size() *
                                       kMusicCoCaEmbeddingDim);
    for (size_t i = 0; i < comp_arrays.size(); ++i) {
      memcpy(flat_components.data() + i * kMusicCoCaEmbeddingDim,
             comp_arrays[i].data<float>(),
             kMusicCoCaEmbeddingDim * sizeof(float));
    }

    pca_loaded_ = false; // Reset to ensure consistent state
    pca_component_count_ = (int)comp_arrays.size();
    for (int i = 0; i < pca_component_count_; ++i) {
      memcpy(pca_components_[i],
             flat_components.data() + i * kMusicCoCaEmbeddingDim,
             kMusicCoCaEmbeddingDim * sizeof(float));
    }
    pca_loaded_ = true;

    std::cout << "[MagentaRT] Loaded PCA corpus mean and "
              << pca_component_count_ << " components" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MagentaRT] Failed to load PCA file: " << e.what()
              << std::endl;
    return false;
  }
}

bool MLXEngine::Impl::reblend_musiccoca_tokens(const float *weights, int count,
                                               const float *pca_coeffs,
                                               int pca_count) {
  if (!quantizer_interpreter_)
    return false;

  // Try to acquire the mutex (non-blocking to avoid stalling inference)
  std::unique_lock<std::mutex> lock(musiccoca_mutex_, std::try_to_lock);
  if (!lock.owns_lock())
    return false;

  // Check under lock: if a fetch thread is active, cached_embeddings_ may be
  // mid-update.  The flag is set/cleared under musiccoca_mutex_ so this is
  // race-free.
  if (is_musiccoca_fetching_.load(std::memory_order_relaxed))
    return false;

  // Recompute PCA embeddings if coefficients provided
  if (pca_coeffs && pca_count > 0) {
    int n = std::min(pca_count, (int)kMaxPCAComponents);
    for (int i = 0; i < n; ++i) {
      current_pca_coeffs_[i].store(pca_coeffs[i], std::memory_order_relaxed);
    }
    for (int i = n; i < (int)kMaxPCAComponents; ++i) {
      current_pca_coeffs_[i].store(0.0f, std::memory_order_relaxed);
    }
  }

  // Collect active embeddings with the new weights
  float acc_embedding[kMusicCoCaEmbeddingDim] = {};
  float weight_sum = 0.0f;
  int num_active = 0;

  int n = std::min(count, (int)kMaxPrompts);
  for (int i = 0; i < n; ++i) {
    if (embedding_valid_[i] && weights[i] > 0.0f) {
      weight_sum += weights[i];
      num_active++;
    }
  }

  if (num_active == 0 || weight_sum < 0.0001f)
    return false;

  for (int i = 0; i < n; ++i) {
    if (embedding_valid_[i] && weights[i] > 0.0f) {
      float norm_w = weights[i] / weight_sum;
      for (int j = 0; j < kMusicCoCaEmbeddingDim; ++j) {
        acc_embedding[j] += norm_w * cached_embeddings_[i][j];
      }
    }
  }

  // Apply PCA components at the very end before quantization
  if (pca_loaded_) {
    for (int c = 0; c < pca_component_count_; ++c) {
      float coeff = current_pca_coeffs_[c].load(std::memory_order_relaxed);
      if (coeff == 0.0f)
        continue;
      for (int j = 0; j < kMusicCoCaEmbeddingDim; ++j) {
        acc_embedding[j] += coeff * pca_components_[c][j];
      }
    }
  }

  // Quantize the blended embedding
  std::vector<int> tokens_result;
  if (!quantize_embedding(acc_embedding, tokens_result))
    return false;

  musiccoca_tokens_ = tokens_result;
  return true;
}

std::string MLXEngine::Impl::get_cached_text(int index) {
  std::lock_guard<std::mutex> lock(musiccoca_mutex_);
  if (index >= 0 && index < (int)kMaxPrompts) {
    return cached_texts_[index];
  }
  return "";
}

void MLXEngine::Impl::add_log(const std::string &msg) {
  std::lock_guard<std::mutex> lock(log_mutex_);
  log_lines_.push_back(msg);
}

std::vector<std::string> MLXEngine::Impl::get_logs() {
  std::lock_guard<std::mutex> lock(log_mutex_);
  auto logs = log_lines_;
  log_lines_.clear();
  return logs;
}

// ─── Public MLXEngine facade (forwarders to Impl) ───────────────────────────

MLXEngine::MLXEngine() : impl_(std::make_unique<Impl>()) {}
MLXEngine::~MLXEngine() = default;
MLXEngine::MLXEngine(MLXEngine &&) noexcept = default;
MLXEngine &MLXEngine::operator=(MLXEngine &&) noexcept = default;

// Lifecycle
bool MLXEngine::init_assets(const char *d, const char *s) {
  return impl_->init_assets(d, s);
}
bool MLXEngine::load_model(const char *p) {
  return impl_->load_model(p);
}
bool MLXEngine::load_prefill_model(const char *ss, const char *pf) {
  return impl_->load_prefill_model(ss, pf);
}
bool MLXEngine::prefill_state(const float *s, int n, int trim_front_frames,
                              int trim_back_frames,
                              std::function<void(const std::string &)> cb,
                              std::vector<float> *out_audio_L,
                              std::vector<float> *out_audio_R,
                              bool mask_musiccoca_during_prefill) {
  return impl_->prefill_state(s, n, trim_front_frames, trim_back_frames,
                              std::move(cb), out_audio_L, out_audio_R,
                              mask_musiccoca_during_prefill);
}
bool MLXEngine::prefill_state_from_tokens(
    const int32_t *tokens, int num_frames,
    std::function<void(const std::string &)> cb,
    std::vector<float> *out_audio_L, std::vector<float> *out_audio_R,
    bool mask_musiccoca_during_prefill) {
  return impl_->prefill_state_from_tokens(tokens, num_frames, std::move(cb),
                                          out_audio_L, out_audio_R,
                                          mask_musiccoca_during_prefill);
}
bool MLXEngine::prefill_silence(int duration_frames, bool reset_first,
                                std::function<void(const std::string &)> cb,
                                std::vector<float> *out_audio_L,
                                std::vector<float> *out_audio_R) {
  return impl_->prefill_silence(duration_frames, reset_first, std::move(cb),
                                out_audio_L, out_audio_R);
}
void MLXEngine::unload() { impl_->unload(); }
void MLXEngine::reset_state() { impl_->reset_state(); }
bool MLXEngine::save_state(const char *p) { return impl_->save_state(p); }
bool MLXEngine::load_state(const char *p) { return impl_->load_state(p); }
void MLXEngine::reset_to_factory() { impl_->reset_to_factory(); }
bool MLXEngine::is_loaded() const { return impl_->transformer_fn_.has_value(); }

// Generation
bool MLXEngine::generate_frame(float *L, float *R, std::int32_t *tokens_out) {
  return impl_->generate_frame(L, R, tokens_out);
}
const FrameMetrics &MLXEngine::last_metrics() const {
  return impl_->last_metrics_;
}

// Prompts — text
void MLXEngine::set_text_prompt(const std::string &t) {
  impl_->set_text_prompt(t);
}
void MLXEngine::set_text_prompts(const std::vector<std::string> &t,
                                 const std::vector<float> &w) {
  impl_->set_text_prompts(t, w);
}
void MLXEngine::set_musiccoca_tokens_masked() {
  impl_->set_musiccoca_tokens_masked();
}
int MLXEngine::get_text_encoder_status() const {
  return impl_->text_encoder_status_.load(std::memory_order_relaxed);
}
int MLXEngine::get_rvq_depth() const { return kNumRVQLevels; }

int MLXEngine::get_prompt_status(int index) const {
  if (index < 0 || index >= (int)kMaxPrompts)
    return 0;
  return impl_->prompt_statuses_[index].load(std::memory_order_relaxed);
}
int MLXEngine::get_quantizer_status() const {
  return impl_->quantizer_status_.load(std::memory_order_relaxed);
}
void MLXEngine::add_log(const std::string &m) { impl_->add_log(m); }
std::vector<std::string> MLXEngine::get_logs() { return impl_->get_logs(); }
bool MLXEngine::reblend_musiccoca_tokens(const float *w, int c, const float *p,
                                         int pc) {
  return impl_->reblend_musiccoca_tokens(w, c, p, pc);
}
int MLXEngine::get_active_prompt_count() const {
  return impl_->active_prompt_count_;
}
std::string MLXEngine::get_cached_text(int i) {
  return impl_->get_cached_text(i);
}

// Prompts — PCA
bool MLXEngine::load_pca_data(const float *m, const float *c, int n) {
  return impl_->load_pca_data(m, c, n);
}
bool MLXEngine::load_pca_file(const char *p) { return impl_->load_pca_file(p); }
bool MLXEngine::is_pca_loaded() const { return impl_->pca_loaded_; }
int MLXEngine::pca_component_count() const {
  return impl_->pca_component_count_;
}
int MLXEngine::pca_centroid_count() const {
  return static_cast<int>(impl_->custom_centroids_.size());
}

// Sampling parameters
void MLXEngine::set_temperature(float t) {
  impl_->temperature_.store(t, std::memory_order_relaxed);
}
float MLXEngine::get_temperature() const {
  return impl_->temperature_.load(std::memory_order_relaxed);
}
void MLXEngine::set_top_k(int k) {
  impl_->top_k_.store(k, std::memory_order_relaxed);
}
int MLXEngine::get_top_k() const {
  return impl_->top_k_.load(std::memory_order_relaxed);
}

void MLXEngine::set_cfg_musiccoca(float v) {
  impl_->cfg_musiccoca_.store(v, std::memory_order_relaxed);
}
float MLXEngine::get_cfg_musiccoca() const {
  return impl_->cfg_musiccoca_.load(std::memory_order_relaxed);
}
void MLXEngine::set_cfg_notes(float v) {
  impl_->cfg_notes_.store(v, std::memory_order_relaxed);
}
float MLXEngine::get_cfg_notes() const {
  return impl_->cfg_notes_.load(std::memory_order_relaxed);
}
void MLXEngine::set_cfg_drums(float v) {
  impl_->cfg_drums_.store(v, std::memory_order_relaxed);
}
float MLXEngine::get_cfg_drums() const {
  return impl_->cfg_drums_.load(std::memory_order_relaxed);
}
void MLXEngine::set_unmask_width(int w) {
  impl_->unmask_width_.store(w, std::memory_order_relaxed);
}
int MLXEngine::get_unmask_width() const {
  return impl_->unmask_width_.load(std::memory_order_relaxed);
}
void MLXEngine::set_seed_rotation(int r) {
  impl_->seed_rotation_.store(r, std::memory_order_relaxed);
}
int MLXEngine::get_seed_rotation() const {
  return impl_->seed_rotation_.load(std::memory_order_relaxed);
}

// MIDI notes
void MLXEngine::set_note_on(int n) { impl_->note_tracker_->noteOn(n); }
void MLXEngine::set_note_off(int n) { impl_->note_tracker_->noteOff(n); }

// Drumless
void MLXEngine::set_drumless(bool on) {
  impl_->drumless_.store(on, std::memory_order_relaxed);
}
bool MLXEngine::get_drumless() const {
  return impl_->drumless_.load(std::memory_order_relaxed);
}
void MLXEngine::set_onset_mode(int mode) {
  impl_->onset_mode_.store(mode, std::memory_order_relaxed);
}
int MLXEngine::get_onset_mode() const {
  return impl_->onset_mode_.load(std::memory_order_relaxed);
}

// Audio prompts
void MLXEngine::set_audio_embedding(int i, const float *e) {
  impl_->set_audio_embedding(i, e);
}
void MLXEngine::set_audio_prompt_samples(int i, const std::string &f,
                                         const float *s, std::size_t c) {
  impl_->set_audio_prompt_samples(i, f, s, c);
}
bool MLXEngine::get_audio_embedding(int i, float *o) const {
  return impl_->get_audio_embedding(i, o);
}

} // namespace core
} // namespace magentart
