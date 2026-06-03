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

// Implementation of `magentart::core::RealtimeRunner` — the audio-thread-safe
// wrapper around `MLXEngine`. Header declares; this file carries the bodies
// so we can include `<mlx/mlx.h>` and the portable `AutoreleasePool` helper
// without leaking them to consumers.

#include <magentart/realtime_runner.h>
#include <magentart/detail/autorelease_pool.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>

#include <mlx/mlx.h>

namespace magentart {
namespace core {

namespace mx = mlx::core;

RealtimeRunner::RealtimeRunner() {
    // Default: equal blend of the first two prompt slots.
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
        blend_weights_[i].store((i < 2) ? 0.5f : 0.0f, std::memory_order_relaxed);
    }
    for (int i = 0; i < 128; ++i) {
        note_states_[i].store(false, std::memory_order_relaxed);
    }
    reset_env_.set_attack_samples(1920.0f);
    reset_env_.value.store(1.0f, std::memory_order_relaxed);
    midi_env_.set_attack_samples(1920.0f);
    midi_env_.set_release_samples(48000.0f);
    midi_env_.value.store(0.0f, std::memory_order_relaxed);
}

RealtimeRunner::~RealtimeRunner() {
    stop();
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

bool RealtimeRunner::load_musiccoca_model(const char* resource_dir, const char* subfolder) {
    bool was_running = running_.load(std::memory_order_relaxed);
    if (was_running) stop();

    bool success = engine_.init_assets(resource_dir, subfolder);

    if (was_running && success) start();
    return success;
}

bool RealtimeRunner::load_model(const char* mlxfn_path) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stop_locked();
    bool success = engine_.load_model(mlxfn_path);
    if (success) start_locked();
    return success;
}

bool RealtimeRunner::load_prefill_model(const char* spectrostream_mlxfn_path,
                                        const char* prefill_mlxfn_path) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return engine_.load_prefill_model(spectrostream_mlxfn_path, prefill_mlxfn_path);
}

bool RealtimeRunner::prefill_state(const float* audio_samples, int num_samples,
                                   std::function<void(const std::string&)> log_callback) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stop_locked();
    // Drop the first and last 1s (25 frames @ 25 hz) of encoded SpectroStream
    // tokens, which we suspect may contain "edge artifacts".
    // TODO: Determine if this is necessary, and if we can reduce the number
    // of dropped frames on either side.
    //
    // We deliberately do NOT capture the prefill audio for ring-buffer
    // priming: the captured frames are the model's per-step *predictions*
    // during teacher-forced prefill, which diverge from the natural
    // decoder trajectory and sound artifacted at the input length scale
    // (>20 s) typical for audio prefill. Instead, the engine internally
    // checkpoints the post-prefill state (transformer_initial_state_ is
    // updated), and we restart the runner with `start_locked` — the
    // 3-frame ring-buffer priming kicks in and produces clean real-time
    // generation from the prefilled state. Subsequent reset_state()
    // calls land back on this checkpoint, so the user can prefill once
    // and try multiple prompts via repeated resets.
    constexpr int kTrimFrames = 25;
    bool success = engine_.prefill_state(audio_samples, num_samples,
                                         kTrimFrames, kTrimFrames,
                                         std::move(log_callback),
                                         /*out_audio_L=*/nullptr,
                                         /*out_audio_R=*/nullptr,
                                         /*mask_musiccoca_during_prefill=*/false);
    if (success) {
        skip_next_transport_reset_.store(true, std::memory_order_relaxed);
        start_locked(/*skip_reset=*/true);
    }
    return success;
}

bool RealtimeRunner::prefill_silence(int duration_frames,
                                     std::function<void(const std::string&)> log_callback) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stop_locked();
    // `prefill_silence` resets the model state, masks MusicCoCa, and feeds
    // silent tokens straight through `prefill_state_from_tokens` (no
    // encoder needed since silent tokens are token-exact and
    // yield no decoder trajectory drift). Like the audio path, we don't
    // capture audio for ring-buffer priming — the engine checkpoints the
    // post-silent-prefill state, and the runner's 3-frame priming
    // generates clean real-time audio from the silent context on restart.
    bool success = engine_.prefill_silence(duration_frames,
                                            /*reset_first=*/true,
                                            std::move(log_callback),
                                            /*out_audio_L=*/nullptr,
                                            /*out_audio_R=*/nullptr);
    if (success) {
        skip_next_transport_reset_.store(true, std::memory_order_relaxed);
        start_locked(/*skip_reset=*/true);
    }
    return success;
}

void RealtimeRunner::unload() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stop_locked();
    engine_.unload();
}

void RealtimeRunner::start() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    start_locked();
}

void RealtimeRunner::stop() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    stop_locked();
}

void RealtimeRunner::start_locked(bool skip_reset) {
    // Safe even if thread is already running: stop first.
    if (thread_.joinable()) {
        running_.store(false, std::memory_order_relaxed);
        thread_.join();
    }
    if (!engine_.is_loaded()) return;
    if (!skip_reset) {
        engine_.reset_state();
    }

    // Reblend prompt tokens using explicit blend weights before pre-generating
    float weights[kMaxPrompts] = {};
    for (int i = 0; i < (int)kMaxPrompts; ++i) {
        weights[i] = blend_weights_[i].load(std::memory_order_relaxed);
    }
    float pca_coeffs[kMaxPCAComponents];
    for (int i = 0; i < (int)kMaxPCAComponents; ++i) {
        pca_coeffs[i] = pca_coeff_[i].load(std::memory_order_relaxed);
    }
    if (engine_.reblend_musiccoca_tokens(weights, kMaxPrompts,
                                     pca_coeffs, kMaxPCAComponents)) {
        last_blended_generation_ = position_generation_.load(std::memory_order_relaxed);
    }

    ring_L_.reset();
    ring_R_.reset();
    // 3-frame priming to prevent initial audio underruns
    for (int i = 0; i < 3; ++i) {
        detail::AutoreleasePool pool;
        engine_.generate_frame(audio_buf_L_, audio_buf_R_);
        ring_L_.write(audio_buf_L_, kFrameSamples);
        ring_R_.write(audio_buf_R_, kFrameSamples);
    }
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&RealtimeRunner::inference_loop, this);
}

void RealtimeRunner::stop_locked() {
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
}

// ─── Audio output ───────────────────────────────────────────────────────────

bool RealtimeRunner::read_audio_stereo(float* destL, float* destR,
                                       std::size_t count, bool blocking) {
    // Bypass: immediate silence, don't drain ring buffer.
    if (bypass_.load(std::memory_order_relaxed) ||
        host_bypass_.load(std::memory_order_relaxed)) {
        std::memset(destL, 0, count * sizeof(float));
        std::memset(destR, 0, count * sizeof(float));
        return true;
    }

    // During offline render, wait for the inference thread to produce data
    // instead of padding with zeros (which would create gaps in the export).
    // Wait for min(count, virtual_capacity) to avoid deadlock when the host
    // requests more frames than the ring buffer can hold at once.
    if (blocking) {
        std::size_t target = std::min(count, ring_L_.get_virtual_capacity());
        while (ring_L_.available() < target &&
               running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    bool okL = ring_L_.read(destL, count);
    bool okR = ring_R_.read(destR, count);

    // A real-time underrun = the inference thread couldn't keep pace, so the
    // ring buffer zero-padded. Count it as a dropped frame. Skip the blocking
    // (offline render) path: there the audio thread deliberately waits for
    // data, so a short read isn't a real-time dropout.
    if (!blocking && !(okL && okR)) {
        dropped_frame_count_.fetch_add(1, std::memory_order_relaxed);
    }

    bool is_muted = mute_.load(std::memory_order_relaxed);
    float vol_db = volume_db_.load(std::memory_order_relaxed);
    float target_gain = is_muted ? 0.0f : std::pow(10.0f, vol_db / 20.0f);

    // Principled one-pole smoothing filter coefficient.
    // alpha = 1.0 - exp(-1.0 / (time_constant * sample_rate))
    constexpr float time_constant = 0.01f;  // 10 ms
    constexpr float sample_rate = 48000.0f;
    const float alpha = 1.0f - std::exp(-1.0f / (time_constant * sample_rate));
    const float one_minus_alpha = 1.0f - alpha;

    bool gate_enabled = midi_gate_enabled_.load(std::memory_order_relaxed);
    float midi_target = (active_note_count_.load(std::memory_order_relaxed) > 0) ? 1.0f : 0.0f;

    for (std::size_t i = 0; i < count; ++i) {
        smoothed_gain_ = one_minus_alpha * smoothed_gain_ + alpha * target_gain;

        float reset_gain = reset_env_.tick(1.0f);
        float env_gain = reset_gain;
        if (gate_enabled) {
            float midi_gain = midi_env_.tick(midi_target);
            env_gain *= midi_gain;
        }

        destL[i] *= smoothed_gain_ * env_gain;
        destR[i] *= smoothed_gain_ * env_gain;
    }

    if (is_recording_.load(std::memory_order_relaxed)) {
        std::size_t idx = recorded_samples_.load(std::memory_order_relaxed);
        if (idx + count <= recording_buf_L_.size()) {
            for (std::size_t i = 0; i < count; ++i) {
                recording_buf_L_[idx + i] = destL[i];
                recording_buf_R_[idx + i] = destR[i];
            }
            recorded_samples_.fetch_add(count, std::memory_order_relaxed);
        }
    }

    return okL && okR;
}

// ─── MIDI notes ─────────────────────────────────────────────────────────────

void RealtimeRunner::set_note_on(int n) {
    engine_.set_note_on(n);
    if (n >= 0 && n < 128) {
        if (!note_states_[n].exchange(true, std::memory_order_relaxed)) {
            active_note_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void RealtimeRunner::set_note_off(int n) {
    engine_.set_note_off(n);
    if (n >= 0 && n < 128) {
        if (note_states_[n].exchange(false, std::memory_order_relaxed)) {
            active_note_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

// ─── Prompts ────────────────────────────────────────────────────────────────

void RealtimeRunner::set_audio_prompt(int index, const std::string& path) {
    if (path.empty()) {
        engine_.set_audio_embedding(index, nullptr);
    } else {
        float dummy_embedding[kMusicCoCaEmbeddingDim];
        for (int i = 0; i < kMusicCoCaEmbeddingDim; ++i) {
            dummy_embedding[i] = 0.1f * (i % 10);
        }
        engine_.set_audio_embedding(index, dummy_embedding);
    }
}

void RealtimeRunner::set_audio_embedding(int index, const float* embedding) {
    engine_.set_audio_embedding(index, embedding);
}

// ─── Reset & state ──────────────────────────────────────────────────────────

void RealtimeRunner::reset() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    bool was_running = running_.load(std::memory_order_relaxed);
    if (was_running) stop_locked();

    engine_.reset_state();
    ring_L_.reset();
    ring_R_.reset();

    if (was_running) start_locked();
}

bool RealtimeRunner::save_state(const char* path) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    bool was_running = running_.load(std::memory_order_relaxed);
    if (was_running) stop_locked();

    bool success = engine_.save_state(path);
    if (success) {
        engine_.load_state(path);
    }

    if (was_running) start_locked(true);
    return success;
}

bool RealtimeRunner::load_state(const char* path) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    bool was_running = running_.load(std::memory_order_relaxed);
    if (was_running) stop_locked();

    bool success = engine_.load_state(path);

    if (was_running) start_locked();
    return success;
}

void RealtimeRunner::reset_to_factory() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    bool was_running = running_.load(std::memory_order_relaxed);
    if (was_running) stop_locked();

    engine_.reset_to_factory();
    ring_L_.reset();
    ring_R_.reset();

    if (was_running) start_locked();
}

EngineMetrics RealtimeRunner::get_metrics() const {
    auto m = engine_.last_metrics();
    float total = frame_total_ms_.load(std::memory_order_relaxed);
    return {m.transformer_ms,
            total > 0 ? total : m.total_ms,
            ring_L_.available(),
            get_buffer_size(),
            transport_flags_.load(std::memory_order_relaxed),
            dropped_frame_count_.load(std::memory_order_relaxed)};
}

// ─── Inference loop ─────────────────────────────────────────────────────────

void RealtimeRunner::inference_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        detail::AutoreleasePool pool;

        if (bypass_.load(std::memory_order_relaxed) ||
            host_bypass_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Read the pending reset request in a single atomic exchange so we
        // can decide on the request kind without a TOCTOU window between
        // "is anything pending?" and "what kind is it?". Then, only for
        // transport-initiated resets, optionally consume the
        // skip_next_transport_reset_ flag set by a recent prefill — user-
        // initiated resets always fire regardless of any prefill state.
        const ResetRequest req = reset_request_.exchange(ResetRequest::None,
                                                          std::memory_order_relaxed);
        bool do_reset = false;
        if (req == ResetRequest::User) {
            do_reset = true;
        } else if (req == ResetRequest::Transport) {
            const bool skip = skip_next_transport_reset_.exchange(
                false, std::memory_order_relaxed);
            do_reset = !skip;
            // If we suppressed, the prefilled context is preserved and we
            // fall through to normal generation. If a user reset is
            // requested *after* this iteration, it will fire normally on
            // the next iteration — the skip flag is one-shot.
        }
        if (do_reset) {
            engine_.reset_state();
            ring_L_.reset();
            ring_R_.reset();
            // 3-frame priming to prevent audio underruns on reset
            for (int i = 0; i < 3; ++i) {
                engine_.generate_frame(audio_buf_L_, audio_buf_R_);
                ring_L_.write(audio_buf_L_, kFrameSamples);
                ring_R_.write(audio_buf_R_, kFrameSamples);
            }
            continue;
        }

        // Re-blend musiccoca tokens if blend weights or PCA coefficients changed.
        std::uint32_t current_gen = position_generation_.load(std::memory_order_relaxed);
        if (current_gen != last_blended_generation_) {
            float weights[kMaxPrompts] = {};
            for (int i = 0; i < (int)kMaxPrompts; ++i) {
                weights[i] = blend_weights_[i].load(std::memory_order_relaxed);
            }
            float pca_coeffs[kMaxPCAComponents];
            for (int i = 0; i < (int)kMaxPCAComponents; ++i) {
                pca_coeffs[i] = pca_coeff_[i].load(std::memory_order_relaxed);
            }
            if (engine_.reblend_musiccoca_tokens(weights, kMaxPrompts,
                                             pca_coeffs, kMaxPCAComponents)) {
                last_blended_generation_ = current_gen;
            }
        }

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        bool ok = engine_.generate_frame(audio_buf_L_, audio_buf_R_);
        auto t1 = clock::now();
        frame_total_ms_.store(
            std::chrono::duration<float, std::milli>(t1 - t0).count(),
            std::memory_order_relaxed);

        if (ok) {
            // Wait until there is space in the buffer instead of dropping
            // the frame. GPU keepalive: tiny GPU ops keep macOS from
            // downclocking the GPU during idle gaps (causes permanent
            // ~8 ms latency increase after ~20 sec of running plugin otherwise).
            while (ring_L_.free_space() < kFrameSamples &&
                   running_.load(std::memory_order_relaxed)) {
                auto dummy = mx::array({0.0f}) + mx::array({0.0f});
                mx::eval(dummy);
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            if (running_.load(std::memory_order_relaxed)) {
                ring_L_.write(audio_buf_L_, kFrameSamples);
                ring_R_.write(audio_buf_R_, kFrameSamples);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void RealtimeRunner::start_recording() {
    std::size_t max_samples = 5 * 60 * 48000; // 5 minutes at 48kHz
    recording_buf_L_.resize(max_samples, 0.0f);
    recording_buf_R_.resize(max_samples, 0.0f);
    recorded_samples_.store(0, std::memory_order_relaxed);
    is_recording_.store(true, std::memory_order_relaxed);
}

void RealtimeRunner::stop_recording() {
    is_recording_.store(false, std::memory_order_relaxed);
}

void RealtimeRunner::clear_recording() {
    recorded_samples_.store(0, std::memory_order_relaxed);
}

bool RealtimeRunner::get_recorded_audio(float* destL, float* destR, std::size_t start_idx, std::size_t count) const {
    std::size_t total = recorded_samples_.load(std::memory_order_relaxed);
    if (start_idx + count > total) return false;

    std::copy(recording_buf_L_.begin() + start_idx, recording_buf_L_.begin() + start_idx + count, destL);
    std::copy(recording_buf_R_.begin() + start_idx, recording_buf_R_.begin() + start_idx + count, destR);
    return true;
}

std::vector<float> RealtimeRunner::get_waveform_peaks(int num_buckets) const {
    // Reduces millions of samples into 'num_buckets' (e.g. 200) peak values.
    // Divides the audio into even chunks and calculates the maximum absolute
    // amplitude in each chunk. Safe to call from the UI thread.
    std::size_t total = recorded_samples_.load(std::memory_order_relaxed);
    std::vector<float> peaks(num_buckets, 0.0f);
    if (total == 0) return peaks;

    std::size_t step = total / num_buckets;
    if (step == 0) step = 1;

    for (int i = 0; i < num_buckets; ++i) {
        float max_val = 0.0f;
        std::size_t end = std::min(total, (std::size_t)(i + 1) * step);
        for (std::size_t j = i * step; j < end; ++j) {
            float vL = std::abs(recording_buf_L_[j]);
            float vR = std::abs(recording_buf_R_[j]);
            max_val = std::max({max_val, vL, vR});
        }
        peaks[i] = max_val;
    }
    return peaks;
}

}  // namespace core
}  // namespace magentart
