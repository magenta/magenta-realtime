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

/// @file ring_buffer.h
/// @brief Lock-free single-producer / single-consumer ring buffer for audio.
///
/// The inference thread writes `kFrameSamples` at a time into the buffer; the
/// audio thread reads variable-size chunks out. Both threads use relaxed loads
/// on their own cursor and acquire/release on the peer's cursor — no locks, no
/// allocations, suitable for use inside an audio callback.
///
/// Thread safety: exactly one producer thread and exactly one consumer thread.
/// Calling `write` from two threads at once, or `read` from two threads at
/// once, is undefined behavior. `reset()` must be called while both are quiet.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>

namespace magentart {
namespace core {

class RingBuffer {
public:
    /// Maximum buffered samples (~4 model blocks). Must be a power of two.
    static constexpr size_t kCapacity = 8192;

    RingBuffer() : buffer_{}, write_pos_(0), read_pos_(0), virtual_capacity_(2048) {}

    /// Cap the effective capacity below `kCapacity`. Used to tune latency vs.
    /// back-pressure on the inference thread.
    void set_virtual_capacity(size_t cap) {
        if (cap > kCapacity) cap = kCapacity;
        virtual_capacity_.store(cap, std::memory_order_relaxed);
    }

    size_t get_virtual_capacity() const {
        return virtual_capacity_.load(std::memory_order_relaxed);
    }

    /// Samples available to the consumer right now.
    size_t available() const {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_relaxed);
    }

    /// Samples the producer may write right now without blocking.
    size_t free_space() const {
        size_t avail = available();
        size_t cap = get_virtual_capacity();
        return (cap > avail) ? (cap - avail) : 0;
    }

    /// Producer-only. Write `count` samples. Returns false if there isn't
    /// enough free space (no partial writes).
    bool write(const float* data, size_t count) {
        if (free_space() < count) return false;
        write_unchecked(data, count);
        return true;
    }

    /// Consumer-only. Read `count` samples into `dest`. Underflow pads with
    /// zeroes so the caller always gets `count` samples written. Returns false
    /// iff an underrun occurred (so callers can log / count glitches).
    bool read(float* dest, size_t count) {
        size_t avail = available();
        size_t to_read = (avail < count) ? avail : count;
        return read_n(dest, to_read, count);
    }

    /// Hard-reset cursors. Call only when both producer and consumer are
    /// paused — concurrent reset is undefined behavior.
    void reset() {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

    /// Consumer-only. Skip all currently buffered samples. Safe while the
    /// producer is still running.
    void drain() {
        read_pos_.store(write_pos_.load(std::memory_order_acquire),
                        std::memory_order_release);
    }

    /// Producer-only. Write `count` samples to each of two buffers, or to
    /// neither if either lacks room. Two separate `write` calls can leave the
    /// buffers holding different amounts when the capacity changes in between.
    static bool write_pair(RingBuffer& a, RingBuffer& b, const float* data_a,
                           const float* data_b, size_t count) {
        if (std::min(a.free_space(), b.free_space()) < count) return false;
        a.write_unchecked(data_a, count);
        b.write_unchecked(data_b, count);
        return true;
    }

    /// Consumer-only. Read `count` samples from two buffers that the producer
    /// fills one after the other, advancing both read cursors by the same
    /// amount. A read that lands between the producer's two writes then sees
    /// the shorter of the two, so an underrun can't leave the channels offset
    /// from each other. Returns false iff an underrun occurred.
    static bool read_pair(RingBuffer& a, RingBuffer& b, float* dest_a,
                          float* dest_b, size_t count) {
        size_t to_read = std::min({a.available(), b.available(), count});
        a.read_n(dest_a, to_read, count);
        b.read_n(dest_b, to_read, count);
        return to_read == count;
    }

    /// Consumer-only. Skip the buffered samples of two buffers that the
    /// producer fills one after the other, by the same amount for both. Safe
    /// while the producer is still running.
    static void drain_pair(RingBuffer& a, RingBuffer& b) {
        size_t n = std::min(a.available(), b.available());
        a.skip(n);
        b.skip(n);
    }

private:
    /// Producer-only. `write` without the free-space check, for callers that
    /// have already checked that `count` samples fit.
    void write_unchecked(const float* data, size_t count) {
        size_t pos = write_pos_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) {
            buffer_[(pos + i) & (kCapacity - 1)] = data[i];
        }
        write_pos_.store(pos + count, std::memory_order_release);
    }

    /// Consumer-only. Copy `to_read` samples (which must not exceed
    /// `available()` or `count`) into `dest`, zero-pad the rest of `count` and
    /// advance the read cursor by `to_read`. Returns false iff
    /// `to_read < count`.
    bool read_n(float* dest, size_t to_read, size_t count) {
        size_t pos = read_pos_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < to_read; ++i) {
            dest[i] = buffer_[(pos + i) & (kCapacity - 1)];
        }
        for (size_t i = to_read; i < count; ++i) {
            dest[i] = 0.0f;
        }
        read_pos_.store(pos + to_read, std::memory_order_release);

        return to_read == count;
    }

    /// Consumer-only. Skip `n` buffered samples, which must not exceed
    /// `available()`.
    void skip(size_t n) {
        read_pos_.store(read_pos_.load(std::memory_order_relaxed) + n,
                        std::memory_order_release);
    }

    float buffer_[kCapacity];
    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
    std::atomic<size_t> virtual_capacity_;
};

}  // namespace core
}  // namespace magentart
