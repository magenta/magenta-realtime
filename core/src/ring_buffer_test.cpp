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

#include <magentart/ring_buffer.h>
#include <cstdio>

using magentart::core::RingBuffer;

static constexpr size_t kFrame = 256;

static bool failed = false;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s\n", what);
        failed = true;
    }
}

// Writes `first`, `first + 1`, ... into `rb`.
static void write_ramp(RingBuffer& rb, float first, size_t count) {
    float data[kFrame];
    for (size_t i = 0; i < count; ++i) data[i] = first + static_cast<float>(i);
    expect(rb.write(data, count), "write fits");
}

static void test_read_zero_pads_a_short_read() {
    RingBuffer rb;
    write_ramp(rb, 1.0f, 3);
    float out[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    expect(!rb.read(out, 8), "a short read reports an underrun");
    expect(rb.available() == 0, "a short read consumes what is there");
    expect(out[0] == 1.0f && out[2] == 3.0f, "a short read copies the available samples");
    expect(out[3] == 0.0f && out[7] == 0.0f, "a short read zero-pads the rest");
}

// The inference thread writes the left buffer and then the right one. A read
// that lands in between sees the left one a frame ahead.
static void test_read_pair_stays_aligned_when_left_is_ahead() {
    RingBuffer left, right;
    write_ramp(left, 1.0f, kFrame);

    float out_l[2 * kFrame], out_r[2 * kFrame];
    bool ok = RingBuffer::read_pair(left, right, out_l, out_r, 2 * kFrame);
    expect(!ok, "read_pair reports the underrun");
    expect(left.available() == kFrame, "left keeps its unread frame");

    write_ramp(right, 1.0f, kFrame);
    write_ramp(left, 1.0f + kFrame, kFrame);
    write_ramp(right, 1.0f + kFrame, kFrame);

    ok = RingBuffer::read_pair(left, right, out_l, out_r, 2 * kFrame);
    expect(ok, "second read_pair has enough data");
    bool aligned = true;
    for (size_t i = 0; i < 2 * kFrame; ++i) {
        if (out_l[i] != out_r[i] || out_l[i] != 1.0f + static_cast<float>(i)) {
            aligned = false;
        }
    }
    expect(aligned, "left and right return the same samples after an underrun");
}

static void test_read_pair_advances_both_by_the_shorter() {
    RingBuffer left, right;
    write_ramp(left, 1.0f, kFrame);
    write_ramp(right, 1.0f, kFrame / 2);

    float out_l[kFrame], out_r[kFrame];
    expect(!RingBuffer::read_pair(left, right, out_l, out_r, kFrame), "short read is an underrun");
    expect(left.available() == kFrame / 2, "left advanced by the shorter amount");
    expect(right.available() == 0, "right is fully consumed");
    expect(out_l[kFrame / 2 - 1] == static_cast<float>(kFrame / 2), "left data intact");
    expect(out_l[kFrame / 2] == 0.0f && out_r[kFrame / 2] == 0.0f, "both zero-pad");
}

static void test_read_pair_full_read() {
    RingBuffer left, right;
    write_ramp(left, 1.0f, kFrame);
    write_ramp(right, 1.0f, kFrame);

    float out_l[kFrame], out_r[kFrame];
    expect(RingBuffer::read_pair(left, right, out_l, out_r, kFrame), "full read has no underrun");
    expect(left.available() == 0 && right.available() == 0, "both consumed");
    expect(out_l[kFrame - 1] == static_cast<float>(kFrame), "left data intact");
}

// A drain between the producer's two writes must not leave one buffer holding
// a frame that the other has already thrown away.
static void test_drain_pair_stays_aligned_when_left_is_ahead() {
    RingBuffer left, right;
    write_ramp(left, 1.0f, kFrame);
    write_ramp(right, 1.0f, kFrame);
    write_ramp(left, 1.0f + kFrame, kFrame);

    RingBuffer::drain_pair(left, right);
    expect(right.available() == 0, "right is drained");
    expect(left.available() == kFrame, "left keeps the frame right has not received");

    write_ramp(right, 1.0f + kFrame, kFrame);
    float out_l[kFrame], out_r[kFrame];
    expect(RingBuffer::read_pair(left, right, out_l, out_r, kFrame), "read after drain has enough data");
    bool aligned = true;
    for (size_t i = 0; i < kFrame; ++i) {
        if (out_l[i] != out_r[i] || out_l[i] != 1.0f + kFrame + static_cast<float>(i)) {
            aligned = false;
        }
    }
    expect(aligned, "left and right return the same samples after a drain");
}

// A capacity change between two separate writes can refuse one and not the
// other. write_pair checks both buffers first, so a frame lands in both or in
// neither.
static void test_write_pair_is_all_or_nothing() {
    RingBuffer left, right;
    left.set_virtual_capacity(kFrame);
    right.set_virtual_capacity(kFrame);
    write_ramp(right, 1.0f, kFrame);

    float data[kFrame];
    for (size_t i = 0; i < kFrame; ++i) data[i] = 5.0f;
    expect(!RingBuffer::write_pair(left, right, data, data, kFrame), "refused while the right buffer is full");
    expect(left.available() == 0, "left stays empty when the right one refuses");

    float scratch[kFrame];
    right.read(scratch, kFrame);
    expect(RingBuffer::write_pair(left, right, data, data, kFrame), "accepted once both have room");
    expect(left.available() == kFrame && right.available() == kFrame, "both hold the frame");
}

int main() {
    std::printf("Verifying RingBuffer stereo reads, writes and drains...\n");
    test_read_zero_pads_a_short_read();
    test_read_pair_stays_aligned_when_left_is_ahead();
    test_read_pair_advances_both_by_the_shorter();
    test_read_pair_full_read();
    test_drain_pair_stays_aligned_when_left_is_ahead();
    test_write_pair_is_all_or_nothing();

    if (failed) {
        std::printf("[FAIL] RingBuffer stereo reads, writes and drains\n");
        return 1;
    }
    std::printf("[PASS] RingBuffer stereo reads, writes and drains\n");
    return 0;
}
