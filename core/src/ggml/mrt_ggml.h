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

// ggml port of the MagentaRT-2 depthformer.
//
// The numpy model in magenta_rt/ggml/numpy_model.py is the reference
// specification for everything here; see its docstring for the non-obvious
// semantics (unscaled-query sink logits, Praxis query scaling, etc.).

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <map>
#include <string>
#include <vector>

namespace mrt {

struct Hparams {
    int32_t temporal_layers = 0;
    int32_t temporal_dim = 0;
    int32_t temporal_ffn = 0;
    int32_t temporal_heads = 0;
    int32_t temporal_dh = 0;
    // Cache slots = max_past_horizon + 1. The JAX local attention concatenates
    // the current frame onto its max_past_horizon buffer and attends over
    // both, so the cache must hold one more frame than the horizon. Sizing it
    // at max_past_horizon evicts the oldest frame one step early -- invisible
    // until the buffer first wraps (step 41 for mrt2_small).
    int32_t temporal_cache_slots = 0;
    int32_t temporal_max_past_horizon = 0;

    int32_t depth_layers = 0;
    int32_t depth_dim = 0;
    int32_t depth_ffn = 0;
    int32_t depth_heads = 0;
    int32_t depth_dh = 0;
    int32_t depth_cache_slots = 0;
    int32_t depth_max_past_horizon = 0;

    int32_t encoder_dim = 0;
    int32_t num_codebooks = 0;
    int32_t codebook_size = 0;
    int32_t num_reserved_tokens = 0;
    int32_t sos_id = 0;
    float   soft_cap_logits = 30.0f;
};

// A right-aligned view is unnecessary: the model uses NoPE, so attention is
// permutation-invariant over cached keys. The cache is therefore a plain
// circular buffer plus a validity mask, which avoids shifting every step.
struct KVCache {
    struct ggml_tensor * k = nullptr;  // [dh, window, heads]
    struct ggml_tensor * v = nullptr;  // [window, dh, heads]  (transposed)
    int32_t              filled = 0;
    int32_t              pos = 0;
};

struct Model {
    Hparams hparams;

    struct ggml_context * ctx_w = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;
    struct ggml_context * ctx_kv = nullptr;

    std::map<std::string, struct ggml_tensor *> tensors;

    std::vector<KVCache> self_cache;   // temporal self-attention, per layer
    std::vector<KVCache> cross_cache;  // temporal cross-attention, per layer
    std::vector<KVCache> depth_cache;  // depth self-attention, per layer

    // Causal history for the codec's convs, in graph-walk order. Each holds the
    // (kernel_time - 1) previous frames of that conv's input.
    std::vector<struct ggml_tensor *> codec_hist;
    // Inverse STFT as a real-IDFT matmul: ggml has no FFT op, and a
    // [481 x 960] matrix costs ~0.5 MFLOP/frame/channel - negligible here and
    // portable to every backend.
    struct ggml_tensor * idft_re = nullptr;   // [481, 960]
    struct ggml_tensor * idft_im = nullptr;   // [481, 960]
    struct ggml_tensor * istft_window = nullptr;  // [960]
    ggml_backend_buffer_t buf_codec = nullptr;
    struct ggml_context * ctx_codec = nullptr;

    struct ggml_tensor * get(const std::string & name) const;
    bool                 has(const std::string & name) const;
};

// Loads weights + hparams from a GGUF produced by magenta_rt/ggml/convert.py.
bool load_model(const std::string & gguf_path, ggml_backend_t backend, Model & model);

void free_model(Model & model);

// Allocates the streaming KV caches (call after load_model).
bool alloc_caches(Model & model);
void reset_caches(Model & model);

// ---------------------------------------------------------------------------
// Graph building
// ---------------------------------------------------------------------------

// A built graph plus the input/output handles the caller needs.
struct Graph {
    struct ggml_context * ctx = nullptr;
    struct ggml_cgraph *  gf = nullptr;
    std::vector<struct ggml_tensor *> inputs;
    struct ggml_tensor *  out = nullptr;
};

// Encoder: conditioning token indices (already offset on the host) -> [enc_dim].
// `mulan_idx` is [12] I32 and `reg_idx` is [132] I32.
Graph build_encoder_graph(const Model & model, int n_mulan, int n_regular);

// Temporal decoder: one streaming step.
//   inputs[0] = prev_frame tokens [num_codebooks] I32
//   inputs[1] = encoder output [enc_dim] F32
//   inputs[2] = attention mask [temporal_cache_slots + 1] F32 (0 keep, -INF drop; slot 0 is the sink)
// `write_pos` is the circular-buffer slot this step writes. The model uses NoPE,
// so attention is order-invariant and a circular buffer needs no shifting.
Graph build_temporal_graph(const Model & model, int write_pos);

// Depth decoder: one RVQ step.
//   inputs[0] = depth input [temporal_dim] (temporal output, then embedded tokens)
//   inputs[1] = attention mask [depth_cache_slots] F32
// Returns logits [vocab], before the tanh soft cap.
Graph build_depth_graph(const Model & model, int write_pos);

// ---------------------------------------------------------------------------
// SpectroStream codec decoder
// ---------------------------------------------------------------------------
//
// Layout: ggml_conv_2d wants data as [W, H, C, N] and kernels as
// [KW, KH, IC, OC]. The codec's spatial axes are (time, frequency) with the
// Flax kernel stored [KH=time, KW=freq, IC, OC], so in ggml ne0 = frequency
// and ne1 = time. Activations are therefore ne = (bins, frames, channels).
//
// All regular convs have time stride 1, so sequence_layers' "semicausal"
// padding degenerates to fully causal (kh-1 on the left, 0 on the right);
// streaming supplies that left context from per-conv history instead of zeros.

// Codec front end: 12 RVQ codes -> [bins=5, frames=1, channels=512].
//   inputs[0] = codes [num_codebooks] I32 (non-unique, i.e. already reduced)
Graph build_codec_input_graph(const Model & model);

// Full codec decoder: 12 RVQ codes -> 1920 stereo samples for one frame.
//   inputs[0] = codes [num_codebooks] I32 (non-unique)
// Streaming state lives in Model::codec_hist and is updated by the graph.
// stop_after: -1 runs the whole decoder; 0 stops after the input residual
// unit; b>=1 stops after decoder_{b-1}. Used to bisect parity failures.
Graph build_codec_graph(const Model & model, int stop_after = -1);

// Allocates the codec's per-conv causal history buffers.
bool alloc_codec_state(Model & model);

void free_graph(Graph & g);

}  // namespace mrt
