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

// LLM-only latency benchmark for the ggml depthformer: one streaming frame is
// one temporal step plus num_codebooks depth steps.
//
// This does NOT include the SpectroStream codec, so it is not directly
// comparable to benchmark_mlxfn (which decodes to audio).

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "mrt_ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

int argmax(const std::vector<float> & v, int lo, int hi) {
    int best = lo;
    float bv = -INFINITY;
    for (int i = lo; i < hi && i < (int) v.size(); i++) {
        if (v[i] > bv) { bv = v[i]; best = i; }
    }
    return best;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string gguf_path = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) +
          "/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small_f32.gguf";
    const int n_frames = argc > 2 ? atoi(argv[2]) : 100;
    const int n_warmup = 10;
    // Extra args: "rebuild" measures the cost of rebuilding graphs each frame;
    // "cpu" forces the CPU backend instead of the best available.
    bool rebuild_each_step = false, force_cpu = false, with_codec = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "rebuild") rebuild_each_step = true;
        if (a == "cpu")     force_cpu = true;
        if (a == "codec")   with_codec = true;
    }

    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    printf("Backend: %s\n", ggml_backend_name(backend));

    mrt::Model model;
    if (!mrt::load_model(gguf_path, backend, model)) return 1;
    if (!mrt::alloc_caches(model))                   return 1;
    const auto & hp = model.hparams;

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // Pre-build every graph we will ever need: the temporal write position
    // cycles over the cache slots, and the depth cache is reset each frame so
    // its write position is always the RVQ index.
    std::vector<mrt::Graph> temporal, depth;
    double build_ms = 0.0;
    if (!rebuild_each_step) {
        const double t0 = now_ms();
        for (int p = 0; p < hp.temporal_cache_slots; p++) {
            temporal.push_back(mrt::build_temporal_graph(model, p));
        }
        for (int r = 0; r < hp.num_codebooks; r++) {
            depth.push_back(mrt::build_depth_graph(model, r));
        }
        build_ms = now_ms() - t0;
        printf("Pre-built %zu temporal + %zu depth graphs in %.1f ms\n",
               temporal.size(), depth.size(), build_ms);
    }

    // Codec: one graph, reused every frame (its state lives in the model).
    mrt::Graph codec;
    ggml_gallocr_t codec_alloc = nullptr;
    std::vector<int32_t> codes(hp.num_codebooks);
    std::vector<float> frames;
    if (with_codec) {
        if (!mrt::alloc_codec_state(model)) return 1;
        codec = mrt::build_codec_graph(model);
        codec_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(codec_alloc, codec.gf);
        frames.resize(ggml_nelements(codec.out));
        printf("Codec graph: %d nodes\n", ggml_graph_n_nodes(codec.gf));
        printf("Codec enabled: output ne = [%lld, %lld, %lld]\n",
               (long long) codec.out->ne[0], (long long) codec.out->ne[1],
               (long long) codec.out->ne[2]);
    }

    std::vector<float> enc_out(hp.encoder_dim, 0.05f);  // stand-in conditioning
    std::vector<int32_t> prev(hp.num_codebooks, hp.sos_id);
    std::vector<float> logits, emb_row(hp.temporal_dim);
    std::vector<float> temporal_out(hp.temporal_dim);

    auto run_frame = [&](int frame) {
        const int wpos = frame % hp.temporal_cache_slots;
        const int filled = frame + 1 < hp.temporal_cache_slots ? frame + 1 : hp.temporal_cache_slots;

        std::vector<float> mask(hp.temporal_cache_slots + 1, -INFINITY);
        mask[0] = 0.0f;  // sink
        for (int i = 0; i < filled; i++) mask[1 + i] = 0.0f;

        mrt::Graph gt = rebuild_each_step ? mrt::build_temporal_graph(model, wpos)
                                          : temporal[wpos];
        ggml_gallocr_alloc_graph(allocr, gt.gf);
        ggml_backend_tensor_set(gt.inputs[0], prev.data(), 0, prev.size() * sizeof(int32_t));
        ggml_backend_tensor_set(gt.inputs[1], enc_out.data(), 0, enc_out.size() * sizeof(float));
        ggml_backend_tensor_set(gt.inputs[2], mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gt.gf);
        ggml_backend_tensor_get(gt.out, temporal_out.data(), 0, ggml_nbytes(gt.out));
        if (rebuild_each_step) mrt::free_graph(gt);

        // Depth loop: 12 sequential RVQ steps, greedy argmax within each
        // codebook's valid range (stands in for the sampler).
        std::vector<float> depth_in = temporal_out;
        for (int r = 0; r < hp.num_codebooks; r++) {
            std::vector<float> dmask(hp.depth_cache_slots, -INFINITY);
            for (int j = 0; j <= r && j < hp.depth_cache_slots; j++) dmask[j] = 0.0f;

            mrt::Graph gd = rebuild_each_step ? mrt::build_depth_graph(model, r) : depth[r];
            ggml_gallocr_alloc_graph(allocr, gd.gf);
            ggml_backend_tensor_set(gd.inputs[0], depth_in.data(), 0, depth_in.size() * sizeof(float));
            ggml_backend_tensor_set(gd.inputs[1], dmask.data(), 0, dmask.size() * sizeof(float));
            ggml_backend_graph_compute(backend, gd.gf);

            logits.resize(ggml_nelements(gd.out));
            ggml_backend_tensor_get(gd.out, logits.data(), 0, ggml_nbytes(gd.out));
            if (rebuild_each_step) mrt::free_graph(gd);

            const int lo = hp.num_reserved_tokens + r * hp.codebook_size;
            const int tok = argmax(logits, lo, lo + hp.codebook_size);
            prev[r] = tok;

            struct ggml_tensor * emb = model.get("dec.emb");
            ggml_backend_tensor_get(emb, emb_row.data(),
                                    (size_t) tok * emb->nb[1], hp.temporal_dim * sizeof(float));
            const float s = sqrtf((float) hp.temporal_dim);
            depth_in.assign(hp.temporal_dim, 0.0f);
            for (int i = 0; i < hp.temporal_dim; i++) depth_in[i] = emb_row[i] * s;
        }

        if (with_codec) {
            // Depthformer emits "unique" indices; the codec wants per-level codes.
            for (int r = 0; r < hp.num_codebooks; r++) {
                codes[r] = (prev[r] - hp.num_reserved_tokens) % hp.codebook_size;
            }
            ggml_gallocr_alloc_graph(codec_alloc, codec.gf);
            ggml_backend_tensor_set(codec.inputs[0], codes.data(), 0,
                                    codes.size() * sizeof(int32_t));
            ggml_backend_graph_compute(backend, codec.gf);
            ggml_backend_tensor_get(codec.out, frames.data(), 0, ggml_nbytes(codec.out));
            // Overlap-add is a few thousand host adds; negligible beside the graph.
        }
    };

    for (int i = 0; i < n_warmup; i++) run_frame(i);
    mrt::reset_caches(model);

    const double t0 = now_ms();
    for (int i = 0; i < n_frames; i++) run_frame(i);
    const double total = now_ms() - t0;

    const double per_frame = total / n_frames;
    printf("\n");
    printf("Mode:            %s\n", rebuild_each_step ? "rebuild graphs each frame" : "pre-built graphs");
    printf("Frames:          %d\n", n_frames);
    printf("Per frame:       %.3f ms  (1 temporal + %d depth steps%s)\n",
           per_frame, hp.num_codebooks, with_codec ? " + codec" : "");
    printf("Realtime budget: 40.000 ms/frame at 25 Hz  ->  %.2fx realtime\n", 40.0 / per_frame);
    printf("BENCHMARK_RESULT llm_only ms_per_frame=%.4f\n", per_frame);

    if (!rebuild_each_step) {
        for (auto & g : temporal) mrt::free_graph(g);
        for (auto & g : depth)    mrt::free_graph(g);
    }
    if (codec_alloc) { mrt::free_graph(codec); ggml_gallocr_free(codec_alloc); }
    ggml_gallocr_free(allocr);
    mrt::free_model(model);
    ggml_backend_free(backend);
    return 0;
}
