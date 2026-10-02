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

// Free-running generation with the ggml depthformer. Writes RVQ tokens as
// int32 [frames, num_codebooks]; decode them to audio with
// scripts/decode_tokens.py (the codec is not ported yet).

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "mrt_ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kMulanPerRvqVocab = 1031;

const std::vector<int> kRegularVocab = [] {
    std::vector<int> v;
    v.insert(v.end(), 128, 11);   // pianoroll
    v.push_back(9);               // drum pianoroll
    v.insert(v.end(), 2, 47);     // cfg conditioning
    v.push_back(15);              // cfg drums
    return v;
}();

// Mirrors depthformer._sample_categorical_with_temperature: mask to the RVQ's
// valid range, keep top-k, then Gumbel-max. Note the temperature scales the
// noise rather than dividing the logits, matching the JAX implementation.
int sample_token(std::vector<float> & logits, int lo, int hi, float soft_cap,
                 float temperature, int top_k, std::mt19937 & rng) {
    const float kNeg = -0.7f * 3.4028235e38f;
    std::vector<float> cand;
    cand.reserve(hi - lo);
    for (int i = lo; i < hi; i++) {
        logits[i] = std::tanh(logits[i] / soft_cap) * soft_cap;
        cand.push_back(logits[i]);
    }
    float kth = -INFINITY;
    if (top_k > 0 && top_k < (int) cand.size()) {
        std::nth_element(cand.begin(), cand.end() - top_k, cand.end());
        kth = *(cand.end() - top_k);
    }

    std::uniform_real_distribution<float> uni(1e-20f, 1.0f);
    float best_v = -INFINITY;
    int   best_i = lo;
    for (int i = lo; i < hi; i++) {
        float l = logits[i] < kth ? kNeg : logits[i];
        const float g = -std::log(-std::log(uni(rng)));
        l += g * temperature;
        if (l > best_v) { best_v = l; best_i = i; }
    }
    return best_i;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: mrt_generate <model.gguf> <ref_dir> <out.tokens.bin> "
                "[frames] [seed] [cpu]\n");
        return 1;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir   = argv[2];
    const std::string out_path  = argv[3];
    const int   n_frames = argc > 4 ? atoi(argv[4]) : 250;   // 10 s at 25 Hz
    const unsigned seed  = argc > 5 ? (unsigned) atoi(argv[5]) : 0;
    bool force_cpu = false;
    for (int i = 6; i < argc; i++) if (std::string(argv[i]) == "cpu") force_cpu = true;

    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    printf("Backend: %s\n", ggml_backend_name(backend));

    mrt::Model model;
    if (!mrt::load_model(gguf_path, backend, model)) return 1;
    if (!mrt::alloc_caches(model))                   return 1;
    const auto & hp = model.hparams;

    // Conditioning tokens (style + notes), with the host-side channel offsets.
    std::vector<int32_t> tokens(144);
    {
        std::ifstream f(ref_dir + "/cond_tokens.i32.bin", std::ios::binary);
        if (!f) { fprintf(stderr, "missing cond_tokens.i32.bin in %s\n", ref_dir.c_str()); return 1; }
        f.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(int32_t));
    }
    const int n_mulan = 12, n_regular = (int) tokens.size() - 12;
    std::vector<int32_t> mulan_idx(n_mulan), reg_idx(n_regular);
    for (int i = 0; i < n_mulan; i++) mulan_idx[i] = tokens[i] + i * kMulanPerRvqVocab;
    for (int i = 0, off = 0; i < n_regular; i++) {
        reg_idx[i] = tokens[12 + i] + off;
        off += kRegularVocab[i];
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    mrt::Graph genc = mrt::build_encoder_graph(model, n_mulan, n_regular);
    ggml_gallocr_alloc_graph(allocr, genc.gf);
    ggml_backend_tensor_set(genc.inputs[0], mulan_idx.data(), 0, mulan_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(genc.inputs[1], reg_idx.data(), 0, reg_idx.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, genc.gf);
    std::vector<float> enc_out(ggml_nelements(genc.out));
    ggml_backend_tensor_get(genc.out, enc_out.data(), 0, ggml_nbytes(genc.out));
    mrt::free_graph(genc);

    std::vector<mrt::Graph> temporal, depth;
    for (int p = 0; p < hp.temporal_cache_slots; p++) temporal.push_back(mrt::build_temporal_graph(model, p));
    for (int r = 0; r < hp.num_codebooks; r++)        depth.push_back(mrt::build_depth_graph(model, r));

    std::mt19937 rng(seed);
    std::vector<int32_t> prev(hp.num_codebooks, hp.sos_id);
    std::vector<int32_t> out;
    out.reserve((size_t) n_frames * hp.num_codebooks);
    std::vector<float> temporal_out(hp.temporal_dim), logits, emb_row(hp.temporal_dim);

    for (int frame = 0; frame < n_frames; frame++) {
        const int wpos = frame % hp.temporal_cache_slots;
        const int filled = std::min(frame + 1, hp.temporal_cache_slots);
        std::vector<float> mask(hp.temporal_cache_slots + 1, -INFINITY);
        mask[0] = 0.0f;
        for (int i = 0; i < filled; i++) mask[1 + i] = 0.0f;

        mrt::Graph & gt = temporal[wpos];
        ggml_gallocr_alloc_graph(allocr, gt.gf);
        ggml_backend_tensor_set(gt.inputs[0], prev.data(), 0, prev.size() * sizeof(int32_t));
        ggml_backend_tensor_set(gt.inputs[1], enc_out.data(), 0, enc_out.size() * sizeof(float));
        ggml_backend_tensor_set(gt.inputs[2], mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gt.gf);
        ggml_backend_tensor_get(gt.out, temporal_out.data(), 0, ggml_nbytes(gt.out));

        std::vector<float> depth_in = temporal_out;
        for (int r = 0; r < hp.num_codebooks; r++) {
            std::vector<float> dmask(hp.depth_cache_slots, -INFINITY);
            for (int j = 0; j <= r && j < hp.depth_cache_slots; j++) dmask[j] = 0.0f;

            mrt::Graph & gd = depth[r];
            ggml_gallocr_alloc_graph(allocr, gd.gf);
            ggml_backend_tensor_set(gd.inputs[0], depth_in.data(), 0, depth_in.size() * sizeof(float));
            ggml_backend_tensor_set(gd.inputs[1], dmask.data(), 0, dmask.size() * sizeof(float));
            ggml_backend_graph_compute(backend, gd.gf);

            logits.resize(ggml_nelements(gd.out));
            ggml_backend_tensor_get(gd.out, logits.data(), 0, ggml_nbytes(gd.out));

            const int lo = hp.num_reserved_tokens + r * hp.codebook_size;
            const int tok = sample_token(logits, lo, lo + hp.codebook_size,
                                         hp.soft_cap_logits, 1.3f, 40, rng);
            prev[r] = tok;
            out.push_back(tok);

            struct ggml_tensor * emb = model.get("dec.emb");
            ggml_backend_tensor_get(emb, emb_row.data(),
                                    (size_t) tok * emb->nb[1], hp.temporal_dim * sizeof(float));
            const float s = sqrtf((float) hp.temporal_dim);
            for (int i = 0; i < hp.temporal_dim; i++) depth_in[i] = emb_row[i] * s;
        }
        if ((frame + 1) % 50 == 0) printf("  %d/%d frames\n", frame + 1, n_frames);
    }

    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(out.data()), out.size() * sizeof(int32_t));
    printf("Wrote %zu tokens (%d frames, %.2f s) to %s\n",
           out.size(), n_frames, n_frames / 25.0, out_path.c_str());

    for (auto & g : temporal) mrt::free_graph(g);
    for (auto & g : depth)    mrt::free_graph(g);
    ggml_gallocr_free(allocr);
    mrt::free_model(model);
    ggml_backend_free(backend);
    return 0;
}
