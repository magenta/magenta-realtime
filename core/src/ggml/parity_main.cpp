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

// Parity harness: runs the ggml depthformer and compares each stage against the
// JAX fp32 references produced by scripts/generate_ggml_reference.py.

#include "mrt_ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace {

struct RefEntry {
    std::vector<int64_t> shape;
    std::string          file;
};

// Parses the TSV manifest: "name<TAB>ndim<TAB>d0 d1 ...<TAB>file".
std::map<std::string, RefEntry> load_manifest(const std::string & dir) {
    std::map<std::string, RefEntry> out;
    std::ifstream f(dir + "/manifest.txt");
    if (!f) {
        fprintf(stderr, "parity: cannot open %s/manifest.txt\n", dir.c_str());
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string name, ndim_s, dims_s, file;
        if (!std::getline(ss, name, '\t'))   continue;
        if (!std::getline(ss, ndim_s, '\t')) continue;
        if (!std::getline(ss, dims_s, '\t')) continue;
        if (!std::getline(ss, file, '\t'))   continue;
        RefEntry e;
        std::stringstream ds(dims_s);
        int64_t d;
        while (ds >> d) e.shape.push_back(d);
        e.file = file;
        out[name] = e;
    }
    return out;
}

std::vector<float> read_bin(const std::string & path, size_t n) {
    std::vector<float> out(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "parity: cannot open %s\n", path.c_str());
        return {};
    }
    f.read(reinterpret_cast<char *>(out.data()), n * sizeof(float));
    return out;
}

int g_failures = 0;

// MusicCoCa per-RVQ vocab (codebook 1024 + 6 reserved + 1 dropout).
constexpr int kMulanPerRvqVocab = 1031;

// Per-channel vocab sizes for the 132 non-MusicCoCa conditioning channels:
// 128 pianoroll (4+6+1), 1 drum pianoroll (2+6+1), 2 CFG (41+6), 1 CFG drums (9+6).
const std::vector<int> kRegularVocab = [] {
    std::vector<int> v;
    v.insert(v.end(), 128, 11);
    v.push_back(9);
    v.insert(v.end(), 2, 47);
    v.push_back(15);
    return v;
}();

bool compare(const char * label, const std::vector<float> & got,
             const std::vector<float> & want, float tol = 1e-3f) {
    if (got.size() != want.size()) {
        printf("  %-40s SIZE MISMATCH got=%zu want=%zu\n", label, got.size(), want.size());
        g_failures++;
        return false;
    }
    float max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        const float d = std::abs(got[i] - want[i]);
        if (d > max_abs) max_abs = d;
    }
    const bool ok = max_abs <= tol;
    printf("  %-40s max_abs=%.4e %s\n", label, max_abs, ok ? "PASS" : "FAIL");
    if (!ok) g_failures++;
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string gguf_path = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) +
          "/Documents/Magenta/magenta-rt-v2/models/mrt2_small/mrt2_small_f32.gguf";
    const std::string ref_dir = argc > 2 ? argv[2] : "tests/testdata/ggml/mrt2_small";

    bool force_cpu = false;
    for (int i = 3; i < argc; i++) if (std::string(argv[i]) == "cpu") force_cpu = true;
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    printf("Backend: %s\n", ggml_backend_name(backend));

    mrt::Model model;
    if (!mrt::load_model(gguf_path, backend, model)) return 1;
    if (!mrt::alloc_caches(model))                   return 1;

    auto manifest = load_manifest(ref_dir);
    printf("parity: %zu reference tensors\n\n", manifest.size());

    auto ref = [&](const std::string & name) -> std::vector<float> {
        auto it = manifest.find(name);
        if (it == manifest.end()) {
            fprintf(stderr, "parity: no reference '%s'\n", name.c_str());
            g_failures++;
            return {};
        }
        size_t n = 1;
        for (int64_t d : it->second.shape) n *= (size_t) d;
        return read_bin(ref_dir + "/" + it->second.file, n);
    };

    // Conditioning tokens, plus the host-side channel offsets. The offsets are
    // pure index arithmetic over model constants, so they are applied here
    // rather than in the graph.
    std::vector<int32_t> tokens(144);
    {
        std::ifstream f(ref_dir + "/cond_tokens.i32.bin", std::ios::binary);
        f.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(int32_t));
    }
    const int n_mulan = 12, n_regular = (int) tokens.size() - 12;
    std::vector<int32_t> mulan_idx(n_mulan), reg_idx(n_regular);
    for (int i = 0; i < n_mulan; i++) {
        mulan_idx[i] = tokens[i] + i * kMulanPerRvqVocab;
    }
    for (int i = 0, off = 0; i < n_regular; i++) {
        reg_idx[i] = tokens[12 + i] + off;
        off += kRegularVocab[i];
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    printf("-- encoder --\n");
    mrt::Graph genc = mrt::build_encoder_graph(model, n_mulan, n_regular);
    ggml_gallocr_reserve(allocr, genc.gf);
    ggml_gallocr_alloc_graph(allocr, genc.gf);
    ggml_backend_tensor_set(genc.inputs[0], mulan_idx.data(), 0, mulan_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(genc.inputs[1], reg_idx.data(), 0, reg_idx.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, genc.gf);

    std::vector<float> enc_out(ggml_nelements(genc.out));
    ggml_backend_tensor_get(genc.out, enc_out.data(), 0, ggml_nbytes(genc.out));
    compare("encoder", enc_out, ref("step0.depthformer.encoder.body.step"));
    mrt::free_graph(genc);

    // ---- temporal decoder, step 0 ----------------------------------------
    printf("\n-- temporal decoder --\n");
    {
        const auto & hp = model.hparams;
        const int window = hp.temporal_cache_slots;

        // Step 0: only the slot just written is valid; the sink (index 0) is
        // always valid.
        std::vector<float> mask(window + 1, -INFINITY);
        mask[0] = 0.0f;              // sink
        const int write_pos = 0;
        mask[1 + write_pos] = 0.0f;  // the frame written this step

        std::vector<int32_t> prev(hp.num_codebooks, hp.sos_id);

        mrt::Graph gt = mrt::build_temporal_graph(model, write_pos);
        ggml_gallocr_reserve(allocr, gt.gf);
        ggml_gallocr_alloc_graph(allocr, gt.gf);
        ggml_backend_tensor_set(gt.inputs[0], prev.data(), 0, prev.size() * sizeof(int32_t));
        ggml_backend_tensor_set(gt.inputs[1], enc_out.data(), 0, enc_out.size() * sizeof(float));
        ggml_backend_tensor_set(gt.inputs[2], mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gt.gf);

        std::vector<float> out(ggml_nelements(gt.out));
        ggml_backend_tensor_get(gt.out, out.data(), 0, ggml_nbytes(gt.out));
        compare("temporal_outputs", out, ref("step0.depthformer.decoder.temporal_outputs"));
        mrt::free_graph(gt);
    }

    // ---- depth decoder: all RVQ steps, teacher-forced ---------------------
    printf("\n-- depth decoder (teacher-forced) --\n");
    {
        const auto & hp = model.hparams;
        std::vector<float> depth_in = ref("step0.depthformer.decoder.temporal_outputs");
        std::vector<float> emb_row(hp.temporal_dim);

        for (int rvq = 0; rvq < hp.num_codebooks; rvq++) {
            std::vector<float> mask(hp.depth_cache_slots, -INFINITY);
            for (int j = 0; j <= rvq && j < hp.depth_cache_slots; j++) mask[j] = 0.0f;

            mrt::Graph gd = mrt::build_depth_graph(model, rvq);
            ggml_gallocr_reserve(allocr, gd.gf);
            ggml_gallocr_alloc_graph(allocr, gd.gf);
            ggml_backend_tensor_set(gd.inputs[0], depth_in.data(), 0, depth_in.size() * sizeof(float));
            ggml_backend_tensor_set(gd.inputs[1], mask.data(), 0, mask.size() * sizeof(float));
            ggml_backend_graph_compute(backend, gd.gf);

            std::vector<float> logits(ggml_nelements(gd.out));
            ggml_backend_tensor_get(gd.out, logits.data(), 0, ggml_nbytes(gd.out));
            char label[64];
            snprintf(label, sizeof(label), "depth_logits rvq=%d", rvq);
            compare(label, logits, ref("step0.depthformer.decoder.depth_logits." + std::to_string(rvq)));
            mrt::free_graph(gd);

            // Teacher forcing: feed JAX's sampled token back, embedded and scaled.
            auto samp = ref("step0.depthformer.decoder.depth_samples." + std::to_string(rvq));
            const int token = (int) samp[0];
            struct ggml_tensor * emb = model.get("dec.emb");
            ggml_backend_tensor_get(emb, emb_row.data(),
                                    (size_t) token * emb->nb[1], hp.temporal_dim * sizeof(float));
            const float scale = sqrtf((float) hp.temporal_dim);
            for (int i = 0; i < hp.temporal_dim; i++) depth_in[i] = emb_row[i] * scale;
        }
    }

    // ---- streaming: step 1 reuses the caches written by step 0 ------------
    if (manifest.count("step1.depthformer.decoder.temporal_outputs")) {
        printf("\n-- streaming (step 1) --\n");
        const auto & hp = model.hparams;
        const int window = hp.temporal_cache_slots;
        const int write_pos = 1;

        // Slots 0 and 1 now hold real frames; the sink is always valid.
        std::vector<float> mask(window + 1, -INFINITY);
        mask[0] = 0.0f;
        mask[1 + 0] = 0.0f;
        mask[1 + write_pos] = 0.0f;

        // The previous frame is step 0's sampled RVQ tokens.
        std::vector<int32_t> prev(hp.num_codebooks);
        for (int j = 0; j < hp.num_codebooks; j++) {
            auto samp = ref("step0.depthformer.decoder.depth_samples." + std::to_string(j));
            prev[j] = (int32_t) samp[0];
        }

        mrt::Graph gt = mrt::build_temporal_graph(model, write_pos);
        ggml_gallocr_reserve(allocr, gt.gf);
        ggml_gallocr_alloc_graph(allocr, gt.gf);
        ggml_backend_tensor_set(gt.inputs[0], prev.data(), 0, prev.size() * sizeof(int32_t));
        ggml_backend_tensor_set(gt.inputs[1], enc_out.data(), 0, enc_out.size() * sizeof(float));
        ggml_backend_tensor_set(gt.inputs[2], mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gt.gf);

        std::vector<float> out(ggml_nelements(gt.out));
        ggml_backend_tensor_get(gt.out, out.data(), 0, ggml_nbytes(gt.out));
        compare("step1 temporal_outputs", out, ref("step1.depthformer.decoder.temporal_outputs"));
        mrt::free_graph(gt);
    }

    // ---- codec front end --------------------------------------------------
    const bool have_codec_refs = manifest.count("step0.layers_1.layer") > 0;
    if (!have_codec_refs) {
        printf("\n-- codec: skipped (no codec references in %s) --\n", ref_dir.c_str());
    }
    if (have_codec_refs) {
        printf("\n-- codec input stage --\n");
        const auto & hp = model.hparams;
        // layers_1 emits the non-unique per-level codes.
        auto codes_f = ref("step0.layers_1.layer");
        std::vector<int32_t> codes(hp.num_codebooks);
        for (int i = 0; i < hp.num_codebooks; i++) codes[i] = (int32_t) codes_f[i];

        mrt::Graph gc = mrt::build_codec_input_graph(model);
        ggml_gallocr_alloc_graph(allocr, gc.gf);
        ggml_backend_tensor_set(gc.inputs[0], codes.data(), 0, codes.size() * sizeof(int32_t));
        ggml_backend_graph_compute(backend, gc.gf);

        std::vector<float> got(ggml_nelements(gc.out));
        ggml_backend_tensor_get(gc.out, got.data(), 0, ggml_nbytes(gc.out));

        // Reference is [1,1,5,512] = (frames, bins, channels) with channels
        // contiguous; ggml holds (bins, frames, channels) with bins contiguous.
        auto want_nhwc = ref("step0.soundstream.decoder.input_layers_residual_unit.conv2d_3x3_a.layers_0.layer");
        // that reference is elu(x); undo by comparing elu(got) instead
        std::vector<float> got_t(got.size());
        const int bins = 5, ch = 512;
        for (int b = 0; b < bins; b++)
            for (int c = 0; c < ch; c++)
                got_t[b * ch + c] = got[b + c * bins];
        for (auto & v : got_t) v = v > 0 ? v : std::expm1(v);   // elu
        compare("codec input stage (elu)", got_t, want_nhwc);
        mrt::free_graph(gc);
    }

    // ---- full codec decoder ----------------------------------------------
    if (have_codec_refs && mrt::alloc_codec_state(model)) {
        printf("\n-- codec decoder --\n");
        const auto & hp = model.hparams;
        auto codes_f = ref("step0.layers_1.layer");
        std::vector<int32_t> codes(hp.num_codebooks);
        for (int i = 0; i < hp.num_codebooks; i++) codes[i] = (int32_t) codes_f[i];

        const int stop = getenv("MRT_CODEC_STOP") ? atoi(getenv("MRT_CODEC_STOP")) : -1;
        mrt::Graph gc = mrt::build_codec_graph(model, stop);
        // Dedicated allocator: the codec graph is much larger than the LLM
        // graphs, and sharing an allocator across them risks buffer reuse.
        ggml_gallocr_t calloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(calloc_, gc.gf);
        ggml_gallocr_alloc_graph(calloc_, gc.gf);
        ggml_backend_tensor_set(gc.inputs[0], codes.data(), 0, codes.size() * sizeof(int32_t));
        ggml_backend_graph_compute(backend, gc.gf);

        printf("  output ne = [%lld, %lld, %lld]\n",
               (long long) gc.out->ne[0], (long long) gc.out->ne[1], (long long) gc.out->ne[2]);
        std::vector<float> got(ggml_nelements(gc.out));
        ggml_backend_tensor_get(gc.out, got.data(), 0, ggml_nbytes(gc.out));

        // numpy_codec (validated against JAX to ~7e-5) supplies the pre-ISTFT
        // reference, stored in ggml order (F, T, C).
        if (getenv("MRT_DUMP")) {
            std::ofstream df(getenv("MRT_DUMP"), std::ios::binary);
            df.write(reinterpret_cast<const char *>(got.data()), got.size() * sizeof(float));
            printf("  dumped %zu floats to %s\n", got.size(), getenv("MRT_DUMP"));
        }
        char fn[128];
        snprintf(fn, sizeof(fn), "/codec_stage%d.bin", stop);
        // stop == -1 is the complete decoder, i.e. through the inverse STFT.
        auto want = read_bin(ref_dir + (stop == -1 ? "/codec_frames_frame0.bin" : fn), got.size());
        if (!want.empty()) compare(stop == -1 ? "codec full (incl. ISTFT)" : "codec stage", got, want);
        ggml_gallocr_free(calloc_);
        mrt::free_graph(gc);
    }

    ggml_gallocr_free(allocr);
    mrt::free_model(model);
    ggml_backend_free(backend);
    printf("\n%s\n", g_failures ? "FAILURES" : "OK");
    return g_failures ? 1 : 0;
}
