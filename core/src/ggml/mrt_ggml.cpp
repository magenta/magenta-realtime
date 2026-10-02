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

#include "mrt_ggml.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cmath>

namespace mrt {

namespace {

const char * kArch = "magenta-rt2";

int32_t kv_u32(gguf_context * ctx, const std::string & key, int32_t fallback = -1) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        if (fallback >= 0) {
            return fallback;
        }
        fprintf(stderr, "mrt: missing required metadata key '%s'\n", key.c_str());
        return -1;
    }
    return (int32_t) gguf_get_val_u32(ctx, id);
}

float kv_f32(gguf_context * ctx, const std::string & key, float fallback) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    return id < 0 ? fallback : gguf_get_val_f32(ctx, id);
}

std::string k(const char * suffix) { return std::string(kArch) + "." + suffix; }

}  // namespace

struct ggml_tensor * Model::get(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "mrt: tensor not found: '%s'\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

bool Model::has(const std::string & name) const {
    return tensors.find(name) != tensors.end();
}

bool load_model(const std::string & gguf_path, ggml_backend_t backend, Model & model) {
    struct ggml_context *   ctx_disk = nullptr;
    struct gguf_init_params gp = { /*no_alloc =*/ false, /*ctx =*/ &ctx_disk };

    gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gguf) {
        fprintf(stderr, "mrt: failed to open '%s'\n", gguf_path.c_str());
        return false;
    }

    Hparams & hp = model.hparams;
    hp.temporal_layers = kv_u32(gguf, k("temporal.block_count"));
    hp.temporal_dim    = kv_u32(gguf, k("temporal.embedding_length"));
    hp.temporal_ffn    = kv_u32(gguf, k("temporal.feed_forward_length"));
    hp.temporal_heads  = kv_u32(gguf, k("temporal.attention.head_count"));
    hp.temporal_dh     = kv_u32(gguf, k("temporal.attention.key_length"));
    hp.temporal_max_past_horizon = kv_u32(gguf, k("temporal.attention.max_past_horizon"));
    hp.temporal_cache_slots = hp.temporal_max_past_horizon + 1;
    hp.depth_layers    = kv_u32(gguf, k("depth.block_count"));
    hp.depth_dim       = kv_u32(gguf, k("depth.embedding_length"));
    hp.depth_ffn       = kv_u32(gguf, k("depth.feed_forward_length"));
    hp.depth_heads     = kv_u32(gguf, k("depth.attention.head_count"));
    hp.depth_dh        = kv_u32(gguf, k("depth.attention.key_length"));
    hp.depth_max_past_horizon = kv_u32(gguf, k("depth.attention.max_past_horizon"));
    hp.depth_cache_slots = hp.depth_max_past_horizon + 1;
    hp.encoder_dim     = kv_u32(gguf, k("encoder.embedding_length"));
    hp.num_codebooks   = kv_u32(gguf, k("num_codebooks"));
    hp.codebook_size   = kv_u32(gguf, k("codebook_size"));
    hp.num_reserved_tokens = kv_u32(gguf, k("num_reserved_tokens"));
    hp.sos_id          = kv_u32(gguf, k("sos_id"), 0);
    hp.soft_cap_logits = kv_f32(gguf, k("soft_cap_logits"), 30.0f);

    const int n_tensors = (int) gguf_get_n_tensors(gguf);

    // A context large enough for the tensor structs (data lives in the backend
    // buffer allocated below).
    struct ggml_init_params ip = {
        /*mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 64),
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    model.ctx_w = ggml_init(ip);
    model.backend = backend;

    for (int i = 0; i < n_tensors; i++) {
        const char *         name = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor * src  = ggml_get_tensor(ctx_disk, name);
        struct ggml_tensor * dst  = ggml_dup_tensor(model.ctx_w, src);
        ggml_set_name(dst, name);
        model.tensors[name] = dst;
    }

    model.buf_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, backend);
    if (!model.buf_w) {
        fprintf(stderr, "mrt: failed to allocate %d weight tensors\n", n_tensors);
        return false;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char *         name = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor * src  = ggml_get_tensor(ctx_disk, name);
        ggml_backend_tensor_set(model.tensors[name], src->data, 0, ggml_nbytes(src));
    }

    printf("mrt: loaded %d tensors from %s\n", n_tensors, gguf_path.c_str());
    printf("mrt: temporal %dL d=%d ffn=%d heads=%d dh=%d cache=%d\n",
           hp.temporal_layers, hp.temporal_dim, hp.temporal_ffn,
           hp.temporal_heads, hp.temporal_dh, hp.temporal_cache_slots);
    printf("mrt: depth    %dL d=%d ffn=%d heads=%d dh=%d cache=%d\n",
           hp.depth_layers, hp.depth_dim, hp.depth_ffn,
           hp.depth_heads, hp.depth_dh, hp.depth_cache_slots);
    printf("mrt: encoder d=%d  codebooks=%d codebook_size=%d reserved=%d\n",
           hp.encoder_dim, hp.num_codebooks, hp.codebook_size,
           hp.num_reserved_tokens);

    ggml_free(ctx_disk);
    gguf_free(gguf);
    return true;
}

bool alloc_caches(Model & model) {
    const Hparams & hp = model.hparams;
    const int n_caches = hp.temporal_layers * 2 + hp.depth_layers;

    struct ggml_init_params ip = {
        /*mem_size   =*/ ggml_tensor_overhead() * (n_caches * 2 + 16),
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    model.ctx_kv = ggml_init(ip);

    auto make = [&](std::vector<KVCache> & out, int layers, int heads, int dh, int window) {
        out.resize(layers);
        for (int i = 0; i < layers; i++) {
            // k: [dh, window, heads]; v is stored transposed as [window, dh, heads]
            // so the context can be produced by a single mul_mat.
            out[i].k = ggml_new_tensor_3d(model.ctx_kv, GGML_TYPE_F32, dh, window, heads);
            out[i].v = ggml_new_tensor_3d(model.ctx_kv, GGML_TYPE_F32, window, dh, heads);
        }
    };

    make(model.self_cache,  hp.temporal_layers, hp.temporal_heads, hp.temporal_dh, hp.temporal_cache_slots);
    make(model.cross_cache, hp.temporal_layers, hp.temporal_heads, hp.temporal_dh, hp.temporal_cache_slots);
    make(model.depth_cache, hp.depth_layers,    hp.depth_heads,    hp.depth_dh,    hp.depth_cache_slots);

    model.buf_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
    if (!model.buf_kv) {
        fprintf(stderr, "mrt: failed to allocate KV caches\n");
        return false;
    }
    reset_caches(model);
    return true;
}

void reset_caches(Model & model) {
    auto clear = [&](std::vector<KVCache> & caches) {
        for (auto & c : caches) {
            std::vector<char> zeros(ggml_nbytes(c.k) > ggml_nbytes(c.v) ? ggml_nbytes(c.k)
                                                                        : ggml_nbytes(c.v), 0);
            ggml_backend_tensor_set(c.k, zeros.data(), 0, ggml_nbytes(c.k));
            ggml_backend_tensor_set(c.v, zeros.data(), 0, ggml_nbytes(c.v));
            c.filled = 0;
            c.pos = 0;
        }
    };
    clear(model.self_cache);
    clear(model.cross_cache);
    clear(model.depth_cache);
}

namespace {

struct ggml_context * new_graph_ctx(size_t nodes = 8192) {
    struct ggml_init_params ip = {
        /*mem_size   =*/ ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false),
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    return ggml_init(ip);
}

// Sum over ne1 (the "rows" axis), returning [ne0]. ggml_sum_rows reduces ne0,
// so transpose first.
struct ggml_tensor * sum_over_rows(struct ggml_context * ctx, struct ggml_tensor * x) {
    struct ggml_tensor * t = ggml_cont(ctx, ggml_transpose(ctx, x));  // [n1, n0]
    return ggml_sum_rows(ctx, t);                                     // [1, n0]
}

// LayerNorm with learned scale and bias.
struct ggml_tensor * layer_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                struct ggml_tensor * scale, struct ggml_tensor * bias,
                                float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, scale);
    return ggml_add(ctx, x, bias);
}

}  // namespace

Graph build_encoder_graph(const Model & model, int n_mulan, int n_regular) {
    Graph g;
    g.ctx = new_graph_ctx();
    struct ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    struct ggml_tensor * mulan_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_mulan);
    struct ggml_tensor * reg_idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_regular);
    ggml_set_name(mulan_idx, "mulan_idx");
    ggml_set_name(reg_idx, "reg_idx");
    g.inputs = { mulan_idx, reg_idx };

    // Branch A: dequantize MusicCoCa tokens, sum over the RVQ axis, project.
    struct ggml_tensor * a = ggml_get_rows(ctx, model.get("enc.mulan.emb"), mulan_idx);
    a = sum_over_rows(ctx, a);                                   // [1, 768]
    a = ggml_reshape_1d(ctx, a, a->ne[1]);                       // [768]
    a = ggml_mul_mat(ctx, model.get("enc.mulan.adapter"), a);    // [256]

    // Branch B: per-channel embedding lookup, mean over channels.
    struct ggml_tensor * b = ggml_get_rows(ctx, model.get("enc.reg.emb"), reg_idx);
    b = sum_over_rows(ctx, b);                                   // [1, 256]
    b = ggml_reshape_1d(ctx, b, b->ne[1]);                       // [256]
    b = ggml_scale(ctx, b, 1.0f / (float) n_regular);

    // sl.CombinationMode.MEAN over the two branches.
    struct ggml_tensor * y = ggml_scale(ctx, ggml_add(ctx, a, b), 0.5f);
    y = layer_norm(ctx, y, model.get("enc.ln.scale"), model.get("enc.ln.bias"), 1e-6f);

    g.out = y;
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}


namespace {

constexpr float kEps = 1e-6f;
constexpr float kRSoftplus0 = 1.442695041f;

struct ggml_tensor * rms_scale(struct ggml_context * ctx, struct ggml_tensor * x,
                               struct ggml_tensor * scale) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, kEps), scale);
}

// Praxis query scaling: r_softplus_0 * (1/sqrt(dh)) * softplus(per_dim_scale).
struct ggml_tensor * query_scale_vec(struct ggml_context * ctx,
                                     struct ggml_tensor * per_dim_scale, int dh) {
    struct ggml_tensor * sp = ggml_softplus(ctx, per_dim_scale);
    return ggml_scale(ctx, sp, kRSoftplus0 / sqrtf((float) dh));
}

// One attention sublayer: pre-norm, attend over the cache, output projection,
// post-norm, residual. `kv_src` supplies K/V (the encoder output for cross
// attention); when null, K/V come from the pre-normed input (self attention).
struct ggml_tensor * attention_block(struct ggml_context * ctx, const Model & m,
                                     const std::string & p, struct ggml_tensor * x,
                                     struct ggml_tensor * kv_src, const KVCache & cache,
                                     struct ggml_tensor * mask, int heads, int dh,
                                     int window, int write_pos,
                                     struct ggml_cgraph * gf) {
    struct ggml_tensor * pre = rms_scale(ctx, x, m.get(p + ".pre"));
    struct ggml_tensor * ksrc = kv_src ? kv_src : pre;

    struct ggml_tensor * q = ggml_mul_mat(ctx, m.get(p + ".q"), pre);
    struct ggml_tensor * k = ggml_mul_mat(ctx, m.get(p + ".k"), ksrc);
    struct ggml_tensor * v = ggml_mul_mat(ctx, m.get(p + ".v"), ksrc);

    // [h*dh] -> [dh, 1, heads]
    q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, dh, heads, 1), 0, 2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, dh, heads, 1), 0, 2, 1, 3);
    v = ggml_permute(ctx, ggml_reshape_3d(ctx, v, dh, heads, 1), 0, 2, 1, 3);
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Write K/V into the circular cache at `write_pos`.
    struct ggml_tensor * k_dst = ggml_view_3d(ctx, cache.k, dh, 1, heads,
                                              cache.k->nb[1], cache.k->nb[2],
                                              (size_t) write_pos * cache.k->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_dst));

    // v cache is [window, dh, heads], so one step writes a strided column.
    struct ggml_tensor * v_row = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [1, dh, heads]
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, cache.v, 1, dh, heads,
                                              cache.v->nb[1], cache.v->nb[2],
                                              (size_t) write_pos * cache.v->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_row, v_dst));

    // Logits over the cache use the scaled query...
    struct ggml_tensor * qs = ggml_mul(ctx, q, query_scale_vec(ctx, m.get(p + ".pds"), dh));
    struct ggml_tensor * logits = ggml_mul_mat(ctx, cache.k, qs);   // [window, 1, heads]

    // ...while sink logits deliberately use the *unscaled* query. The depth
    // transformer has no sinks, so this is conditional.
    const bool has_sink = m.has(p + ".sink_k");
    struct ggml_tensor * sink_logit = nullptr;
    if (has_sink) {
        struct ggml_tensor * sink_k = ggml_reshape_3d(ctx, m.get(p + ".sink_k"), dh, 1, heads);
        sink_logit = ggml_mul_mat(ctx, sink_k, q);                    // [1, 1, heads]
    }

    struct ggml_tensor * attn = nullptr;
    if (has_sink) {
        struct ggml_tensor * all = ggml_concat(ctx, sink_logit, logits, 0);  // [window+1,1,heads]
        all = ggml_add(ctx, all, mask);
        struct ggml_tensor * probs = ggml_soft_max(ctx, all);

        struct ggml_tensor * p_sink = ggml_view_3d(ctx, probs, 1, 1, heads,
                                                   probs->nb[1], probs->nb[2], 0);
        struct ggml_tensor * p_real = ggml_cont(ctx,
            ggml_view_3d(ctx, probs, window, 1, heads, probs->nb[1], probs->nb[2],
                         probs->nb[0]));

        struct ggml_tensor * sink_v = ggml_reshape_3d(ctx, m.get(p + ".sink_v"), dh, 1, heads);
        attn = ggml_mul_mat(ctx, cache.v, p_real);                    // [dh, 1, heads]
        attn = ggml_add(ctx, attn, ggml_mul(ctx, sink_v, p_sink));
    } else {
        struct ggml_tensor * probs = ggml_soft_max(ctx, ggml_add(ctx, logits, mask));
        attn = ggml_mul_mat(ctx, cache.v, probs);                     // [dh, 1, heads]
    }

    // [dh, 1, heads] -> [h*dh]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));       // [dh, heads, 1]
    attn = ggml_reshape_1d(ctx, attn, (int64_t) heads * dh);

    struct ggml_tensor * o = ggml_mul_mat(ctx, m.get(p + ".o"), attn);
    return ggml_add(ctx, x, rms_scale(ctx, o, m.get(p + ".post")));
}

struct ggml_tensor * ffn_block(struct ggml_context * ctx, const Model & m,
                               const std::string & p, struct ggml_tensor * x) {
    struct ggml_tensor * h = rms_scale(ctx, x, m.get(p + ".pre"));
    h = ggml_add(ctx, ggml_mul_mat(ctx, m.get(p + ".w1"), h), m.get(p + ".b1"));
    h = ggml_gelu(ctx, h);   // tanh approximation, matching jax.nn.gelu
    h = ggml_add(ctx, ggml_mul_mat(ctx, m.get(p + ".w2"), h), m.get(p + ".b2"));
    return ggml_add(ctx, x, rms_scale(ctx, h, m.get(p + ".post")));
}

}  // namespace

Graph build_temporal_graph(const Model & model, int write_pos) {
    const Hparams & hp = model.hparams;
    Graph g;
    g.ctx = new_graph_ctx(16384);
    struct ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, 16384, false);

    struct ggml_tensor * prev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.num_codebooks);
    struct ggml_tensor * enc  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.encoder_dim);
    struct ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                   hp.temporal_cache_slots + 1, 1, 1);
    ggml_set_name(prev, "prev_tokens");
    ggml_set_name(enc, "enc_out");
    ggml_set_name(mask, "mask");
    g.inputs = { prev, enc, mask };

    // Shared decoder embedding, scaled by sqrt(D), averaged over the RVQ axis.
    struct ggml_tensor * e = ggml_get_rows(ctx, model.get("dec.emb"), prev);  // [D, Q]
    e = sum_over_rows(ctx, e);                                                // [1, D]
    e = ggml_reshape_1d(ctx, e, e->ne[1]);
    struct ggml_tensor * x = ggml_scale(
        ctx, e, sqrtf((float) hp.temporal_dim) / (float) hp.num_codebooks);

    for (int i = 0; i < hp.temporal_layers; i++) {
        const std::string b = "t." + std::to_string(i);
        x = attention_block(ctx, model, b + ".sa", x, nullptr, model.self_cache[i],
                            mask, hp.temporal_heads, hp.temporal_dh,
                            hp.temporal_cache_slots, write_pos, g.gf);
        x = attention_block(ctx, model, b + ".ca", x, enc, model.cross_cache[i],
                            mask, hp.temporal_heads, hp.temporal_dh,
                            hp.temporal_cache_slots, write_pos, g.gf);
        x = ffn_block(ctx, model, b + ".ffn", x);
    }

    g.out = x;
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}


Graph build_depth_graph(const Model & model, int write_pos) {
    const Hparams & hp = model.hparams;
    Graph g;
    g.ctx = new_graph_ctx(8192);
    struct ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.temporal_dim);
    struct ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.depth_cache_slots, 1, 1);
    ggml_set_name(in, "depth_in");
    ggml_set_name(mask, "depth_mask");
    g.inputs = { in, mask };

    struct ggml_tensor * x = ggml_mul_mat(ctx, model.get("d.adapter"), in);  // [768]
    for (int i = 0; i < hp.depth_layers; i++) {
        const std::string b = "d." + std::to_string(i);
        x = attention_block(ctx, model, b + ".sa", x, nullptr, model.depth_cache[i],
                            mask, hp.depth_heads, hp.depth_dh, hp.depth_cache_slots,
                            write_pos, g.gf);
        x = ffn_block(ctx, model, b + ".ffn", x);
    }
    x = layer_norm(ctx, x, model.get("d.ln.scale"), model.get("d.ln.bias"), kEps);
    x = ggml_add(ctx, ggml_mul_mat(ctx, model.get("d.logits.w"), x), model.get("d.logits.b"));

    g.out = x;
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}


namespace {

// Zero-insertion upsampling: spreads x along `dim` by `rate`, filling the new
// slots with zeros (x lands on multiples of `rate`). ggml has no interleave op,
// so this concatenates zeros on a spare axis and reshapes.
struct ggml_tensor * zero_upsample(struct ggml_context * ctx, struct ggml_tensor * x,
                                   int dim, int rate) {
    if (rate == 1) return x;
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];
    struct ggml_tensor * y;
    if (dim == 1) {                       // time
        y = ggml_reshape_4d(ctx, x, F, 1, T, C);
        struct ggml_tensor * z = ggml_scale(ctx, y, 0.0f);
        struct ggml_tensor * acc = y;
        for (int i = 1; i < rate; i++) acc = ggml_concat(ctx, acc, z, 1);
        return ggml_reshape_3d(ctx, ggml_cont(ctx, acc), F, T * rate, C);
    }
    y = ggml_reshape_4d(ctx, x, 1, F, T, C);   // frequency
    struct ggml_tensor * z = ggml_scale(ctx, y, 0.0f);
    struct ggml_tensor * acc = y;
    for (int i = 1; i < rate; i++) acc = ggml_concat(ctx, acc, z, 0);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, acc), F * rate, T, C);
}

// Causal conv over time: history supplies the (kh-1) left context frames that a
// non-streaming implementation would zero-pad. Frequency padding is symmetric.
struct ggml_tensor * causal_conv2d(struct ggml_context * ctx,
                                   std::vector<struct ggml_tensor *> & updates,
                                   struct ggml_tensor * w, struct ggml_tensor * b,
                                   struct ggml_tensor * x, struct ggml_tensor * hist,
                                   int s_freq, int s_time, int p_freq) {
    struct ggml_tensor * in = x;
    if (hist) {
        in = ggml_concat(ctx, hist, x, 1);
        // Save the tail for the next frame. The copy is deferred to the end of
        // the graph: expanding it inline interleaves a write to `hist` with
        // reads of the concat that depends on it.
        const int64_t keep = hist->ne[1];
        struct ggml_tensor * tail = ggml_view_3d(
            ctx, in, in->ne[0], keep, in->ne[2], in->nb[1], in->nb[2],
            (size_t) (in->ne[1] - keep) * in->nb[1]);
        updates.push_back(ggml_cpy(ctx, ggml_cont(ctx, tail), hist));
    }
    struct ggml_tensor * y = ggml_conv_2d(ctx, w, in, s_freq, s_time, p_freq, 0, 1, 1);
    if (b) y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]));
    return y;
}

}  // namespace

namespace {

// One decoder residual block: name, (time stride, freq stride).
struct BlockSpec { const char * name; int st; int sf; };

const BlockSpec kBlocks[] = {
    {"decoder_0", 2, 1}, {"decoder_1", 2, 2}, {"decoder_2", 1, 2},
    {"decoder_3", 1, 2}, {"decoder_4", 1, 3}, {"decoder_5", 1, 2},
    {"decoder_6", 1, 2},
};
constexpr int kNumBlocks = 7;
constexpr int kSplitAfter = 0;      // channel groups split after decoder_0
constexpr int kGroups = 2;
constexpr int kInputBins = 5;
constexpr int kBaseChannels = 512;

int kern_t(int st) { return st > 1 ? 2 * st : 3; }
int kern_f(int sf) { return sf > 1 ? 2 * sf : 3; }

// Frequency padding for a transposed conv expressed as upsample + conv.
// Left/right differ when (kernel - stride) is odd.
void transpose_freq_pad(int kf, int sf, int * pl, int * pr) {
    const int start = (kf - sf) / 2;
    *pl = (kf - 1) - start;
    *pr = (kf - 1) - *pl;
}

// Walks the decoder, reporting each causal conv's history shape as
// (bins, frames, channels). Order must match build_codec_graph exactly: the
// channel groups run one after the other, not interleaved per block.
template <typename F>
void walk_codec(F && visit) {
    static const int kOutCh[kNumBlocks] = {1024, 256, 256, 256, 128, 128, 64};

    // input_layers_residual_unit: two stride-1 3x3 convs.
    visit(kInputBins, 2, kBaseChannels);
    visit(kInputBins, 2, kBaseChannels);

    // decoder_0 runs on the full channel width, before the split.
    {
        const BlockSpec & s = kBlocks[0];
        const int out_bins = kInputBins * s.sf;
        visit(out_bins, kern_t(s.st) - 1, kBaseChannels);   // transposed conv
        visit(out_bins, 2, kOutCh[0]);                      // conv2d_3x3
    }

    const int bins0 = kInputBins * kBlocks[0].sf;
    for (int g = 0; g < kGroups; g++) {
        int bins = bins0;
        int ch = kOutCh[0] / kGroups;                       // channels after split
        for (int b = 1; b < kNumBlocks; b++) {
            const BlockSpec & s = kBlocks[b];
            const int out_bins = bins * s.sf;
            visit(out_bins, kern_t(s.st) - 1, ch);          // transposed conv
            visit(out_bins, 2, kOutCh[b]);                  // conv2d_3x3
            bins = out_bins;
            ch = kOutCh[b];
        }
    }
    // base_conv_last (7x7) per group, on the final geometry.
    int fbins = bins0;
    for (int b = 1; b < kNumBlocks; b++) fbins *= kBlocks[b].sf;
    for (int g = 0; g < kGroups; g++) visit(fbins, 6, kOutCh[kNumBlocks - 1]);
}

}  // namespace

bool alloc_codec_state(Model & model) {
    int n = 0;
    walk_codec([&](int, int, int) { n++; });

    struct ggml_init_params ip = {
        /*mem_size   =*/ ggml_tensor_overhead() * (n + 16),
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    model.ctx_codec = ggml_init(ip);
    model.codec_hist.clear();
    walk_codec([&](int bins, int frames, int ch) {
        model.codec_hist.push_back(
            ggml_new_tensor_3d(model.ctx_codec, GGML_TYPE_F32, bins, frames, ch));
    });

    const int n_fft = 960, n_bins = n_fft / 2 + 1;
    model.idft_re = ggml_new_tensor_2d(model.ctx_codec, GGML_TYPE_F32, n_bins, n_fft);
    model.idft_im = ggml_new_tensor_2d(model.ctx_codec, GGML_TYPE_F32, n_bins, n_fft);
    model.istft_window = ggml_new_tensor_1d(model.ctx_codec, GGML_TYPE_F32, n_fft);

    model.buf_codec = ggml_backend_alloc_ctx_tensors(model.ctx_codec, model.backend);
    if (!model.buf_codec) {
        fprintf(stderr, "mrt: failed to allocate codec state\n");
        return false;
    }
    for (auto * t : model.codec_hist) {
        std::vector<char> z(ggml_nbytes(t), 0);
        ggml_backend_tensor_set(t, z.data(), 0, ggml_nbytes(t));
    }
    // Real IDFT: x[n] = (1/N) * ( X0 + 2*sum_{k=1}^{N/2-1}(Re_k cos - Im_k sin)
    //                              + X_{N/2} cos(pi n) )
    {
        std::vector<float> re((size_t) n_bins * n_fft), im((size_t) n_bins * n_fft);
        for (int nn = 0; nn < n_fft; nn++) {
            for (int k = 0; k < n_bins; k++) {
                const double ang = 2.0 * M_PI * k * nn / n_fft;
                const double scale = (k == 0 || k == n_fft / 2) ? 1.0 : 2.0;
                re[(size_t) nn * n_bins + k] = (float) (scale * cos(ang) / n_fft);
                im[(size_t) nn * n_bins + k] = (float) (-scale * sin(ang) / n_fft);
            }
        }
        ggml_backend_tensor_set(model.idft_re, re.data(), 0, re.size() * sizeof(float));
        ggml_backend_tensor_set(model.idft_im, im.data(), 0, im.size() * sizeof(float));

        // Griffin & Lim eq. 7 inverse window (periodic Hann / summed overlaps).
        const int hop = 480;
        std::vector<float> fw(n_fft), den(hop, 0.0f), win(n_fft);
        for (int i = 0; i < n_fft; i++) fw[i] = 0.5f - 0.5f * cosf(2.0f * (float) M_PI * i / n_fft);
        for (int i = 0; i < n_fft; i++) den[i % hop] += fw[i] * fw[i];
        for (int i = 0; i < n_fft; i++) win[i] = fw[i] / (den[i % hop] > 1e-10f ? den[i % hop] : 1.0f);
        ggml_backend_tensor_set(model.istft_window, win.data(), 0, win.size() * sizeof(float));
    }

    printf("mrt: codec state = %d history buffers\n", n);
    return true;
}


Graph build_codec_input_graph(const Model & model) {
    const int n_codebooks = model.hparams.num_codebooks;
    const int bins = 5, channels = 512, feat_dim = 256;

    Graph g;
    g.ctx = new_graph_ctx(4096);
    struct ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor * codes = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_codebooks);
    ggml_set_name(codes, "codes");
    g.inputs = { codes };

    // Dequantize: sum the per-level codebook embeddings.
    // ss.quant.emb has ne = (dim, entries, quantizers).
    struct ggml_tensor * qemb = model.get("ss.quant.emb");
    struct ggml_tensor * feat = nullptr;
    for (int j = 0; j < n_codebooks; j++) {
        struct ggml_tensor * book = ggml_view_2d(ctx, qemb, qemb->ne[0], qemb->ne[1],
                                                 qemb->nb[1], (size_t) j * qemb->nb[2]);
        struct ggml_tensor * idx = ggml_view_1d(ctx, codes, 1, (size_t) j * codes->nb[0]);
        struct ggml_tensor * e = ggml_get_rows(ctx, book, idx);      // [dim, 1]
        e = ggml_reshape_1d(ctx, e, feat_dim);
        feat = feat ? ggml_add(ctx, feat, e) : e;
    }

    // 1x1 convs are plain matmuls; the stored kernels have ne = (1, 1, IC, OC).
    auto conv1x1 = [&](const char * name, struct ggml_tensor * x, int ic, int oc) {
        struct ggml_tensor * w = ggml_reshape_2d(ctx, model.get(std::string(name) + ".w"), ic, oc);
        return ggml_add(ctx, ggml_mul_mat(ctx, w, x), model.get(std::string(name) + ".b"));
    };

    const int proj = bins * channels;                                // 2560
    struct ggml_tensor * main = conv1x1("ss.input_layer.conv1x1_first", feat, feat_dim, proj);
    struct ggml_tensor * sc = conv1x1("ss.input_layer.shortcut_layer.conv1x1_b1", feat, feat_dim, proj);
    sc = ggml_elu(ctx, sc);
    sc = conv1x1("ss.input_layer.shortcut_layer.conv1x1_b2", sc, proj, proj);

    struct ggml_tensor * x = ggml_add(ctx, main, sc);                // [2560]

    // Reshape([bins, channels]) is bin-major with channels contiguous, so in
    // ggml that is ne = (channels, bins); transpose to (bins, channels) and
    // insert the frame axis to reach (bins, frames=1, channels).
    x = ggml_reshape_2d(ctx, x, channels, bins);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                      // (bins, channels)
    x = ggml_reshape_3d(ctx, x, bins, 1, channels);

    g.out = x;
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

namespace {

// Nearest-neighbour repeat along time (dim 1) / frequency (dim 0).
struct ggml_tensor * nn_upsample(struct ggml_context * ctx, struct ggml_tensor * x,
                                 int st, int sf) {
    const int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];
    if (st > 1) {
        struct ggml_tensor * y = ggml_reshape_4d(ctx, x, F, 1, T, C);
        struct ggml_tensor * acc = y;
        for (int i = 1; i < st; i++) acc = ggml_concat(ctx, acc, y, 1);
        x = ggml_reshape_3d(ctx, ggml_cont(ctx, acc), F, T * st, C);
    }
    if (sf > 1) {
        const int64_t T2 = x->ne[1];
        struct ggml_tensor * y = ggml_reshape_4d(ctx, x, 1, F, T2, C);
        struct ggml_tensor * acc = y;
        for (int i = 1; i < sf; i++) acc = ggml_concat(ctx, acc, y, 0);
        x = ggml_reshape_3d(ctx, ggml_cont(ctx, acc), F * sf, T2, C);
    }
    return x;
}

// Transposed conv as: zero-upsample -> causal conv with the *original* kernel.
// (JAX's conv_transpose flips the kernel; the flip cancels in this form.)
struct ggml_tensor * transposed_conv(struct ggml_context * ctx,
                                     std::vector<struct ggml_tensor *> & updates,
                                     const Model & m, const std::string & p,
                                     struct ggml_tensor * x, struct ggml_tensor * hist,
                                     int st, int sf, int kt, int kf) {
    struct ggml_tensor * up = zero_upsample(ctx, x, 1, st);
    up = zero_upsample(ctx, up, 0, sf);
    const int64_t out_f = up->ne[0];

    int pl, pr;
    transpose_freq_pad(kf, sf, &pl, &pr);
    const int P = pl > pr ? pl : pr;

    struct ggml_tensor * y = causal_conv2d(ctx, updates, m.get(p + ".w"), m.get(p + ".b"),
                                           up, hist, /*s_freq=*/1, /*s_time=*/1, P);
    if (P != pl) {   // asymmetric: shift the window (decoder_4 only)
        y = ggml_cont(ctx, ggml_view_3d(ctx, y, out_f, y->ne[1], y->ne[2],
                                        y->nb[1], y->nb[2], (size_t) (P - pl) * y->nb[0]));
    } else if (y->ne[0] != out_f) {
        y = ggml_cont(ctx, ggml_view_3d(ctx, y, out_f, y->ne[1], y->ne[2],
                                        y->nb[1], y->nb[2], 0));
    }
    return y;
}

}  // namespace

Graph build_codec_graph(const Model & model, int stop_after) {
    Graph g;
    g.ctx = new_graph_ctx(32768);
    struct ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, 32768, false);

    const int n_codebooks = model.hparams.num_codebooks;
    const int bins = kInputBins, channels = kBaseChannels, feat_dim = 256;

    struct ggml_tensor * codes = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_codebooks);
    ggml_set_name(codes, "codes");
    g.inputs = { codes };

    struct ggml_tensor * qemb = model.get("ss.quant.emb");
    struct ggml_tensor * feat = nullptr;
    for (int j = 0; j < n_codebooks; j++) {
        struct ggml_tensor * book = ggml_view_2d(ctx, qemb, qemb->ne[0], qemb->ne[1],
                                                 qemb->nb[1], (size_t) j * qemb->nb[2]);
        struct ggml_tensor * idx = ggml_view_1d(ctx, codes, 1, (size_t) j * codes->nb[0]);
        struct ggml_tensor * e = ggml_reshape_1d(ctx, ggml_get_rows(ctx, book, idx), feat_dim);
        feat = feat ? ggml_add(ctx, feat, e) : e;
    }

    auto conv1x1 = [&](const std::string & name, struct ggml_tensor * x, int ic, int oc) {
        struct ggml_tensor * w = ggml_reshape_2d(ctx, model.get(name + ".w"), ic, oc);
        return ggml_add(ctx, ggml_mul_mat(ctx, w, x), model.get(name + ".b"));
    };

    const int proj = bins * channels;
    struct ggml_tensor * main = conv1x1("ss.input_layer.conv1x1_first", feat, feat_dim, proj);
    struct ggml_tensor * sc0 = conv1x1("ss.input_layer.shortcut_layer.conv1x1_b1", feat, feat_dim, proj);
    sc0 = conv1x1("ss.input_layer.shortcut_layer.conv1x1_b2", ggml_elu(ctx, sc0), proj, proj);
    struct ggml_tensor * x = ggml_add(ctx, main, sc0);
    x = ggml_reshape_2d(ctx, x, channels, bins);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_reshape_3d(ctx, x, bins, 1, channels);

    if (stop_after == -3) { g.out = x; ggml_build_forward_expand(g.gf, g.out); return g; }

    std::vector<struct ggml_tensor *> updates;   // deferred history writes
    int hi = 0;   // history index, must follow walk_codec's order
    auto hist = [&]() { return model.codec_hist[hi++]; };

    // input_layers_residual_unit (stride 1, transposed=true -> 3x3_a then 3x3)
    {
        const std::string p = "ss.input_layers_residual_unit.";
        struct ggml_tensor * mn = causal_conv2d(ctx, updates, model.get(p + "conv2d_3x3_a.w"),
                                                model.get(p + "conv2d_3x3_a.b"),
                                                ggml_elu(ctx, x), hist(), 1, 1, 1);
        if (stop_after == -2) { g.out = mn; ggml_build_forward_expand(g.gf, g.out); return g; }
        mn = causal_conv2d(ctx, updates, model.get(p + "conv2d_3x3.w"), model.get(p + "conv2d_3x3.b"),
                           ggml_elu(ctx, mn), hist(), 1, 1, 1);
        x = ggml_add(ctx, mn, x);
    }
    if (stop_after == 0) { g.out = x; ggml_build_forward_expand(g.gf, g.out); return g; }

    auto run_block = [&](int b, struct ggml_tensor * in) {
        const BlockSpec & s = kBlocks[b];
        const std::string p = std::string("ss.") + s.name + ".";
        const int kt = kern_t(s.st), kf = kern_f(s.sf);

        struct ggml_tensor * mn = ggml_elu(ctx, in);
        if (s.st == 1 && s.sf == 1) {
            mn = causal_conv2d(ctx, updates, model.get(p + "conv2d_3x3_a.w"),
                               model.get(p + "conv2d_3x3_a.b"), mn, hist(), 1, 1, 1);
        } else {
            char tn[64];
            snprintf(tn, sizeof(tn), "conv2dtranspose_%dx%d", kt, kf);
            mn = transposed_conv(ctx, updates, model, p + tn, mn, hist(), s.st, s.sf, kt, kf);
        }
        mn = causal_conv2d(ctx, updates, model.get(p + "conv2d_3x3.w"), model.get(p + "conv2d_3x3.b"),
                           ggml_elu(ctx, mn), hist(), 1, 1, 1);

        struct ggml_tensor * sc = in;
        if (model.has(p + "shortcut_layer.conv1x1.w")) {
            struct ggml_tensor * w = model.get(p + "shortcut_layer.conv1x1.w");
            sc = ggml_conv_2d(ctx, w, sc, 1, 1, 0, 0, 1, 1);
            struct ggml_tensor * b = model.get(p + "shortcut_layer.conv1x1.b");
            sc = ggml_add(ctx, sc, ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]));
        }
        if (s.st != 1 || s.sf != 1) sc = nn_upsample(ctx, sc, s.st, s.sf);
        return ggml_add(ctx, mn, sc);
    };

    x = run_block(0, x);                                  // decoder_0 -> 1024 ch
    if (stop_after == 1) { g.out = x; ggml_build_forward_expand(g.gf, g.out); return g; }

    // Channel groups share weights; run the same stack on each half.
    std::vector<struct ggml_tensor *> outs;
    const int64_t half = x->ne[2] / kGroups;
    for (int gi = 0; gi < kGroups; gi++) {
        struct ggml_tensor * xg = ggml_cont(ctx, ggml_view_3d(
            ctx, x, x->ne[0], x->ne[1], half, x->nb[1], x->nb[2],
            (size_t) gi * half * x->nb[2]));
        for (int b = 1; b < kNumBlocks; b++) {
            xg = run_block(b, xg);
            // Diagnostic stop: return group 0 only, so the history walk stays
            // in sync (breaking mid-loop would desync group 1's indices).
            if (stop_after == 20 + gi * 10 + b) {
                g.out = xg; ggml_build_forward_expand(g.gf, g.out);
                for (auto * u : updates) ggml_build_forward_expand(g.gf, u);
                return g;
            }
            if (gi == 0 && stop_after == b + 1) {
                g.out = xg; ggml_build_forward_expand(g.gf, g.out);
                for (auto * u : updates) ggml_build_forward_expand(g.gf, u);
                return g;
            }
        }
        outs.push_back(xg);
    }
    // base_conv_last needs its own history per group, allocated after the blocks.
    for (int gi = 0; gi < kGroups; gi++) {
        outs[gi] = causal_conv2d(ctx, updates, model.get("ss.output_layer.base_conv_last.w"),
                                 model.get("ss.output_layer.base_conv_last.b"),
                                 ggml_elu(ctx, outs[gi]), hist(), 1, 1, 3);
    }
    x = ggml_concat(ctx, outs[0], outs[1], 2);            // -> [480, 4, 4]

    if (stop_after == -1 && model.idft_re) {
        // Inverse STFT. Channels are interleaved (re0, im0, re1, im1); split the
        // real and imaginary parts so the transform is two matmuls per channel
        // instead of an interleave ggml cannot express cheaply.
        std::vector<struct ggml_tensor *> chans;
        for (int c = 0; c < 2; c++) {
            auto part = [&](int idx) {
                struct ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(
                    ctx, x, x->ne[0], x->ne[1], x->nb[1], (size_t) idx * x->nb[2]));
                return ggml_pad(ctx, v, 1, 0, 0, 0);        // 480 -> 481 bins
            };
            struct ggml_tensor * fr = ggml_add(
                ctx, ggml_mul_mat(ctx, model.idft_re, part(2 * c)),
                     ggml_mul_mat(ctx, model.idft_im, part(2 * c + 1)));   // [960, frames]
            fr = ggml_mul(ctx, fr, ggml_reshape_2d(ctx, model.istft_window, 960, 1));
            chans.push_back(fr);
        }
        x = ggml_concat(ctx, chans[0], chans[1], 2);        // [960, frames, 2]
    }

    g.out = x;
    ggml_build_forward_expand(g.gf, g.out);
    for (auto * u : updates) ggml_build_forward_expand(g.gf, u);
    return g;
}


void free_graph(Graph & g) {
    if (g.ctx) ggml_free(g.ctx);
    g.ctx = nullptr; g.gf = nullptr; g.out = nullptr; g.inputs.clear();
}

void free_model(Model & model) {
    if (model.buf_codec) ggml_backend_buffer_free(model.buf_codec);
    if (model.ctx_codec) ggml_free(model.ctx_codec);
    model.buf_codec = nullptr; model.ctx_codec = nullptr;
    if (model.buf_w)  ggml_backend_buffer_free(model.buf_w);
    if (model.buf_kv) ggml_backend_buffer_free(model.buf_kv);
    if (model.ctx_w)  ggml_free(model.ctx_w);
    if (model.ctx_kv) ggml_free(model.ctx_kv);
    model.buf_w = nullptr; model.buf_kv = nullptr;
    model.ctx_w = nullptr; model.ctx_kv = nullptr;
    model.tensors.clear();
}

}  // namespace mrt
