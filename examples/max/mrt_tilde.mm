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

// mrt2~ — MaxMSP signal external for the realtime music model in this repo.
// Mirrors the AU plugin's parameter set as Max messages; no prompt surface — each
// of the 6 prompt slots is set independently via `prompt N "text" weight`.

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"

// Max headers leak a few names that clash with C++ STL when present in the same
// TU. Undef the worst offenders before pulling in the engine headers.
#ifdef error
#undef error
#endif
#ifdef post
// keep post — we use it
#endif

#include <magentart/realtime_runner.h>
#include "../common/cpp/magenta_paths.h"

#include <array>
#include <iostream>
#include <new>
#include <streambuf>
#include <string>
#include <vector>

namespace {

using magentart::core::RealtimeRunner;
using magentart::core::kMaxPrompts;

class MaxPostStreamBuf : public std::streambuf {
public:
    MaxPostStreamBuf(const std::string& prefix) : prefix_(prefix) {}
protected:
    virtual int overflow(int c) override {
        if (c == EOF) return EOF;
        if (c == '\n') {
            post("%s%s", prefix_.c_str(), buffer_.c_str());
            buffer_.clear();
        } else {
            buffer_ += static_cast<char>(c);
        }
        return c;
    }
private:
    std::string buffer_;
    std::string prefix_;
};

constexpr int kNumPromptSlots = static_cast<int>(kMaxPrompts);  // 6
constexpr double kEngineSampleRate = 48000.0;

struct t_mrt {
    t_pxobject ob;
    RealtimeRunner* engine;
    std::array<std::string, kNumPromptSlots>* prompt_text;
    std::array<float, kNumPromptSlots>*       prompt_weight;
    std::vector<float>* bufL;
    std::vector<float>* bufR;
    bool sr_warned;
    bool assets_loaded;
    bool model_loaded;
};

t_class* s_mrt_class = nullptr;

void mrt_resync_prompts(t_mrt* x) {
    // Send all prompt texts and weights to the engine (including zero-weight
    // slots — PR #280 made weights the source of truth; zero is valid).
    std::vector<std::string> texts;
    std::vector<float> weights;
    texts.reserve(kNumPromptSlots);
    weights.reserve(kNumPromptSlots);
    for (int i = 0; i < kNumPromptSlots; ++i) {
        texts.push_back((*x->prompt_text)[i]);
        weights.push_back((*x->prompt_weight)[i]);
    }
    x->engine->set_text_prompts(texts, weights);

    // Also set the blend weights so the inference loop picks them up.
    for (int i = 0; i < kNumPromptSlots; ++i) {
        x->engine->set_blend_weight(i, (*x->prompt_weight)[i]);
    }
}

void* mrt_new(t_symbol* s, long argc, t_atom* argv) {
    t_mrt* x = static_cast<t_mrt*>(object_alloc(s_mrt_class));
    if (!x) return nullptr;

    dsp_setup(reinterpret_cast<t_pxobject*>(x), 0);  // 0 audio inlets
    outlet_new(x, "signal");                          // R outlet (right)
    outlet_new(x, "signal");                          // L outlet (left)
    // Max wires outlets in reverse declaration order, so the first declared is
    // outs[1] and the second is outs[0]. We swap during perform64 below.

    x->engine = new RealtimeRunner();
    x->prompt_text = new std::array<std::string, kNumPromptSlots>();
    x->prompt_weight = new std::array<float, kNumPromptSlots>();
    x->bufL = new std::vector<float>();
    x->bufR = new std::vector<float>();
    x->sr_warned = false;
    x->assets_loaded = false;
    x->model_loaded = false;

    for (int i = 0; i < kNumPromptSlots; ++i) {
        (*x->prompt_text)[i].clear();
        (*x->prompt_weight)[i] = 0.0f;
    }

    @autoreleasepool {
        MaxPostStreamBuf cout_buf("mrt2~: ");
        MaxPostStreamBuf cerr_buf("mrt2~: ERROR: ");
        std::streambuf* old_cout = std::cout.rdbuf(&cout_buf);
        std::streambuf* old_cerr = std::cerr.rdbuf(&cerr_buf);

        // Optional args: [assets_dir] [model_path]
        // If no args are given, auto-load from ~/Documents/Magenta/magenta-rt-v2/.
        if (argc >= 1 && atom_gettype(argv) == A_SYM) {
            const char* dir = atom_getsym(argv)->s_name;
            post("mrt2~: loading assets from %s", dir);
            x->assets_loaded = x->engine->init_assets(dir);
            if (!x->assets_loaded) {
                object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to init assets");
                for (const auto& log : x->engine->get_logs()) {
                    object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
                }
            }
        } else {
            std::string default_dir = magentart::paths::get_resources_dir();
            post("mrt2~: loading assets from %s", default_dir.c_str());
            x->assets_loaded = x->engine->init_assets(default_dir.c_str());
            if (!x->assets_loaded) {
                object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to init assets from default path");
                for (const auto& log : x->engine->get_logs()) {
                    object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
                }
            }
        }
        if (argc >= 2 && atom_gettype(argv + 1) == A_SYM) {
            const char* mpath = atom_getsym(argv + 1)->s_name;
            post("mrt2~: loading model from %s", mpath);
            x->model_loaded = x->engine->load_model(mpath);
            if (!x->model_loaded) {
                object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to load model");
                for (const auto& log : x->engine->get_logs()) {
                    object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
                }
            }
        } else if (argc < 2) {
            std::string mlxfn = magentart::paths::find_mlxfn_in_dir(magentart::paths::get_default_model_dir());
            if (!mlxfn.empty()) {
                post("mrt2~: loading default model from %s", mlxfn.c_str());
                x->model_loaded = x->engine->load_model(mlxfn.c_str());
                if (!x->model_loaded) {
                    object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to load default model");
                    for (const auto& log : x->engine->get_logs()) {
                        object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
                    }
                }
            } else {
                post("mrt2~: no default model found at %s — send 'model <path>' to load one",
                     magentart::paths::get_default_model_dir().c_str());
            }
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }

    return x;
}

void mrt_free(t_mrt* x) {
    dsp_free(reinterpret_cast<t_pxobject*>(x));
    if (x->engine) {
        x->engine->stop();
        x->engine->unload();
        delete x->engine;
        x->engine = nullptr;
    }
    delete x->prompt_text;
    delete x->prompt_weight;
    delete x->bufL;
    delete x->bufR;
}

void mrt_assist(t_mrt*, void*, long m, long a, char* s) {
    if (m == ASSIST_INLET) {
        snprintf(s, 256, "messages: prompt N \"text\" w | temperature | topk | volume | reset | ...");
    } else {
        snprintf(s, 256, "(signal) %s output", a == 0 ? "left" : "right");
    }
}

// ---------------------------------------------------------------------------
// DSP

void mrt_perform64(t_mrt* x, t_object*, double**, long,
                   double** outs, long, long sampleframes, long, void*) {
    auto& bufL = *x->bufL;
    auto& bufR = *x->bufR;
    if (static_cast<long>(bufL.size()) < sampleframes) bufL.resize(sampleframes);
    if (static_cast<long>(bufR.size()) < sampleframes) bufR.resize(sampleframes);

    x->engine->read_audio_stereo(bufL.data(), bufR.data(),
                                 static_cast<size_t>(sampleframes), /*blocking=*/false);

    // Outlets are declared in reverse, so outs[0]=right, outs[1]=left.
    double* outR = outs[0];
    double* outL = outs[1];
    for (long i = 0; i < sampleframes; ++i) {
        outL[i] = static_cast<double>(bufL[i]);
        outR[i] = static_cast<double>(bufR[i]);
    }
}

void mrt_dsp64(t_mrt* x, t_object* dsp64, short*, double samplerate, long, long) {
    if (!x->sr_warned && samplerate != kEngineSampleRate) {
        object_warn(reinterpret_cast<t_object*>(x),
                    "mrt2~: host SR is %.0f Hz but model produces 48000 Hz; output will play at the wrong speed. Set Max's SR to 48000.",
                    samplerate);
        x->sr_warned = true;
    }
    object_method(dsp64, gensym("dsp_add64"), x, (method)mrt_perform64, 0, NULL);
}

// ---------------------------------------------------------------------------
// Message handlers (all dispatched via A_GIMME for uniform parsing)

void mrt_assets(t_mrt* x, t_symbol*, long argc, t_atom* argv) {
    if (argc < 1 || atom_gettype(argv) != A_SYM) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: assets requires a directory path");
        return;
    }
    const char* dir = atom_getsym(argv)->s_name;
    post("mrt2~: loading assets from %s", dir);
    @autoreleasepool {
        MaxPostStreamBuf cout_buf("mrt2~: ");
        MaxPostStreamBuf cerr_buf("mrt2~: ERROR: ");
        std::streambuf* old_cout = std::cout.rdbuf(&cout_buf);
        std::streambuf* old_cerr = std::cerr.rdbuf(&cerr_buf);

        x->assets_loaded = x->engine->init_assets(dir);
        if (!x->assets_loaded) {
            object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to init assets");
            for (const auto& log : x->engine->get_logs()) {
                object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
            }
        } else {
            post("mrt2~: assets loaded.");
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
}

void mrt_model(t_mrt* x, t_symbol*, long argc, t_atom* argv) {
    if (argc < 1 || atom_gettype(argv) != A_SYM) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: model requires a path to a .mlxfn file");
        return;
    }
    const char* path = atom_getsym(argv)->s_name;
    post("mrt2~: loading model %s", path);
    @autoreleasepool {
        MaxPostStreamBuf cout_buf("mrt2~: ");
        MaxPostStreamBuf cerr_buf("mrt2~: ERROR: ");
        std::streambuf* old_cout = std::cout.rdbuf(&cout_buf);
        std::streambuf* old_cerr = std::cerr.rdbuf(&cerr_buf);

        x->model_loaded = x->engine->load_model(path);
        if (!x->model_loaded) {
            object_error(reinterpret_cast<t_object*>(x), "mrt2~: failed to load model");
            for (const auto& log : x->engine->get_logs()) {
                object_error(reinterpret_cast<t_object*>(x), "  [Engine Log] %s", log.c_str());
            }
        } else {
            post("mrt2~: model loaded.");
        }

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
    }
}

void mrt_prompt(t_mrt* x, t_symbol*, long argc, t_atom* argv) {
    if (argc < 1 || (atom_gettype(argv) != A_LONG && atom_gettype(argv) != A_FLOAT)) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: prompt requires <slot> [\"text\" weight]");
        return;
    }
    long slot = atom_getlong(argv);
    if (slot < 0 || slot >= kNumPromptSlots) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: prompt slot %ld out of range [0, %d]", slot, kNumPromptSlots - 1);
        return;
    }
    if (argc == 1) {
        // Clear slot.
        (*x->prompt_text)[slot].clear();
        (*x->prompt_weight)[slot] = 0.0f;
    } else {
        if (atom_gettype(argv + 1) != A_SYM) {
            object_error(reinterpret_cast<t_object*>(x), "mrt2~: prompt text must be a symbol");
            return;
        }
        const char* text = atom_getsym(argv + 1)->s_name;
        float weight = 1.0f;
        if (argc >= 3) weight = atom_getfloat(argv + 2);
        if (weight < 0.0f) weight = 0.0f;
        if (weight > 1.0f) weight = 1.0f;
        (*x->prompt_text)[slot].assign(text);
        (*x->prompt_weight)[slot] = weight;
    }
    mrt_resync_prompts(x);
}

void mrt_temperature(t_mrt* x, double f) { x->engine->set_temperature(static_cast<float>(f)); }
void mrt_topk(t_mrt* x, long k)            { x->engine->set_top_k(static_cast<int>(k)); }
void mrt_cfgmusiccoca(t_mrt* x, double f)  { x->engine->set_cfg_musiccoca(static_cast<float>(f)); }
void mrt_cfgnotes(t_mrt* x, double f)      { x->engine->set_cfg_notes(static_cast<float>(f)); }

void mrt_cfgdrums(t_mrt* x, double f)      { x->engine->set_cfg_drums(static_cast<float>(f)); }
void mrt_unmaskwidth(t_mrt* x, long w)     { x->engine->set_unmask_width(static_cast<int>(w)); }
void mrt_volume(t_mrt* x, double db)       { x->engine->set_volume_db(static_cast<float>(db)); }
void mrt_mute(t_mrt* x, long m)            { x->engine->set_mute(m != 0); }
void mrt_bypass(t_mrt* x, long b)          { x->engine->set_bypass(b != 0); }
void mrt_drumless(t_mrt* x, long m)        { x->engine->set_drumless(m != 0); }
void mrt_midigate(t_mrt* x, long g)        { x->engine->set_midi_gate_enabled(g != 0); }
void mrt_noteon(t_mrt* x, long n)          { x->engine->set_note_on(static_cast<int>(n)); }
void mrt_noteoff(t_mrt* x, long n)         { x->engine->set_note_off(static_cast<int>(n)); }

void mrt_buffersize(t_mrt* x, long n) {
    // AU plugin exposes 2048 / 4096 / 8192. Bigger = more inference headroom
    // (fewer underruns) at the cost of more output latency.
    if (n < 1920) {
        object_warn(reinterpret_cast<t_object*>(x),
                    "mrt2~: buffersize %ld is below the 1920-sample frame size; clamping to 1920.", n);
        n = 1920;
    }
    x->engine->set_buffer_size(static_cast<size_t>(n));
    post("mrt2~: buffer size = %ld samples (%.1f ms @ 48 kHz)",
         n, static_cast<double>(n) * 1000.0 / 48000.0);
}

void mrt_reset(t_mrt* x) {
    x->engine->reset();
    post("mrt2~: state reset.");
}



void mrt_pca(t_mrt* x, t_symbol*, long argc, t_atom* argv) {
    if (argc < 2) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: pca <axis> <value>");
        return;
    }
    long axis = atom_getlong(argv);
    float v = atom_getfloat(argv + 1);
    x->engine->set_pca_coeff(static_cast<int>(axis), v);
}

void mrt_pcafile(t_mrt* x, t_symbol*, long argc, t_atom* argv) {
    if (argc < 1 || atom_gettype(argv) != A_SYM) {
        object_error(reinterpret_cast<t_object*>(x), "mrt2~: pcafile requires a path");
        return;
    }
    const char* path = atom_getsym(argv)->s_name;
    bool ok = x->engine->load_pca_file(path);
    if (!ok) object_error(reinterpret_cast<t_object*>(x), "mrt2~: pca file load failed");
    else post("mrt2~: pca file loaded (%d components, %d centroids)",
              x->engine->pca_component_count(), x->engine->pca_centroid_count());
}

}  // anonymous namespace

extern "C" void C74_EXPORT ext_main(void*) {
    t_class* c = class_new("mrt2~",
                           (method)mrt_new,
                           (method)mrt_free,
                           static_cast<long>(sizeof(t_mrt)),
                           0L,
                           A_GIMME, 0);

    class_addmethod(c, (method)mrt_dsp64,        "dsp64",        A_CANT,  0);
    class_addmethod(c, (method)mrt_assist,       "assist",       A_CANT,  0);

    class_addmethod(c, (method)mrt_assets,       "assets",       A_GIMME, 0);
    class_addmethod(c, (method)mrt_model,        "model",        A_GIMME, 0);
    class_addmethod(c, (method)mrt_prompt,       "prompt",       A_GIMME, 0);

    class_addmethod(c, (method)mrt_temperature,  "temperature",  A_FLOAT, 0);
    class_addmethod(c, (method)mrt_topk,         "topk",         A_LONG,  0);
    class_addmethod(c, (method)mrt_cfgmusiccoca, "cfgmusiccoca", A_FLOAT, 0);
    class_addmethod(c, (method)mrt_cfgnotes,     "cfgnotes",     A_FLOAT, 0);

    class_addmethod(c, (method)mrt_cfgdrums,     "cfgdrums",     A_FLOAT, 0);
    class_addmethod(c, (method)mrt_unmaskwidth,  "unmaskwidth",  A_LONG,  0);
    class_addmethod(c, (method)mrt_volume,       "volume",       A_FLOAT, 0);
    class_addmethod(c, (method)mrt_mute,         "mute",         A_LONG,  0);
    class_addmethod(c, (method)mrt_bypass,       "bypass",       A_LONG,  0);
    class_addmethod(c, (method)mrt_drumless,     "drumless",     A_LONG,  0);
    class_addmethod(c, (method)mrt_midigate,     "midigate",     A_LONG,  0);
    class_addmethod(c, (method)mrt_noteon,       "noteon",       A_LONG,  0);
    class_addmethod(c, (method)mrt_noteoff,      "noteoff",      A_LONG,  0);
    class_addmethod(c, (method)mrt_buffersize,   "buffersize",   A_LONG,  0);
    class_addmethod(c, (method)mrt_reset,        "reset",        0);

    class_addmethod(c, (method)mrt_pca,          "pca",          A_GIMME, 0);
    class_addmethod(c, (method)mrt_pcafile,      "pcafile",      A_GIMME, 0);

    class_dspinit(c);
    class_register(CLASS_BOX, c);
    s_mrt_class = c;
}
