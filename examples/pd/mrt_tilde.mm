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

// mrt2~ — Pure Data signal external for the realtime music model in this repo.
// Mirrors the MaxMSP external (`examples/max/mrt_tilde.mm`); same engine, same
// message surface, translated to PD's calling convention. No prompt surface —
// each of the 6 prompt slots is set independently via `prompt N text… weight`.

#include "m_pd.h"

// PD's m_pd.h is C-only; engine headers are C++ and the realtime runner uses
// Objective-C autorelease pools internally, so this TU is .mm + ARC.
#include <magentart/realtime_runner.h>
#include "../common/cpp/magenta_paths.h"

#include <array>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

using magentart::core::RealtimeRunner;
using magentart::core::kMaxPrompts;

constexpr int    kNumPromptSlots   = static_cast<int>(kMaxPrompts);  // 6
constexpr double kEngineSampleRate = 48000.0;

// Single class pointer used by the setup function.
t_class* s_mrt_class = nullptr;

struct t_mrt {
    t_object x_obj;
    RealtimeRunner* engine;
    std::array<std::string, kNumPromptSlots>* prompt_text;
    std::array<float, kNumPromptSlots>*       prompt_weight;
    std::vector<float>* bufL;
    std::vector<float>* bufR;
    bool sr_warned;
    bool assets_loaded;
    bool model_loaded;
    t_outlet* out_l;
    t_outlet* out_r;
};

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

// ---------------------------------------------------------------------------
// Lifecycle

void* mrt_new(t_symbol*, int argc, t_atom* argv) {
    // pd_new returns a t_pd* (a class-pointer cell); the rest of the struct
    // is allocated immediately after it. PD code-bases conventionally
    // reinterpret-cast — no inheritance is involved.
    t_mrt* x = reinterpret_cast<t_mrt*>(pd_new(s_mrt_class));
    if (!x) return nullptr;

    x->engine        = new RealtimeRunner();
    x->prompt_text   = new std::array<std::string, kNumPromptSlots>();
    x->prompt_weight = new std::array<float, kNumPromptSlots>();
    x->bufL          = new std::vector<float>();
    x->bufR          = new std::vector<float>();
    x->sr_warned     = false;
    x->assets_loaded = false;
    x->model_loaded  = false;

    for (int i = 0; i < kNumPromptSlots; ++i) {
        (*x->prompt_text)[i].clear();
        (*x->prompt_weight)[i] = 0.0f;
    }

    // Two signal outlets, declared L→R; PD respects declaration order.
    x->out_l = outlet_new(&x->x_obj, &s_signal);
    x->out_r = outlet_new(&x->x_obj, &s_signal);

    @autoreleasepool {
        // Optional creation args: [assets_dir] [model_path] (mirrors Max external).
        // If no assets_dir is provided, default to ~/Documents/Magenta/magenta-rt-v2/resources.
        if (argc >= 1 && argv[0].a_type == A_SYMBOL) {
            const char* dir = atom_getsymbol(&argv[0])->s_name;
            post("mrt2~: loading assets from %s", dir);
            x->assets_loaded = x->engine->init_assets(dir);
            if (!x->assets_loaded) {
                pd_error(x, "mrt2~: failed to init assets");
                for (const auto& log : x->engine->get_logs()) {
                    pd_error(x, "  [Engine Log] %s", log.c_str());
                }
            }
        } else {
            std::string default_dir = magentart::paths::get_resources_dir();
            post("mrt2~: no assets_dir provided, defaulting to %s", default_dir.c_str());
            x->assets_loaded = x->engine->init_assets(default_dir.c_str());
            if (!x->assets_loaded) {
                pd_error(x, "mrt2~: failed to init assets from default path");
                for (const auto& log : x->engine->get_logs()) {
                    pd_error(x, "  [Engine Log] %s", log.c_str());
                }
            }
        }
        if (argc >= 2 && argv[1].a_type == A_SYMBOL) {
            const char* mpath = atom_getsymbol(&argv[1])->s_name;
            post("mrt2~: loading model from %s", mpath);
            x->model_loaded = x->engine->load_model(mpath);
            if (!x->model_loaded) {
                pd_error(x, "mrt2~: failed to load model");
                for (const auto& log : x->engine->get_logs()) {
                    pd_error(x, "  [Engine Log] %s", log.c_str());
                }
            }
        } else if (argc < 2) {
            std::string mlxfn = magentart::paths::find_mlxfn_in_dir(magentart::paths::get_default_model_dir());
            if (!mlxfn.empty()) {
                post("mrt2~: loading default model from %s", mlxfn.c_str());
                x->model_loaded = x->engine->load_model(mlxfn.c_str());
                if (!x->model_loaded) {
                    pd_error(x, "mrt2~: failed to load default model");
                    for (const auto& log : x->engine->get_logs()) {
                        pd_error(x, "  [Engine Log] %s", log.c_str());
                    }
                }
            } else {
                post("mrt2~: no default model found at %s — send 'model <path>' to load one",
                     magentart::paths::get_default_model_dir().c_str());
            }
        }
    }

    return x;
}

void mrt_free(t_mrt* x) {
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

// ---------------------------------------------------------------------------
// DSP

t_int* mrt_perform(t_int* w) {
    t_mrt* x         = reinterpret_cast<t_mrt*>(w[1]);
    t_sample* outL   = reinterpret_cast<t_sample*>(w[2]);
    t_sample* outR   = reinterpret_cast<t_sample*>(w[3]);
    int n            = static_cast<int>(w[4]);

    auto& bufL = *x->bufL;
    auto& bufR = *x->bufR;
    if (static_cast<int>(bufL.size()) < n) bufL.resize(n);
    if (static_cast<int>(bufR.size()) < n) bufR.resize(n);

    x->engine->read_audio_stereo(bufL.data(), bufR.data(),
                                 static_cast<size_t>(n), /*blocking=*/false);

    // PD's t_sample is float, matching engine output — direct copy.
    for (int i = 0; i < n; ++i) {
        outL[i] = bufL[i];
        outR[i] = bufR[i];
    }
    return (w + 5);
}

void mrt_dsp(t_mrt* x, t_signal** sp) {
    if (!x->sr_warned && static_cast<int>(sp[0]->s_sr) != static_cast<int>(kEngineSampleRate)) {
        post("mrt2~: WARNING — host SR is %d Hz but model produces 48000 Hz; output will play at the wrong speed. Set Pd's SR to 48000.",
             static_cast<int>(sp[0]->s_sr));
        x->sr_warned = true;
    }
    // sp[0]/sp[1] are the two signal outlets (no signal inlet declared).
    dsp_add(mrt_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

// ---------------------------------------------------------------------------
// Message handlers

void mrt_assets(t_mrt* x, t_symbol*, int argc, t_atom* argv) {
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "mrt2~: assets requires a directory path");
        return;
    }
    const char* dir = atom_getsymbol(&argv[0])->s_name;
    post("mrt2~: loading assets from %s", dir);
    @autoreleasepool {
        x->assets_loaded = x->engine->init_assets(dir);
        if (!x->assets_loaded) {
            pd_error(x, "mrt2~: failed to init assets");
            for (const auto& log : x->engine->get_logs()) {
                pd_error(x, "  [Engine Log] %s", log.c_str());
            }
        } else {
            post("mrt2~: assets loaded.");
        }
    }
}

void mrt_model(t_mrt* x, t_symbol*, int argc, t_atom* argv) {
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "mrt2~: model requires a path to a .mlxfn file");
        return;
    }
    const char* path = atom_getsymbol(&argv[0])->s_name;
    post("mrt2~: loading model %s", path);
    @autoreleasepool {
        x->model_loaded = x->engine->load_model(path);
        if (!x->model_loaded) {
            pd_error(x, "mrt2~: failed to load model");
            for (const auto& log : x->engine->get_logs()) {
                pd_error(x, "  [Engine Log] %s", log.c_str());
            }
        } else {
            post("mrt2~: model loaded.");
        }
    }
}

// PD has no string-quoting in message boxes, so we accept `prompt N text… w`
// where `text…` is one or more symbols and `w` is the trailing float weight.
void mrt_prompt(t_mrt* x, t_symbol*, int argc, t_atom* argv) {
    if (argc < 1 || argv[0].a_type != A_FLOAT) {
        pd_error(x, "mrt2~: prompt requires <slot> [text… weight]");
        return;
    }
    int slot = static_cast<int>(atom_getfloat(&argv[0]));
    if (slot < 0 || slot >= kNumPromptSlots) {
        pd_error(x, "mrt2~: prompt slot %d out of range [0, %d]", slot, kNumPromptSlots - 1);
        return;
    }
    if (argc == 1) {
        // Clear slot.
        (*x->prompt_text)[slot].clear();
        (*x->prompt_weight)[slot] = 0.0f;
    } else {
        // Trailing float = weight (if present); everything between slot and
        // weight = text symbols joined by spaces.
        int last = argc - 1;
        float weight = 1.0f;
        bool has_weight = (argv[last].a_type == A_FLOAT);
        if (has_weight) {
            weight = atom_getfloat(&argv[last]);
        }
        int text_end = has_weight ? last : argc;

        std::string text;
        for (int i = 1; i < text_end; ++i) {
            if (argv[i].a_type != A_SYMBOL) continue;
            if (!text.empty()) text += " ";
            text += atom_getsymbol(&argv[i])->s_name;
        }
        if (text.empty()) {
            pd_error(x, "mrt2~: prompt text must include at least one symbol");
            return;
        }
        if (weight < 0.0f) weight = 0.0f;
        if (weight > 1.0f) weight = 1.0f;
        (*x->prompt_text)[slot].assign(text);
        (*x->prompt_weight)[slot] = weight;
    }
    mrt_resync_prompts(x);
}

void mrt_temperature (t_mrt* x, t_floatarg f) { x->engine->set_temperature(f); }
void mrt_topk        (t_mrt* x, t_floatarg f) { x->engine->set_top_k(static_cast<int>(f)); }
void mrt_cfgmusiccoca(t_mrt* x, t_floatarg f) { x->engine->set_cfg_musiccoca(f); }
void mrt_cfgnotes    (t_mrt* x, t_floatarg f) { x->engine->set_cfg_notes(f); }

void mrt_cfgdrums    (t_mrt* x, t_floatarg f) { x->engine->set_cfg_drums(f); }
void mrt_unmaskwidth (t_mrt* x, t_floatarg f) { x->engine->set_unmask_width(static_cast<int>(f)); }
void mrt_volume      (t_mrt* x, t_floatarg f) { x->engine->set_volume_db(f); }
void mrt_mute        (t_mrt* x, t_floatarg f) { x->engine->set_mute(f != 0.0f); }
void mrt_bypass      (t_mrt* x, t_floatarg f) { x->engine->set_bypass(f != 0.0f); }
void mrt_drumless    (t_mrt* x, t_floatarg f) { x->engine->set_drumless(f != 0.0f); }
void mrt_midigate    (t_mrt* x, t_floatarg f) { x->engine->set_midi_gate_enabled(f != 0.0f); }
void mrt_noteon      (t_mrt* x, t_floatarg f) { x->engine->set_note_on(static_cast<int>(f)); }
void mrt_noteoff     (t_mrt* x, t_floatarg f) { x->engine->set_note_off(static_cast<int>(f)); }

void mrt_buffersize(t_mrt* x, t_floatarg f) {
    long n = static_cast<long>(f);
    if (n < 1920) {
        post("mrt2~: buffersize %ld is below the 1920-sample frame size; clamping to 1920.", n);
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



void mrt_pca(t_mrt* x, t_symbol*, int argc, t_atom* argv) {
    if (argc < 2) {
        pd_error(x, "mrt2~: pca <axis> <value>");
        return;
    }
    int axis = static_cast<int>(atom_getfloat(&argv[0]));
    float v  = atom_getfloat(&argv[1]);
    x->engine->set_pca_coeff(axis, v);
}

void mrt_style_embedding(t_mrt* x, t_symbol* s, int argc, t_atom* argv) {
    // Expect: set_style_embedding <slot> <f0> <f1> ... <f767>
    // Total arguments: 1 (slot) + 768 (embedding values) = 769
    if (argc != 769 || argv[0].a_type != A_FLOAT) {
        pd_error(x, "mrt2~: set_style_embedding requires <slot> and 768 floats");
        return;
    }

    int slot = static_cast<int>(atom_getfloat(&argv[0]));
    if (slot < 0 || slot >= kNumPromptSlots) {
        pd_error(x, "mrt2~: slot %d out of range", slot);
        return;
    }

    std::vector<float> embedding(768);
    for (int i = 0; i < 768; ++i) {
        if (argv[i + 1].a_type != A_FLOAT) {
            pd_error(x, "mrt2~: argument %d is not a float", i + 1);
            return;
        }
        embedding[i] = atom_getfloat(&argv[i + 1]);
    }

    // 1. Set the raw embedding in the engine
    x->engine->set_audio_embedding(slot, embedding.data());

    // 2. Trigger reblend so the engine immediately incorporates the new embedding
    // We update weights array to ensure this slot is active (weight 1.0)
    (*x->prompt_weight)[slot] = 1.0f;
    
    // Call reblend.
    x->engine->reblend_musiccoca_tokens(x->prompt_weight->data(), kNumPromptSlots);

    post("mrt2~: set raw style embedding for slot %d", slot);
}

void mrt_pcafile(t_mrt* x, t_symbol*, int argc, t_atom* argv) {
    if (argc < 1 || argv[0].a_type != A_SYMBOL) {
        pd_error(x, "mrt2~: pcafile requires a path");
        return;
    }
    const char* path = atom_getsymbol(&argv[0])->s_name;
    bool ok = x->engine->load_pca_file(path);
    if (!ok) pd_error(x, "mrt2~: pca file load failed");
    else post("mrt2~: pca file loaded (%d components, %d centroids)",
              x->engine->pca_component_count(), x->engine->pca_centroid_count());
}

}  // anonymous namespace

// PD discovers the setup function by stripping the trailing `~` from the
// external name and appending `_setup` → `mrt2_tilde_setup`.
extern "C" void mrt2_tilde_setup(void) {
    s_mrt_class = class_new(gensym("mrt2~"),
                            reinterpret_cast<t_newmethod>(mrt_new),
                            reinterpret_cast<t_method>(mrt_free),
                            sizeof(t_mrt),
                            CLASS_DEFAULT,
                            A_GIMME, 0);

    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_dsp),
                    gensym("dsp"), A_CANT, 0);

    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_assets),
                    gensym("assets"),       A_GIMME, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_model),
                    gensym("model"),        A_GIMME, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_prompt),
                    gensym("prompt"),       A_GIMME, 0);

    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_temperature),
                    gensym("temperature"),  A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_topk),
                    gensym("topk"),         A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_cfgmusiccoca),
                    gensym("cfgmusiccoca"),     A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_cfgnotes),
                    gensym("cfgnotes"),     A_FLOAT, 0);

    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_cfgdrums),
                    gensym("cfgdrums"),     A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_unmaskwidth),
                    gensym("unmaskwidth"),  A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_volume),
                    gensym("volume"),       A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_mute),
                    gensym("mute"),         A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_bypass),
                    gensym("bypass"),       A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_drumless),
                    gensym("drumless"),     A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_midigate),
                    gensym("midigate"),     A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_noteon),
                    gensym("noteon"),       A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_noteoff),
                    gensym("noteoff"),      A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_buffersize),
                    gensym("buffersize"),   A_FLOAT, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_reset),
                    gensym("reset"),        A_NULL);

    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_pca),
                    gensym("pca"),          A_GIMME, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_style_embedding),
                    gensym("set_style_embedding"), A_GIMME, 0);
    class_addmethod(s_mrt_class, reinterpret_cast<t_method>(mrt_pcafile),
                    gensym("pcafile"),      A_GIMME, 0);
}
