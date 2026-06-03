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

// MRT2 — SuperCollider UGen plugin for the realtime music model in this
// repo. Mirrors the MaxMSP (`examples/max/mrt_tilde.mm`) and PD
// (`examples/pd/mrt_tilde.mm`) externals: same engine, same exposed parameter
// set. State changes arrive from sclang as `/u_cmd` OSC messages and dispatch
// through SC's per-UGen unit command mechanism (`DefineUnitCmd`), each
// landing in an atomic / mutex-protected RealtimeRunner setter — same proven
// path used by AU, standalone, Max, and PD.
//
// File extension is .mm (Objective-C++) because magentart::core's
// realtime_runner.h pulls in C++ that uses Objective-C autorelease pools
// internally; building this TU as Objective-C++ keeps the module ABI
// consistent with the rest of the hosts.

#include "SC_PlugIn.h"
#include <magentart/realtime_runner.h>
#include "../common/cpp/magenta_paths.h"

#include <array>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static InterfaceTable* ft;

namespace {
using magentart::core::RealtimeRunner;
using magentart::core::kMaxPrompts;

constexpr int    kNumPromptSlots   = static_cast<int>(kMaxPrompts);  // 6
constexpr double kEngineSampleRate = 48000.0;
}  // namespace

// SC zero-fills the unit struct on allocation, so the C++ pointer members are
// nullptr until Ctor `new`s them. Plain heap allocation is fine in Ctor/Dtor
// (SC explicitly permits non-RT allocation there); the audio thread itself
// (`MRT2UGen_next`) never allocates.
// Note: the C++ Unit class is named MRT2UGen so its registration name on
// the server matches the sclang-side `MRT2UGen : MultiOutUGen` subclass.
// The end-user-facing wrapper class in MRT2.sc is `MRT2` (no UGen
// suffix) and never appears here.
struct MRT2UGen : public Unit {
    RealtimeRunner* engine;
    std::array<std::string, kNumPromptSlots>* prompt_text;
    std::array<float,       kNumPromptSlots>* prompt_weight;
    bool sr_warned;
    bool assets_loaded;
    bool model_loaded;
};

static void MRT2_resync_prompts(MRT2UGen* unit) {
    // Send all prompt texts and weights to the engine (including zero-weight
    // slots — PR #280 made weights the source of truth; zero is valid).
    std::vector<std::string> texts;
    std::vector<float> weights;
    texts.reserve(kNumPromptSlots);
    weights.reserve(kNumPromptSlots);
    for (int i = 0; i < kNumPromptSlots; ++i) {
        texts.push_back((*unit->prompt_text)[i]);
        weights.push_back((*unit->prompt_weight)[i]);
    }
    unit->engine->set_text_prompts(texts, weights);

    // Also set the blend weights so the inference loop picks them up.
    for (int i = 0; i < kNumPromptSlots; ++i) {
        unit->engine->set_blend_weight(i, (*unit->prompt_weight)[i]);
    }
}

// ----------------------------------------------------------------------------
// DSP

extern "C" {
    void MRT2UGen_Ctor(MRT2UGen* unit);
    void MRT2UGen_Dtor(MRT2UGen* unit);
    void MRT2UGen_next(MRT2UGen* unit, int inNumSamples);
}

void MRT2UGen_next(MRT2UGen* unit, int inNumSamples) {
    // SC's t_sample is float (matches engine output) — write straight to the
    // output buffers, no intermediate copy needed. Both OUT(0) and OUT(1) are
    // valid because the sclang class declares 2 outputs via initOutputs.
    float* outL = OUT(0);
    float* outR = OUT(1);
    unit->engine->read_audio_stereo(outL, outR,
                                    static_cast<size_t>(inNumSamples),
                                    /*blocking=*/false);
}

void MRT2UGen_Ctor(MRT2UGen* unit) {
    unit->engine        = new RealtimeRunner();
    unit->prompt_text   = new std::array<std::string, kNumPromptSlots>();
    unit->prompt_weight = new std::array<float, kNumPromptSlots>();
    unit->sr_warned     = false;
    unit->assets_loaded = false;
    unit->model_loaded  = false;
    for (int i = 0; i < kNumPromptSlots; ++i) {
        (*unit->prompt_text)[i].clear();
        (*unit->prompt_weight)[i] = 0.0f;
    }

    if (!unit->sr_warned && static_cast<int>(SAMPLERATE) != static_cast<int>(kEngineSampleRate)) {
        Print("MRT2: WARNING — server SR is %d Hz but model produces 48000 Hz; output will play at the wrong speed. Set Server's SR to 48000.\n",
              static_cast<int>(SAMPLERATE));
        unit->sr_warned = true;
    }

    @autoreleasepool {
        // Auto-load assets and default model from ~/Documents/Magenta/magenta-rt-v2/.
        // Users can override via /u_cmd messages (assets, model) from sclang.
        std::string default_assets = magentart::paths::get_resources_dir();
        Print("MRT2: loading assets from %s\n", default_assets.c_str());
        unit->assets_loaded = unit->engine->init_assets(default_assets.c_str());
        if (!unit->assets_loaded) {
            Print("MRT2: failed to init assets from default path\n");
            for (const auto& log : unit->engine->get_logs()) {
                Print("  [Engine Log] %s\n", log.c_str());
            }
        }

        std::string mlxfn = magentart::paths::find_mlxfn_in_dir(magentart::paths::get_default_model_dir());
        if (!mlxfn.empty()) {
            Print("MRT2: loading default model from %s\n", mlxfn.c_str());
            unit->model_loaded = unit->engine->load_model(mlxfn.c_str());
            if (!unit->model_loaded) {
                Print("MRT2: failed to load default model\n");
                for (const auto& log : unit->engine->get_logs()) {
                    Print("  [Engine Log] %s\n", log.c_str());
                }
            } else {
                Print("MRT2: model loaded.\n");
            }
        } else {
            Print("MRT2: no default model found at %s — send 'model' command to load one\n",
                  magentart::paths::get_default_model_dir().c_str());
        }
    }

    SETCALC(MRT2UGen_next);
    // Produce one frame of silence so downstream UGens have something defined
    // for their initial sample. Engine isn't running yet (no model loaded), so
    // we just zero the first output sample on each channel.
    OUT(0)[0] = 0.0f;
    OUT(1)[0] = 0.0f;
}

void MRT2UGen_Dtor(MRT2UGen* unit) {
    if (unit->engine) {
        unit->engine->stop();
        unit->engine->unload();
        delete unit->engine;
        unit->engine = nullptr;
    }
    delete unit->prompt_text;
    delete unit->prompt_weight;
}

// ----------------------------------------------------------------------------
// Unit commands. All have signature `void(Unit*, sc_msg_iter*)` and dispatch
// from `/u_cmd <nodeID> <ugenIdx> <cmdName> <args…>`. sc_msg_iter exposes
// geti / getf / gets for int / float / string args.

extern "C" {

void MRT2_cmd_assets(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    const char* dir = args->gets();
    if (!dir || !*dir) {
        Print("MRT2: assets requires a directory path\n");
        return;
    }
    Print("MRT2: loading assets from %s\n", dir);
    @autoreleasepool {
        unit->assets_loaded = unit->engine->init_assets(dir);
        if (!unit->assets_loaded) {
            Print("MRT2: failed to init assets\n");
            for (const auto& log : unit->engine->get_logs()) {
                Print("  [Engine Log] %s\n", log.c_str());
            }
        } else {
            Print("MRT2: assets loaded.\n");
        }
    }
}

void MRT2_cmd_model(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    const char* path = args->gets();
    if (!path || !*path) {
        Print("MRT2: model requires a path to a .mlxfn file\n");
        return;
    }
    Print("MRT2: loading model %s\n", path);
    @autoreleasepool {
        unit->model_loaded = unit->engine->load_model(path);
        if (!unit->model_loaded) {
            Print("MRT2: failed to load model\n");
            for (const auto& log : unit->engine->get_logs()) {
                Print("  [Engine Log] %s\n", log.c_str());
            }
        } else {
            Print("MRT2: model loaded.\n");
        }
    }
}

void MRT2_cmd_prompt(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    int slot = args->geti();
    if (slot < 0 || slot >= kNumPromptSlots) {
        Print("MRT2: prompt slot %d out of range [0, %d]\n", slot, kNumPromptSlots - 1);
        return;
    }
    const char* text = args->gets();
    if (!text) {
        Print("MRT2: prompt %d requires text (or use prompt_clear)\n", slot);
        return;
    }
    float weight = 1.0f;
    if (args->remain() > 0) weight = args->getf();
    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;
    (*unit->prompt_text)[slot].assign(text);
    (*unit->prompt_weight)[slot] = weight;
    MRT2_resync_prompts(unit);
}

void MRT2_cmd_prompt_clear(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    int slot = args->geti();
    if (slot < 0 || slot >= kNumPromptSlots) {
        Print("MRT2: prompt_clear slot %d out of range [0, %d]\n", slot, kNumPromptSlots - 1);
        return;
    }
    (*unit->prompt_text)[slot].clear();
    (*unit->prompt_weight)[slot] = 0.0f;
    MRT2_resync_prompts(unit);
}

void MRT2_cmd_temperature(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_temperature(args->getf());
}
void MRT2_cmd_topk(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_top_k(args->geti());
}
void MRT2_cmd_cfgmusiccoca(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_cfg_musiccoca(args->getf());
}
void MRT2_cmd_cfgnotes(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_cfg_notes(args->getf());
}

void MRT2_cmd_cfgdrums(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_cfg_drums(args->getf());
}
void MRT2_cmd_unmaskwidth(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_unmask_width(args->geti());
}
void MRT2_cmd_volume(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_volume_db(args->getf());
}
void MRT2_cmd_mute(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_mute(args->geti() != 0);
}
void MRT2_cmd_bypass(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_bypass(args->geti() != 0);
}

void MRT2_cmd_buffersize(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    int n = args->geti();
    if (n < 1920) {
        Print("MRT2: buffersize %d is below the 1920-sample frame size; clamping to 1920.\n", n);
        n = 1920;
    }
    unit->engine->set_buffer_size(static_cast<size_t>(n));
    Print("MRT2: buffer size = %d samples (%.1f ms @ 48 kHz)\n",
          n, static_cast<double>(n) * 1000.0 / 48000.0);
}

void MRT2_cmd_reset(Unit* u, sc_msg_iter*) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->reset();
    Print("MRT2: state reset.\n");
}

void MRT2_cmd_noteon(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_note_on(args->geti());
}
void MRT2_cmd_noteoff(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_note_off(args->geti());
}
void MRT2_cmd_midigate(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_midi_gate_enabled(args->geti() != 0);
}
void MRT2_cmd_drumless(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    unit->engine->set_drumless(args->geti() != 0);
}

void MRT2_cmd_pca(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    int axis = args->geti();
    float v  = args->getf();
    unit->engine->set_pca_coeff(axis, v);
}

void MRT2_cmd_pcafile(Unit* u, sc_msg_iter* args) {
    auto* unit = reinterpret_cast<MRT2UGen*>(u);
    const char* path = args->gets();
    if (!path) {
        Print("MRT2: pcafile requires a path\n");
        return;
    }
    bool ok = unit->engine->load_pca_file(path);
    if (!ok) Print("MRT2: pca file load failed\n");
    else     Print("MRT2: pca file loaded (%d components, %d centroids)\n",
                   unit->engine->pca_component_count(), unit->engine->pca_centroid_count());
}

}  // extern "C"

// ----------------------------------------------------------------------------
// Plugin registration. scsynth calls this at load time and looks up
// `api_version` + `server_version` symbols inside the .scx, both stamped here
// by the PluginLoad macro.

PluginLoad(MRT2UGens) {
    ft = inTable;
    DefineDtorUnit(MRT2UGen);

    DefineUnitCmd("MRT2UGen", "assets",       MRT2_cmd_assets);
    DefineUnitCmd("MRT2UGen", "model",        MRT2_cmd_model);
    DefineUnitCmd("MRT2UGen", "prompt",       MRT2_cmd_prompt);
    DefineUnitCmd("MRT2UGen", "prompt_clear", MRT2_cmd_prompt_clear);
    DefineUnitCmd("MRT2UGen", "temperature",  MRT2_cmd_temperature);
    DefineUnitCmd("MRT2UGen", "topk",         MRT2_cmd_topk);
    DefineUnitCmd("MRT2UGen", "cfgmusiccoca", MRT2_cmd_cfgmusiccoca);
    DefineUnitCmd("MRT2UGen", "cfgnotes",     MRT2_cmd_cfgnotes);

    DefineUnitCmd("MRT2UGen", "unmaskwidth",  MRT2_cmd_unmaskwidth);
    DefineUnitCmd("MRT2UGen", "volume",       MRT2_cmd_volume);
    DefineUnitCmd("MRT2UGen", "mute",         MRT2_cmd_mute);
    DefineUnitCmd("MRT2UGen", "bypass",       MRT2_cmd_bypass);
    DefineUnitCmd("MRT2UGen", "buffersize",   MRT2_cmd_buffersize);
    DefineUnitCmd("MRT2UGen", "reset",        MRT2_cmd_reset);
    DefineUnitCmd("MRT2UGen", "noteon",       MRT2_cmd_noteon);
    DefineUnitCmd("MRT2UGen", "noteoff",      MRT2_cmd_noteoff);
    DefineUnitCmd("MRT2UGen", "midigate",     MRT2_cmd_midigate);
    DefineUnitCmd("MRT2UGen", "drumless",     MRT2_cmd_drumless);
    DefineUnitCmd("MRT2UGen", "cfgdrums",     MRT2_cmd_cfgdrums);
    DefineUnitCmd("MRT2UGen", "pca",          MRT2_cmd_pca);
    DefineUnitCmd("MRT2UGen", "pcafile",      MRT2_cmd_pcafile);
}
