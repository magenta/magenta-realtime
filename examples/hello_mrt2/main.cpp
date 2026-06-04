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

// hello_mrt2 — shortest-path consumer of magentart::core.
//
// Loads an exported Magenta RealTime v2 model, sets a text prompt, generates
// N 1920-sample stereo frames, and writes `out.wav` in the current directory.
//
// Run:
//   hello_mrt2 <mlxfn_path> [resource_dir] [num_frames]
// where `resource_dir` contains a `musiccoca/` subfolder with the TFLite
// assets (defaults to ~/Documents/Magenta/magenta-rt-v2/resources if not provided).

#include <magentart/mlx_engine.h>

#include "../common/cpp/magenta_paths.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace magentart::core;

static bool write_wav(const std::string& path,
                      const std::vector<float>& interleaved,
                      int sample_rate,
                      int num_channels) {
    uint32_t num_frames = static_cast<uint32_t>(interleaved.size()) / num_channels;
    uint16_t bits_per_sample = 32;
    uint16_t block_align = num_channels * (bits_per_sample / 8);
    uint32_t byte_rate = sample_rate * block_align;
    uint32_t data_size = num_frames * block_align;
    uint32_t chunk_size = 36 + data_size;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    uint32_t subchunk1_size = 16;
    f.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    uint16_t audio_format = 3;  // IEEE float
    uint16_t nc = static_cast<uint16_t>(num_channels);
    uint32_t sr = static_cast<uint32_t>(sample_rate);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&nc), 2);
    f.write(reinterpret_cast<const char*>(&sr), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    f.write(reinterpret_cast<const char*>(interleaved.data()),
            interleaved.size() * sizeof(float));
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <mlxfn_path> [resource_dir] [num_frames] [--prompt prompt_text] [--output out_path] [--force] [--prefill-silence] [--spectrostream-encoder path] [--prefill-duration duration]\n"
            "  mlxfn_path    path to exported .mlxfn model directory\n"
            "  resource_dir  path containing a `musiccoca/` subfolder with TFLite assets\n"
            "                (default: %s)\n"
            "  num_frames    frames to generate (default 100 = 4.0s)\n"
            "  --prompt, -p  text prompt (default: 'a jazz piano trio')\n"
            "  --output, -o  output path (default: 'out.wav')\n"
            "  --force, -f   overwrite existing output file\n"
            "  --prefill-silence prefill state with silent audio before generation\n"
            "  --spectrostream-encoder path path to exported spectrostream_encoder.mlxfn\n"
            "                (default: %s/spectrostream_encoder.mlxfn)\n"
            "  --prefill-duration duration duration of silent prefill in seconds (default: 1.64)\n",
            argv[0],
            magentart::paths::get_resources_dir().c_str(),
            magentart::paths::get_spectrostream_dir().c_str());
        return 1;
    }
    std::string mlxfn_path = argv[1];
    std::string resource_dir = (argc >= 3) ? argv[2] : magentart::paths::get_resources_dir();
    int num_frames = 100;

    std::string prompt = "a jazz piano trio";
    std::string out_path = "out.wav";
    bool force_overwrite = false;
    bool is_prefill_silence = false;
    std::string spectrostream_encoder_path = magentart::paths::get_spectrostream_dir() + "/spectrostream_encoder.mlxfn";
    double prefill_duration = 1.64;
    float temperature = 1.0f;
    int top_k = 100;

    float cfg_musiccoca = 3.0f;
    float cfg_notes = 5.0f;
    float cfg_drums = 1.0f;
    bool drumless = false;
    int unmask_width = 0;
    int seed_rotation = 0;
    std::string model_subfolder = "musiccoca";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--subfolder" || arg == "-s") {
            if (i + 1 < argc) {
                model_subfolder = argv[++i];
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--prompt" || arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[++i];
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg.rfind("--prompt=", 0) == 0) {
            prompt = arg.substr(9);
        } else if (arg.rfind("-p=", 0) == 0) {
            prompt = arg.substr(3);
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                out_path = argv[++i];
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg.rfind("--output=", 0) == 0) {
            out_path = arg.substr(9);
        } else if (arg.rfind("-o=", 0) == 0) {
            out_path = arg.substr(3);
        } else if (arg == "--force" || arg == "-f") {
            force_overwrite = true;
        } else if (arg == "--prefill-silence") {
            is_prefill_silence = true;
        } else if (arg == "--spectrostream-encoder") {
            if (i + 1 < argc) {
                spectrostream_encoder_path = argv[++i];
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--prefill-duration") {
            if (i + 1 < argc) {
                prefill_duration = std::atof(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--temperature") {
            if (i + 1 < argc) {
                temperature = std::stof(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--top-k") {
            if (i + 1 < argc) {
                top_k = std::atoi(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }

        } else if (arg == "--cfg-musiccoca") {
            if (i + 1 < argc) {
                cfg_musiccoca = std::stof(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--cfg-notes") {
            if (i + 1 < argc) {
                cfg_notes = std::stof(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--cfg-drums") {
            if (i + 1 < argc) {
                cfg_drums = std::stof(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--drumless") {
            drumless = true;
        } else if (arg == "--unmask-width") {
            if (i + 1 < argc) {
                unmask_width = std::atoi(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--seed-rotation") {
            if (i + 1 < argc) {
                seed_rotation = std::atoi(argv[++i]);
            } else {
                std::fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else {
            num_frames = std::atoi(arg.c_str());
        }
    }

    std::filesystem::path p(out_path);
    if (std::filesystem::exists(p) && !force_overwrite) {
        std::fprintf(stderr, "Error: Output file '%s' already exists. Use --force or -f to overwrite.\n", out_path.c_str());
        return 1;
    }
    if (p.has_parent_path() && !std::filesystem::exists(p.parent_path())) {
        std::fprintf(stderr, "Error: Parent directory '%s' does not exist.\n", p.parent_path().c_str());
        return 1;
    }

    MLXEngine engine;
    if (!engine.init_assets(resource_dir.c_str(), model_subfolder.c_str())) {
        std::fprintf(stderr, "Failed to init TFLite assets from %s\n", resource_dir.c_str());
        return 1;
    }
    if (!engine.load_model(mlxfn_path.c_str())) {
        std::fprintf(stderr, "Failed to load model %s\n", mlxfn_path.c_str());
        return 1;
    }

    if (is_prefill_silence) {
        if (!engine.load_prefill_model(spectrostream_encoder_path.c_str(), nullptr)) {
            std::fprintf(stderr, "Failed to load prefill model from %s\n", spectrostream_encoder_path.c_str());
            return 1;
        }
        const int maxFrames = static_cast<int>(prefill_duration * 48000);
        std::vector<float> samples(maxFrames * 2, 0.0f);
        std::printf("Starting silent prefill (%.2fs)...\n", prefill_duration);
        if (!engine.prefill_state(samples.data(), maxFrames,
                                  /*trim_front_frames=*/0, /*trim_back_frames=*/0,
                                  [](const std::string& msg) {
            std::printf("  [Prefill Log] %s\n", msg.c_str());
        })) {
            std::fprintf(stderr, "Failed to prefill state\n");
            return 1;
        }
        std::printf("Prefill complete.\n");
    }

    engine.set_temperature(temperature);
    engine.set_top_k(top_k);

    engine.set_cfg_musiccoca(cfg_musiccoca);
    engine.set_cfg_notes(cfg_notes);
    engine.set_cfg_drums(cfg_drums);
    engine.set_drumless(drumless);
    engine.set_unmask_width(unmask_width);
    engine.set_seed_rotation(seed_rotation);
    engine.reset_state();

    engine.set_text_prompt(prompt);

    // MusicCoCa text encoding runs on a background thread. Status 1 = in-flight,
    // 2 = success, 3 = error. Block until both encoder and quantizer settle.
    while (engine.get_text_encoder_status() == 1 ||
           engine.get_quantizer_status() == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::printf("Prompt ready. Generating %d frames...\n", num_frames);

    std::vector<float> interleaved;
    interleaved.reserve(static_cast<size_t>(num_frames) * kFrameSamples * kNumChannels);
    std::vector<float> L(kFrameSamples), R(kFrameSamples);
    for (int f = 0; f < num_frames; ++f) {
        if (!engine.generate_frame(L.data(), R.data())) {
            std::fprintf(stderr, "generate_frame failed at frame %d\n", f);
            return 1;
        }
        for (size_t i = 0; i < kFrameSamples; ++i) {
            interleaved.push_back(L[i]);
            interleaved.push_back(R[i]);
        }
        if ((f + 1) % 10 == 0) {
            std::printf("  frame %d / %d\n", f + 1, num_frames);
        }
    }

    // out_path is already defined and verified
    if (!write_wav(out_path, interleaved, 48000, static_cast<int>(kNumChannels))) {
        std::fprintf(stderr, "Failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("Wrote %s (%.2f seconds)\n",
                out_path.c_str(),
                static_cast<double>(interleaved.size() / kNumChannels) / 48000.0);
    return 0;
}
