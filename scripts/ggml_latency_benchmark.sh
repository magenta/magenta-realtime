#!/bin/bash
# Copyright 2026 Google LLC
#
# LLM-only latency benchmark for the ggml port (no SpectroStream codec yet, so
# not directly comparable to latency_benchmark.sh / benchmark_mlxfn).
#
# Usage:  scripts/ggml_latency_benchmark.sh [cpu] [rebuild]

set -e

MODEL=mrt2_small
DTYPE=q8
STEPS=100
MAGENTA_HOME=~/Documents/Magenta/magenta-rt-v2
GGUF=${MAGENTA_HOME}/models/${MODEL}/${MODEL}_${DTYPE}.gguf

# convert flax checkpoint -> gguf
# python -m magenta_rt.ggml.convert \
#   --checkpoint ${MAGENTA_HOME}/checkpoints/${MODEL}.safetensors \
#   --output ${GGUF} --model ${MODEL} --dtype ${DTYPE}

# build (own build dir: the repo root's build/ belongs to the top-level project)
cmake -B core/src/ggml/build -S core/src/ggml -DGGML_METAL=ON
cmake --build core/src/ggml/build -j --target mrt_bench

# benchmark
./core/src/ggml/build/mrt_bench ${GGUF} ${STEPS} "$@"
