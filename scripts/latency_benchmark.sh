#!/bin/bash
# Copyright 2026 Google LLC

MLXFN_NAME=mrt2_base_test
MODEL=mrt2_base

# export mlxfn
mrt mlx export --skip-restore --output-name=${MLXFN_NAME} --model=${MODEL} --num-codebooks=12 --bits=8 --num-cfgs=0

# benchmark
./benchmark_build/benchmark_mlxfn ~/Documents/Magenta/magenta-rt-v2/models/mrt2_base 100
