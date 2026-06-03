#!/bin/bash

# Copyright 2026 Google LLC

CHKP_NAME=mrt2_small
MLXFN_NAME=mrt2_small_test
MODEL=mrt2_small

# download checkpoints
mrt checkpoints download ${CHKP_NAME}.safetensors

# generate a 4s sample in jax
mrt jax generate  --model=${CHKP_NAME}

# export mlxfn with quantization
mrt mlx export --checkpoint=${CHKP_NAME}.safetensors --output-name=${MLXFN_NAME} \
    --model=${MODEL} \
    --bits=8 --num-codebooks=12 --num-cfgs=0

# bulk generate
python scripts/bulk_generate.py --size=${MODEL}
