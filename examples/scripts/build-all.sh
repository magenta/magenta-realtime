#!/bin/bash
# Copyright 2026 Google LLC

# Build and deploy all example applications and plugins to their respective locations,
# signing them using the Developer ID Application certificate.
#
# Usage: ./scripts/build-all.sh [0-3]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CMAKE_CMD="$REPO_ROOT/.venv/bin/cmake"
BUILD_DIR="$REPO_ROOT/build"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
MODE=""

for arg in "$@"; do
  if [[ "$arg" =~ ^[0-3]$ ]]; then
    MODE="$arg"
  fi
done

if [ -z "$MODE" ]; then
  echo "Which targets would you like to build?"
  echo "  0) All apps & targets"
  echo "  1) Plugin"
  echo "  2) Jam"
  echo "  3) Collider"
  echo ""
  read -p "Enter choice [0-3, default 0]: " MODE
  MODE="${MODE:-0}"
fi

if [[ ! "$MODE" =~ ^[0-3]$ ]]; then
  echo "Invalid option: $MODE"
  exit 1
fi

echo "Building and codesigning MRT project targets..."
echo "Using CMake: $CMAKE_CMD"
echo "Build directory: $BUILD_DIR"
echo "Parallel jobs: $CORES"
echo ""

build_target() {
    local target=$1
    local name=$2
    echo "================================================================================"
    echo "Building $name (target: $target)..."
    echo "================================================================================"
    "$CMAKE_CMD" --build "$BUILD_DIR" --target "$target" -j "$CORES"
    echo "✓ Finished $name"
    echo ""
}

# 1. Standalone / AUv3
if [[ "$MODE" == 0 || "$MODE" == 1 ]]; then
  build_target "deploy_mrt2_au" "AUv3"
  build_target "deploy_mrt2_standalone" "Standalone"
fi

# 2. Jam
if [[ "$MODE" == 0 || "$MODE" == 2 ]]; then
  build_target "deploy_mrt2_jam" "Jam App"
fi

# 3. Collider
if [[ "$MODE" == 0 || "$MODE" == 3 ]]; then
  build_target "deploy_mrt2_collider" "Collider App"
fi

# 5. Audio Host Externals (Max, Pd, SuperCollider)
if [[ "$MODE" == 0 ]]; then
  build_target "deploy_mrt2_max" "Max MSP external (mrt~.mxo)"
  build_target "deploy_mrt2_pd" "Pure Data external (mrt~.pd_darwin)"
  build_target "deploy_mrt2_sc" "SuperCollider UGen (MRT2.scx)"
fi

# 6. CLI examples
if [[ "$MODE" == 0 ]]; then
  build_target "hello_mrt2" "Minimal CLI (hello_mrt2)"
fi

echo "================================================================================"
echo "✓ Selected MRT projects successfully built, codesigned, and deployed!"
echo "================================================================================"
