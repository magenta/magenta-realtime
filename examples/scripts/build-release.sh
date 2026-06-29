#!/bin/bash
# Copyright 2026 Google LLC

# Build all MRT2 applications, plugins, and externals, sign them,
# and package them into the 5 ZIP files required for web distribution.
#
# Usage: ./examples/scripts/build-release.sh [output_directory] [--keychain-profile <profile>]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CMAKE_CMD="$REPO_ROOT/.venv/bin/cmake"
BUILD_DIR="$REPO_ROOT/build"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")

# Parse arguments
OUT_DIR=""
KEYCHAIN_PROFILE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keychain-profile)
            KEYCHAIN_PROFILE="$2"
            shift
            ;;
        -*)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
        *)
            if [ -z "$OUT_DIR" ]; then
                OUT_DIR="$1"
            else
                echo "Unknown parameter passed: $1"
                exit 1
            fi
            ;;
    esac
    shift
done

# Set default OUT_DIR if not provided
OUT_DIR="${OUT_DIR:-$REPO_ROOT/dist}"

# Convert OUT_DIR to absolute path if it is relative
if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$(cd "$(dirname "$OUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUT_DIR")"
fi

STAGE_DIR="$BUILD_DIR/stage_web_dist"

# Verify CMake exists
if [ ! -x "$CMAKE_CMD" ]; then
    echo "ERROR: CMake command not found or not executable at '$CMAKE_CMD'"
    echo "Please ensure you have configured your environment and run setup first."
    exit 1
fi

echo "================================================================================"
echo "Building, Notarizing, and packaging MRT2 for web distribution..."
echo "Build directory:  $BUILD_DIR"
echo "Output directory: $OUT_DIR"
echo "Parallel jobs:    $CORES"
echo "================================================================================"

# Ensure clean output and staging directories
mkdir -p "$OUT_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Ensure deployment parent directories exist before trying to write to them
mkdir -p "$HOME/Documents/Max 9/Library"
mkdir -p "$HOME/Documents/Pd/externals"
mkdir -p "$HOME/Library/Application Support/SuperCollider/Extensions"

# 1. Build and notarize all required targets via notarize-all.sh
if [ -n "$KEYCHAIN_PROFILE" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "Running notarization for all targets..."
    echo "--------------------------------------------------------------------------------"
    bash "$SCRIPT_DIR/notarize-all.sh" --keychain-profile "$KEYCHAIN_PROFILE"
    echo "✓ Notarization completed successfully!"
else
    echo "--------------------------------------------------------------------------------"
    echo "No keychain profile provided. Running unnotarized build..."
    echo "--------------------------------------------------------------------------------"
    bash "$SCRIPT_DIR/build-all.sh" 0
    echo "✓ Unnotarized build completed successfully!"
fi
echo ""

# Source paths of built/deployed files
APP_DIR="$HOME/Applications"
COLLIDER_APP="$APP_DIR/MRT2 - Collider.app"
JAM_APP="$APP_DIR/MRT2 - Jam.app"
STANDALONE_APP="$APP_DIR/MRT2.app"
AU_APP="$APP_DIR/MRT2 (AU).app"

MAX_EXTERNAL="$HOME/Documents/Max 9/Library/mrt2~.mxo"
PD_EXTERNAL="$HOME/Documents/Pd/externals/mrt2~"
SC_EXTERNAL="$HOME/Library/Application Support/SuperCollider/Extensions/MRT2"

# Helper to verify files exist before zipping
verify_exists() {
    local path=$1
    local name=$2
    if [ ! -e "$path" ]; then
        echo "ERROR: $name not found at '$path'."
        echo "Did the build target run successfully?"
        exit 1
    fi
}

echo "================================================================================"
echo "Creating ZIP files..."
echo "================================================================================"

# --- 1. MRT2 - Collider.zip ---
echo "Creating: MRT2 - Collider.zip..."
verify_exists "$COLLIDER_APP" "MRT2 - Collider App"
rm -f "$OUT_DIR/MRT2 - Collider.zip"
(cd "$APP_DIR" && zip -ryq "$OUT_DIR/MRT2 - Collider.zip" "MRT2 - Collider.app")
echo "✓ Created MRT2 - Collider.zip"

# --- 2. MRT2 - Jam.zip ---
echo "Creating: MRT2 - Jam.zip..."
verify_exists "$JAM_APP" "MRT2 - Jam App"
rm -f "$OUT_DIR/MRT2 - Jam.zip"
(cd "$APP_DIR" && zip -ryq "$OUT_DIR/MRT2 - Jam.zip" "MRT2 - Jam.app")
echo "✓ Created MRT2 - Jam.zip"

# --- 3. MRT2 (Plugin & App).zip ---
echo "Creating: MRT2 (Plugin & App).zip..."
verify_exists "$STANDALONE_APP" "MRT2 Standalone App"
verify_exists "$AU_APP" "MRT2 AU Host App"
verify_exists "$REPO_ROOT/examples/mrt2/auv3/INSTALL.md" "AUv3 INSTALL.md"

STAGE_PLUGIN_APP="$STAGE_DIR/plugin_app"
mkdir -p "$STAGE_PLUGIN_APP"
ditto "$STANDALONE_APP" "$STAGE_PLUGIN_APP/MRT2.app"
ditto "$AU_APP" "$STAGE_PLUGIN_APP/MRT2 (AU).app"
cp "$REPO_ROOT/examples/mrt2/auv3/INSTALL.md" "$STAGE_PLUGIN_APP/INSTALL.md"

rm -f "$OUT_DIR/MRT2 (Plugin & App).zip"
(cd "$STAGE_PLUGIN_APP" && zip -ryq "$OUT_DIR/MRT2 (Plugin & App).zip" "MRT2.app" "MRT2 (AU).app" "INSTALL.md")
echo "✓ Created MRT2 (Plugin & App).zip"

# --- 4. MRT2 Bundle.zip ---
echo "Creating: MRT2 Bundle.zip..."
verify_exists "$JAM_APP" "MRT2 - Jam App"
verify_exists "$COLLIDER_APP" "MRT2 - Collider App"
verify_exists "$AU_APP" "MRT2 AU Host App"
verify_exists "$REPO_ROOT/examples/mrt2/auv3/INSTALL.md" "AUv3 INSTALL.md"

STAGE_BUNDLE="$STAGE_DIR/bundle"
mkdir -p "$STAGE_BUNDLE/AudioUnit"
ditto "$JAM_APP" "$STAGE_BUNDLE/MRT2 - Jam.app"
ditto "$COLLIDER_APP" "$STAGE_BUNDLE/MRT2 - Collider.app"
ditto "$AU_APP" "$STAGE_BUNDLE/AudioUnit/MRT2 (AU).app"
cp "$REPO_ROOT/examples/mrt2/auv3/INSTALL.md" "$STAGE_BUNDLE/AudioUnit/INSTALL.md"

rm -f "$OUT_DIR/MRT2 Bundle.zip"
(cd "$STAGE_BUNDLE" && zip -ryq "$OUT_DIR/MRT2 Bundle.zip" "MRT2 - Jam.app" "MRT2 - Collider.app" "AudioUnit")
echo "✓ Created MRT2 Bundle.zip"

# --- 5. CreativeCodeExternals.zip ---
echo "Creating: CreativeCodeExternals.zip..."
verify_exists "$MAX_EXTERNAL" "Max MSP External"
verify_exists "$PD_EXTERNAL" "Pure Data External"
verify_exists "$SC_EXTERNAL" "SuperCollider UGen"

STAGE_EXTERNALS="$STAGE_DIR/CreativeCodeExternals"
mkdir -p "$STAGE_EXTERNALS/max"
mkdir -p "$STAGE_EXTERNALS/pd"
mkdir -p "$STAGE_EXTERNALS/sc"

# Copy externals directly instead of zipping them individually
echo "  Staging Max files..."
verify_exists "$REPO_ROOT/examples/max/mrt2~.maxhelp" "Max Help Patch"
verify_exists "$REPO_ROOT/examples/max/README.md" "Max README"
ditto "$MAX_EXTERNAL" "$STAGE_EXTERNALS/max/mrt2~.mxo"
cp "$REPO_ROOT/examples/max/mrt2~.maxhelp" "$STAGE_EXTERNALS/max/mrt2~.maxhelp"
cp "$REPO_ROOT/examples/max/README.md" "$STAGE_EXTERNALS/max/README.md"

echo "  Staging Pd files..."
verify_exists "$REPO_ROOT/examples/pd/README.md" "Pd README"
ditto "$PD_EXTERNAL" "$STAGE_EXTERNALS/pd"
cp "$REPO_ROOT/examples/pd/README.md" "$STAGE_EXTERNALS/pd/README.md"

echo "  Staging SuperCollider files..."
verify_exists "$REPO_ROOT/examples/sc/README.md" "SC README"
ditto "$SC_EXTERNAL" "$STAGE_EXTERNALS/sc"
cp "$REPO_ROOT/examples/sc/README.md" "$STAGE_EXTERNALS/sc/README.md"

echo "  Staging root INSTALL.md..."
verify_exists "$REPO_ROOT/examples/max/INSTALL.md" "Max INSTALL.md"
cp "$REPO_ROOT/examples/max/INSTALL.md" "$STAGE_EXTERNALS/INSTALL.md"

rm -f "$OUT_DIR/CreativeCodeExternals.zip"
(cd "$STAGE_DIR" && zip -ryq "$OUT_DIR/CreativeCodeExternals.zip" "CreativeCodeExternals")
echo "✓ Created CreativeCodeExternals.zip"

# Clean up staging
rm -rf "$STAGE_DIR"

echo "================================================================================"
echo "✓ All 5 ZIP packages successfully created in: $OUT_DIR"
ls -la "$OUT_DIR"/*.zip
echo ""
echo "Next step: Upload these ZIP files to Google Cloud Storage:"
echo "  gcloud storage cp \"$OUT_DIR\"/*.zip gs://magenta-rt-public/magenta-rt-2/downloads"
echo "================================================================================"
