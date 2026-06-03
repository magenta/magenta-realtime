#!/bin/bash
# Copyright 2026 Google LLC

# Script to notarize all MRT projects and externals into separate ZIP files.
# Requires a pre-configured notarytool keychain profile (default name: "notarytool-creds").

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CMAKE_CMD="$REPO_ROOT/.venv/bin/cmake"
BUILD_DIR="$REPO_ROOT/build"

KEYCHAIN_PROFILE="notarytool-creds"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keychain-profile) KEYCHAIN_PROFILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "================================================================================"
echo "Notarizing all MRT targets using profile: $KEYCHAIN_PROFILE"
echo "================================================================================"

# Verify keychain credentials exist
if ! xcrun notarytool history --keychain-profile "$KEYCHAIN_PROFILE" >/dev/null 2>&1; then
    echo "ERROR: Keychain profile '$KEYCHAIN_PROFILE' not found or credentials invalid."
    echo "Please set it up first by running:"
    echo "  xcrun notarytool store-credentials \"$KEYCHAIN_PROFILE\" \\"
    echo "      --apple-id \"<AppleID>\" --team-id <TEAMID> --password <app-specific-password>"
    exit 1
fi

notarize_cmake_target() {
    local target=$1
    local name=$2
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Notarizing $name (target: $target)..."
    echo "--------------------------------------------------------------------------------"
    "$CMAKE_CMD" --build "$BUILD_DIR" --target "$target"
}

notarize_manual_bundle() {
    local source_path=$1
    local zip_name=$2
    local label=$3
    local is_bundle=$4 # true if we can staple the bundle directly
    local build_target=$5 # CMake target to compile/deploy first
    local zip_path="$BUILD_DIR/$zip_name"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Building and Notarizing $label (manual package)..."
    echo "--------------------------------------------------------------------------------"

    echo "Building and deploying target $build_target..."
    "$CMAKE_CMD" --build "$BUILD_DIR" --target "$build_target"

    if [ ! -e "$source_path" ]; then
        echo "Error: Source path not found after build: $source_path"
        exit 1
    fi

    echo "Packaging $source_path -> $zip_path..."
    rm -f "$zip_path"
    ditto -c -k --keepParent "$source_path" "$zip_path"

    echo "Submitting to Apple Notary Service..."
    xcrun notarytool submit "$zip_path" --keychain-profile "$KEYCHAIN_PROFILE" --wait

    if [ "$is_bundle" = "true" ]; then
        echo "Stapling notarization ticket to bundle..."
        xcrun stapler staple "$source_path"
        xcrun stapler validate "$source_path"

        # Re-zip with the stapled ticket inside
        echo "Re-packaging stapled bundle..."
        rm -f "$zip_path"
        ditto -c -k --keepParent "$source_path" "$zip_path"
    else
        echo "Note: Stapling is not applicable to non-app bundles. Notarization ticket registered online."
    fi

    echo "✓ Finished notarizing $label. Output: $zip_path"
}

# 1. Notarize the 4 App Bundles via CMake targets
notarize_cmake_target "notarize_mrt2_standalone" "Standalone"
notarize_cmake_target "notarize_mrt2_au" "AUv3"
notarize_cmake_target "notarize_mrt2_jam" "Jam App"
notarize_cmake_target "notarize_mrt2_collider" "Collider App"

# 2. Notarize the 3 Audio Externals manually
notarize_manual_bundle "$HOME/Documents/Max 9/Library/mrt~.mxo" "MRT2_Max.zip" "Max MSP External" "true" "deploy_mrt2_max"
notarize_manual_bundle "$HOME/Documents/Pd/externals/mrt~" "MRT2_Pd.zip" "Pure Data External" "false" "deploy_mrt2_pd"
notarize_manual_bundle "$HOME/Library/Application Support/SuperCollider/Extensions/MRT2" "MRT2_SuperCollider.zip" "SuperCollider UGen" "false" "deploy_mrt2_sc"

echo ""
echo "================================================================================"
echo "✓ All ZIP files successfully notarized and located in: $BUILD_DIR"
ls -la "$BUILD_DIR"/*.zip
echo "================================================================================"
