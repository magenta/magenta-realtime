#!/bin/bash

# Copyright 2026 Google LLC

# Exit on error
set -e

DATE_PREFIX=$(date +'%y%m%d')
APP_DIR="$HOME/Applications"
OUT_DIR="$APP_DIR/$DATE_PREFIX"

# Array of source apps
APPS=(
    "$APP_DIR/MRT2.app"
    "$APP_DIR/MRT2 (AU).app"
    "$APP_DIR/MRT2 - Jam.app"
    "$APP_DIR/MRT2 - Collider.app"
)

echo "Starting compression of Magenta RT apps..."
echo "Date prefix: $DATE_PREFIX"
echo "Output directory: $OUT_DIR"

# Ensure output subfolder exists
mkdir -p "$OUT_DIR"

for app in "${APPS[@]}"; do
    if [ -d "$app" ]; then
        # Get the base name (e.g. "Magenta RT AUv3 2")
        base_name=$(basename "$app" .app)

        # Replace spaces with underscores for clean filenames
        clean_name=$(echo "$base_name" | tr ' ' '_')

        zip_file="$OUT_DIR/${DATE_PREFIX}-${clean_name}.zip"

        echo "----------------------------------------"
        echo "Compressing: $base_name"
        echo "To: $zip_file"

        # cd into APP_DIR first so zip doesn't contain absolute path structures
        (cd "$APP_DIR" && zip -rq "$zip_file" "$(basename "$app")")

        echo "Successfully created: $(basename "$zip_file")"
    else
        echo "----------------------------------------"
        echo "Warning: App not found at '$app'. Skipping."
    fi
done

echo "----------------------------------------"
echo "All done! Zips are located in $OUT_DIR."

# Tag the current commit so we know exactly which code produced these builds
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG_NAME="apps-${DATE_PREFIX}"
git -C "$SCRIPT_DIR" tag -f "$TAG_NAME" -m "Release ${DATE_PREFIX}"
echo "Tagged current commit as: $TAG_NAME"
