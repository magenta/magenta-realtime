#!/bin/bash
# Copyright 2026 Google LLC

# Clear app defaults for Jam, Collider, and Standalone 1 & 2.
defaults delete com.magentart.jam 2>/dev/null
defaults delete com.google.mrt2.collider 2>/dev/null
defaults delete com.google.magentart.standalone 2>/dev/null
defaults delete com.google.magentart.standalone2 2>/dev/null
echo "Cleared app defaults successfully."
