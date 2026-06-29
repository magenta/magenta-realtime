#!/bin/bash
# Copyright 2026 Google LLC

# Clear app defaults for Jam, Collider, Standalone, and AUv3.
defaults delete com.google.mrt2.jam 2>/dev/null
defaults delete com.google.mrt2.collider 2>/dev/null
defaults delete com.google.mrt2.standalone 2>/dev/null
defaults delete com.google.mrt2.au 2>/dev/null
defaults delete com.google.mrt2.au.host 2>/dev/null
echo "Cleared app defaults successfully."
