# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.3] - 2026-07-30

- Fix prefilling forced token bug
- Increase modularity on the MRT2 model input/conditioning configs
- Add mrt checkpoints download CLI
- Improvements and bug fixes to examples

## [2.0.2] - 2026-06-04

### Added

- First release of Magenta RealTime 2.
- - Model weights via Hugging Face
- - PyPI package `magenta-rt[mlx]` supports JAX and optionally MLX
- - C++ inference library
- New examples:
- - Jam v0.0.1
- - Collider v0.0.1
- - Audio Unit (AUv3) v0.0.1
- - Standalone v0.0.1
- - Pure Data v0.0.1
- - SuperCollider v0.0.1
- - Max/MSP v0.0.1

### Removed

- Removed Magenta RealTime 1. It can be found at the branch [v1_legacy](https://github.com/magenta/magenta-realtime/commits/v1_legacy).
