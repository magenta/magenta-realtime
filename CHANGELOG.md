# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `flax.nnx` inference backend (`magenta_rt.nnx` package) for `mrt2` models.
  - Streaming `step` API with `MagentaRT2State` handles for continuation, support for mixed precision (`bfloat16`), and Linen-format safetensors weight loading.
  - Batch/offline generation support with `nnx.scan` (using `scan=True`).
  - Added CLI subcommand `mrt nnx generate`.
- Added a vendored minimal copy of the `AudioTree` container under `magenta_rt/_vendor/audiotree/` for fallback support when the PyPI package is not installed.
- Added demo notebook `notebooks/demo_magenta_rt_v2_nnx.ipynb` and scripts to compare latency between JAX Linen and JAX NNX.

### Changed

- **Breaking Change**: Replaced the old custom `Waveform` container (`[B, T, C]`, `tokens`) with `audiotree.AudioTree` (`[B, C, T]`, `codes`).
  - Layout changed from time-major `[B, T, C]` to channel-major `[B, C, T]`.
  - `wav.samples` (array) is now `audio_tree.waveform`.
  - `audio_tree.samples` is now the sample count (int) instead of the waveform array.
  - `wav.tokens` is now `audio_tree.codes`.
  - `as_mono`/`as_stereo` changed to `to_mono`/`to_stereo`.
  - `apply_gain`/`peak_normalize`/`compute_rms` moved from container methods to `magenta_rt.audio` functions.
  - Migrated JAX and MLX inference backends to use `AudioTree`.
  - Moved shared conditioning broadcast helpers to `magenta_rt.conditioning`.

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
