# Overview

Magenta RealTime 2 ships a bundle of native apps, plugins, and extensions built on the same portable C++ inference engine (`magentart::core`).

### [Jam (App)](jam_app.md), try this first!
An easy way to jump straight into playing and jamming with the model, packed with presets. Runs as a self-contained standalone app.

### [Collider (App)](collider.md)
Mix and mash prompts on a 2D surface and let your ideas collide to create new genres and sonic mixtures. Runs as a self-contained standalone app.

### [MRT2 (Plugin & App)](audio_unit_plugin.md)
DAW integration and raw access to the MRT2 model itself for maximum control. Ships as both an AUv3 instrument plugin and a [Standalone app](standalone_app.md).

### Creative Coding (Extensions)
Extensions for popular creative coding environments, giving you the tools to quickly build your own experiences:
* **Max MSP**: `mrt~.mxo`
* **Pure Data**: `mrt~.pd_darwin`
* **SuperCollider**: `MRT2.scx`

When you're ready to share a build with someone else, see
[Distributing the macOS apps](distributing.md).

If you want to build your own new application using the `magentart::core` inference engine, see the [Developer Guide](developer.md).

## Prerequisites

All three apps need Node (for the React UI build):

```bash
brew install node
```

They also need the shared resources (MusicCoCa, SpectroStream) and at least one
pre-exported model. Download them with the `mrt models` CLI — see
[Models & checkpoints](../models.md).

## Build & deploy

The build pattern is identical for every app — only the CMake target changes:

```bash
source .venv/bin/activate
cmake . -B build
cmake --build build --target <deploy-target> -j10
```

| App               | Deploy target            | Deployed to                          |
|-------------------|--------------------------|--------------------------------------|
| Audio Unit plugin | `deploy_mrt2_au`         | `~/Applications/MRT2 (AU).app`       |
| Standalone        | `deploy_mrt2_standalone` | `~/Applications/MRT2.app`            |
| Jam               | `deploy_mrt2_jam`        | `~/Applications/MRT2 - Jam.app`      |
| Collider          | `deploy_mrt2_collider`   | `~/Applications/MRT2 - Collider.app` |

Each `deploy_*` target signs the bundle with an ad-hoc signature
(`CODESIGN_IDENTITY=-`), which is fine on your own machine. To run a build on
another Mac, see [Distributing the macOS apps](distributing.md).

The per-app pages cover anything specific to that app — how to register
or launch it, and how to use it.
