# Magenta RealTime 2

[![CLI Tests](https://github.com/magenta/magenta-realtime/actions/workflows/cli_tests.yml/badge.svg)](https://github.com/magenta/magenta-realtime/actions/workflows/cli_tests.yml)

Magenta RealTime 2 is a state-of-the-art open-weights model for real-time music generation.
Use this project to run inference, build a DAW plugin, or embed the model into new applications of your imagination.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`unlock;1.5em;sd-mr-1` Open weights
An open-weights real-time music generation model.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` JAX & MLX
On-device inference implementations for both JAX and MLX.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` C++ inference engine
Efficient streaming audio generation library on Apple Silicon.
:::

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Example applications
AUv3 plugin, standalone app, Jam, and Collider.
Download them [here](https://magenta.withgoogle.com/mrt2).
:::
::::

## Repo Highlights

- `magenta_rt/` — Python inference library (JAX / MLX backends).
- `cpp/` — C++ inference library (`magentart::core`).
- `examples/mrt2/auv3/` — macOS AUv3 plugin for DAW users.
- `examples/mrt2/standalone/` — macOS standalone app.
- `examples/jam/` — specialized standalone app.
- `examples/collider/` — standalone app for dynamically mixing prompts.
- `notebooks/` - demo notebooks for inference in Python.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Get started
installation
models
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Inference
inference
exporting
benchmark
testing
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: macOS apps
apps/index
apps/audio_unit_plugin
apps/standalone_app
apps/jam_app
apps/collider
apps/distributing
apps/developer
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: About
changelog
```
