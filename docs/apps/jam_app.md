# Jam App

Jam is an easy way to jump straight into playing and jamming with the model, packed with presets. Compared to the main MRT2 app, Jam has a more minimal interface. It supports the same models, with built-in audio output and MIDI input. See [macOS apps](index.md) for the shared prerequisites and build pattern.

## Build & deploy

```bash
source .venv/bin/activate
cmake . -B build
cmake --build build --target deploy_mrt2_jam -j10
```

After a successful build, the app is deployed to `~/Applications/MRT2 - Jam.app`.
