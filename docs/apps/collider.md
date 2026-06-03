# Collider

Collider is a standalone app to mix and mash prompts on a 2D surface and let your ideas collide to create new genres and sonic mixtures. It reuses the [standalone app](standalone_app.md)'s inference engine and audio output. See [macOS apps](index.md) for the shared prerequisites and build pattern.

Its distinguishing feature is the **Prompt Surface**: a 2D canvas where you place
several text prompts as points and move a query location around them.
The model blends all the prompts at once, weighting each by how close the query is to it — so dragging toward one prompt morphs the output toward that style, and sitting between several mixes them.
A generation-params panel (temperature, top-k, CFG weight for MusicCoCa, unmask width, buffer size) rounds out the app.

## Build & deploy

```bash
source .venv/bin/activate
cmake . -B build
cmake --build build --target deploy_mrt2_collider -j10
```

After a successful build, the app is deployed to
`~/Applications/Collider.app`:

```bash
open ~/Applications/MRT2\ -\ Collider.app
```

## Use Collider

1. Launch **Collider.app**.
2. Click **"Load Model…"** and select the exported model folder or `.mlxfn` file.
3. Place text prompts on the Prompt Surface and drag the query location to blend between them. Audio plays through the system default output.
4. Open *Settings* (Cmd+,) to tune generation parameters and select the audio output device.
