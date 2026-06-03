# Standalone App

The standalone app provides raw access to the MRT2 model itself for maximum control with its own audio output and MIDI input — no DAW required. See [macOS apps](index.md) for the shared prerequisites and build pattern.

## Build & deploy

```bash
source .venv/bin/activate
cmake . -B build
cmake --build build --target deploy_mrt2_standalone -j10
```

After a successful build, the app is deployed to
`~/Applications/MRT2.app`:

```bash
open ~/Applications/MRT2.app
```

## Use the standalone app

1. Launch **MRT2.app**.
2. Click **"Load Model…"** (or use *File → Load Model*) and select the exported model folder or `.mlxfn` file.
3. Audio plays through the system default output. The app creates a virtual MIDI port called **"Magenta RT Input"** — any MIDI source can route to it.
4. Open *Settings* (Cmd+,) to select an audio output device and connect to physical MIDI sources.

```{note}
The standalone app persists the last loaded model path, text prompts, and
prompt surface layout in `NSUserDefaults`. On relaunch it will auto-load the
previous model.
```
