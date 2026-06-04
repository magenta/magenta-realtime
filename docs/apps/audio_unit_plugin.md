# Audio Unit Plugin

The Audio Unit plugin provides DAW integration and raw access to the MRT2 model itself for maximum control. It is an AUv3 instrument that runs inside a DAW. See [macOS apps](index.md) for the shared prerequisites and build pattern.

## Prerequisites

```bash
brew install node
```

## Build & deploy

```bash
source .venv/bin/activate
cmake . -B build
cmake --build build --target deploy_mrt2_au -j10
```

After a successful build, the plugin is deployed to
`~/Applications/MRT2 (AU).app`. Open it once to register the Audio Unit:

```bash
open ~/Applications/MRT2\ \(AU\).app
```

```{note}
If the plugin fails to load in your DAW, macOS's `pluginkit` may have cached a
stale or broken build (e.g., from your Downloads folder). You can verify
registered plugin paths by running `pluginkit -mAv | grep -i mrt`. If it
points to an incorrect path, unregister the bad one with
`pluginkit -r <bad_path>` and explicitly register the correct one with
`pluginkit -a "/Applications/MRT2 (AU).app/Contents/PlugIns/MRT2_AU.appex"`.
```

## Use the plugin

1. In **Ableton Live**, go to *Settings → Plug-Ins* and enable **"Use Audio Unit v3"**, then **"Rescan Plug-Ins"**.
2. Load the MagentaRT instrument on a track.
3. Click **"Load Model…"** and select the exported model *folder* (e.g. `~/Documents/Magenta/magenta-rt-v2/models/mrt2_base/`).
