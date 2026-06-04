# mrt2~ — Pure Data external for the realtime music model

A Pure Data signal external (`mrt2~`) that wraps the realtime music model in this
repo. Stereo audio output (no audio in); all parameters that the AU plugin
exposes as sliders are exposed here as Pd messages. Mirrors the MaxMSP external
under `../max/` — same engine, same message surface, translated to Pd's
calling convention.

## Build

The Pd target is wired into the top-level `CMakeLists.txt`, so the existing
CMake build directory builds it alongside the AU plugin, standalone app, and
MaxMSP external.

```sh
mkdir -p build
cd build
cmake ..
make mrt2_pd -j8
```

The build produces `build/examples/pd/mrt2~.pd_darwin` plus a colocated `mlx.metallib`.

## Install

Pd looks for externals in its search paths (Preferences → Path). The simplest
install puts the external + its metallib + help patch into a `mrt2~/`
subdirectory under your user externals folder:

```sh
ditto build/examples/pd "$HOME/Documents/Pd/externals/mrt2~"
cp examples/pd/mrt2~-help.pd "$HOME/Documents/Pd/externals/mrt2~/mrt2~-help.pd"
```

Or use the optional CMake target:

```sh
make deploy_mrt2_pd          # copies into the path set by PD_LIBRARY_DIR
```

You can override the destination at configure time:

```sh
cmake .. -DPD_LIBRARY_DIR="$HOME/pd-externals"
```

Then add `~/Documents/Pd/externals/mrt2~` (or whatever you used) to Pd's search
path. `mlx.metallib` must remain next to `mrt2~.pd_darwin` — MLX uses `dladdr`
to locate its kernels at runtime.

## Use

In Pd:

1. Set the audio sample rate to **48000 Hz** (Media → Audio Settings). The
   model produces audio at 48 kHz; running Pd at 44.1 kHz makes output play
   ~8.8% slow.
2. Drop `[mrt2~]` in a patcher and wire its two outlets to `[dac~]`.
   On creation the external automatically loads resources and the default
   model (`mrt2_base`) from `~/Documents/Magenta/magenta-rt-v2/`. You can
   override via creation args (`[mrt2~ /path/to/resources /path/to/model.mlxfn]`)
   or messages (`assets <dir>`, `model <path>`).
3. Send a prompt:
   - `prompt 0 piano 1.0` — sets prompt slot 0
4. Open `mrt2~-help.pd` for a working example patcher.

## Message reference

| Message | Args | Effect |
|---|---|---|
| `assets <dir>` | symbol | Load TFLite encoders + tokenizer from `<dir>` (point at `resources/` containing `musiccoca/`). |
| `model <path>` | symbol | Load `.mlxfn` transformer. |
| `prompt <N> <text…> <weight>` | float, sym…, float | Slot `N` (0–5), weight ∈ [0, 1]. Multi-word texts are joined with spaces (no quoting needed in Pd). |
| `prompt <N>` | float | Clear slot `N`. |
| `temperature <f>` | float | Sampling temperature (default 1.0). |
| `topk <i>` | float (cast to int) | Top-K sampling (default 100). |
| `cfgmusiccoca <f>` / `cfgnotes <f>` / `cfgdrums <f>` | float | Classifier-free guidance scales. |
| `unmaskwidth <i>` | float (cast to int) | MIDI unmask width. |
| `volume <db>` | float | Output volume in dB (default 0). |
| `mute <0/1>` | float | Mute output (smoothed). |
| `bypass <0/1>` | float | Pause inference and output silence. |
| `buffersize <samples>` | float | Inference→audio ring buffer capacity. Bigger = more headroom against dropouts, more output latency. AU plugin uses 2048 / 4096 / 8192 (≈43 / 85 / 170 ms @ 48 kHz); default 8192. |
| `reset` | — | Reset transformer state. |
| `noteon <i>` / `noteoff <i>` | float | MIDI-style note on/off. |
| `midigate <0/1>` | float | Enable MIDI gate envelope. |
| `drumless <0/1>` | float | Drumless mode toggle. |

The prompt surface (cursor X/Y, prompt X/Y, falloff) is intentionally not exposed —
you set per-slot weights directly via `prompt`.

### Multi-word prompts

Pd message boxes don't have quoted strings, so the `prompt` parser treats every
symbol atom between the slot index and the trailing weight float as part of the
prompt text:

```
prompt 0 lo-fi hip hop 1.0     →  slot 0, text = "lo-fi hip hop", weight = 1.0
prompt 1 jazz 0.5              →  slot 1, text = "jazz",          weight = 0.5
prompt 2 strings               →  slot 2, text = "strings",       weight = 1.0
```

If you omit the trailing float, weight defaults to 1.0.

## Known limitations

- macOS / Apple Silicon only. The model uses MLX (Metal) and TFLite.
- Sample rate is hardcoded at 48 kHz internally; running Pd at any other SR
  plays the output at the wrong speed.
- One model checkpoint per object instance. The `mlx.metallib` is colocated
  next to `mrt2~.pd_darwin`; loading multiple `mrt2~` instances in one patcher
  is supported but each holds its own engine + ~1 GB of weights.
- First-time `assets`/`model` messages briefly stall the Pd UI while loading
  TFLite + MLX state; subsequent prompt updates are streamed asynchronously.
- Audio prompts and the prefill API are not yet exposed; only text prompts.
- No latency reporting to Pd (Pd has no PDC channel for generator objects).
