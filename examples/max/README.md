# mrt2~ — MaxMSP external for the realtime music model

A MaxMSP signal external (`mrt2~`) that wraps the realtime music model in this
repo. Stereo audio output (no audio in); all parameters that the AU plugin
exposes as sliders are exposed here as Max messages.

## Build

The MaxMSP target is wired into the top-level `CMakeLists.txt`, so the existing
CMake build directory builds it alongside the AU plugin, standalone app, and
the other host examples.

```sh
mkdir -p build
cd build
cmake ..
make mrt2_max -j8
```

The build produces `build/examples/max/mrt2~.mxo`.

## Install

Copy `mrt2~.mxo` into a directory Max searches. The simplest place is
`~/Documents/Max 9/Library/` (or `Max 8` if you're on the older version):

```sh
ditto build/examples/max/mrt2~.mxo "$HOME/Documents/Max 9/Library/mrt2~.mxo"
```

Or use the optional CMake target:

```sh
make deploy_mrt2_max          # copies into the path set by MAX_LIBRARY_DIR
```

You can override the destination at configure time:

```sh
cmake .. -DMAX_LIBRARY_DIR="$HOME/Documents/Max 8/Library"
```

## Use

In Max:

1. Set the audio sample rate to **48000 Hz** (Options → Audio Status). The model
   produces audio at 48 kHz; running Max at 44.1 kHz makes output play ~8.8% slow.
2. Drop `[mrt2~]` in a patcher and wire its two outlets to `[ezdac~]`.
   On creation the external automatically loads resources and the default
   model (`mrt2_base`) from `~/Documents/Magenta/magenta-rt-v2/`. You can
   override via creation args (`[mrt2~ /path/to/resources /path/to/model.mlxfn]`)
   or messages (`assets <dir>`, `model <path>`).
3. Send a prompt:
   - `prompt 0 "heavy metal" 1.0` — sets prompt slot 0
4. Open `mrt2~.maxhelp` for a working example patcher.

## Message reference

| Message | Args | Effect |
|---|---|---|
| `assets <dir>` | symbol | Load TFLite encoders + tokenizer from `<dir>` (point at `resources/` containing `musiccoca/`). |
| `model <path>` | symbol | Load `.mlxfn` transformer. |
| `prompt <N> "<text>" <weight>` | int, sym, float | Slot `N` (0–5), weight ∈ [0, 1]. |
| `prompt <N>` | int | Clear slot `N`. |
| `temperature <f>` | float | Sampling temperature (default 1.0). |
| `topk <i>` | int | Top-K sampling (default 100). |
| `cfgmusiccoca <f>` / `cfgnotes <f>` / `cfgdrums <f>` | float | Classifier-free guidance scales. |
| `unmaskwidth <i>` | int | MIDI unmask width. |
| `volume <db>` | float | Output volume in dB (default 0). |
| `mute <0/1>` | int | Mute output (smoothed). |
| `bypass <0/1>` | int | Pause inference and output silence. |
| `buffersize <samples>` | int | Inference→audio ring buffer capacity. Bigger = more headroom against dropouts, more output latency. AU plugin uses 2048 / 4096 / 8192 (≈43 / 85 / 170 ms @ 48 kHz); default 8192. |
| `reset` | — | Reset transformer state. |
| `noteon <i>` / `noteoff <i>` | int | MIDI-style note on/off. |
| `midigate <0/1>` | int | Enable MIDI gate envelope. |
| `drumless <0/1>` | int | Drumless mode toggle. |

The prompt surface (cursor X/Y, prompt X/Y, falloff) is intentionally not exposed —
you set per-slot weights directly via `prompt`.

## Known limitations

- macOS / Apple Silicon only. The model uses MLX (Metal) and TFLite.
- Sample rate is hardcoded at 48 kHz internally; running Max at any other SR
  plays the output at the wrong speed.
- One model checkpoint per object instance. The `mlx.metallib` is bundled into
  the `.mxo` next to the binary; loading multiple `mrt2~` instances in one
  patcher is supported but each holds its own engine + ~1 GB of weights.
- First-time `assets`/`model` messages briefly stall the Max UI while loading
  TFLite + MLX state; subsequent prompt updates are streamed asynchronously.
- Audio prompts and the prefill API are not yet exposed; only text prompts.
