# MRT2 — SuperCollider UGen for the realtime music model

A SuperCollider scsynth UGen plugin (`MRT2.scx`) that wraps the realtime
music model in this repo, plus a small sclang client class (`MRT2.sc`).
Stereo audio output (no audio in); every parameter that the AU plugin exposes
as a slider is exposed here as a per-UGen `/u_cmd` and surfaced as an idiomatic
sclang method on the `MRT2` wrapper. Mirrors the MaxMSP external under
`../max/` and the Pure Data external under `../pd/` — same engine, same
parameter surface, translated to scsynth's UGen + unit-command convention.

## Build

The SuperCollider target is wired into the top-level `CMakeLists.txt`, so the
existing CMake build directory builds it alongside the AU plugin, the standalone
app, and the MaxMSP/PD externals.

```sh
mkdir -p build
cd build
cmake ..
make mrt2_sc -j8
```

The build produces `build/examples/sc/MRT2.scx` plus a colocated
`mlx.metallib`. The first configure step shallow-clones SuperCollider 3.13 for
its public headers (no SC build runs); to point at an existing source tree
instead, configure with `-DSC_PATH=/path/to/supercollider`.

## Install

SuperCollider scans `~/Library/Application Support/SuperCollider/Extensions`
recursively at server boot. The simplest install drops the binary + metallib +
sclang class + example into a `MRT2/` subdirectory there:

```sh
ditto build/examples/sc "$HOME/Library/Application Support/SuperCollider/Extensions/MRT2"
cp examples/sc/MRT2.sc examples/sc/example.scd "$HOME/Library/Application Support/SuperCollider/Extensions/MRT2/"
```

Or use the optional CMake target:

```sh
make deploy_mrt2_sc          # copies into SC_EXTENSIONS_DIR
```

Override the destination at configure time:

```sh
cmake .. -DSC_EXTENSIONS_DIR="$HOME/sc-externals"
```

After installing, recompile the sclang class library (Language → Recompile
Class Library, or `Cmd-Shift-L`). `mlx.metallib` must remain next to
`MRT2.scx` — MLX uses `dladdr` to locate its kernels at runtime.

## Use

In SuperCollider:

1. Set the audio hardware sample rate to **48 000 Hz** (macOS Audio MIDI Setup
   → Format → 48000 Hz). The model produces audio at 48 kHz; running scsynth
   at 44.1 kHz makes output play ~8.8 % slow.
2. Configure and boot the server with stereo output:
   ```supercollider
   s.options.sampleRate = 48000;
   s.options.numOutputBusChannels = 2;
   s.boot;
   ```
3. Make an instance and send a prompt:
    ```supercollider
    ~mrt = MRT2.new(s);
    // Assets and default model (mrt2_base) auto-load from
    // ~/Documents/Magenta/magenta-rt-v2/. Override with:
    //   ~mrt.assets("/custom/path/resources");
    //   ~mrt.model("/custom/path/model.mlxfn");
    ~mrt.prompt(0, "piano", 1.0);
    ```
4. See `example.scd` for a full session.

## Command reference

Every method on the `MRT2` wrapper sends a single `/u_cmd <synthID>
<ugenIdx> <name> <args…>` OSC packet to the server, where it dispatches into
a per-UGen unit command registered by the .scx.

| sclang method | Args | Effect |
|---|---|---|
| `assets(path)` | string | Load TFLite encoders + tokenizer from `path` (point at `resources/` containing `musiccoca/`). |
| `model(path)` | string | Load `.mlxfn` transformer. |
| `prompt(slot, text, weight = 1.0)` | int, string, float | Slot `0–5`, weight `[0, 1]`. Strings can be multi-word — no escaping. |
| `clearPrompt(slot)` | int | Clear the slot. |
| `temperature(v)` | float | Sampling temperature (default 1.3). |
| `topk(v)` | int | Top-K sampling (default 40). |
| `cfgMusicCoCa(v)` / `cfgNotes(v)` / `cfgDrums(v)` | float | Classifier-free guidance scales. |
| `unmaskWidth(v)` | int | MIDI unmask width. |
| `volume(db)` | float | Output volume in dB (default 0). |
| `mute(0/1)` | int | Mute output (smoothed). |
| `bypass(0/1)` | int | Pause inference and output silence. |
| `bufferSize(samples)` | int | Inference→audio ring-buffer capacity. Bigger = more headroom against dropouts, more output latency. AU plugin uses 2048 / 4096 / 8192 (≈43 / 85 / 170 ms @ 48 kHz); default 8192. |
| `reset` | — | Reset transformer state. |
| `noteOn(n)` / `noteOff(n)` | int | MIDI-style note on/off. |
| `midiGate(0/1)` | int | Enable MIDI gate envelope. |
| `drumless(0/1)` | int | Drumless mode toggle. |
| `free` | — | Free the underlying synth. Subsequent commands warn and no-op. |

The prompt surface (cursor X/Y, prompt X/Y, falloff) is intentionally not exposed
— set per-slot weights directly via `prompt`.

## Embedding inside your own SynthDef

`MRT2.new(s)` auto-builds a no-op SynthDef of the form
`Out.ar(outBus, MRT2UGen.ar)` and runs it. To embed the UGen inside a
larger SynthDef (e.g. with onboard effects), reference `MRT2UGen.ar`
directly, then construct the wrapper without auto-spawning and point it at
your synth + ugen index:

```supercollider
SynthDef(\mymrt, {
    var sig = MRT2UGen.ar;
    sig = LPF.ar(sig, 8000);
    Out.ar(0, sig);
}).add;

~syn = Synth(\mymrt);
~mrt = MRT2.new(s);          // builds its own def — discard
~mrt.free;                        // free the auto-spawned synth
~mrt.synth_(~syn).ugenIdx_(0);    // re-aim at your synth (UGen idx may differ)
```

(For most use cases the default wrapper is enough.)

## Known limitations

- macOS / Apple Silicon only. The model uses MLX (Metal) and TFLite.
- Sample rate is hardcoded at 48 kHz internally; running scsynth at any other
  SR plays the output at the wrong speed.
- One model checkpoint per UGen instance. Multiple `MRT2UGen` instances
  in one server are supported but each holds its own engine + ~1 GB of weights.
- First-time `assets`/`model` commands briefly stall the server while loading
  TFLite + MLX state; subsequent prompt updates are streamed asynchronously.
- Audio prompts and the prefill API are not yet exposed; only text prompts.
- No latency reporting back to sclang.
