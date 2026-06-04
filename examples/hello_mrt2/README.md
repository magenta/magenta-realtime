# hello_mrt2

The shortest command-line interface (CLI) that uses `magentart::core`: load a model, set a text prompt,
generate a short clip, write a WAV.

## Build

From the repo root:

```bash
cmake . -B build
cmake --build build --target hello_mrt2 -j10
```

The binary lands at `build/examples/hello_mrt2/hello_mrt2`.

## Run

You'll need:

- an exported `.mlxfn` model directory (see
  [docs/exporting.md](../../docs/exporting.md) for how to build one)
- a resource directory containing the MusicCoCa TFLite assets in a subfolder (defaults to `musiccoca`, override with `--subfolder`)

```bash
./build/examples/hello_mrt2/hello_mrt2 \
    path/to/model.mlxfn \
    ~/Documents/Magenta/magenta-rt-v2/resources \
    100 \       # optional: frames to generate (default 100 ≈ 4.0 seconds)
    --subfolder musiccoca \ # optional: model subfolder under resources (default: musiccoca)
    --prompt "ambient pads with sub bass" \ # optional: prompt text
    --output output.wav \ # optional: custom output path
    --force \    # optional: force overwrite if output path exists
    --prefill-silence \ # optional: prefill state with silent audio before generation
    --spectrostream-encoder path/to/spectrostream_encoder.mlxfn \ # optional: custom path to spectrostream encoder
    --prefill-duration 1.64 \ # optional: duration of silent prefill in seconds (default 1.64)
    --temperature 1.0 \ # optional: sampling temperature (default 1.0)
    --top-k 100 \ # optional: sampling top-k (default 100)
    --cfg-musiccoca 3.0 \ # optional: classifier-free guidance scale for musiccoca (default 3.0)
    --cfg-notes 5.0 \ # optional: classifier-free guidance scale for notes (default 5.0)
    --cfg-drums 1.0 \ # optional: classifier-free guidance scale for drums (default 1.0)
    --unmask-width 0 \ # optional: note pitch window radius (default 0)
    --seed-rotation 0 # optional: seed rotation for variations (default 0)
```

The program writes `out.wav` in the current directory by default.

### Silent Prefill Expectations

When you pass `--prefill-silence`, `--prefill-duration` seconds (default
1.64) of stereo silence are SpectroStream-encoded and walked through the
transformer one frame at a time before generation starts. The model's
attention KV caches end up populated with the codec's representation of
silence; the first generated frames continue from that state. No token
trimming is applied (`trim_front_frames=0`, `trim_back_frames=0`) — the
caller's silent buffer is fed verbatim.

To investigate the prefill path with real audio, you can encode a user-supplied WAV, trim SpectroStream's head/tail tokens, and save the prefill output audio.

## Use it as a starting point

`main.cpp` exercises the three calls a new app is most likely to care about:

1. `engine.init_assets(resource_dir)` — load the MusicCoCa TFLite models.
2. `engine.load_model(mlxfn_path)` — load the MLX transformer and model state (empty).
3. `engine.generate_frame(L, R)` — produce one 1920-sample stereo frame
   (48 kHz, 40 ms).

Prompts, sampling parameters, and MIDI notes are additional method calls on
`MLXEngine` — see [`core/include/magentart/mlx_engine.h`](../../core/include/magentart/mlx_engine.h)
for the full API. For real-time audio-callback use, wrap the engine in a
`RealtimeRunner` (see
[`core/include/magentart/realtime_runner.h`](../../core/include/magentart/realtime_runner.h)).
