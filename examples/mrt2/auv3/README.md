# State and Prefill (the bank panel + Reset)

The AUv3 manages the model's transformer state through a
**bank panel** plus the transport **Reset** button. The semantics are
intentional but non-obvious, so this section spells out what each
control does, what data it touches, and how they compose.

> The original single-button layout (*Save State* / *Reset State* /
> *Reset Model*) still ships in the **experimental** plugin — see
> [`examples/mrt2_experimental/auv3/README.md`](../../mrt2_experimental/auv3/README.md).
> This document describes the `mrt2` UI.

## The two state vectors

Internally the engine tracks two parallel state vectors:

- **`transformer_state_`** — the *live* state. Every call to
  `generate_frame` reads from and writes to it. This is what the model
  is "currently thinking."
- **`transformer_initial_state_`** — the *reset target*. The transport
  **Reset** copies this into `transformer_state_`. By default it's the
  freshly loaded model's factory initial state, but every bank action
  below moves it.

Conceptually, `transformer_initial_state_` is a single-slot "checkpoint"
the engine carries with it. The bank panel is the set of controls that
populate that slot — from an on-disk save, from a prefill pass, or from
the factory state.

## The bank panel

The panel has two columns.

### User banks (Bank 1–3)

Three persistent save slots, stored as `bank_<N>.safetensors` under
`~/Documents/Magenta/magenta-rt-v2/banks/`. A filled dot means the slot
holds a saved state.

- **Save** (save icon) — dumps the **live `transformer_state_`** into
  that slot's file via `save_state`. Pure dump; doesn't change what the
  model is playing.
- **Load** (replay icon, enabled once the slot is filled) — calls
  `load_state` on that file, which validates the loaded shapes against
  the live model (rejecting state from a different model variant with a
  clear error) and writes the data into the **reset target**
  (`transformer_initial_state_`). It does **not** change the live state
  on its own — press the transport **Reset** to apply it. The Reset
  tooltip shows which bank is currently armed (e.g. "Reset from Bank 2").

### Presets (Empty, Custom)

- **Empty** (replay icon) — `reset_to_factory()`: restores the reset
  target to the model's factory initial state (the
  `<model>_state.safetensors` arrays loaded at model-load time) and
  applies it to the live state immediately. Undoes the checkpointing
  sideeffect of a prefill or a prior bank load, without requiring a
  full model reload. The factory snapshot is held in memory as a shallow
  `mx::array` copy, so this is cheap (no disk I/O, no recompilation).
- **Custom** (upload icon, then replay icon) — **Audio Prefill**. Upload
  a `.wav`/audio file (up to 28 seconds; the SpectroStream encoder is traced
  at that fixed length); it's encoded to RVQ tokens, the unreliable
  head/tail of the sequence is trimmed (1 s each side), and the tokens
  are fed through the transformer one frame at a time.

  **Important: prefill checkpoints itself.** After a successful prefill,
  the reset target is overwritten with the post-prefill state, so
  subsequent **Reset** presses land back at the prefilled context,
  **not** the factory state. This is by design: prefill is expensive (it
  can take a few seconds), and users typically want to try several
  MusicCoCa prompts on top of the same musical context.

A **Silent Prefill** path also exists in the engine — it masks MusicCoCa
and prefills ~22 s of silent RVQ tokens to saturate every layer's
local-attention window with silence (the steady-state silent token is
encoded through SpectroStream once and cached, so subsequent silent
prefills skip the encoder). It surfaces as a "Silence" bank row, but
that row is **currently hidden** in this build.

## The transport Reset button

The **Reset** button in the bottom transport bar applies the current
reset target to the live state (an edge-triggered model parameter that
maps to `reset_state()`). Whatever you last armed — a user-bank load,
*Empty*, or a prefill — is what Reset returns to. This is also where
seed rotation (if set) takes effect, letting you draw a fresh variation
from the same starting state.

## Recommended workflows

### Try multiple prompts on the same audio context

1. Set a starter prompt (or skip).
2. On the **Custom** row, upload the source clip. After ~5–10 s the
   model is checkpointed at the post-prefill state and real-time
   generation resumes.
3. Listen. If you don't like the continuation, change the prompt.
4. Press **Reset** — the live state returns to the prefill
   checkpoint and the model continues from there with the new prompt.
5. Repeat step 4 as many times as you like; the prefill is not
   redone, only the active conditioning changes.

### Snapshot a moment you like and return to it later — the seamless option

> **This is the only workflow with a provably bit-exact boundary.**
> An empirical round-trip test confirmed that under greedy sampling, generating audio
> through the model and then resuming from a saved state produces
> 100 frames of continuation that are *byte-identical* to what the
> uninterrupted run would have produced (`np.array_equal: True`,
> `max_abs_diff = 0.0`). The byte-exactness claim is a greedy-only
> test claim — under the AUv3's default temp=1.3 / top-k=40 sampling,
> the same save/load mechanism still resumes from the *same* engine
> state (the saved-state file faithfully captures everything,
> including the RNG), so the trajectory continues coherently from the
> moment the snapshot was taken. The boundary is acoustically
> seamless under user-facing defaults, even though byte-exactness can
> only be proven under greedy. For the "I want the model to keep
> playing from this exact point later" use case, this is the
> seamless path — not audio prefill.

1. While generation is running, press **Save** on one of the user banks
   (Bank 1–3) to dump the live state into that slot.
2. Continue using the plugin (change prompts, prefill different
   audio, whatever).
3. To return: press **Load** on that bank (`load_state` writes it into
   the reset target), then press **Reset** to apply.

The two prefill paths (audio prefill, silent prefill) are *not*
byte-exact — they reach a state similar to but not equal to what
natural generation would produce. For model-generated audio with a
matching prompt, that gap manifests as a perceptible rhythm drop at
the boundary even though peak/RMS amplitude metrics look smooth. The
audio-prefill path is for "feed me an audio file I have on disk and
keep going" workflows where some boundary mismatch is acceptable; the
save/load path is for "resume the model's own output exactly."

### Recover the model's factory initial state

Press **Empty**. The reset target reverts to the
`<model>_state.safetensors` payload and the live state follows. No
model reload required.

## What the user *hears* immediately after prefill

The runner restarts after prefill and seeds its ring buffer with three
priming frames generated from the post-prefill state. Real-time
generation then takes over. The runner deliberately does **not** play
back audio captured during the prefill loop itself — those would be
the model's per-step *predictions* on a teacher-forced trajectory, and
they accumulate audible artifacts at typical prefill lengths
(currently up to 28 seconds).

### Why token prefill behaves differently

The codebase exposes a separate path, `prefill_state_from_tokens`,
that takes RVQ tokens directly (e.g. captured from earlier
`generate_frame` calls) instead of audio. That path *does* support
clean prompt-swaps because the tokens stay in the model's natural
distribution. There's no AUv3 control wired to it — the user banks
above resume via full-state save/load (`save_state` / `load_state`),
not token prefill.

# React UI Architecture

The Audio Unit interface is built with React, Vite, and Tailwind CSS, located in the `examples/mrt2/react_ui` directory (shared with the standalone host).

- **Vite Plugin Singlefile:** The entire React application is bundled into a single standalone `index.html` file during the build process to simplify local loading.
- **WKWebView Bridge:** The native Audio Unit (`MagentaRT_AudioUnit.mm`) hosts a `WKWebView` that displays the `index.html` file.
- **Two-way Communication:**
  - **JS to Native:** React sends parameter updates (e.g. slider movements) using `window.webkit.messageHandlers.auHost.postMessage()`.
  - **Native to JS:** The Audio Unit pushes state and metric updates down to React by evaluating `window.updateState(...)` via `evaluateJavaScript`.
- **Entitlements:** The App Extension strictly requires `com.apple.security.network.client` and `com.apple.security.network.server` in `Entitlements.plist` to allow WebKit's internal IPC processes to function within the AU sandbox.

# Debug Logging

The AUv3 host has a developer-only on-screen debug overlay (`NSTextField`) and a `mrt_debug.log` file writer in your selected models directory (or `~/Library/Application Support/MagentaRT/models/` by default). Both are gated behind the `MAGENTART_DEBUG_LOG` CMake option (off by default). Enable with:

```bash
cmake . -B build -DMAGENTART_DEBUG_LOG=ON
cmake --build build --target deploy_mrt2_au
```

Release builds ship with the overlay and file writer compiled out; the React UI's in-app log panel continues to work in both modes.

The standalone host has no on-screen dev overlay and no disk log writer — its `NSLog` output goes to Console.app in the usual way — so `MAGENTART_DEBUG_LOG` is a no-op there.
