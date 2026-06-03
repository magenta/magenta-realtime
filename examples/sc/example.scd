// example.scd — MRT2 SuperCollider example session.
//
// Mirrors mrt2~.maxhelp / mrt2~-help.pd: a sequence of self-contained blocks,
// each runnable on its own (Cmd-Return on the block).
//
// IMPORTANT: set the audio hardware sample rate to 48000 Hz BEFORE booting.
// macOS: Audio MIDI Setup → built-in output → Format → 48000 Hz. The model
// produces audio at 48 kHz; running scsynth at 44.1 kHz makes output play
// ~8.8% slow.


// ── 0. Boot ─────────────────────────────────────────────────────────────────
(
s.options.sampleRate = 48000;
s.options.numOutputBusChannels = 2;
s.boot;
)

// ── 1. Create instance ──────────────────────────────────────────────────────
// Assets and default model (mrt2_base) auto-load from
// ~/Documents/Magenta/magenta-rt-v2/. Override with:
//   ~mrt.assets("/custom/resources");
//   ~mrt.model("/custom/model.mlxfn");
(
~mrt = MRT2.new(s);
)


// ── 1a. Set Buffer Size, increase if you hear clicking ──────────────────────
(
~mrt.bufferSize(8192)
)

// ── 2. Prompts ──────────────────────────────────────────────────────────────
(
~mrt.prompt(0, "piano",      1.0);
~mrt.prompt(1, "jazz drums", 0.0);
~mrt.prompt(2, "dubstep",    0.0);
~mrt.prompt(3, "synth pad",  0.0);
)

// Tweak weights
~mrt.prompt(1, "jazz drums", 0.6);
~mrt.prompt(2, "dubstep",    0.5);

// Clear a slot
~mrt.clearPrompt(2);

// ── 3. Sampling ─────────────────────────────────────────────────────────────
~mrt.temperature(1.3);
~mrt.topk(40);

// ── 4. Guidance (CFG) ───────────────────────────────────────────────────────
~mrt.cfgMusicCoCa(1.6);   // Style
~mrt.cfgNotes(2.4);        // Notes
~mrt.cfgDrums(4.0);        // Drums

// ── 5. Output & Config ──────────────────────────────────────────────────────
~mrt.volume(0);            // dB
~mrt.mute(0);
~mrt.bypass(0);
~mrt.drumless(0);
~mrt.bufferSize(8192);     // samples @ 48 kHz (≈170 ms)

// Reset transformer state
~mrt.reset;

// ── 6. MIDI ─────────────────────────────────────────────────────────────────
// Jam mode (unmaskWidth 0) vs Solo mode (unmaskWidth 127)
~mrt.unmaskWidth(0);       // Jam
~mrt.unmaskWidth(127);     // Solo

// MIDI gate + note on/off
~mrt.midiGate(1);
fork {
    ~mrt.noteOn(60); 0.5.wait;
    ~mrt.noteOn(64); 0.5.wait;
    ~mrt.noteOn(67); 1.0.wait;
    ~mrt.noteOff(60); ~mrt.noteOff(64); ~mrt.noteOff(67);
};
~mrt.midiGate(0);

// ── 7. Shutdown ─────────────────────────────────────────────────────────────
(
~mrt.free;
s.quit;
)
