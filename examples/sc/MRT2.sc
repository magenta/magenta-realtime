// MRT2.sc — sclang client for the MRT2 scsynth UGen plugin.
//
// Two classes live here:
//   MRT2UGen  : MultiOutUGen   — the actual UGen wrapping the C++ Unit
//                                     (used inside the SynthDef built below)
//   MRT2                       — user-facing wrapper; creates the Synth
//                                     and exposes one method per /u_cmd
//
// Typical use:
//     ~mrt = MRT2.new(s);
//     ~mrt.assets("/path/to/magenta-rt-v2/resources");
//     ~mrt.model("/path/to/checkpoint.mlxfn");
//     ~mrt.prompt(0, "piano", 1.0);
//     ...
//     ~mrt.free;


MRT2UGen : MultiOutUGen {
    *ar {
        ^this.multiNew('audio')
    }
    init { |... theInputs|
        inputs = theInputs;
        ^this.initOutputs(2, rate)
    }
    checkInputs { ^this.checkValidInputs }
}

MRT2 {
    var <server, <synth, <ugenIdx, <outBus;

    *new { |server, outBus = 0|
        ^super.new.init(server ? Server.default, outBus.asInteger)
    }

    init { |argServer, argOutBus|
        var defName;
        server = argServer;
        outBus = argOutBus;
        defName = ("magentart_out" ++ outBus).asSymbol;
        // SynthDef has no Control UGens (no \-args), so MRT2UGen is the
        // first UGen in the synth's optimized graph and ugenIdx = 0. If you
        // embed MRT2UGen.ar in your own SynthDef, set `ugenIdx`
        // explicitly via `.ugenIdx_(n)`.
        SynthDef(defName, {
            Out.ar(outBus, MRT2UGen.ar)
        }).add;
        ugenIdx = 0;
        // .add sends the SynthDef asynchronously; wait for the server to
        // acknowledge it before creating the Synth.
        fork {
            server.sync;
            synth = Synth(defName, target: server);
        };
    }

    ugenIdx_ { |n| ugenIdx = n.asInteger }

    free {
        if(synth.notNil) {
            synth.free;
            synth = nil;
        }
    }

    isRunning { ^synth.notNil }

    sendCmd { |... args|
        if(synth.isNil) {
            "MRT2: synth is not running; call MRT2.new first.".warn;
            ^this
        };
        server.sendMsg(*(['/u_cmd', synth.nodeID, ugenIdx] ++ args));
    }

    // ── Assets / model ────────────────────────────────────────────────────
    assets { |path| this.sendCmd("assets", path.asString) }
    model  { |path|
        this.sendCmd("model", path.asString)
    }

    // ── Prompts ───────────────────────────────────────────────────────────
    // slot  ∈ [0, 5];  text is any string (no quoting/escaping needed);
    // weight ∈ [0, 1] — the C++ side clamps.
    prompt { |slot, text, weight = 1.0|
        this.sendCmd("prompt", slot.asInteger, text.asString, weight.asFloat)
    }
    clearPrompt { |slot| this.sendCmd("prompt_clear", slot.asInteger) }

    // ── Sampling parameters ───────────────────────────────────────────────
    temperature { |val|  this.sendCmd("temperature", val.asFloat) }
    topk        { |val|  this.sendCmd("topk", val.asInteger) }
    cfgMusicCoCa{ |val|  this.sendCmd("cfgmusiccoca", val.asFloat) }
    cfgNotes    { |val|  this.sendCmd("cfgnotes", val.asFloat) }
    cfgDrums    { |val|  this.sendCmd("cfgdrums", val.asFloat) }
    unmaskWidth { |val|  this.sendCmd("unmaskwidth", val.asInteger) }

    // ── Output ────────────────────────────────────────────────────────────
    volume     { |db|        this.sendCmd("volume", db.asFloat) }
    mute       { |on = 1|    this.sendCmd("mute", on.asInteger) }
    bypass     { |on = 1|    this.sendCmd("bypass", on.asInteger) }
    bufferSize { |samples|   this.sendCmd("buffersize", samples.asInteger) }
    reset      {             this.sendCmd("reset") }

    // ── MIDI ──────────────────────────────────────────────────────────────
    noteOn   { |note|     this.sendCmd("noteon",  note.asInteger) }
    noteOff  { |note|     this.sendCmd("noteoff", note.asInteger) }
    midiGate { |on = 1|   this.sendCmd("midigate", on.asInteger) }

    // ── Drums ─────────────────────────────────────────────────────────────
    drumless  { |on = 1|         this.sendCmd("drumless", on.asInteger) }

    // ── PCA ───────────────────────────────────────────────────────────────
    pcaFile { |path|         this.sendCmd("pcafile", path.asString) }
    pca     { |axis, value|  this.sendCmd("pca", axis.asInteger, value.asFloat) }
}
