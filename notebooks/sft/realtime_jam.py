# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-time streaming "jam" for an NNX LoRA Magenta-RT model, with LIVE
adapter-strength control from a browser.

The model streams audio continuously — a functional ``jax.jit`` ``step`` (the
model is split once into a constant parameter partition and a *stream* partition
— KV/codec caches, decode state, sampling rng — and only the stream is threaded
and donated, so the params are never duplicated in+out per frame) runs
comfortably under the 40 ms / 25 Hz real-time budget for both ``mrt2_small`` and
``mrt2_base`` (bf16) on an RTX 4080 — so ``mrt2_base`` bf16 streams in real time
without int8. It plays to your speakers via ``sounddevice``
(through WSLg's PulseAudio on WSL2 → Windows), and serves a small web control
panel with:

  * **adapter strength** slider — live, *no recompile*
  * **temperature** and **top_k** sliders — live (traced ``jax.jit`` args; the
    mrt2 sampler thresholds top_k via a dynamic ``sort``/``take_along_axis``,
    not a static ``lax.top_k`` — same runtime-dynamic top_k as the MRT2 apps)
  * **seed** + **Reset** — restart the stream from a fresh state

Why it's glitch-free: for plain LoRA the adapter adds
``delta = scale·strength·(x @ A @ B)``, so folding strength into the ``lora_b``
weight (``B ← strength·B₀``) changes the effective strength by editing a
*parameter* leaf. The step takes the parameter partition as a (non-donated)
traced argument, so re-splitting to pick up the edited ``lora_b`` flows the new
weights in with no graphdef change → no ``jax.jit`` recompile. (DoRA bakes
strength inside a norm, so the trick doesn't apply — DoRA adapters are rejected;
use a plain-LoRA run.)

Conditioning (this session): **unconditional** — every content channel is the
learned dropout token, exactly how a ``--mask_conditioning`` run was trained.
``--prompt "<text>"`` instead conditions on one MusicCoCa text embedding.

Usage:

    python notebooks/sft/realtime_jam.py \\
        --model mrt2_small \\
        --checkpoint ~/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors \\
        --adapters  ~/Documents/Magenta/magenta-rt-v2/runs/electronic_masked_lora_r16/sft_nnx_adapters_step_200.safetensors \\
        --strength 0.6

then open the printed http://localhost:<port> in your Windows browser.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# mrt2_base in bf16 (4.8 GB) + the codec + the depthformer's ~1.9 GB forward
# transient exceed JAX's default 0.75 GPU-memory fraction (~12 GB) on a 16 GB
# card → RESOURCE_EXHAUSTED. Raise the default here, BEFORE any JAX import (jax
# is imported lazily inside the functions below), so the bare launch command
# just works; export XLA_PYTHON_CLIENT_MEM_FRACTION yourself to override.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import jax.numpy as jnp
from flax import nnx
import sounddevice as sd
import json as _json
import logging
from flask import Flask, jsonify, request
from flask_sock import Sock

from magenta_rt import conditioning, config as _cfg
from magenta_rt.nnx.model import MagentaRT2Sampler
from magenta_rt.nnx.quantize import quantize_in_place
from magenta_rt.nnx.musiccoca import MusicCoCa
from magenta_rt.sft.lora_nnx import LoRAAdapter
from magenta_rt.sft.lora_io import load_lora_adapters, load_lora_weights


def discover_adapters(adapters_arg: str):
    """Resolve ``--adapters`` (a file OR a directory) to a sorted checkpoint list.

    Returns ``([(step, path), ...] sorted by step, initial_path)``. The trainer
    writes ``sft_nnx_adapters_step_<N>.safetensors`` every ``save_every_steps``;
    pointing ``--adapters`` at the run dir (or at any one of those files) exposes
    them all in the UI dropdown. A non-matching explicit file still works as a
    single-entry list (step parsed as -1)."""
    p = adapters_arg
    search_dir = p if os.path.isdir(p) else os.path.dirname(p) or "."
    found = []
    for f in glob.glob(os.path.join(search_dir, "sft_nnx_adapters_step_*.safetensors")):
        m = re.search(r"step_(\d+)\.safetensors$", os.path.basename(f))
        if m:
            found.append((int(m.group(1)), os.path.abspath(f)))
    found.sort()
    if os.path.isfile(p):
        ap = os.path.abspath(p)
        if ap not in {q for _, q in found}:
            step = -1
            m = re.search(r"step_(\d+)\.safetensors$", os.path.basename(p))
            if m:
                step = int(m.group(1))
            found.append((step, ap))
            found.sort()
        initial = ap
    elif found:
        initial = found[-1][1]   # directory → start at the latest checkpoint
    else:
        raise SystemExit(
            f"[jam] no adapter checkpoints found at {p!r} "
            "(expected sft_nnx_adapters_step_*.safetensors).")
    return found, initial

SAMPLE_RATE = 48_000
FRAME_SAMPLES = 1920          # one 25 Hz frame = 40 ms of audio
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Shared, thread-safe control state (UI thread writes, gen thread reads).
# ---------------------------------------------------------------------------
@dataclass
class Controls:
    strength: float = 0.6
    temperature: float = 1.0
    top_k: int = 40
    gain: float = 1.0              # output gain applied host-side before the clip
    seed: int = 0
    reset_requested: bool = True   # arm an initial streaming reset
    running: bool = True
    # Checkpoint switching: the UI thread sets ``pending_adapter`` (a path); the
    # generation thread loads it (in-place lora_a/lora_b swap, recompile-free),
    # records it in ``current_adapter``, and clears pending.
    pending_adapter: Optional[str] = None
    current_adapter: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self):
        with self.lock:
            return (self.strength, self.temperature, self.top_k, self.gain,
                    self.seed, self.reset_requested, self.pending_adapter)


# ---------------------------------------------------------------------------
# Model runtime
# ---------------------------------------------------------------------------
def snapshot_lora_b(model):
    """Snapshot each adapter's current ``lora_b`` as ``B₀`` so live strength can
    be applied as ``B ← s·B₀`` (dynamic state mutation → no jit recompile).
    Re-run after a checkpoint switch to re-baseline against the new weights."""


    originals = []
    def _walk(node):
        for attr in list(vars(node)):
            if attr.startswith("_"):
                continue
            child = getattr(node, attr)
            if isinstance(child, LoRAAdapter):
                originals.append((child, jnp.asarray(child.lora_b[...])))
            elif isinstance(child, nnx.Module):
                _walk(child)
    _walk(model)
    return originals


# ---------------------------------------------------------------------------
# Keyboard note input — port of core/src/midi_note_tracker.h
# ---------------------------------------------------------------------------
# The pianoroll_with_onsets conditioning wants a freshly-struck note (value 2 =
# onset) marked differently from a held one (1 = sustain). A key tapped+released
# within one 40ms inference frame must still register its onset for exactly one
# frame (the ONSET_RELEASED latch) or it is lost. evaluate_frame() runs ONCE per
# frame and advances the per-pitch state machine.
_N_IDLE, _N_ONSET, _N_SUSTAIN, _N_ONSET_RELEASED = 0, 1, 2, 3


class NoteTracker:
    """Per-pitch onset/sustain tracker (see ``core/src/midi_note_tracker.h``).

    The WebSocket thread calls :meth:`note_on` / :meth:`note_off`; the generation
    thread calls :meth:`evaluate_frame` EXACTLY ONCE per inference frame to read
    the per-pitch pianoroll value (0 off / 1 sustain / 2 onset) and advance the
    state machine.
    """

    def __init__(self, n_pitches: int = 128):
        self._state = np.zeros(n_pitches, dtype=np.int8)  # all _N_IDLE
        self._lock = threading.Lock()

    def note_on(self, pitch: int) -> None:
        if 0 <= pitch < self._state.shape[0]:
            with self._lock:
                self._state[pitch] = _N_ONSET   # (re)articulate, even from SUSTAIN

    def note_off(self, pitch: int) -> None:
        if 0 <= pitch < self._state.shape[0]:
            with self._lock:
                # Onset not yet seen by a frame → latch it so the next frame still
                # emits one onset (fast tap); otherwise release to idle.
                if self._state[pitch] == _N_ONSET:
                    self._state[pitch] = _N_ONSET_RELEASED
                else:
                    self._state[pitch] = _N_IDLE

    def evaluate_frame(self) -> np.ndarray:
        """``[n_pitches]`` int8 pianoroll values for this frame; advances states."""
        with self._lock:
            st = self._state
            # Capture masks BEFORE mutating (onset→sustain must not then read as sustain).
            onset = st == _N_ONSET
            sustain = st == _N_SUSTAIN
            released = st == _N_ONSET_RELEASED
            out = np.zeros_like(st)
            out[onset] = 2
            out[sustain] = 1
            out[released] = 2
            st[onset] = _N_SUSTAIN
            st[released] = _N_IDLE
            return out


def build_runtime(args):
    """Load the sampler + plain-LoRA adapters (no merge) and the conditioning.

    Returns ``(mrt, lora_b_originals, source_tokens, adapters, initial_path)``
    where ``adapters`` is the sorted ``[(step, path), ...]`` checkpoint list.
    """


    adapters, initial = discover_adapters(args.adapters)
    print(f"[jam] {len(adapters)} adapter checkpoint(s): "
          f"steps {[s for s, _ in adapters]}")

    print(f"[jam] loading {args.model} + codec "
          f"({'bf16' if args.bf16 else 'fp32'}) …")
    pdt = jnp.bfloat16 if args.bf16 else None
    mrt = MagentaRT2Sampler.from_preset(
        args.model, int16_outputs=False, param_dtype=pdt, rngs=nnx.Rngs(args.seed))
    # host-load avoids an on-device fp32 checkpoint spike (needed for bf16 base).
    mrt.load_checkpoint(args.checkpoint, host=args.bf16)

    meta = load_lora_adapters(mrt.depthformer, initial)  # inject + load, NO merge
    print(f"[jam] initial adapter: {os.path.basename(initial)} "
          f"rank={meta.get('rank')} alpha={meta.get('alpha')} "
          f"dora={meta.get('dora')} targets={meta.get('targets')}")
    if str(meta.get("dora")).lower() == "true":
        raise SystemExit(
            "[jam] DoRA adapters can't use the recompile-free strength trick "
            "(strength is inside the DoRA norm). Use a plain-LoRA run.")

    # Weight-only int8 (after LoRA inject, so it quantizes the int8 base inside
    # each LoRAAdapter.base AND the un-wrapped Linears; the bf16 lora_a/lora_b
    # delta is untouched → QLoRA-style int8-base inference). ~RT for mrt2_base.
    if args.bits:
        nq = quantize_in_place(mrt.depthformer, bits=args.bits)
        print(f"[jam] quantized {nq} Linears to int{args.bits} (weight-only)")

    originals = snapshot_lora_b(mrt.depthformer)
    if not originals:
        raise SystemExit("[jam] no LoRA adapters found on the model.")
    print(f"[jam] {len(originals)} LoRA adapters wrapped for live strength.")

    # Conditioning: unconditional (content → dropout token) unless --prompt.
    off = _cfg.NUM_RESERVED_TOKENS + 1
    cfgs = [int((args.cfg_musiccoca + 1.0) / 0.2),
            int((args.cfg_notes + 1.0) / 0.2),
            int((args.cfg_drums + 1.0) / 1.0)]
    if args.prompt:
        print(f"[jam] embedding MusicCoCa prompt {args.prompt!r} …")
        mc = MusicCoCa()
        # use_mapper=True (text->audio space) to match the canonical generate
        # path AND the SFT `--style_prompt` training embedding (embed_style_prompt
        # runs `embed_prompt --use-mapper`); without it the prompt tokens diverge
        # from what the model was trained on.
        emb = np.asarray(mc.embed_text(args.prompt, use_mapper=True), np.float32)
        style = [int(t) for t in np.asarray(mc.tokenize(emb[None]))[0]]
    else:
        style = [-1] * 12  # unconditional → dropout token after offset
        print("[jam] conditioning: UNCONDITIONAL (content = dropout token)")
    row = conditioning.build_conditioning_rows(
        batch_style=[style], notes=None, drums=None, cfgs=cfgs,
        num_musiccoca=12, num_notes=128, drum_tokens=1, cfg_tokens=3, offset=off)
    source_tokens = jnp.asarray(row)  # [1, 1, C]
    # Pianoroll block in the source row + the token offset (value v → token v+off),
    # matching the build_conditioning_rows layout above (musiccoca 12, notes 128).
    note_cols = (12, 12 + 128, off)
    return mrt, originals, source_tokens, adapters, initial, note_cols


class JaxJitStepper:
    """Functional ``jax.jit`` streaming step (replaces the per-call ``nnx.jit``).

    The model is split ONCE into a constant parameter partition and a *stream*
    partition (``nnx.split(model, nnx.Param, ...)`` → the KV/codec caches, decode
    state and sampling rng). A plain ``jax.jit`` body merges them, runs
    ``model.step``, and splits the updated stream back out (the merge/split trace
    to no-ops). ``params`` is passed as a *non-donated* constant — never
    duplicated in+out per frame, so ``mrt2_base`` bf16 fits — and only the
    ``stream`` is donated (from the 2nd frame on; a fresh stream's 1st frame is
    not donated, since its buffers alias the just-split model). This avoids the
    per-call ``nnx.jit`` Python split/merge of the whole module graph, which was
    the dominant per-step overhead.

    ``temperature`` / ``top_k`` stay *traced* args (live sliders, no recompile).
    Live LoRA strength is also recompile-free: ``B ← s·B₀`` edits ``lora_b`` (a
    ``Param``) on the model, and :meth:`refresh_params` re-splits to pass the new
    params on the next frame (same shapes). :meth:`rearm` re-splits after a fresh
    ``init_streaming`` (adapter switch / reset). Because the 1st frame is never
    donated, the model's own armed caches are never deleted, so re-splitting in
    :meth:`refresh_params` / :meth:`rearm` is always safe.
    """

    def __init__(self, mrt):


        self._nnx = nnx
        self._mrt = mrt
        self._graphdef = self._params = self._stream = None
        self._fresh = True

        def _step(params, stream, source_tokens, temperature, top_k):
            model = nnx.merge(self._graphdef, params, stream)
            tree = model.step(source_tokens=source_tokens,
                              temperature=temperature, top_k=top_k)
            _, _, new_stream = nnx.split(model, nnx.Param, ...)
            return tree, new_stream

        self._step_donate = jax.jit(_step, donate_argnums=1)
        self._step_nodonate = jax.jit(_step)

    def _split(self):
        return self._nnx.split(self._mrt, self._nnx.Param, ...)

    def rearm(self):
        """Re-split after ``mrt.init_streaming`` (fresh stream; next frame won't
        donate)."""
        graphdef, self._params, self._stream = self._split()
        if self._graphdef is None:       # structure is invariant across streams,
            self._graphdef = graphdef    # so reuse the trace-time graphdef
        self._fresh = True

    def refresh_params(self):
        """Pick up a live ``lora_b`` edit (strength) without touching the stream."""
        _, self._params, _ = self._split()

    def step(self, source_tokens, temperature, top_k):
        fn = self._step_nodonate if self._fresh else self._step_donate
        tree, self._stream = fn(self._params, self._stream,
                                source_tokens, temperature, top_k)
        self._fresh = False
        return tree


def apply_strength(originals, strength: float):
    """B ← strength·B₀ on every adapter (dynamic state → no recompile)."""
    for adapter, b0 in originals:
        adapter.lora_b.set_value((strength * b0).astype(b0.dtype))


# ---------------------------------------------------------------------------
# Generation thread: produce 40 ms audio frames into a bounded queue.
# ---------------------------------------------------------------------------
def generation_loop(mrt, originals, source_tokens, controls: Controls,
                    audio_q: queue.Queue, status: dict, note_input=None):


    stepper = JaxJitStepper(mrt)
    # Keyboard pianoroll: (tracker, lo, hi, off). When set, splice the live
    # per-frame note values into the masked pianoroll block of source_tokens.
    tracker = lo = hi = note_off = None
    idle_unconditional = False   # piano mode: un-pressed pitches → dropout token
    if note_input is not None:
        tracker, lo, hi, note_off, idle_unconditional = note_input
    cur_strength = None
    cur_adapter = controls.current_adapter
    status["adapter"] = os.path.basename(cur_adapter) if cur_adapter else ""
    n = 0
    t_log = time.time()
    while controls.running:
        strength, temperature, top_k, gain, seed, reset_req, pending_adapter = controls.snapshot()

        if pending_adapter and pending_adapter != cur_adapter:
            # Switch checkpoints: reload lora_a/lora_b into the existing wrappers
            # (values only → no jit recompile), re-baseline B₀, and start a fresh
            # stream so the new checkpoint is auditioned from a clean state. The
            # brief pause here just drains a little of the audio buffer.
            status["loading"] = os.path.basename(pending_adapter)
            load_lora_weights(mrt.depthformer, pending_adapter, strength=1.0)
            originals = snapshot_lora_b(mrt.depthformer)
            cur_adapter = pending_adapter
            mrt.init_streaming(batch_size=1, rngs=nnx.Rngs(seed))
            stepper.rearm()                              # re-split the fresh stream
            with audio_q.mutex:
                audio_q.queue.clear()
            with controls.lock:
                controls.current_adapter = pending_adapter
                controls.pending_adapter = None
            cur_strength = None                          # force strength re-apply
            status.pop("loading", None)
            status["adapter"] = os.path.basename(cur_adapter)

        if reset_req:                                    # fresh stream + seed
            mrt.init_streaming(batch_size=1, rngs=nnx.Rngs(seed))
            stepper.rearm()                              # re-split the fresh stream
            with audio_q.mutex:
                audio_q.queue.clear()
            with controls.lock:
                controls.reset_requested = False
            cur_strength = None                          # force re-apply below

        if strength != cur_strength:                     # live, no recompile
            apply_strength(originals, strength)
            stepper.refresh_params()                     # pass the edited lora_b in
            cur_strength = strength

        # Splice the live keyboard pianoroll into the (otherwise fixed) source
        # row — values only, so step_fn stays recompile-free (traced arg).
        src = source_tokens
        if tracker is not None:
            vals = tracker.evaluate_frame()        # [128] of 0/1/2, advances states
            toks = vals + note_off                 # 0→off(7), 1→sustain(8), 2→onset(9)
            if idle_unconditional:                 # piano mode: un-pressed pitch →
                toks = np.where(vals == 0, note_off - 1, toks)  # dropout token (6)
            src = source_tokens.at[..., lo:hi].set(
                jnp.asarray(toks, source_tokens.dtype))

        # temperature and top_k are traced args → both change live, no recompile.
        tree = stepper.step(src, jnp.asarray(temperature, jnp.float32),
                            jnp.asarray(top_k, jnp.int32))
        wav = np.asarray(jax.block_until_ready(tree.waveform))  # [1, 2, 1920]
        chunk = np.ascontiguousarray(wav[0].T, dtype=np.float32)  # [1920, 2]
        if gain != 1.0:
            chunk *= gain          # host-side output gain (model audio is quiet, rms ~0.04)
        np.clip(chunk, -1.0, 1.0, out=chunk)   # safety net for the rare peak > 1
        audio_q.put(chunk)                               # blocks when full → throttles to RT

        n += 1
        if time.time() - t_log >= 2.0:
            rms = float(np.sqrt(np.mean(chunk**2)))
            status.update(frames=n, buffer=audio_q.qsize(), rms=rms,
                          strength=cur_strength, top_k=top_k)
            t_log = time.time()


def make_audio_callback(audio_q: queue.Queue, status: dict):
    def callback(outdata, frames, time_info, cb_status):
        try:
            outdata[:] = audio_q.get_nowait()
        except queue.Empty:
            outdata.fill(0.0)                            # underrun → silence
            status["underruns"] = status.get("underruns", 0) + 1
    return callback


# ---------------------------------------------------------------------------
# Flask control panel
# ---------------------------------------------------------------------------
_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Magenta-RT LoRA jam</title>
<style>body{font-family:system-ui;margin:2rem;max-width:560px}
label{display:block;margin:1rem 0 .25rem;font-weight:600}
input[type=range]{width:100%}.row{display:flex;gap:1rem;align-items:center}
.val{font-variant-numeric:tabular-nums;min-width:3.5rem}
button{padding:.5rem 1rem;font-size:1rem;margin-top:1rem}
#status{margin-top:1.5rem;color:#555;white-space:pre}</style></head><body>
<h2>Magenta-RT NNX LoRA — live jam</h2>
<label>Checkpoint (training step) <span class=val id=ckptcur></span></label>
<select id=ckpt onchange=selectCkpt() style="width:100%;padding:.4rem;font-size:1rem"></select>
<label>Adapter strength <span class=val id=sv></span></label>
<input type=range id=strength min=0 max=1 step=0.01>
<label>Temperature <span class=val id=tv></span></label>
<input type=range id=temperature min=0.1 max=2 step=0.01>
<label>top_k <span class=val id=kv></span></label>
<input type=range id=top_k min=1 max=256 step=1>
<label>Output gain <span class=val id=gv></span></label>
<input type=range id=gain min=0 max=8 step=0.1>
<div class=row><label>Seed</label><input type=number id=seed style=width:8rem>
<button onclick=reset()>Reset stream</button></div>
<div id=status></div>
<div id=khint style="display:none;margin-top:.5rem;color:#888"><small>⌨ Keyboard (C major): <b>A S D F G H J K</b> = C D E F G A B C — hold to sustain. Click the page first so it has focus. Raise <b>--cfg-notes</b> for stronger note steering.</small></div>
<script>
const $=id=>document.getElementById(id);
let _last=0,_t=null;
function _send(){_last=Date.now();fetch('/control',{method:'POST',
 headers:{'Content-Type':'application/json'},
 body:JSON.stringify({strength:+$('strength').value,temperature:+$('temperature').value,
 top_k:+$('top_k').value,gain:+$('gain').value})});}
function push(){  // throttle drag → at most ~1 POST/150ms (+ trailing), so a
 const now=Date.now();clearTimeout(_t);  // slider drag can't flood the gen thread
 if(now-_last>150)_send();else _t=setTimeout(_send,150);}
for(const id of ['strength','temperature','top_k','gain']){
 $(id).oninput=e=>{$({strength:'sv',temperature:'tv',top_k:'kv',gain:'gv'}[id]).textContent=e.target.value;push();};
 $(id).onchange=_send;}  // always send the final value on release
function reset(){fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({seed:+$('seed').value})});}
function selectCkpt(){fetch('/select_checkpoint',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({step:+$('ckpt').value})});}
async function init(){const s=await (await fetch('/state')).json();
 $('strength').value=s.strength;$('sv').textContent=s.strength;
 $('temperature').value=s.temperature;$('tv').textContent=s.temperature;
 $('top_k').value=s.top_k;$('kv').textContent=s.top_k;
 $('gain').value=s.gain;$('gv').textContent=s.gain;$('seed').value=s.seed;
 const sel=$('ckpt');sel.innerHTML='';
 for(const st of (s.checkpoints||[])){const o=document.createElement('option');
  o.value=st;o.textContent='step '+st;sel.appendChild(o);}
 sel.value=s.current_step;$('ckptcur').textContent=s.current_step;if(s.keys)initKeys();}
const _KMAP={a:60,s:62,d:64,f:65,g:67,h:69,j:71,k:72};let _ws=null,_held=new Set();
function initKeys(){const h=$('khint');if(h)h.style.display='block';
 (function open(){_ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/notes');
  _ws.onclose=()=>setTimeout(open,1000);})();
 const send=(on,p)=>{if(_ws&&_ws.readyState===1)_ws.send(JSON.stringify({on:on,p:p}));};
 addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase(),p=_KMAP[k];
  if(p===undefined||_held.has(k))return;_held.add(k);send(1,p);e.preventDefault();});  // dedupe auto-repeat
 addEventListener('keyup',e=>{const k=e.key.toLowerCase(),p=_KMAP[k];if(p===undefined)return;_held.delete(k);send(0,p);});}
async function poll(){const s=await (await fetch('/state')).json();
 $('ckptcur').textContent=(s.loading?('loading '+s.loading+'…'):s.current_step);
 $('status').textContent='adapter '+(s.adapter||'?')+'  frames '+(s.frames||0)+
 '  buffer '+(s.buffer||0)+'  rms '+(s.rms||0).toFixed(3)+
 '  underruns '+(s.underruns||0);}
init();setInterval(poll,2000);
</script></body></html>"""


def _step_of(path: str) -> int:
    m = re.search(r"step_(\d+)\.safetensors$", os.path.basename(path or ""))
    return int(m.group(1)) if m else -1


def make_app(controls: Controls, status: dict, adapters, tracker=None):


    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # silence per-request logs

    app = Flask(__name__)
    step_to_path = {s: p for s, p in adapters}
    steps = [s for s, _ in adapters]

    if tracker is not None:
        # WebSocket for low-latency key events: {"on":0|1,"p":pitch}. The browser
        # debounces auto-repeat; the gen thread reads the tracker once per frame.
        sock = Sock(app)

        @sock.route("/notes")
        def notes(ws):
            while True:
                msg = ws.receive()
                if msg is None:
                    break
                d = _json.loads(msg)
                p = int(d["p"])
                tracker.note_on(p) if d.get("on") else tracker.note_off(p)

    @app.get("/")
    def index():
        return _PAGE

    @app.get("/state")
    def state():
        with controls.lock:
            s = dict(strength=controls.strength, temperature=controls.temperature,
                     top_k=controls.top_k, gain=controls.gain, seed=controls.seed,
                     current_step=_step_of(controls.current_adapter),
                     pending=bool(controls.pending_adapter))
        s.update(status)
        s["checkpoints"] = steps
        s["keys"] = tracker is not None
        return jsonify(s)

    @app.post("/select_checkpoint")
    def select_checkpoint():
        d = request.get_json(force=True)
        path = step_to_path.get(int(d["step"]))
        if path is None:
            return ("unknown checkpoint step", 400)
        with controls.lock:
            controls.pending_adapter = path
        return ("", 204)

    @app.post("/control")
    def control():
        d = request.get_json(force=True)
        with controls.lock:
            controls.strength = float(d["strength"])
            controls.temperature = float(d["temperature"])
            controls.top_k = int(d["top_k"])
            controls.gain = float(d.get("gain", controls.gain))
        return ("", 204)

    @app.post("/reset")
    def reset():
        d = request.get_json(force=True)
        with controls.lock:
            controls.seed = int(d.get("seed", controls.seed))
            controls.reset_requested = True
        return ("", 204)

    return app


# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mrt2_small")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapters", required=True,
                   help="plain-LoRA adapter .safetensors, OR a run directory — "
                        "all sft_nnx_adapters_step_*.safetensors are exposed in "
                        "the UI checkpoint dropdown (starts at the latest).")
    p.add_argument("--bf16", action="store_true",
                   help="store base weights in bf16 + host-load the checkpoint "
                        "(required to fit mrt2_base in 16 GB).")
    p.add_argument("--bits", type=int, default=None, choices=[8],
                   help="weight-only int8 quantization of the depthformer "
                        "(matches the int8 the MRT2 apps ship): extra speed + "
                        "memory headroom. The donated bf16 step already streams "
                        "mrt2_base under the 40ms real-time budget, so int8 is "
                        "now optional. Composes with LoRA.")
    p.add_argument("--strength", type=float, default=0.6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--gain", type=float, default=1.0,
                   help="output gain applied host-side before the [-1,1] clip; "
                        "the model audio is quiet (rms ~0.04, not peak-normalized) "
                        "so raise this (e.g. 2-4) for a louder mix. Live-adjustable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=None,
                   help="MusicCoCa text prompt (omit → unconditional, matching a "
                        "--mask_conditioning run)")
    p.add_argument("--cfg-musiccoca", type=float, default=3.0)
    p.add_argument("--cfg-notes", type=float, default=1.0)
    p.add_argument("--cfg-drums", type=float, default=1.0)
    p.add_argument("--buffer-seconds", type=float, default=3.0,
                   help="audio pre-buffer (latency vs underrun robustness): the "
                        "queue holds buffer_seconds*25 frames, so this caps BOTH "
                        "keypress->sound latency and jitter tolerance. mrt2_base "
                        "bf16 runs ~25ms/frame (RTF ~0.6) on an Nvidia RTX 4080, "
                        "~15ms under the 40ms real-time budget — measured "
                        "streaming cleanly down to 0.12 (3 frames), and even "
                        "0.04 (the 1-frame floor) is usable there with only an "
                        "occasional underrun at that zero-slack edge. The default "
                        "3.0 is conservative; drop it (~0.1-0.2) for live "
                        "keyboard play. (mrt2_small is ~15ms/frame, more slack.)")
    p.add_argument("--keys", action="store_true",
                   help="enable computer-keyboard note input (ASDFGHJK = C major, "
                        "hold to sustain) via a WebSocket + the midi_note_tracker "
                        "onset/sustain state machine; splices a live pianoroll into "
                        "the conditioning. Needs a notes-trained adapter (e.g. "
                        "base_electronic_music_notes); pair with a higher --cfg-notes "
                        "and a small --buffer-seconds for playability.")
    p.add_argument("--piano", action="store_true",
                   help="like --keys, but un-pressed pitches default to the "
                        "UNCONDITIONAL dropout token (the model keeps improvising; "
                        "your held keys only inject notes), rather than explicit "
                        "'off'. Implies keyboard input.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--selftest", action="store_true",
                   help="build + run a few steps + verify live strength has no "
                        "recompile and changes output; then exit (no UI/audio).")
    return p.parse_args(argv)


def _selftest(mrt, originals, source_tokens):
    """Validate: jit step works; the lora_b rescale actually changes the weights
    and the generated audio (over multiple frames, where strength-0 vs strength-1
    streams diverge); and a strength change does NOT recompile."""


    stepper = JaxJitStepper(mrt)
    temp = jnp.asarray(1.0, jnp.float32)
    tk = jnp.asarray(40, jnp.int32)

    # (1) the rescale really mutates lora_b: 0·B₀ = 0, 1·B₀ = B₀.
    adapter, b0 = originals[0]
    apply_strength(originals, 0.0); z = float(jnp.abs(adapter.lora_b[...]).max())
    apply_strength(originals, 1.0); f = float(jnp.abs(adapter.lora_b[...]).max())
    print(f"[selftest] lora_b max-abs: strength0={z:.5f} strength1={f:.5f}")
    assert z == 0.0 and f > 0.0, "lora_b rescale not applied"

    def gen(strength, frames=60):
        apply_strength(originals, strength)
        mrt.init_streaming(batch_size=1, rngs=nnx.Rngs(0))   # same seed → deterministic
        stepper.rearm()
        out = [np.asarray(jax.block_until_ready(
                   stepper.step(source_tokens, temp, tk).waveform))
               for _ in range(frames)]
        return np.concatenate(out, axis=-1)

    gen(1.0, frames=3)  # warmup/compile
    a0, a1, a06 = gen(0.0), gen(1.0), gen(0.6)
    d01 = float(np.abs(a0 - a1).max())

    # a strength change must NOT recompile (step stays ~tens of ms, not seconds).
    apply_strength(originals, 0.42); stepper.refresh_params()
    t0 = time.time(); jax.block_until_ready(stepper.step(source_tokens, temp, tk).waveform)
    dt = (time.time() - t0) * 1000

    print(f"[selftest] |strength0 - strength1| max over 60 frames = {d01:.4f} "
          f"(>0 ⇒ adapter changes the audio)")
    print(f"[selftest] step after strength change: {dt:.1f}ms (recompile-free if tens of ms)")
    print(f"[selftest] finite: s0={np.isfinite(a0).all()} s0.6={np.isfinite(a06).all()} "
          f"s1={np.isfinite(a1).all()}")
    assert d01 > 1e-3, "strength has no audible effect over 60 frames"
    assert dt < 1000, "step took >1s after strength change — likely recompiled"
    print("[selftest] PASS: live strength works and is recompile-free.")


def main(argv=None):
    args = _parse_args(argv)
    mrt, originals, source_tokens, adapters, initial, note_cols = build_runtime(args)

    if args.selftest:
        _selftest(mrt, originals, source_tokens)
        return

    # Keyboard note input (opt-in): a tracker shared by the WS thread (note_on/off)
    # and the gen thread (evaluate_frame once per frame → spliced pianoroll).
    tracker = note_input = None
    if args.keys or args.piano:
        tracker = NoteTracker(128)
        lo, hi, off = note_cols
        note_input = (tracker, lo, hi, off, args.piano)
        mode = "piano (un-pressed → unconditional)" if args.piano else "explicit (un-pressed → off)"
        print(f"[jam] keyboard input ON, {mode}; ASDFGHJK = C major; "
              f"cfg_notes={args.cfg_notes} — raise it for stronger steering, "
              f"use a notes-trained adapter.")



    controls = Controls(strength=args.strength, temperature=args.temperature,
                        top_k=args.top_k, gain=args.gain, seed=args.seed)
    controls.current_adapter = initial   # the checkpoint build_runtime loaded
    status: dict = {}
    audio_q: queue.Queue = queue.Queue(maxsize=max(1, int(args.buffer_seconds * 25)))

    gen = threading.Thread(
        target=generation_loop,
        args=(mrt, originals, source_tokens, controls, audio_q, status, note_input),
        daemon=True)
    gen.start()

    # Pre-roll: let the generator compile (first jit step) and fill the buffer
    # BEFORE opening audio, so there's no startup silence / underrun burst.
    print("[jam] warming up (jit compile + buffer fill)…")
    t_warm = time.time()
    while audio_q.qsize() < audio_q.maxsize and controls.running:
        if time.time() - t_warm > 120:
            print("[jam] warmup timeout — opening audio anyway.")
            break
        time.sleep(0.2)
    print(f"[jam] warmed up in {time.time() - t_warm:.1f}s "
          f"(buffer {audio_q.qsize()}/{audio_q.maxsize}).")

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, channels=2, dtype="float32",
        blocksize=FRAME_SAMPLES, callback=make_audio_callback(audio_q, status))
    stream.start()
    print(f"[jam] audio stream open ({SAMPLE_RATE} Hz stereo, "
          f"{args.buffer_seconds:.1f}s buffer).")

    app = make_app(controls, status, adapters, tracker=tracker)
    print(f"\n  ┌─ open  http://{args.host}:{args.port}  in your browser "
          f"(Windows side) ─┐")
    print(f"  └─ strength={args.strength}  temp={args.temperature}  "
          f"top_k={args.top_k}  seed={args.seed} ─┘\n")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        controls.running = False
        stream.stop(); stream.close()


if __name__ == "__main__":
    main()
