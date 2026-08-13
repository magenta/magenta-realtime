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

"""``.mlxfn`` export for pure-MLX streaming inference.

Mirrors :mod:`magenta_rt.mlx.export` but routes through
:mod:`magenta_rt.mlx_pure`. Builds an ``MagentaRT2Sampler`` system,
optionally loads weights from a real safetensors checkpoint via
``load_from_safetensors``, optionally quantizes, then traces the
streaming step with ``mx.exporter`` to a ``.mlxfn`` file plus a
companion ``_state.safetensors`` snapshot of the initial cache
state.

C++ engine calling convention
-----------------------------

The traced signature follows the same contract as
:mod:`magenta_rt.mlx.export` (the sl exporter), which is what the C++
runtime (``core/src/mlx_engine.cpp``, behind the ``examples/`` hosts)
binds against positionally: 9 leading args
``(cond, temperature, top_k, cfg_musiccoca, cfg_notes, cfg_drums,
neg_musiccoca, neg_notes, forced_tokens, *state)``, conditioning rows
*without* the 3 CFG token slots (``[1, 1, 141]`` for the production
specs: style 12 + notes 128 + drums 1), and the runtime float CFG
scales discretized into the CFG token slots *inside* the trace
(:func:`_discretize_cfg_token`). Keep the two exporters' signatures in
lock-step — the engine has no format probe and a silent divergence
shifts every argument after the mismatch.

Zero-element conv buffers (``pad_left == 0``) are excluded from the
flat state so the shipped ``state_<i>`` safetensors keys stay
contiguous — the engine loads ``state_0..state_N`` and stops at the
first gap.

.. warning:: Validated so far via a Python re-import exercising the
   exact C++ argument construction (9 leading args, ``[1, 1, 141]``
   rows, contiguous state load, multi-step state threading, both
   forced-token traces); an end-to-end run through the actual C++
   engine is still outstanding. Known gap: the engine locates
   ``previous_frame`` by shape probe (``[1, 1, RVQ]`` or
   ``[1, B, 1, RVQ]``) which does NOT match this export's batch-first
   ``[B, 1, RVQ]`` slot — generation works, but C++ prefill /
   ``tokens_out`` stay unavailable until the engine reads the
   manifest's ``"previous_frame"`` role (which this exporter already
   writes) or extends its probe. The RNG probe (the only uint32
   ``[..., 2]`` array) does match.

Streaming correctness
---------------------

The exported function consumes and returns a single flat list of
``mx.array``\\s covering *all* per-step state on both sides of the
pipeline:

* **Depthformer**: ``SamplerState`` (rng, previous_frame, step) plus
  each ``LocalKVCache``'s ``keys`` / ``values`` / ``offset``. The
  offset is promoted from Python ``int`` to ``mx.array(int32)`` so
  the in-cache ``mx.put_along_axis`` write path runs under tracing
  and slot indices aren't baked at trace time.

* **Codec**: every ``Conv2D`` / ``Conv2DTranspose`` left-context
  buffer inside ``spectrostream.decoder`` (from the ``ParallelChannels``
  batch-stacked groups too — ``nn.Module.modules()`` walks through
  them) and the ``OverlapAddCache.buffer`` on
  ``spectrostream._istft_cache``.

``MagentaRT2Sampler.init_cache`` eagerly allocates every lazy cache —
zero-allocating + sink-priming the depthformer KV caches and
allocating the codec's conv / InverseSTFT buffers via zero-input
passes, then zeroing them. The captured flat state is therefore both
*fully allocated* (correct shapes/dtypes for the traced function) and
*content-neutral* — no generation content (``previous_frame``, KV,
codec buffers, RNG, step counter) is baked into ``_state.safetensors``.
The exported ``.mlxfn`` still matches the eager streaming path
bit-exactly across an unbounded number of calls.

``SpectroStreamDecoder._lookahead_remaining`` remains a Python ``int``;
``init_cache`` drains it to 0, so the trace observes the
``drop > 0 == False`` branch and no mutation participates in the
trace. Because the shipped state is neutral (not post-warmup), a host
can generate straight from it — the only artifact is the decoder's
~``lookahead_length``-frame opening transient, which a prefill erases.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .conv import Conv2D, Conv2DTranspose
from .depthformer import SamplerState, TemporalCaches
from .generate import _build_source_tokens
from .model import MagentaRT2Sampler
from .. import paths


_DTYPE_TO_STR = {
    mx.float32: "float32", mx.bfloat16: "bfloat16", mx.float16: "float16",
    mx.int32: "int32", mx.int64: "int64", mx.uint32: "uint32",
    mx.bool_: "bool",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def save_flat_state(
    flat_state: list[mx.array],
    state_path: str,
    *,
    roles: dict[int, str] | None = None,
) -> None:
    """Save ``flat_state`` to ``state_path`` (safetensors) with a
    sidecar ``.json`` manifest. ``mx.save_safetensors`` rejects
    0-element arrays (the lazy left-context buffers for convs with
    ``pad_left=0``), so they're recorded in the manifest only and
    reconstructed by :func:`load_flat_state`.

    ``roles`` optionally tags entries by flat-state index with a
    semantic name (e.g. ``{PREVIOUS_FRAME_FLAT_IDX: "previous_frame"}``)
    so consumers like the C++ engine can locate slots by name instead
    of guessing from array shapes.
    """
    roles = roles or {}
    manifest = []
    nonempty = {}
    for i, arr in enumerate(flat_state):
        entry = {"shape": list(arr.shape), "dtype": _DTYPE_TO_STR[arr.dtype]}
        if arr.size > 0:
            key = f"state_{i}"
            nonempty[key] = arr
            entry["key"] = key
        if i in roles:
            entry["role"] = roles[i]
        manifest.append(entry)
    mx.save_safetensors(state_path, nonempty)
    with open(state_path + ".json", "w") as f:
        json.dump(manifest, f)


def load_flat_state(state_path: str) -> list[mx.array]:
    """Inverse of :func:`save_flat_state`."""
    with open(state_path + ".json") as f:
        manifest = json.load(f)
    stored = mx.load(state_path)
    flat: list[mx.array] = []
    for entry in manifest:
        if "key" in entry:
            flat.append(stored[entry["key"]])
        else:
            flat.append(mx.zeros(tuple(entry["shape"]),
                                 dtype=_STR_TO_DTYPE[entry["dtype"]]))
    return flat

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESOURCE_DIR = REPO_ROOT / "examples" / "auv3" / "resources"
TRANSFORMER_MLXFN = "MagentaRT_transformer_pure.mlxfn"
TRANSFORMER_STATE = "MagentaRT_transformer_pure_state.safetensors"

DEFAULT_MODEL = "mrt2_small"
DEFAULT_CHECKPOINT = (
    "mrt2_small.safetensors"
)


# ---------------------------------------------------------------------------
# Flatten / unflatten the SamplerState pytree + LocalKVCache arrays
# ---------------------------------------------------------------------------
#
# Layout of the flat ``mx.array`` list:
#
#     [rng, previous_frame, step,
#      self_caches[0].keys, self_caches[0].values, self_caches[0].offset,
#      self_caches[1].keys, ...,
#      cross_caches[0].keys, cross_caches[0].values, cross_caches[0].offset,
#      ...]
#
# Each ``LocalKVCache.offset`` is promoted from Python ``int`` to
# ``mx.array(scalar, dtype=int32)`` before tracing so ``mx.exporter``
# sees it as runtime state (see ``_promote_cache_offsets`` below).


def _promote_cache_offsets(state) -> None:
    """In-place: convert each cache's Python-int ``offset`` to an
    ``mx.array`` scalar so it travels through the trace as runtime
    state. Idempotent.
    """
    for cache in (*state.temporal.self_caches, *state.temporal.cross_caches):
        if not isinstance(cache.offset, mx.array):
            cache.offset = mx.array(int(cache.offset), dtype=mx.int32)


def _flatten_state(state) -> list[mx.array]:
    flat: list[mx.array] = [state.rng, state.previous_frame, state.step]
    for cache in state.temporal.self_caches:
        flat.append(cache.keys)
        flat.append(cache.values)
        flat.append(cache.offset)
    for cache in state.temporal.cross_caches:
        flat.append(cache.keys)
        flat.append(cache.values)
        flat.append(cache.offset)
    return flat


# Index of ``state.previous_frame`` in the flat list above (and thus in
# ``_flatten_state(...) + _flatten_codec(...)``, since the codec arrays
# are appended after). Recorded in the manifest as a semantic ``role``
# so the C++ engine locates the slot by name rather than by a shape
# heuristic. Must track the ``_flatten_state`` layout.
PREVIOUS_FRAME_FLAT_IDX = 1


def _discretize_cfg_token(value, step, max_bin, offset):
    """MLX-op CFG-scale binning, identical to ``magenta_rt.mlx.export``'s.

    Bins a float CFG scale in [-1.0, 7.0] to a conditioning token index,
    then shifts it by ``offset`` (the conditioning vocab's reserved-token
    count + 1 for the dropout token). Runs *inside* the traced function so
    the C++ runtime can pass raw float scales at runtime — part of the
    shared ``.mlxfn`` calling convention (see the module docstring).
    Operates on a traced ``mx.array`` scalar of shape ``[1]`` and returns
    an int32 token of shape ``[1]``.
    """
    clamped = mx.clip(value, -1.0, 7.0)
    bin_index = mx.round((clamped - (-1.0)) / step)
    bin_index = mx.clip(bin_index, 0.0, float(max_bin))
    return bin_index.astype(mx.int32) + offset


def _install_state(flat: list[mx.array], ref_state):
    """In-place: install ``flat`` back into ``ref_state``'s caches and
    return a fresh ``SamplerState`` NamedTuple referencing them.
    """
    rng = flat[0]
    prev = flat[1]
    step = flat[2]
    cursor = 3
    for cache in ref_state.temporal.self_caches:
        cache.keys = flat[cursor]
        cache.values = flat[cursor + 1]
        cache.offset = flat[cursor + 2]
        cursor += 3
    for cache in ref_state.temporal.cross_caches:
        cache.keys = flat[cursor]
        cache.values = flat[cursor + 1]
        cache.offset = flat[cursor + 2]
        cursor += 3
    if cursor != len(flat):
        raise ValueError(
            f"unexpected flat-state length: used {cursor}, got {len(flat)}"
        )
    return SamplerState(
        rng=rng,
        previous_frame=prev,
        temporal=TemporalCaches(
            self_caches=ref_state.temporal.self_caches,
            cross_caches=ref_state.temporal.cross_caches,
        ),
        step=step,
    )


# ---------------------------------------------------------------------------
# Codec-side streaming state walker
# ---------------------------------------------------------------------------
#
# Beyond the depthformer's SamplerState, the spectrostream decoder
# carries per-step streaming state as Python attributes on its
# submodules: every Conv2D / Conv2DTranspose left-context buffer
# (lazily allocated via Conv2DCache) and the InverseSTFT's
# OverlapAddCache buffer. ``nn.Module.modules()`` walks through
# ParallelChannels' batch-stacked inner module too, so we don't need
# a separate pass for it.


def _decoder_streaming_caches(spectrostream):
    """Deterministic iterator over the decoder's Conv2DCache slots.

    Yields each ``Conv2DCache`` whose ``buffer`` is allocated (i.e.,
    after at least one streaming call has run through that conv).
    Iteration order matches ``nn.Module.modules()``, which is stable
    across calls because it walks ``_modules`` in declaration order.

    Zero-element buffers (convs with ``pad_left == 0``, which never
    carry left context) are skipped: they are structurally empty
    forever, ``mx.save_safetensors`` can't store them, and skipping
    them keeps the shipped ``state_<i>`` keys contiguous — the C++
    engine loads ``state_0..state_N`` and stops at the first gap.
    """
    for m in spectrostream.decoder.modules():
        if isinstance(m, (Conv2D, Conv2DTranspose)):
            cache = m._streaming_cache
            if cache is not None and cache.buffer is not None and cache.buffer.size > 0:
                yield cache


def _flatten_codec(spectrostream) -> list[mx.array]:
    flat: list[mx.array] = []
    for cache in _decoder_streaming_caches(spectrostream):
        flat.append(cache.buffer)
    istft_cache = getattr(spectrostream, "_istft_cache", None)
    if istft_cache is not None and istft_cache.buffer is not None:
        flat.append(istft_cache.buffer)
    return flat


def _install_codec(spectrostream, flat: list[mx.array], cursor: int) -> int:
    for cache in _decoder_streaming_caches(spectrostream):
        cache.buffer = flat[cursor]
        cursor += 1
    istft_cache = getattr(spectrostream, "_istft_cache", None)
    if istft_cache is not None and istft_cache.buffer is not None:
        istft_cache.buffer = flat[cursor]
        cursor += 1
    return cursor


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(
    restore: bool = True,
    model_name: str = DEFAULT_MODEL,
    bits: Optional[int] = None,
    quantize_method: str = "naive",
    gptq_cal_steps: int = 8,
    quantize_group_size: Optional[int] = None,
    num_cfgs: int = 2,
    temperature: float = 1.3,
    top_k: int = 40,
    cfg_musiccoca: float = 3.0,
    cfg_notes: float = 1.0,
    cfg_drums: float = 1.0,
    output_name: Optional[str] = None,
    output_dir: str = paths.models_dir(),
    checkpoint: str = DEFAULT_CHECKPOINT,
):
    """Export an mlx_pure streaming step to ``.mlxfn``."""
    print(f"Building pure-MLX system (model={model_name})…")
    mrt = MagentaRT2Sampler.from_preset(model_name, int16_outputs=False)

    if restore:
        print("Restoring weights via standalone native loader…")
        from .load_weights import load_from_safetensors
        checkpoint_path = REPO_ROOT / "checkpoints" / checkpoint
        load_from_safetensors(mrt, checkpoint_path, model_name=model_name)
    else:
        from .load_weights import init_random_params
        print("Initializing random weights…")
        init_random_params(mrt, seed=0)

    # Source tokens for calibration/tracing. The export is content-neutral,
    # so the MusicCoCa style segment is fully masked (-1) rather than
    # encoded from a prompt; channel layout is [style, notes(128), drums(1),
    # cfgs(3)] (see generate._build_source_tokens).
    input_num_channels = mrt.depthformer.encoder.embedding.num_channels
    if input_num_channels == 1:
        style = None
    else:
        style = [-1] * (input_num_channels - (128 + 1 + 3))

    if bits and bits < 32:
        from .quantize import gptq_calibrate_and_quantize, quantize_in_place
        gs = quantize_group_size or (32 if bits == 4 else 64)
        print(f"Quantizing depthformer to {bits}-bit "
              f"(method={quantize_method}, group_size={gs})…")
        if quantize_method == "naive":
            quantize_in_place(mrt.depthformer, group_size=gs, bits=bits)
        elif quantize_method == "gptq":
            batch_size = 1 if num_cfgs == 0 else num_cfgs + 1
            cal_source = _build_source_tokens(
                style=style, num_cfgs=num_cfgs,
                input_num_channels=input_num_channels,
                num_reserved=mrt.num_reserved_tokens,
                cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes,
            )
            cfg_scales_arg = []
            if num_cfgs >= 1:
                cfg_scales_arg.append(cfg_musiccoca)
            if num_cfgs >= 2:
                cfg_scales_arg.append(cfg_notes)

            def _calibrate(_root):
                cal_state = mrt.make_initial_state(
                    batch_size=batch_size, seed=0,
                )
                for _ in range(max(1, gptq_cal_steps)):
                    _, cal_state = mrt.step(
                        cal_state, source_tokens=cal_source,
                        temperature=temperature, top_k=top_k,
                        cfg_scales=cfg_scales_arg if num_cfgs > 0 else None,
                        cfg_arity=num_cfgs,
                    )

            gptq_calibrate_and_quantize(
                mrt.depthformer, _calibrate, group_size=gs, bits=bits,
            )
        else:
            raise ValueError(f"unknown quantize_method: {quantize_method!r}")

    # ------------------------------------------------------------------
    # Build the streaming-step closure.
    # ------------------------------------------------------------------

    batch_size = 1 if num_cfgs == 0 else num_cfgs + 1
    source_tokens = _build_source_tokens(
        style=style, num_cfgs=num_cfgs,
        input_num_channels=input_num_channels,
        num_reserved=mrt.num_reserved_tokens,
        cfg_musiccoca=cfg_musiccoca, cfg_notes=cfg_notes,
    )

    state = mrt.make_initial_state(batch_size=batch_size, seed=0)

    # Eagerly allocate every lazy cache so the shipped snapshot is
    # *content-neutral*. ``init_cache`` zero-allocates + sink-primes the
    # depthformer KV caches directly and allocates the codec's conv /
    # InverseSTFT buffers via zero-input passes, then zeros them — the
    # streaming setup the old 5-step warmup did, but without baking real
    # generation content (previous_frame, KV, codec buffers, RNG, step
    # counter) into ``_state.safetensors``. The codec lookahead
    # countdown is drained to 0 by ``init_cache`` so the trace still
    # observes the no-op branch (see module docstring).
    print("Allocating streaming caches (neutral, no warmup generation)…")
    mrt.init_cache(state, batch=batch_size)
    # Switch each LocalKVCache into traced-offset mode so the trace
    # threads offsets through as runtime state.
    _promote_cache_offsets(state)
    flat_state = _flatten_state(state) + _flatten_codec(mrt.spectrostream)
    depth_size = len(_flatten_state(state))
    print(f"  flat-state size: {len(flat_state)} arrays "
          f"({depth_size} depthformer + {len(flat_state) - depth_size} codec)")
    ref_state = state  # Holds the live LocalKVCache instances we mutate.

    # Snapshot the neutral state *now*, before the trace dry-run mutates
    # it — this is what ships in ``_state.safetensors``.
    neutral_flat_state = [mx.array(a) for a in flat_state]
    target_arr = state.rng
    seed_tensor_idx = next((i for i, arr in enumerate(flat_state) if arr is target_arr), -1)

    t_val = mx.array([temperature])
    k_val = mx.array([top_k], dtype=mx.int32)
    cfg_musiccoca_val = mx.array([cfg_musiccoca])
    cfg_notes_val = mx.array([cfg_notes])
    cfg_drums_val = mx.array([cfg_drums])

    pos_tokens = source_tokens[0:1]
    neg_musiccoca_tokens = source_tokens[1:2] if num_cfgs > 0 else source_tokens[0:1]
    neg_notes_tokens = source_tokens[2:3] if num_cfgs > 1 else source_tokens[0:1]

    # The traced function's conditioning inputs follow the C++ engine
    # contract (same as magenta_rt.mlx.export): rows WITHOUT the 3 CFG
    # token slots ([style, notes(128), drums(1)], length 141 for the
    # production specs); the CFG tokens are appended in-trace from the
    # runtime float scales by streaming_step. _build_source_tokens
    # bakes the CFG bins into the rows for the eager paths, so strip
    # them here. The single-channel tiny spec has no conditioning
    # layout and is exempt.
    has_cfg_slots = input_num_channels > 1
    cfg_token_offset = mrt.num_reserved_tokens + 1
    if has_cfg_slots:
        pos_tokens = pos_tokens[..., :-3]
        neg_musiccoca_tokens = neg_musiccoca_tokens[..., :-3]
        neg_notes_tokens = neg_notes_tokens[..., :-3]

    rvq_truncation = mrt.depthformer.decoder.num_active_codebooks
    empty_forced_tokens = mx.zeros((1, 0, rvq_truncation), dtype=mx.int32)

    def streaming_step(x_values, temperature_arg, top_k_arg, cfg_musiccoca_arg, cfg_notes_arg, cfg_drums_arg, neg_musiccoca_values, neg_notes_values, forced_tokens, *state_flat):
        flat_list = list(state_flat)
        local_state = _install_state(flat_list[:depth_size], ref_state)
        _install_codec(mrt.spectrostream, flat_list, depth_size)

        if has_cfg_slots:
            # Discretize the runtime float CFG scales into the 3 CFG
            # conditioning token slots and append them to the positive
            # and both negative blocks — same convention as
            # magenta_rt.mlx.export / core/src/mlx_engine.cpp.
            cfg_tokens = mx.concatenate([
                _discretize_cfg_token(cfg_musiccoca_arg, 0.2, 40, cfg_token_offset),
                _discretize_cfg_token(cfg_notes_arg, 0.2, 40, cfg_token_offset),
                _discretize_cfg_token(cfg_drums_arg, 1.0, 8, cfg_token_offset),
            ], axis=-1).reshape(1, 1, 3)
            x_values = mx.concatenate([x_values, cfg_tokens], axis=-1)
            neg_musiccoca_values = mx.concatenate([neg_musiccoca_values, cfg_tokens], axis=-1)
            neg_notes_values = mx.concatenate([neg_notes_values, cfg_tokens], axis=-1)

        if num_cfgs == 0:
            src_tokens = x_values
            cfg_scales_list = None
        elif num_cfgs == 1:
            src_tokens = mx.concatenate([x_values, neg_musiccoca_values], axis=0)
            cfg_scales_list = [cfg_musiccoca_arg]
        else:
            src_tokens = mx.concatenate([x_values, neg_musiccoca_values, neg_notes_values], axis=0)
            cfg_scales_list = [cfg_musiccoca_arg, cfg_notes_arg]

        if forced_tokens.shape[1] > 0 and num_cfgs > 0:
            forced_tokens = mx.repeat(forced_tokens, repeats=num_cfgs + 1, axis=0)

        wave, new_state = mrt.step(
            local_state, source_tokens=src_tokens,
            temperature=temperature_arg, top_k=top_k_arg,
            cfg_scales=cfg_scales_list,
            cfg_arity=num_cfgs,
            forced_tokens=forced_tokens,
        )

        y = wave[0:1]
        y = mx.swapaxes(y, -2, -1)
        y = mx.reshape(mx.flatten(y), (1, 2, -1))

        new_flat = _flatten_state(new_state) + _flatten_codec(mrt.spectrostream)
        return (y, *new_flat)

    # Run streaming_step once more so the trace-flow has materialized
    # at least one cycle of install→step→flatten before mx.exporter
    # walks it (catches any shape mismatches early, in eager Python,
    # with a clear traceback rather than inside the tracer).
    # Run on the *neutral* state — this both settles shapes early and
    # confirms streaming_step works from the exact snapshot we ship.
    # ``flat_state`` is reassigned to the post-step result and then
    # discarded; ``neutral_flat_state`` is what gets traced and saved.
    print("Trace dry-run (1 step through streaming_step)…")
    outputs = streaming_step(pos_tokens, t_val, k_val, cfg_musiccoca_val, cfg_notes_val, cfg_drums_val, neg_musiccoca_tokens, neg_notes_tokens, empty_forced_tokens, *neutral_flat_state)
    wave, *flat_state = outputs
    mx.eval(wave, *flat_state)
    print("dry-run ok.")

    # ------------------------------------------------------------------
    # Output paths and export.
    # ------------------------------------------------------------------

    if output_name:
        export_dir = REPO_ROOT / output_dir / output_name
        os.makedirs(export_dir, exist_ok=True)
        mlxfn_path = str(export_dir / f"{output_name}.mlxfn")
        state_path = str(export_dir / f"{output_name}_state.safetensors")
    else:
        os.makedirs(RESOURCE_DIR, exist_ok=True)
        mlxfn_path = os.path.join(RESOURCE_DIR, TRANSFORMER_MLXFN)
        state_path = os.path.join(RESOURCE_DIR, TRANSFORMER_STATE)

    print(f"Exporting transformer to {mlxfn_path} …")
    with mx.exporter(mlxfn_path, streaming_step, shapeless=False) as exporter:
        exporter(
            pos_tokens,
            t_val,
            k_val,
            cfg_musiccoca_val,
            cfg_notes_val,
            cfg_drums_val,
            neg_musiccoca_tokens,
            neg_notes_tokens,
            empty_forced_tokens,
            *neutral_flat_state,
        )
        forced_tokens = mx.zeros((1, 1, rvq_truncation), dtype=mx.int32)
        exporter(
            pos_tokens,
            t_val,
            k_val,
            cfg_musiccoca_val,
            cfg_notes_val,
            cfg_drums_val,
            neg_musiccoca_tokens,
            neg_notes_tokens,
            forced_tokens,
            *neutral_flat_state,
        )
    print(f"Exported transformer to {mlxfn_path}")

    # State snapshot shipped alongside the .mlxfn. This is the
    # ``init_cache`` snapshot: every lazy cache allocated (correct
    # shapes/dtypes for the traced function), LocalKVCache offsets
    # promoted to traced ``mx.array``s, the codec lookahead countdown
    # drained to 0 — *and* content-neutral (zeros + primed sink slots,
    # ``previous_frame`` = SOS, fresh RNG, step counter 0). Unlike the
    # old post-warmup snapshot it carries no generation content, so a
    # host can generate straight from it (the only artifact is the
    # decoder's ~lookahead_length-frame opening transient, which a
    # prefill erases).

    roles = {
        PREVIOUS_FRAME_FLAT_IDX: "previous_frame",
    }
    if seed_tensor_idx != -1:
        roles[seed_tensor_idx] = "seed"

    save_flat_state(
        neutral_flat_state, state_path,
        roles=roles,
    )
    print(f"Saved {len(flat_state)} state arrays to {state_path} "
          f"(+ {state_path}.json manifest)")
    metadata_path = state_path.replace(".safetensors", "_metadata.json")
    rvq_truncation = mrt.depthformer.decoder.num_active_codebooks
    metadata = {
        "rvq_depth": int(rvq_truncation),
        "seed_tensor_idx": seed_tensor_idx,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported state metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("magenta_rt.mlx_pure.export")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bits", default=None, type=int,
                        choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--quantize-method", default="naive",
                        choices=["naive", "gptq"])
    parser.add_argument("--gptq-cal-steps", default=8, type=int)
    parser.add_argument("--quantize-group-size", default=None, type=int)
    parser.add_argument("--num-cfgs", default=2, type=int, choices=[0, 1, 2])
    parser.add_argument("--temperature", default=1.3, type=float)
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--cfg-musiccoca", default=3.0, type=float)
    parser.add_argument("--cfg-notes", default=1.0, type=float)
    parser.add_argument("--cfg-drums", default=1.0, type=float)
    parser.add_argument("--skip-restore", dest="restore",
                        action="store_false", default=True)
    parser.add_argument("--output-name", default=None, type=str)
    parser.add_argument("--output-dir", default=str(paths.models_dir()))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    main(
        restore=args.restore,
        model_name=args.model,
        bits=args.bits,
        quantize_method=args.quantize_method,
        gptq_cal_steps=args.gptq_cal_steps,
        quantize_group_size=args.quantize_group_size,
        num_cfgs=args.num_cfgs,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_musiccoca=args.cfg_musiccoca,
        cfg_notes=args.cfg_notes,
        cfg_drums=args.cfg_drums,
        output_name=args.output_name,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )
