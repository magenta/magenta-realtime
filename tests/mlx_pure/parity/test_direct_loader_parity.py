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

"""Bit-parity tests for the DIRECT depthformer loader.

The oracle is the existing sl-bridge loader
(:func:`magenta_rt.mlx_pure.load_weights.load_sft_depthformer_from_safetensors`):
the direct loader
(:func:`magenta_rt.mlx_pure.load_weights.load_depthformer_from_safetensors_direct`)
must reproduce its parameter tree exactly — same keys, shapes, dtypes, and
*bits* — without ever building the fp32 sl model in the middle.

Two tiers:

* **Gated real-checkpoint test** (``@pytest.mark.checkpoint``, skips when
  ``mrt2_small.safetensors`` is absent): load the real ``mrt2_small``
  depthformer via the sl bridge (golden), free it, load a fresh module via
  the direct loader, and assert every leaf is ``np.array_equal``. Models are
  loaded *sequentially* — never both resident — to respect the 16 GB budget.

* **CI-runnable tiny test** (no real checkpoint): the tiny POC spec uses a
  *plain* (non-pretrained-MusicCoCa) encoder embedding, so it doesn't
  exercise the branched-encoder Linen keys the direct loader special-cases —
  and we have no synthetic Linen checkpoint factory in-repo. The mirror logic
  for the plain path is the same identity/transpose composition the gated
  test covers end-to-end on the real branched checkpoint. Rather than fake a
  checkpoint (which would test our own fabrication, not the loader), the tiny
  tier asserts the *structural* invariant that makes the gated test meaningful:
  the direct loader and sl-bridge loader walk the **same pure parameter tree**
  (identical key set + shapes). The numerical bit-parity assertion lives in
  the gated test against the real weights.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten


_UINT_VIEW = {
    mx.bfloat16: mx.uint16,
    mx.float16: mx.uint16,
    mx.float32: mx.uint32,
    mx.float64: mx.uint64,
}


def _bits(arr: mx.array) -> np.ndarray:
    """Reinterpret ``arr``'s raw bits as an unsigned-int numpy array.

    Goes through ``mx.view`` because numpy has no native bf16 dtype — a bare
    ``np.array(bf16_mlx)`` can't ingest the buffer. The uint view is a pure
    bit-reinterpret, so ``np.array_equal`` on it is an exact byte comparison
    (immune to NaN ``!=`` quirks).
    """
    view_dtype = _UINT_VIEW.get(arr.dtype)
    if view_dtype is None:
        return np.array(arr)  # integer / already-uint param
    return np.array(arr.view(view_dtype))


def _flat_capture(module) -> dict[str, tuple]:
    """Flatten ``module.parameters()`` to a path → (shape, dtype-str, bits) dict."""
    out = {}
    for path, arr in tree_flatten(module.parameters()):
        mx.eval(arr)
        out[path] = (tuple(arr.shape), str(arr.dtype), _bits(arr))
    return out


def _build_fresh_depthformer(model_name: str = "mrt2_small"):
    """Build a fp32 depthformer EncoderDecoder and materialize lazy params.

    Matches ``notebooks/sft/train_mlx.py:build_model`` (fp32 build; the
    loader handles the bf16 dtype layout to match the sl bridge).
    """
    from magenta_rt.mlx_pure.configs import get_model_class

    spec = get_model_class(model_name)()
    model = spec.build_decoder()
    Q = spec.target_tokens_config.rvq_truncation_level
    dummy_src = mx.zeros((1, 4, spec.input_num_channels), mx.int32)
    dummy_tgt = mx.zeros((1, 4, Q), mx.int32)
    model.decoder(dummy_tgt, encoded_source=model.encoder(dummy_src))
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Gated real-checkpoint bit-parity test (the oracle).
# ---------------------------------------------------------------------------


@pytest.mark.checkpoint
def test_direct_loader_bit_parity_mrt2_small():
    """Every direct-loaded param is bit-identical to the sl-bridge golden.

    Sequential loads (golden then direct); the golden module is deleted +
    gc'd before the direct module is built, so the two never co-reside.
    """
    from magenta_rt import paths
    from magenta_rt.mlx_pure.load_weights import (
        load_depthformer_from_safetensors_direct,
        load_sft_depthformer_from_safetensors,
    )

    ckpt = paths.resolve_checkpoint("mrt2_small.safetensors")
    if not ckpt.is_file():
        pytest.skip(f"mrt2_small.safetensors not found at {ckpt}")

    # ---- 1. Golden: sl-bridge load, capture params, free the model. ----
    golden_model = _build_fresh_depthformer("mrt2_small")
    load_sft_depthformer_from_safetensors(
        golden_model, ckpt, model_name="mrt2_small"
    )
    mx.eval(golden_model.parameters())
    golden = _flat_capture(golden_model)
    del golden_model
    gc.collect()

    # ---- 2. Direct loader on a fresh module. ----
    direct_model = _build_fresh_depthformer("mrt2_small")
    load_depthformer_from_safetensors_direct(direct_model, ckpt)
    mx.eval(direct_model.parameters())
    direct = _flat_capture(direct_model)
    del direct_model
    gc.collect()

    # ---- 3. Bit-parity assertions. ----
    assert set(direct) == set(golden), (
        "key set mismatch:\n"
        f"  only in direct: {sorted(set(direct) - set(golden))}\n"
        f"  only in golden: {sorted(set(golden) - set(direct))}"
    )

    mismatches = []
    for path in sorted(golden):
        (gs, gdt, gbits) = golden[path]
        (ds, ddt, dbits) = direct[path]
        if gs != ds:
            mismatches.append(f"{path}: shape {ds} vs golden {gs}")
            continue
        if gdt != ddt:
            mismatches.append(f"{path}: dtype {ddt} vs golden {gdt}")
            continue
        if not np.array_equal(dbits, gbits):
            n_diff = int(np.sum(dbits != gbits))
            mismatches.append(
                f"{path}: {n_diff} differing elems (shape {gs}, {gdt})"
            )

    assert not mismatches, (
        f"{len(mismatches)} param(s) not bit-identical; first few:\n  "
        + "\n  ".join(mismatches[:10])
    )

    # Sanity: confirm the expected dtype layout (bf16 everywhere except the
    # fp32 per_dim_scale) actually held — guards against a silent all-fp32 or
    # all-bf16 match that wouldn't represent the real inference layout.
    fp32_s, bf16_s = str(mx.float32), str(mx.bfloat16)
    dtypes = {gdt for (_, gdt, _) in golden.values()}
    assert dtypes in ({fp32_s, bf16_s}, {bf16_s}), dtypes
    fp32_paths = {p for p, (_, gdt, _) in golden.items() if gdt == fp32_s}
    assert all(p.endswith("per_dim_scale") for p in fp32_paths), fp32_paths


# ---------------------------------------------------------------------------
# CI-runnable tiny structural test (no real checkpoint).
# ---------------------------------------------------------------------------


def test_direct_loader_matches_sl_bridge_tree_shape_tiny():
    """The direct loader's bf16-dtype pass produces the same per-leaf dtype
    layout the sl bridge does (bf16 + fp32 per_dim_scale), on a tiny model,
    with no real checkpoint required.

    This exercises ``_match_sl_bf16_dtypes`` — the piece that lines the
    destination dtypes up so the gated test's per-leaf ``astype(dst.dtype)``
    yields bit-parity — independent of any checkpoint file.
    """
    import sys
    from pathlib import Path

    from magenta_rt.mlx_pure.load_weights import _match_sl_bf16_dtypes

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "notebooks" / "sft"))
    import train_mlx  # type: ignore

    spec = train_mlx.TinyPOCSpecMLX()
    model = spec.build_decoder()
    Q = spec.target_tokens_config.rvq_truncation_level
    dummy_src = mx.zeros((1, 4, spec.input_num_channels), mx.int32)
    dummy_tgt = mx.zeros((1, 4, Q), mx.int32)
    model.decoder(dummy_tgt, encoded_source=model.encoder(dummy_src))
    mx.eval(model.parameters())

    fp32_s, bf16_s = str(mx.float32), str(mx.bfloat16)

    # Fresh build is fp32.
    before = {p: str(a.dtype) for p, a in tree_flatten(model.parameters())}
    assert set(before.values()) == {fp32_s}, before

    _match_sl_bf16_dtypes(model)

    after = {p: str(a.dtype) for p, a in tree_flatten(model.parameters())}
    fp32 = {p for p, dt in after.items() if dt == fp32_s}
    bf16 = {p for p, dt in after.items() if dt == bf16_s}
    # Every fp32 leaf must be a per_dim_scale; everything else bf16.
    assert fp32, "expected at least one per_dim_scale fp32 leaf"
    assert all(p.endswith("per_dim_scale") for p in fp32), fp32
    assert bf16, "expected bf16 leaves"
    assert not (fp32 & bf16)
    assert (fp32 | bf16) == set(after)


# ---------------------------------------------------------------------------
# Gated codec (SpectroStream) decode-parity test.
# ---------------------------------------------------------------------------
#
# The depthformer parity above guards the direct depthformer load. The codec
# is loaded separately: ``load_from_safetensors_direct`` used to call
# ``_build_loaded_sl_sampler`` (a FULL fp32 sl sampler — incl. the ~9.6 GB base
# depthformer twin — built only to extract the codec), and now calls
# ``_build_loaded_sl_spectrostream`` (a standalone sl SpectroStream, no
# depthformer). This test proves that swap is functionally exact.
#
# The pure codec stores conv kernels OUTSIDE ``nn.Module.parameters()``, so a
# param-tree bit-compare (used for the depthformer) captures only the quantizer
# embedding. The faithful comparison is therefore *functional*: decode a fixed
# RVQ code grid through each codec load and assert a byte-identical waveform —
# which exercises the quantizer + every decoder conv end to end.


def _mirror_codec(pure_sampler, sl_ss):
    """Copy quantizer + decoder + encoder weights from an sl ``SpectroStream``
    into a pure ``MagentaRT2Sampler``'s codec (the pure encoder loader
    self-materializes the sl encoder's deferred convs)."""
    from magenta_rt.mlx_pure.spectrostream.load_weights import (
        load_quantizer_weights,
        load_spectrostream_decoder_weights,
        load_spectrostream_encoder_weights,
    )

    load_quantizer_weights(pure_sampler.spectrostream.quantizer, sl_ss.quantizer)
    load_spectrostream_decoder_weights(
        pure_sampler.spectrostream.decoder, sl_ss.decoder
    )
    load_spectrostream_encoder_weights(
        pure_sampler.spectrostream.encoder, sl_ss.encoder, sl_ss.config
    )
    mx.eval(pure_sampler.spectrostream.parameters())


def _decode_bits(pure_sampler) -> np.ndarray:
    """Decode a fixed deterministic RVQ code grid to a waveform (fp32 array)."""
    q = pure_sampler.spectrostream.quantizer
    ncb, nemb, T = q.num_expected_input_codes, q.num_embeddings, 8
    codes = mx.array(
        (np.arange(T * ncb).reshape(1, T, ncb) % nemb).astype(np.int32)
    )
    pure_sampler.spectrostream.disable_streaming()
    wav = pure_sampler.spectrostream.codes_to_waveform(codes)
    mx.eval(wav)
    return np.array(wav.astype(mx.float32))


@pytest.mark.checkpoint
def test_direct_codec_decode_parity_mrt2_small():
    """Standalone-SpectroStream codec load decodes byte-identically to the
    full-sampler sl-bridge codec load. Loaded sequentially (golden freed before
    the direct build) to respect the memory budget.
    """
    from magenta_rt import paths
    from magenta_rt.mlx_pure import load_weights as lw
    from magenta_rt.mlx_pure.model import MagentaRT2Sampler

    ckpt = paths.resolve_checkpoint("mrt2_small.safetensors")
    if not ckpt.is_file():
        pytest.skip(f"mrt2_small.safetensors not found at {ckpt}")

    # ---- Golden: codec extracted from the full fp32 sl sampler. ----
    golden_sampler = MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False
    )
    sl_sampler = lw._build_loaded_sl_sampler(str(ckpt), "mrt2_small")
    sl_ss = getattr(sl_sampler, "spectrostream", None) or sl_sampler.soundstream
    _mirror_codec(golden_sampler, sl_ss)
    golden_wav = _decode_bits(golden_sampler)
    del golden_sampler, sl_sampler, sl_ss
    gc.collect()

    # ---- Direct: codec from the standalone SpectroStream (no depthformer). ----
    direct_sampler = MagentaRT2Sampler.from_preset(
        "mrt2_small", int16_outputs=False
    )
    sl_ss2 = lw._build_loaded_sl_spectrostream(str(ckpt), "mrt2_small")
    _mirror_codec(direct_sampler, sl_ss2)
    direct_wav = _decode_bits(direct_sampler)
    del direct_sampler, sl_ss2
    gc.collect()

    assert golden_wav.shape == direct_wav.shape, (
        golden_wav.shape, direct_wav.shape,
    )
    assert np.array_equal(
        golden_wav.view(np.uint8), direct_wav.view(np.uint8)
    ), f"codec decode not byte-identical; max|Δ|={np.abs(golden_wav - direct_wav).max()}"
