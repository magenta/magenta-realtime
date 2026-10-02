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

"""Convert a Flax MagentaRT-2 safetensors checkpoint to GGUF.

Layout conventions
------------------
GGUF stores a tensor's `ne` as the *reverse* of the numpy shape, and
`ggml_mul_mat(w, x)` expects `w.ne == (in, out)`. So a weight must be handed to
the writer as numpy `[out, in]`.

Flax stores dense kernels as `[in, out]` and conv kernels as `[KH, KW, IC, OC]`,
so both need transposing. Every tensor is matched against an explicit rule
below; an unrecognized name is a hard error rather than a silent pass-through,
because a wrong layout produces a model that runs and emits garbage.
"""

import argparse
import json
import os
import re
import struct

import gguf
import numpy as np

from magenta_rt import config as mrt_config

ARCH = "magenta-rt2"

# Mirrors magenta_rt/{jax,mlx}/model.py. Duplicated here so the converter needs
# only magenta_rt.config (dataclasses-only) rather than importing jax or mlx.
MODELS: dict[str, dict] = {
    "mrt2_small": dict(
        encoder=mrt_config.S,
        temporal=mrt_config.L_TPU_OPTIMIZED,
        depth=mrt_config.M_SHALLOW_TPU_OPTIMIZED,
        window=41,
    ),
    "mrt2_base": dict(
        encoder=mrt_config.L_SHALLOW_TPU_OPTIMIZED,
        temporal=mrt_config.XXL_SHALLOW,
        depth=mrt_config.L_SHALLOW_TPU_OPTIMIZED_6,
        window=25,
    ),
}

SOS_ID = 0
SOFT_CAP_LOGITS = 30.0

# stft_spectrostream_40ms_generic_48khz_stereo_config
CODEC = dict(
    sample_rate=48000,
    stft_frame_length=960,
    stft_frame_step=480,
    stft_fft_length=960,
    num_bins=480,
    num_channels=4,
    num_features=256,
    channel_splits=2,
    decoder_base_conv_depth=64,
    decoder_base_conv_size=7,
    decoder_lookahead=1,
    ratios=((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1)),
    mults=(2, 1, 2, 1, 1, 2, 1),
    quantizer_num=64,
    quantizer_size=1024,
    quantizer_dim=256,
)
PARAM_PREFIX = "params/"

# (regex, transform-name). First match wins; order matters.
RULES: list[tuple[str, str]] = [
    # --- attention -----------------------------------------------------------
    # [in, heads, dh] -> [heads*dh, in]
    (r"/attention/(query|key|value)_projection/kernel$", "proj3d"),
    # [d_model, heads, dh]; heads*dh contracts, so flatten without transposing.
    (r"/output_projection/kernel$", "outproj3d"),
    # [1, heads, dh] -> [heads, dh]
    (r"/attention/sink_(key|value)_embeddings$", "sink"),
    (r"/attention/per_dim_scale$", "vector"),
    # --- embeddings (kept [rows, dim]; ggml_get_rows wants ne=(dim, rows)) ----
    (r"/embedding$", "embedding"),
    # --- convolutions --------------------------------------------------------
    (r"/conv/kernel$", "conv4d"),
    (r"/conv/bias$", "vector"),
    # --- norms / biases ------------------------------------------------------
    (r"/(scale|bias)$", "vector"),
    # --- generic dense kernels [in, out] -> [out, in] ------------------------
    (r"/kernel$", "dense2d"),
]


def _read_safetensors(path: str) -> tuple[dict, dict]:
    """Return ({name: ndarray}, metadata) without needing torch."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        metadata = header.pop("__metadata__", {})
        dtypes = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
        out = {}
        for name, info in header.items():
            if info["dtype"] not in dtypes:
                raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            buf = f.read(end - begin)
            arr = np.frombuffer(buf, dtype=dtypes[info["dtype"]])
            out[name] = arr.reshape(info["shape"]).copy()
    return out, metadata


def classify(name: str) -> str:
    for pattern, kind in RULES:
        if re.search(pattern, name):
            return kind
    raise KeyError(f"no layout rule for tensor '{name}'")


def transform(name: str, arr: np.ndarray) -> np.ndarray:
    """Apply the layout transform so the writer receives numpy [out, in, ...]."""
    kind = classify(name)
    if kind == "vector":
        if arr.ndim != 1:
            raise ValueError(f"{name}: expected 1-D, got {arr.shape}")
        out = arr
    elif kind == "embedding":
        if arr.ndim not in (2, 3):
            raise ValueError(f"{name}: expected 2-D/3-D, got {arr.shape}")
        out = arr
    elif kind == "dense2d":
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected 2-D, got {arr.shape}")
        out = arr.T
    elif kind == "proj3d":
        if arr.ndim != 3:
            raise ValueError(f"{name}: expected 3-D, got {arr.shape}")
        in_dim, heads, dh = arr.shape
        out = arr.reshape(in_dim, heads * dh).T
    elif kind == "outproj3d":
        if arr.ndim != 3:
            raise ValueError(f"{name}: expected 3-D, got {arr.shape}")
        d_model, heads, dh = arr.shape
        out = arr.reshape(d_model, heads * dh)
    elif kind == "sink":
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise ValueError(f"{name}: expected [1, heads, dh], got {arr.shape}")
        out = arr[0]
    elif kind == "conv4d":
        if arr.ndim != 4:
            raise ValueError(f"{name}: expected 4-D, got {arr.shape}")
        # [KH, KW, IC, OC] -> [OC, IC, KH, KW]  => ne = (KW, KH, IC, OC)
        out = arr.transpose(3, 2, 0, 1)
    else:
        raise AssertionError(kind)
    return np.ascontiguousarray(out)


# GGUF caps tensor names at 64 chars (GGML_MAX_NAME), while the Flax paths run
# to 73, so they are mapped to a compact scheme the C++ side indexes directly.
NAME_RULES: list[tuple[str, str]] = [
    # encoder
    (r"^depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding$", "enc.mulan.emb"),
    (r"^depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel$", "enc.mulan.adapter"),
    (r"^depthformer/encoder/body/layers_1/branched_regular_embedder/regular_embedder/embedding$", "enc.reg.emb"),
    (r"^depthformer/encoder/body/encoder_ln/(scale|bias)$", r"enc.ln.\1"),
    # decoder embedding
    (r"^depthformer/decoder/decoder_embedding/embedding/embedding$", "dec.emb"),
    # depth head
    (r"^depthformer/decoder/depth_body/depth_input_adapter/kernel$", "d.adapter"),
    (r"^depthformer/decoder/depth_body/final_ln/(scale|bias)$", r"d.ln.\1"),
    (r"^depthformer/decoder/depth_body/to_logits/kernel$", "d.logits.w"),
    (r"^depthformer/decoder/depth_body/to_logits/bias$", "d.logits.b"),
    # soundstream codec
    (r"^soundstream/quantizer/embedding$", "ss.quant.emb"),
]

# body -> prefix, sublayer -> tag
_BODY = {"temporal_body": "t", "depth_body": "d"}
_SUB = {"self_attention": "sa", "cross_attention": "ca"}
_PROJ = {"query": "q", "key": "k", "value": "v"}


def gguf_name(flax_name: str) -> str:
    """Map a Flax path to a short, stable GGUF tensor name."""
    if not flax_name.startswith(PARAM_PREFIX):
        raise ValueError(f"unexpected tensor name '{flax_name}'")
    name = flax_name[len(PARAM_PREFIX):]

    for pattern, repl in NAME_RULES:
        new, n = re.subn(pattern, repl, name)
        if n:
            return _checked(new, flax_name)

    m = re.match(
        r"^depthformer/decoder/(temporal_body|depth_body)/transformer/x_layers_(\d+)/(.+)$",
        name)
    if m:
        body, idx, rest = _BODY[m.group(1)], m.group(2), m.group(3)
        pre = f"{body}.{idx}"
        sub = re.match(r"^(self_attention|cross_attention)/(.+)$", rest)
        if sub:
            tag, tail = _SUB[sub.group(1)], sub.group(2)
            table = {
                "attention/per_dim_scale": "pds",
                "attention/sink_key_embeddings": "sink_k",
                "attention/sink_value_embeddings": "sink_v",
                "output_projection/kernel": "o",
                "pre_norm/scale": "pre",
                "post_norm/scale": "post",
            }
            if tail in table:
                return _checked(f"{pre}.{tag}.{table[tail]}", flax_name)
            proj = re.match(r"^attention/(query|key|value)_projection/kernel$", tail)
            if proj:
                return _checked(f"{pre}.{tag}.{_PROJ[proj.group(1)]}", flax_name)
        ffn = re.match(r"^ffn/(.+)$", rest)
        if ffn:
            table = {
                "ffn_layer1/kernel": "w1", "ffn_layer1/bias": "b1",
                "ffn_layer2/kernel": "w2", "ffn_layer2/bias": "b2",
                "pre_norm/scale": "pre", "post_norm/scale": "post",
            }
            tail = ffn.group(1)
            if tail in table:
                return _checked(f"{pre}.ffn.{table[tail]}", flax_name)

    if name.startswith("soundstream/decoder/"):
        short = name[len("soundstream/decoder/"):]
        short = short.replace("/conv/kernel", ".w").replace("/conv/bias", ".b")
        return _checked("ss." + short.replace("/", "."), flax_name)

    raise KeyError(f"no GGUF name mapping for '{flax_name}'")


def _checked(name: str, original: str) -> str:
    if len(name) >= 64:
        raise ValueError(f"name too long ({len(name)}): '{name}' from '{original}'")
    return name


def add_metadata(writer: "gguf.GGUFWriter", model_name: str) -> None:
    """Record the hyperparameters so the C++ side need not hardcode them."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model '{model_name}'; have {sorted(MODELS)}")
    m = MODELS[model_name]
    ss = mrt_config.SPECTROSTREAM
    writer.add_string(f"{ARCH}.model_name", model_name)

    for part in ("temporal", "depth"):
        spec = m[part]
        p = f"{ARCH}.{part}"
        writer.add_uint32(f"{p}.block_count", spec.num_layers)
        writer.add_uint32(f"{p}.embedding_length", spec.model_dims)
        writer.add_uint32(f"{p}.feed_forward_length", spec.hidden_dims)
        writer.add_uint32(f"{p}.attention.head_count", spec.num_heads)
        writer.add_uint32(f"{p}.attention.key_length", spec.dim_per_head)

    # The encoder body is sl.Identity (see model.py: `body=sl.Identity.Config()`),
    # so despite `encoder_size` naming a ModelSpec, there are no encoder
    # transformer layers in the checkpoint -- only the branched embedders and a
    # final LayerNorm. Only the width is meaningful.
    writer.add_uint32(f"{ARCH}.encoder.block_count", 0)
    writer.add_uint32(f"{ARCH}.encoder.embedding_length", m["encoder"].model_dims)

    writer.add_uint32(f"{ARCH}.temporal.attention.max_past_horizon", m["window"])
    # The depth transformer attends over the RVQ axis, so its horizon is the
    # number of codebooks rather than a frame count.
    writer.add_uint32(f"{ARCH}.depth.attention.max_past_horizon",
                      ss.rvq_truncation_level)

    writer.add_uint32(f"{ARCH}.num_codebooks", ss.rvq_truncation_level)
    writer.add_uint32(f"{ARCH}.codebook_size", ss.codebook_size)
    writer.add_uint32(f"{ARCH}.num_reserved_tokens", ss.num_extra_tokens)
    writer.add_uint32(f"{ARCH}.sos_id", SOS_ID)
    writer.add_float32(f"{ARCH}.soft_cap_logits", SOFT_CAP_LOGITS)
    writer.add_uint32(f"{ARCH}.frame_rate", int(ss.frame_rate))

    for key, value in CODEC.items():
        name = f"{ARCH}.codec.{key}"
        if isinstance(value, tuple):
            flat = [int(x) for pair in value for x in
                    (pair if isinstance(pair, tuple) else (pair,))]
            writer.add_array(name, flat)
        elif isinstance(value, int):
            writer.add_uint32(name, value)
        else:
            raise AssertionError(f"{key}: {type(value)}")


def convert(src: str, dst: str, model_name: str, dtype: str = "f32") -> None:
    tensors, _ = _read_safetensors(src)
    print(f"Read {len(tensors)} tensors from {src}")

    writer = gguf.GGUFWriter(dst, ARCH)
    add_metadata(writer, model_name)
    np_dtype = {"f32": np.float32, "f16": np.float16,
                "q8_0": np.float32, "q8_0_dequant": np.float32}[dtype]

    counts: dict[str, int] = {}
    for flax_name in sorted(tensors):
        arr = transform(flax_name, tensors[flax_name])
        kind = classify(flax_name)
        counts[kind] = counts.get(kind, 0) + 1
        # These stay fp32 regardless of --dtype:
        #  - "vector": norms/biases, tiny and precision-sensitive.
        #  - "sink":   consumed by an elementwise ggml_mul as src[0], which
        #              Metal only implements for F32 (same constraint that
        #              blocks fp16 activations).
        #  - "embedding": rows are read back on the host as float.
        # Only the matmul/conv weights are affected by --dtype, which is where
        # essentially all the size and bandwidth is anyway.
        keep_f32 = kind in ("vector", "sink", "embedding")
        name = gguf_name(flax_name)
        quantizable = (not keep_f32 and arr.ndim == 2 and arr.shape[-1] % 32 == 0)
        if dtype == "q8_0_dequant" and quantizable:
            # Round-trip through Q8_0 and store the result as plain F32. The
            # weight *values* are bit-identical to the q8_0 file; only the
            # storage type differs, so ggml runs mul_mv_f32_f32 instead of
            # mul_mv_q8_0_f32. This isolates "when does dequantization happen"
            # from "what values does quantization produce".
            packed = gguf.quants.quantize(arr.astype(np.float32),
                                          gguf.GGMLQuantizationType.Q8_0)
            deq = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, deq.astype(np.float32).reshape(arr.shape))
            counts["q8_0_dequant"] = counts.get("q8_0_dequant", 0) + 1
            continue
        if dtype == "q8_0" and quantizable:
            # Q8_0 blocks are 32 wide along the row, matching MLX's 8-bit
            # quantization of the same matmul weights (group_size=64 there).
            qdata = gguf.quants.quantize(arr.astype(np.float32),
                                         gguf.GGMLQuantizationType.Q8_0)
            # qdata is the packed byte array; the writer derives the logical
            # shape from it, so raw_shape must not be passed.
            writer.add_tensor(name, qdata,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            counts["q8_0"] = counts.get("q8_0", 0) + 1
            continue
        out_dtype = np.float32 if keep_f32 else np_dtype
        writer.add_tensor(name, arr.astype(out_dtype))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {dst}")
    for k in sorted(counts):
        print(f"  {k:12s} {counts[k]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Flax .safetensors path")
    ap.add_argument("--output", required=True, help="Destination .gguf path")
    ap.add_argument("--model", default="mrt2_small", choices=sorted(MODELS))
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8_0", "q8_0_dequant"])
    args = ap.parse_args()
    convert(args.checkpoint, args.output, args.model, args.dtype)


if __name__ == "__main__":
    main()
