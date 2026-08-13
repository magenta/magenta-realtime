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

"""Extract MusicCoCa weights from the TFLite resources into safetensors.

The five ``.tflite`` files shipped with Magenta-RT are ``jax2tf`` exports of
the original Praxis modules, and the tensor names inside the flatbuffers
retain the full module paths (``x_layers_3/self_attention/query/...``).
This converter walks those flatbuffers and re-keys the constants into the
layout expected by :mod:`magenta_rt.nnx.musiccoca.model`.

Two normalizations are baked in at conversion time so the runtime modules
stay uniform across the two towers:

* The text encoder's query projection has the ``1/sqrt(head_dim)`` scale
  folded into its weights by the exporter (``_scale_query/mul``); the music
  encoder applies ``* 0.125`` at runtime instead. We fold ``0.125`` into the
  music tower's query kernel/bias so neither tower scales at runtime.
* Praxis LayerNorm stores its scale as an offset from 1. The unrolled text
  encoder has the ``+1`` constant-folded by the converter, but the music
  encoder's WHILE-loop weights are stored raw; we add ``1.0`` there.

Run:

    python -m magenta_rt.nnx.musiccoca.convert [--resource-dir DIR] [--output FILE]

Requires ``tensorflow`` (for the TFLite flatbuffer schema); it is only
imported when this module runs, not by the runtime model.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np

from ... import paths

# TFLite TensorType codes we care about.
_DTYPES = {0: np.float32, 2: np.int32, 6: np.bool_, 9: np.int8, 10: np.float64}


def _read_model(path: pathlib.Path):
    from tensorflow.lite.tools import flatbuffer_utils

    return flatbuffer_utils.read_model(str(path))


class _Subgraph:
    """Name-indexed view of one TFLite subgraph's constant tensors."""

    def __init__(self, model, subgraph):
        self.model = model
        self.subgraph = subgraph

    def data(self, index: int) -> np.ndarray:
        tensor = self.subgraph.tensors[index]
        buffer = self.model.buffers[tensor.buffer].data
        if buffer is None:
            raise ValueError(f"tensor #{index} is not a constant")
        array = np.frombuffer(bytes(buffer), dtype=_DTYPES[tensor.type])
        return array.reshape([int(d) for d in tensor.shape])

    def find(self, pattern: str, shape: tuple[int, ...]) -> np.ndarray:
        """Returns the unique constant whose name matches and shape equals.

        TFLite fuses op names with ``;`` when buffers are shared between
        consumers (a key kernel may carry ``value/...`` in its tail), so a
        match against the *first* segment wins; the full name is only a
        fallback when no first-segment match exists.
        """
        first, full = [], []
        for i, tensor in enumerate(self.subgraph.tensors):
            if self.model.buffers[tensor.buffer].data is None:
                continue
            if tuple(int(d) for d in tensor.shape) != shape:
                continue
            name = tensor.name.decode()
            if re.search(pattern, name.split(";")[0]):
                first.append(i)
            elif re.search(pattern, name):
                full.append(i)
        matches = first or full
        if len(matches) != 1:
            raise ValueError(
                f"expected 1 tensor matching {pattern!r} with shape {shape}, "
                f"found {len(first)} first-segment + {len(full)} full-name"
            )
        return self.data(matches[0])

    def arg(self, n: int) -> np.ndarray:
        """Returns the ``jax2tf_arg_<n>`` constant (stacked loop weights)."""
        for i, tensor in enumerate(self.subgraph.tensors):
            if tensor.name.decode().startswith(f"jax2tf_arg_{n}/"):
                return self.data(i)
        raise ValueError(f"jax2tf_arg_{n} not found")


# -----------------------------------------------------------------------------
# Per-model extraction
# -----------------------------------------------------------------------------


def convert_frontend(path: pathlib.Path) -> dict[str, np.ndarray]:
    """audio_preprocessor.tflite → log-mel constants."""
    model = _read_model(path)
    main = _Subgraph(model, model.subgraphs[0])
    # The graph pads the [1024, 128] matrix with one leading zero row (the
    # DC bin) before the matmul against the 1025-bin power spectrum.
    mel = main.find(r"linear_to_mel_weight_matrix/Maximum", (1024, 128))
    mel = np.concatenate([np.zeros((1, 128), np.float32), mel], axis=0)
    window = main.find(r"hann_window", (400,))
    return {"frontend.mel_matrix": mel, "frontend.window": window}


def _tower_layer(
    qkv_post: dict[str, np.ndarray],
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    """Reshapes unrolled TFLite FC weights ([out, in]) to Praxis DNH."""
    d = num_heads * head_dim
    out = {}
    for name in ("q", "k", "v"):
        out[f"{name}.kernel"] = (
            qkv_post[f"{name}.kernel"].T.reshape(d, num_heads, head_dim)
        )
        out[f"{name}.bias"] = qkv_post[f"{name}.bias"].reshape(num_heads, head_dim)
    # post: FC weight is [D out, N*H in]; DNH splits the *input* axis.
    out["post.kernel"] = qkv_post["post.kernel"].reshape(d, num_heads, head_dim)
    out["post.bias"] = qkv_post["post.bias"]
    return out


def convert_text_encoder(path: pathlib.Path) -> dict[str, np.ndarray]:
    """text_encoder.tflite → ``text.*`` params."""
    model = _read_model(path)
    main = _Subgraph(model, model.subgraphs[0])
    params: dict[str, np.ndarray] = {
        "text.token_emb": main.find(r"emb", (16000, 768)),
        "text.pos_emb": main.find(r"position_emb", (1, 128, 768)).reshape(128, 768),
        "text.final_ln.scale": main.find(
            r"exit_stack/ln/broadcast_in_dim1", (1, 1, 768)
        ).reshape(768),
        "text.final_ln.bias": main.find(
            r"exit_stack/ln/broadcast_in_dim(?!1)", (1, 1, 768)
        ).reshape(768),
    }
    blocks = []
    for i in range(12):
        layer = f"x_layers_{i}/"
        raw = {
            # The exporter folded the 1/sqrt(head_dim) query scale into the
            # weights, hence the `_scale_query/mul` names.
            "q.kernel": main.find(layer + r"self_attention/.*_scale_query/mul", (768, 768)),
            "q.bias": main.find(layer + r"self_attention/.*_scale_query/mul", (768,)),
            "k.kernel": main.find(layer + r"self_attention/key/einsum", (768, 768)),
            "k.bias": main.find(layer + r"self_attention/key/add", (768,)),
            "v.kernel": main.find(layer + r"self_attention/value/einsum", (768, 768)),
            "v.bias": main.find(layer + r"self_attention/value/add", (768,)),
            "post.kernel": main.find(layer + r"self_attention/post/einsum", (768, 768)),
            "post.bias": main.find(layer + r"self_attention/post/add", (768,)),
        }
        block = _tower_layer(raw, num_heads=12, head_dim=64)
        block["ln1.scale"] = main.find(
            layer + r"layer_norm/broadcast_in_dim1", (1, 1, 768)
        ).reshape(768)
        block["ln1.bias"] = main.find(
            layer + r"layer_norm/broadcast_in_dim(?!1)", (1, 1, 768)
        ).reshape(768)
        block["ln2.scale"] = main.find(
            layer + r"ff_layer/layer_norm/broadcast_in_dim1", (1, 1, 768)
        ).reshape(768)
        block["ln2.bias"] = main.find(
            layer + r"ff_layer/layer_norm/broadcast_in_dim(?!1)", (1, 1, 768)
        ).reshape(768)
        block["ffn1.kernel"] = main.find(layer + r".*ffn_layer1/linear", (3072, 768)).T
        block["ffn1.bias"] = main.find(layer + r".*ffn_layer1/bias", (3072,))
        block["ffn2.kernel"] = main.find(layer + r".*ffn_layer2/linear", (768, 3072)).T
        block["ffn2.bias"] = main.find(layer + r".*ffn_layer2/bias", (768,))
        blocks.append(block)
    # Stack the unrolled layers along a leading axis, matching the
    # vmap/scan layout used by the runtime modules (and by the music
    # encoder's WHILE-loop weights).
    for key in blocks[0]:
        params[f"text.layers.{key}"] = np.stack([b[key] for b in blocks])
    params.update(_convert_pooler(main, "contrastive_txt_pooler", "text"))
    return params


def _convert_pooler(
    main: _Subgraph, scope: str, prefix: str
) -> dict[str, np.ndarray]:
    """Extracts a CoCa attentional pooler (1 learned query, 12 heads x 256)."""
    return {
        f"{prefix}.pooler.query": main.find(
            scope + r"/pool_attn/.*qk_einsum", (1, 12, 1, 256)
        ).reshape(12, 256),
        f"{prefix}.pooler.key.kernel": main.find(
            scope + r"/pool_attn/key/einsum", (3072, 768)
        ).T.reshape(768, 12, 256),
        f"{prefix}.pooler.key.bias": main.find(
            scope + r"/pool_attn/key/add", (3072,)
        ).reshape(12, 256),
        f"{prefix}.pooler.value.kernel": main.find(
            scope + r"/pool_attn/value/einsum", (3072, 768)
        ).T.reshape(768, 12, 256),
        f"{prefix}.pooler.value.bias": main.find(
            scope + r"/pool_attn/value/add", (3072,)
        ).reshape(12, 256),
        f"{prefix}.pooler.post.kernel": main.find(
            scope + r"/pool_attn/post/einsum", (768, 3072)
        ).reshape(768, 12, 256),
        f"{prefix}.pooler.post.bias": main.find(
            scope + r"/pool_attn/post/add", (768,)
        ),
        f"{prefix}.pooler.ln.scale": main.find(
            scope + r"/pool_attn_ln/broadcast_in_dim1", (1, 1, 768)
        ).reshape(768),
        f"{prefix}.pooler.ln.bias": main.find(
            scope + r"/pool_attn_ln/broadcast_in_dim(?!1)", (1, 1, 768)
        ).reshape(768),
    }


def convert_music_encoder(path: pathlib.Path) -> dict[str, np.ndarray]:
    """music_encoder.tflite → ``audio.*`` params."""
    model = _read_model(path)
    main = _Subgraph(model, model.subgraphs[0])
    # Subgraph #2 is the WHILE body holding the 12 stacked layers.
    body = _Subgraph(model, model.subgraphs[2])

    params: dict[str, np.ndarray] = {
        "audio.patch_proj.kernel": main.find(
            r"patch_projection/linear", (768, 256)
        ).T,
        # The exporter folded patch-projection bias + learned position
        # embedding into a single [1, 496, 768] constant.
        "audio.pos_emb": main.find(
            r"entry_stack/add", (1, 496, 768)
        ).reshape(496, 768),
        "audio.final_ln.scale": main.find(
            r"exit_stack/ln/broadcast_in_dim1", (1, 1, 768)
        ).reshape(768),
        "audio.final_ln.bias": main.find(
            r"exit_stack/ln/broadcast_in_dim(?!1)", (1, 1, 768)
        ).reshape(768),
    }
    params.update(_convert_pooler(main, "contrastive_music_pooler", "audio"))

    # Stacked per-layer weights, mapped by jax2tf arg number (verified
    # against the WHILE-body op walk; parity tests confirm end-to-end).
    stacked = {
        "ffn1.bias": body.arg(46),          # [12, 3072]
        "ffn1.kernel": body.arg(47),        # [12, 768, 3072]
        "ffn2.bias": body.arg(48),          # [12, 768]
        "ffn2.kernel": body.arg(49),        # [12, 3072, 768]
        "ln2.bias": body.arg(50),           # [12, 768]
        "ln2.scale": body.arg(51) + 1.0,    # praxis scale-offset convention
        "ln1.bias": body.arg(52),
        "ln1.scale": body.arg(53) + 1.0,
        "k.bias": body.arg(54),             # [12, 12, 64]
        "k.kernel": body.arg(55),           # [12, 768, 12, 64]
        "post.bias": body.arg(56),
        "post.kernel": body.arg(57),
        "q.bias": body.arg(58) * 0.125,     # fold the runtime query scale
        "q.kernel": body.arg(59) * 0.125,
        "v.bias": body.arg(60),
        "v.kernel": body.arg(61),
    }
    params.update({f"audio.layers.{k}": v for k, v in stacked.items()})
    return params


def convert_quantizer(path: pathlib.Path) -> dict[str, np.ndarray]:
    """pretrained_vector_quantizer.tflite → 12 RVQ codebooks [12, 1024, 768].

    Stages 1-11 keep their codebooks as GATHER tables; stage 12 has no
    residual lookup, so its codebook is recovered from the distance matmul
    weight, which is ``-2 * codebook``.
    """
    from tensorflow.lite.python import schema_py_generated as schema_fb

    model = _read_model(path)
    sg = model.subgraphs[0]
    main = _Subgraph(model, sg)

    def op_name(op) -> int:
        code = model.operatorCodes[op.opcodeIndex]
        return max(code.builtinCode, code.deprecatedBuiltinCode)

    codebooks = []
    fc_inputs = []
    for op in sg.operators:
        code = op_name(op)
        if code == schema_fb.BuiltinOperator.GATHER:
            data = main.data(op.inputs[0])
            if data.shape == (1024, 768):
                codebooks.append(data)
        elif code == schema_fb.BuiltinOperator.FULLY_CONNECTED:
            fc_inputs.append(op)
    if len(codebooks) != 11 or len(fc_inputs) != 12:
        raise ValueError(
            f"unexpected quantizer structure: {len(codebooks)} gathers, "
            f"{len(fc_inputs)} matmuls"
        )
    last = fc_inputs[-1]
    codebooks.append(-0.5 * main.data(last.inputs[1]))
    codebooks = np.stack(codebooks)  # [12, 1024, 768]

    # Cross-check: each stage's distance bias must equal ||c||^2.
    for stage, op in enumerate(fc_inputs):
        bias = main.data(op.inputs[2])
        norms = (codebooks[stage] ** 2).sum(-1)
        if not np.allclose(bias, norms, rtol=1e-3, atol=1e-5):
            raise ValueError(f"stage {stage}: bias != ||codebook||^2")
        weight = main.data(op.inputs[1])
        if not np.allclose(weight, -2.0 * codebooks[stage], rtol=1e-5, atol=1e-7):
            raise ValueError(f"stage {stage}: weight != -2 * codebook")
    return {"quantizer.codebooks": codebooks}


def convert_mapper(path: pathlib.Path) -> dict[str, np.ndarray]:
    """mapper.tflite → ``mapper.*`` params (8 stacked DiT-style layers)."""
    model = _read_model(path)
    main = _Subgraph(model, model.subgraphs[0])
    body = _Subgraph(model, model.subgraphs[2])

    params: dict[str, np.ndarray] = {
        # Learned 128-dim prefix, concatenated twice before the text
        # embedding to form the [1, 1, 1024] conditioning context.
        "mapper.context": main.find(r"arith.constant1$", (1, 1, 128)).reshape(128),
        "mapper.input_proj.kernel": main.find(r"input_projection", (3072, 768)).T,
        "mapper.input_proj.bias": main.find(r"input_projection", (3072,)),
        "mapper.output_proj.kernel": main.find(r"output_projection", (768, 3072)).T,
        "mapper.output_proj.bias": main.find(r"output_projection", (768,)),
    }
    stacked = {
        "cond1.bias": body.arg(2),       # [8, 2, 256]
        "cond1.kernel": body.arg(3),     # [8, 1024, 2, 256]
        "norm1.scale": body.arg(4),      # [8, 256]
        "qkv.kernel": body.arg(5),       # [8, 256, 3, 8, 64]
        "k_sink": body.arg(6),           # [8, 1, 8, 64]
        "v_sink": body.arg(7),           # [8, 1, 8, 64]
        "post.kernel": body.arg(8),      # [8, 256, 8, 64]
        "cond2.bias": body.arg(9),
        "cond2.kernel": body.arg(10),
        "ffn1.bias": body.arg(11),       # [8, 1024]
        "ffn1.kernel": body.arg(12),     # [8, 256, 1024]
        "ffn2.bias": body.arg(13),       # [8, 256]
        "ffn2.kernel": body.arg(14),     # [8, 1024, 256]
        "norm2.scale": body.arg(15),
    }
    params.update({f"mapper.layers.{k}": v for k, v in stacked.items()})
    return params


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

DEFAULT_FILENAME = "musiccoca_nnx.safetensors"


def convert(
    resource_dir: str | pathlib.Path | None = None,
    output: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Converts all five TFLite models; returns the safetensors path."""
    import safetensors.numpy

    resource_dir = pathlib.Path(resource_dir or paths.musiccoca_dir())
    output = pathlib.Path(output or resource_dir / DEFAULT_FILENAME)

    params: dict[str, np.ndarray] = {}
    params.update(convert_frontend(resource_dir / "audio_preprocessor.tflite"))
    params.update(convert_music_encoder(resource_dir / "music_encoder.tflite"))
    params.update(convert_text_encoder(resource_dir / "text_encoder.tflite"))
    params.update(
        convert_quantizer(resource_dir / "pretrained_vector_quantizer.tflite")
    )
    params.update(convert_mapper(resource_dir / "mapper.tflite"))

    params = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in params.items()}
    safetensors.numpy.save_file(params, str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = convert(args.resource_dir, args.output)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
