# Copyright 2025 The MT3 Authors.
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

"""Load pretrained MT3 weights into the MLX model.

Consumes the same safetensors produced by ``python -m
magenta_rt.mt3.download`` (the converted t5x checkpoint) as the nnx
backend — the DenseGeneral kernels are stored flattened 2-D, matching the
checkpoint exactly.
"""

from typing import Optional

import mlx.core as mx

from magenta_rt.mt3.config import MT3Config
from magenta_rt.mt3.download import download_model

from .model import MT3


def _load_attention(attention, params: dict, prefix: str):
    attention.query.kernel = mx.array(params.pop(f"{prefix}/query/kernel"))
    attention.key.kernel = mx.array(params.pop(f"{prefix}/key/kernel"))
    attention.value.kernel = mx.array(params.pop(f"{prefix}/value/kernel"))
    attention.out.kernel = mx.array(params.pop(f"{prefix}/out/kernel"))


def _load_mlp(mlp, params: dict, prefix: str):
    if len(mlp.wi_layers) == 1:
        mlp.wi_layers[0].kernel = mx.array(params.pop(f"{prefix}/wi/kernel"))
    else:
        for i, wi in enumerate(mlp.wi_layers):
            wi.kernel = mx.array(params.pop(f"{prefix}/wi_{i}/kernel"))
    mlp.wo.kernel = mx.array(params.pop(f"{prefix}/wo/kernel"))


def load_weights(model: MT3, params: dict):
    """Load a flat t5x parameter dict into an MLX MT3 model in place."""
    params = dict(params)

    encoder = model.encoder
    encoder.continuous_inputs_projection.kernel = mx.array(
        params.pop("encoder/continuous_inputs_projection/kernel")
    )
    for i, layer in enumerate(encoder.layers):
        prefix = f"encoder/layers_{i}"
        layer.pre_attention_layer_norm.scale = mx.array(
            params.pop(f"{prefix}/pre_attention_layer_norm/scale")
        )
        _load_attention(layer.attention, params, f"{prefix}/attention")
        layer.pre_mlp_layer_norm.scale = mx.array(
            params.pop(f"{prefix}/pre_mlp_layer_norm/scale")
        )
        _load_mlp(layer.mlp, params, f"{prefix}/mlp")
    encoder.encoder_norm.scale = mx.array(params.pop("encoder/encoder_norm/scale"))

    decoder = model.decoder
    decoder.token_embedder.embedding = mx.array(
        params.pop("decoder/token_embedder/embedding")
    )
    for i, layer in enumerate(decoder.layers):
        prefix = f"decoder/layers_{i}"
        layer.pre_self_attention_layer_norm.scale = mx.array(
            params.pop(f"{prefix}/pre_self_attention_layer_norm/scale")
        )
        _load_attention(layer.self_attention, params, f"{prefix}/self_attention")
        layer.pre_cross_attention_layer_norm.scale = mx.array(
            params.pop(f"{prefix}/pre_cross_attention_layer_norm/scale")
        )
        _load_attention(
            layer.encoder_decoder_attention, params, f"{prefix}/encoder_decoder_attention"
        )
        layer.pre_mlp_layer_norm.scale = mx.array(
            params.pop(f"{prefix}/pre_mlp_layer_norm/scale")
        )
        _load_mlp(layer.mlp, params, f"{prefix}/mlp")
    decoder.decoder_norm.scale = mx.array(params.pop("decoder/decoder_norm/scale"))
    if not model.config.logits_via_embedding:
        decoder.logits_dense.kernel = mx.array(params.pop("decoder/logits_dense/kernel"))

    if params:
        raise ValueError(f"Unused checkpoint parameters: {sorted(params.keys())}")


def load_model(
    model_type: str = "mt3",
    load_path: Optional[str] = None,
    config: Optional[MT3Config] = None,
) -> MT3:
    """Load a pretrained MT3 model (MLX).

    Args:
        model_type: 'mt3', 'ismir2021', 'ismir2022_small', or 'ismir2022_base'.
        load_path: Path to a safetensors weights file. If None, downloads the
            pretrained checkpoint.
        config: Model configuration. If None, uses the configuration matching
            ``model_type``.

    Returns:
        MT3 model in eval mode, ready for ``magenta_rt.mlx_pure.mt3.transcribe``.
    """
    from safetensors.numpy import load_file

    if config is None:
        config = MT3Config.from_pretrained(model_type)
    if load_path is None:
        load_path = download_model(model_type)

    model = MT3(config)
    load_weights(model, load_file(load_path))
    model.eval()
    mx.eval(model.parameters())
    return model
