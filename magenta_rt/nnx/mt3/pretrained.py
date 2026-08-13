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

"""Load pretrained MT3 weights into the NNX model."""

from typing import Optional

from jax import numpy as jnp
from flax import nnx

from magenta_rt.mt3.config import MT3Config
from magenta_rt.mt3.download import download_model

from ..checkpoint_utils import load_into_abstract
from .model import MT3


def _load_attention(attention, params: dict, prefix: str):
    attention.query.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/query/kernel")))
    attention.key.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/key/kernel")))
    attention.value.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/value/kernel")))
    attention.out.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/out/kernel")))


def _load_mlp(mlp, params: dict, prefix: str):
    if len(mlp.wi_layers) == 1:
        mlp.wi_layers[0].kernel.set_value(jnp.asarray(params.pop(f"{prefix}/wi/kernel")))
    else:
        for i, wi in enumerate(mlp.wi_layers):
            wi.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/wi_{i}/kernel")))
    mlp.wo.kernel.set_value(jnp.asarray(params.pop(f"{prefix}/wo/kernel")))


def load_weights(model: MT3, params: dict):
    """Load a flat t5x parameter dict into an MT3 model in place."""
    params = dict(params)

    encoder = model.encoder
    encoder.continuous_inputs_projection.kernel.set_value(jnp.asarray(
        params.pop("encoder/continuous_inputs_projection/kernel")
    ))
    for i, layer in enumerate(encoder.layers):
        prefix = f"encoder/layers_{i}"
        layer.pre_attention_layer_norm.scale.set_value(jnp.asarray(
            params.pop(f"{prefix}/pre_attention_layer_norm/scale")
        ))
        _load_attention(layer.attention, params, f"{prefix}/attention")
        layer.pre_mlp_layer_norm.scale.set_value(jnp.asarray(
            params.pop(f"{prefix}/pre_mlp_layer_norm/scale")
        ))
        _load_mlp(layer.mlp, params, f"{prefix}/mlp")
    encoder.encoder_norm.scale.set_value(jnp.asarray(params.pop("encoder/encoder_norm/scale")))

    decoder = model.decoder
    decoder.token_embedder.embedding.set_value(jnp.asarray(
        params.pop("decoder/token_embedder/embedding")
    ))
    for i, layer in enumerate(decoder.layers):
        prefix = f"decoder/layers_{i}"
        layer.pre_self_attention_layer_norm.scale.set_value(jnp.asarray(
            params.pop(f"{prefix}/pre_self_attention_layer_norm/scale")
        ))
        _load_attention(layer.self_attention, params, f"{prefix}/self_attention")
        layer.pre_cross_attention_layer_norm.scale.set_value(jnp.asarray(
            params.pop(f"{prefix}/pre_cross_attention_layer_norm/scale")
        ))
        _load_attention(
            layer.encoder_decoder_attention, params, f"{prefix}/encoder_decoder_attention"
        )
        layer.pre_mlp_layer_norm.scale.set_value(jnp.asarray(
            params.pop(f"{prefix}/pre_mlp_layer_norm/scale")
        ))
        _load_mlp(layer.mlp, params, f"{prefix}/mlp")
    decoder.decoder_norm.scale.set_value(jnp.asarray(params.pop("decoder/decoder_norm/scale")))
    if not model.config.logits_via_embedding:
        decoder.logits_dense.kernel.set_value(jnp.asarray(params.pop("decoder/logits_dense/kernel")))

    if params:
        raise ValueError(f"Unused checkpoint parameters: {sorted(params.keys())}")


def load_model(
    model_type: str = "mt3",
    load_path: Optional[str] = None,
    config: Optional[MT3Config] = None,
) -> MT3:
    """Load a pretrained MT3 model.

    Args:
        model_type: One of:
            - 'mt3': multi-task multitrack model.
            - 'ismir2021': piano-only model with velocities.
            - 'ismir2022_small'/'ismir2022_base': multitrack models trained
              with mixture augmentation.
        load_path: Path to a safetensors weights file. If None, downloads the
            pretrained checkpoint.
        config: Model configuration. If None, uses the configuration matching
            ``model_type``.

    Returns:
        MT3 model in eval mode, ready for inference (see
        ``magenta_rt.nnx.mt3.transcribe``).
    """
    from safetensors.numpy import load_file

    if config is None:
        config = MT3Config.from_pretrained(model_type)
    if load_path is None:
        load_path = download_model(model_type)

    # Build abstractly so the constructor never allocates the random-init
    # weights the checkpoint immediately overwrites.
    params = load_file(load_path)
    model = load_into_abstract(
        lambda: MT3(config, rngs=nnx.Rngs(0)),
        lambda m: load_weights(m, params),
    )
    model.eval()
    return model


if __name__ == "__main__":
    import jax

    model = load_model()
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Loaded MT3 with {num_params:,} parameters")
