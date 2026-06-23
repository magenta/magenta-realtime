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

"""Round-trip coverage for ``magenta_rt.nnx.load_weights.load_system_state_dict``:
reconstruct a Linen-style dict from a freshly-built model, offset every leaf,
load it back, and assert every float parameter changed.

Covers both encoder-embedding layouts: the plain ``tiny`` preset and a small
branched (pretrained-MusicCoCa) mrt2-shaped spec.
"""

import dataclasses

import numpy as np
import pytest
from flax import nnx
import jax.numpy as jnp

from magenta_rt.nnx.model import MagentaRT2Sampler
from magenta_rt.nnx import depthformer as nnx_depthformer
from magenta_rt.nnx.configs import (
    MagentaRT2ModelBase,
    ModelSpec,
    SPECTROSTREAM,
    get_model_class,
)
from magenta_rt.nnx.load_weights import load_system_state_dict
from tests.nnx.parity.test_weight_bridge import _transformer_to_linen


_TINY = ModelSpec(
    num_layers=2, model_dims=16, hidden_dims=32, num_heads=2, dim_per_head=8,
    ffn_use_gated_activation=False,
)


class _SmallBranchedSpec(MagentaRT2ModelBase):
    """mrt2-shaped (branched MusicCoCa embedder + 5-channel input) but tiny
    transformers, so the round-trip stays fast."""

    encoder_size: ModelSpec = _TINY
    decoder_temporal_size: ModelSpec = ModelSpec(
        num_layers=2, model_dims=32, hidden_dims=64, num_heads=2, dim_per_head=16,
        ffn_use_gated_activation=False,
    )
    decoder_depth_size: ModelSpec = _TINY
    spectrostream = dataclasses.replace(
        SPECTROSTREAM, rvq_truncation_level=4, codebook_size=32,
    )


def _depthformer_to_linen(mrt: MagentaRT2Sampler) -> dict:
    """Reconstruct the Linen-style dict for the depthformer subtree."""
    df = mrt.depthformer
    emb = df.encoder.embedding
    body: dict = {}
    if hasattr(emb, "mulan_embedder"):  # branched
        body["layers_1"] = {
            "branched_mulan_embedder": {
                "mulan_embedder": {
                    "mulan_dequantizer": {
                        "embedding": emb.mulan_embedder.mulan_dequantizer.embedding[...]
                    },
                    "depth_input_adapter": {
                        "kernel": emb.mulan_embedder.depth_input_adapter.kernel[...]
                    },
                }
            },
            "branched_regular_embedder": {
                "regular_embedder": {"embedding": emb.regular_embedder.embedding[...]}
            },
        }
    else:
        body["encoder_embedding"] = {"embedding": emb.embedding[...]}
    body["encoder_ln"] = {"scale": df.encoder.encoder_ln.scale[...]}
    if df.encoder.encoder_ln.bias is not None:
        body["encoder_ln"]["bias"] = df.encoder.encoder_ln.bias[...]

    decoder_embedding_dict = {
        "embedding": {"embedding": df.decoder.embedder.embedding.embedding[...]}
    }
    temporal_dict = _transformer_to_linen(df.decoder.temporal, has_sinks=True)
    depth_dict = _transformer_to_linen(df.decoder.depth, has_sinks=False)
    depth_body_dict = {
        "transformer": depth_dict,
        "final_ln": {"scale": df.decoder.final_ln.scale[...]},
        "to_logits": {"kernel": df.decoder.to_logits.kernel[...]},
    }
    if df.decoder.final_ln.bias is not None:
        depth_body_dict["final_ln"]["bias"] = df.decoder.final_ln.bias[...]
    if df.decoder.to_logits.bias is not None:
        depth_body_dict["to_logits"]["bias"] = df.decoder.to_logits.bias[...]
    if df.decoder.depth_input_adapter is not None:
        depth_body_dict["depth_input_adapter"] = {
            "kernel": df.decoder.depth_input_adapter.kernel[...]
        }

    return {
        "depthformer": {
            "encoder": {"body": body},
            "decoder": {
                "decoder_embedding": decoder_embedding_dict,
                "temporal_body": {"transformer": temporal_dict},
                "depth_body": depth_body_dict,
            },
        }
    }


def _add_offset(d, offset=1.0):
    return {k: (_add_offset(v, offset) if isinstance(v, dict) else v + offset)
            for k, v in d.items()}


def _flatten(d, prefix=(), as_numpy=False):
    flat = {}
    for k, v in d.items():
        if hasattr(v, "items"):
            flat.update(_flatten(v, prefix + (k,), as_numpy))
        else:
            try:
                flat[prefix + (k,)] = np.array(v[...]) if as_numpy else v
            except TypeError:
                pass  # skip PRNGKeys etc.
    return flat


def _build(spec):
    enc_dec = nnx_depthformer.EncoderDecoder.from_config(spec, rngs=nnx.Rngs(0))
    target_cfg = spec.target_tokens_config
    return MagentaRT2Sampler(
        depthformer_model=enc_dec, spectrostream=None,
        num_reserved_tokens=target_cfg.num_extra_tokens,
        codebook_size=target_cfg.codebook_size, int16_outputs=False,
    )


@pytest.mark.parametrize("spec_factory", [
    lambda: get_model_class("tiny")(),   # plain encoder embedding
    _SmallBranchedSpec,                   # branched MusicCoCa embedder
], ids=["tiny_plain", "small_branched"])
def test_weight_loading_updates_all_parameters(spec_factory):
    mrt = _build(spec_factory())

    _, initial_state = nnx.split(mrt)
    initial_flat = _flatten(initial_state, as_numpy=True)

    modified = _add_offset(_depthformer_to_linen(mrt), offset=1.0)

    _, test_state = nnx.split(mrt)
    load_system_state_dict(test_state, modified)
    nnx.update(mrt, test_state)

    _, loaded_state = nnx.split(mrt)
    loaded_flat = _flatten(loaded_state)

    not_updated, updated = [], 0
    for path, val_initial in initial_flat.items():
        if path not in loaded_flat:
            continue
        val_loaded = np.asarray(loaded_flat[path][...])
        if val_initial.dtype in (np.float32, np.float16, jnp.bfloat16) and val_initial.size > 1:
            if "step_counter" in path or "previous_frame" in path:
                continue
            if np.array_equal(val_initial, val_loaded):
                not_updated.append(path)
            else:
                updated += 1

    assert not not_updated, f"{len(not_updated)} params NOT updated: {not_updated}"
    assert updated > 0, "no params updated"
