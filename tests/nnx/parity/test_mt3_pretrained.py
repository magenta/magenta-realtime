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

"""Equivalence tests for pretrained MT3 against the original Linen network.

These tests require the pretrained checkpoint (``python -m
magenta_rt.mt3.download``) and, for the parity test, a local clone of
https://github.com/magenta/mt3. Gated like the other real-weight tests.
"""

import pytest as _pytest

pytestmark = _pytest.mark.checkpoint

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from jax import numpy as jnp

from magenta_rt import paths

MT3_REPO = Path.home() / "GitHub" / "mt3"


@pytest.fixture(scope="module")
def pretrained_model():
    if not (paths.mt3_dir() / "mt3_mt3.safetensors").exists():
        pytest.skip("pretrained mt3 checkpoint not downloaded")
    from magenta_rt.nnx.mt3 import load_model

    return load_model("mt3")


@pytest.fixture(scope="module")
def linen_network():
    """Import the original mt3 network without the package __init__ (which
    requires note_seq and other heavy dependencies)."""
    if not MT3_REPO.exists():
        pytest.skip("original mt3 repo not available")
    pkg = types.ModuleType("mt3")
    pkg.__path__ = [str(MT3_REPO / "mt3")]
    sys.modules["mt3"] = pkg
    for name in ["layers", "network"]:
        spec = importlib.util.spec_from_file_location(
            f"mt3.{name}", MT3_REPO / "mt3" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"mt3.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mt3.network"]


def test_logits_match_original_linen_network(pretrained_model, linen_network):
    from safetensors.numpy import load_file

    cfg = pretrained_model.config
    linen_cfg = linen_network.T5Config(
        vocab_size=cfg.vocab_size,
        dtype=jnp.float32,
        emb_dim=cfg.emb_dim,
        num_heads=cfg.num_heads,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        head_dim=cfg.head_dim,
        mlp_dim=cfg.mlp_dim,
        mlp_activations=tuple(cfg.mlp_activations),
        dropout_rate=cfg.dropout_rate,
        logits_via_embedding=cfg.logits_via_embedding,
    )
    linen_model = linen_network.Transformer(config=linen_cfg)

    # Reconstruct the nested Linen param tree from the flat safetensors keys.
    flat = load_file(paths.mt3_dir() / "mt3_mt3.safetensors")
    params = {}
    for key, value in flat.items():
        *parents, leaf = key.split("/")
        d = params
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = jnp.asarray(value)

    rng = np.random.RandomState(0)
    batch, enc_len, dec_len = 2, 64, 32
    enc_in = jnp.asarray(rng.randn(batch, enc_len, 512), jnp.float32)
    dec_in = jnp.asarray(rng.randint(0, cfg.vocab_size, (batch, dec_len)), jnp.int32)
    dec_tgt = jnp.asarray(rng.randint(1, cfg.vocab_size, (batch, dec_len)), jnp.int32)

    ref = linen_model.apply({"params": params}, enc_in, dec_in, dec_tgt, enable_dropout=False)
    got = pretrained_model(enc_in, dec_in, dec_tgt)

    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_transcribe_produces_notes(pretrained_model):
    """End-to-end smoke test: transcribe a synthesized arpeggio."""
    from magenta_rt.nnx.mt3 import transcribe

    sr = 16000

    def tone(pitch, start, dur, total):
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        t = np.arange(int(dur * sr)) / sr
        x = sum((0.6**k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4))
        x = x * np.exp(-2.5 * t) * 0.3
        out = np.zeros(int(total * sr), np.float32)
        i = int(start * sr)
        out[i : i + len(x)] += x.astype(np.float32)
        return out

    onsets = [(60, 0.25), (64, 1.0), (67, 1.75), (72, 2.5)]
    audio = sum(tone(p, s, 1.0, 4.0) for p, s in onsets)

    ns = transcribe(pretrained_model, audio)

    assert len(ns.notes) > 0
    # The model should detect onsets near the synthesized note times (pitch
    # and program may vary on synthetic timbre).
    note_onsets = sorted(n.start_time for n in ns.notes)
    for _, expected_onset in onsets:
        assert any(abs(t - expected_onset) < 0.1 for t in note_onsets), (
            f"no note onset near {expected_onset}s in {note_onsets}"
        )


def test_abstract_load_leaves_nothing_unmaterialized(pretrained_model):
    """``load_model`` builds MT3 with ``nnx.eval_shape``, so every leaf starts as
    a placeholder: parameters come from the checkpoint, RNG state is
    materialized, and computed constants (the sinusoidal position tables) are
    recomputed. Any leaf still abstract here would survive into inference and
    fail deep inside a ``jit`` as "not a valid JAX type" — which is what adding
    a new derived constant to the model, with no matching restore, would do.
    """
    import jax
    from flax import nnx

    state = nnx.state(pretrained_model)
    abstract = [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
        if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    assert not abstract, f"unmaterialized after load: {abstract}"
