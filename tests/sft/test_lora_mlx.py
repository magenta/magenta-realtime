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

"""Tests for the idiomatic-MLX LoRA twin (:mod:`magenta_rt.sft.lora_mlx`).

Fast unit tests (random-weight tiny model, no resources) cover the adapter
contract: zero-init identity, trainable filtering, the fuse/merge round-trip
(the key correctness property — a merged model must be numerically identical to
the adapted one), injection counts, and an end-to-end LoRA train step. A gated
test exercises the real mrt2_small depthformer checkpoint load + a LoRA step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# MLX ships wheels for Apple Silicon only; on Windows / WSL2 / Linux it is
# absent. Skip the whole module cleanly there instead of erroring at collection.
mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten  # noqa: E402

from magenta_rt import paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "sft"))

import train_mlx as sft_mlx  # type: ignore  # noqa: E402

from magenta_rt.sft import lora_mlx as L  # noqa: E402
from magenta_rt.sft.configs import SFTConfig  # noqa: E402


def _tiny():
    return sft_mlx.TinyPOCSpecMLX()


def _dummy_io(spec):
    Q = spec.target_tokens_config.rvq_truncation_level
    src = mx.zeros((1, 4, spec.input_num_channels), mx.int32)
    tgt = mx.zeros((1, 4, Q), mx.int32)
    return src, tgt


def _fwd(model, src, tgt):
    out = model.decoder(tgt, encoded_source=model.encoder(src))
    mx.eval(out)
    return out


def _randomize_adapters(model, scale=0.1):
    """Make every ``lora_b`` non-zero so the adapter delta is non-trivial
    (``lora_b`` is zero-init, which would make round-trip tests vacuous)."""
    for _, mod in model.named_modules():
        if isinstance(mod, (L.LoRALinear, L.LoRAEinsumDense)):
            mod.lora_b = mx.random.normal(mod.lora_b.shape) * scale
    mx.eval(model.parameters())


# ---- Adapter math ----------------------------------------------------------

def test_scale_convention():
    # alpha/rank when alpha != 0, else 1.0 (NNX convention).
    assert L._lora_scale(8, 16.0) == pytest.approx(2.0)
    assert L._lora_scale(4, 0.0) == 1.0


def test_zero_init_identity_ffn():
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)
    n = L.inject_lora(model, rank=4, alpha=8.0, targets=L.default_targets, seed=0)
    assert n > 0
    y1 = _fwd(model, src, tgt)
    # B is zero-init, so the wrapped model is identical at step 0.
    assert mx.allclose(y0, y1, atol=1e-5)


@pytest.mark.parametrize("dora", [False, True])
def test_zero_init_identity_all_linears(dora):
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets,
                  dora=dora, seed=0)
    y1 = _fwd(model, src, tgt)
    # B=0 ⇒ LoRA delta is 0; DoRA's magnitude·W/‖W‖ = W. Identical to base at
    # step 0 (DoRA carries a tiny fp32 normalize/rescale rounding).
    assert mx.allclose(y0, y1, atol=1e-4 if dora else 1e-5)


def test_set_lora_strength_zero_is_base():
    """strength=0 collapses every adapter (LoRA and DoRA) back to the base."""
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    src, tgt = _dummy_io(spec)
    y0 = _fwd(model, src, tgt)
    for dora in (False, True):
        m = sft_mlx.build_model(spec)
        # copy base weights so y0 is the right reference
        m.update(model.parameters())
        L.inject_lora(m, rank=4, alpha=8.0, targets=L.all_linear_targets,
                      dora=dora, seed=2)
        _randomize_adapters(m)  # non-trivial adapter
        assert float(mx.max(mx.abs(_fwd(m, src, tgt) - y0))) > 1e-4  # adapter active
        n = L.set_lora_strength(m, 0.0)
        assert n > 0
        assert mx.allclose(_fwd(m, src, tgt), y0, atol=1e-4)


# ---- Injection / targeting -------------------------------------------------

def test_inject_count_default_vs_all():
    spec = _tiny()
    m_ffn = sft_mlx.build_model(spec)
    n_ffn = L.inject_lora(m_ffn, rank=2, targets=L.default_targets)
    m_all = sft_mlx.build_model(spec)
    n_all = L.inject_lora(m_all, rank=2, targets=L.all_linear_targets)
    # all_linears is a strict superset (FFN + attention output projections).
    assert n_all > n_ffn > 0
    # default wraps only nn.Linear; all_linears adds EinsumDense out-projs.
    ffn_kinds = {
        type(m).__name__
        for _, m in m_ffn.named_modules()
        if isinstance(m, (L.LoRALinear, L.LoRAEinsumDense))
    }
    all_kinds = {
        type(m).__name__
        for _, m in m_all.named_modules()
        if isinstance(m, (L.LoRALinear, L.LoRAEinsumDense))
    }
    assert ffn_kinds == {"LoRALinear"}
    assert all_kinds == {"LoRALinear", "LoRAEinsumDense"}


def test_inject_returns_zero_when_nothing_matches():
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    n = L.inject_lora(model, rank=2, targets=lambda p, m: False)
    assert n == 0


# ---- Trainable filtering ---------------------------------------------------

def test_mark_lora_trainable_filters_to_adapters():
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets, seed=0)
    L.mark_lora_trainable(model)
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert names, "expected some trainable params"
    assert all(n.endswith(("lora_a", "lora_b")) for n in names)
    # Trainable count is strictly less than the full param set (base frozen).
    n_train = sft_mlx.count_params(model.trainable_parameters())
    n_total = sft_mlx.count_params(model.parameters())
    assert 0 < n_train < n_total


# ---- Fuse / merge round-trip (the key correctness property) ----------------

@pytest.mark.parametrize("targets", [L.default_targets, L.all_linear_targets])
@pytest.mark.parametrize("dora", [False, True])
def test_fuse_round_trip(targets, dora):
    spec = _tiny()
    model = sft_mlx.build_model(spec)
    src, tgt = _dummy_io(spec)
    L.inject_lora(model, rank=4, alpha=8.0, targets=targets, dora=dora, seed=1)
    _randomize_adapters(model)

    y_wrapped = _fwd(model, src, tgt)
    n_merged = L.merge_lora_into_base(model)
    y_fused = _fwd(model, src, tgt)

    assert n_merged > 0
    # Merged model must equal the adapted one. Plain LoRA folds the same matmul
    # → bit-exact; DoRA's effective-weight forward vs einsum-with-folded-kernel
    # differ only by fp32 contraction order (tiny).
    # 1e-5 (not 0.0) to match the NNX twin: Metal computes fp32 matmuls in
    # fp32 so exact equality happens to hold here, but the property being
    # asserted is algebraic, and a backend with reduced-precision
    # accumulation would fail an exact bound for no real defect.
    tol = 1e-4 if dora else 1e-5
    assert float(mx.max(mx.abs(y_wrapped - y_fused))) <= tol
    # No adapter modules remain — plain inference module again.
    remaining = [
        p for p, m in model.named_modules()
        if isinstance(m, (L.LoRALinear, L.LoRAEinsumDense))
    ]
    assert remaining == []


# ---- End-to-end LoRA training ----------------------------------------------

def test_lora_train_decreases_loss(tmp_path):
    pytest.importorskip("audiotree")
    from .test_utils import write_fake_tree_dataset

    spec = _tiny()
    write_fake_tree_dataset(
        str(tmp_path), num_files=4, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    config = SFTConfig(
        data_dir=str(tmp_path), batch_size=2, crop_length_seconds=2,
        total_steps=30, learning_rate=1e-2, warmup_steps=0,
        lora_rank=4, lora_alpha=8.0, lora_all_linears=True,
        log_every_steps=50, save_every_steps=0, resume=False,
        output_dir=str(tmp_path / "out"),
    )
    losses = sft_mlx.train(config, spec)
    assert all(np.isfinite(losses))
    assert np.mean(losses[:5]) > np.mean(losses[-5:]), (
        f"LoRA loss did not decrease: first5={np.mean(losses[:5]):.3f} "
        f"last5={np.mean(losses[-5:]):.3f}"
    )


# ---- Gated: real mrt2_small checkpoint load + LoRA step --------------------

CHECKPOINT = paths.resolve_checkpoint("mrt2_small.safetensors")


@pytest.mark.checkpoint
@pytest.mark.slow
def test_real_checkpoint_load_then_lora_step():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT}")
    import mlx.nn as nn
    from magenta_rt.mlx_pure.configs import get_model_class

    spec = get_model_class("mrt2_small")()
    model = sft_mlx.build_model(
        spec, checkpoint_path=str(CHECKPOINT), model_name="mrt2_small",
    )
    # A pretrained-load + LoRA-injected model still runs and trains.
    n = L.inject_lora(model, rank=4, alpha=8.0, targets=L.all_linear_targets)
    assert n > 0
    L.mark_lora_trainable(model)
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert names and all(k.endswith(("lora_a", "lora_b")) for k in names)

    Q = spec.target_tokens_config.rvq_truncation_level
    src = mx.zeros((1, 8, spec.input_num_channels), mx.int32)
    tgt = mx.zeros((1, 8, Q), mx.int32)
    loss_fn = sft_mlx.make_loss_fn(model)
    loss, grads = nn.value_and_grad(model, loss_fn)(src, tgt)
    mx.eval(loss, grads)
    assert np.isfinite(float(loss))
    # Grads exist only for adapter params (base is frozen).
    grad_names = [k for k, _ in tree_flatten(grads)]
    assert grad_names and all(k.endswith(("lora_a", "lora_b")) for k in grad_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
