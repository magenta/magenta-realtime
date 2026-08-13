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

"""End-to-end POC test for the MLX SFT trainer.

Asserts:
  * grain pipeline + mx.array cast on the same data the NNX test uses
  * encoder freeze accounting (trainable_parameters excludes encoder)
  * a few train steps decrease the loss
  * model.save_weights / load_weights round-trip restores identical params
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# MLX ships wheels for Apple Silicon only; on Windows / WSL2 / Linux it is
# absent. Skip the whole module cleanly there instead of erroring at collection.
mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import mlx.utils  # noqa: E402

# Make `notebooks/sft/` importable (train_mlx + its sibling `utils`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "sft"))

import train_mlx as sft_mlx  # type: ignore  # noqa: E402

from magenta_rt.sft import create_audiotree_dataset, to_source_target
from magenta_rt.sft.configs import SFTConfig
from .test_utils import write_fake_tree_dataset


def _make_dataset(tmpdir, spec):
    write_fake_tree_dataset(
        tmpdir,
        num_files=4, frames_per_file=75,
        target_rvq=spec.target_tokens_config.rvq_truncation_level,
        target_codebook_size=spec.target_tokens_config.codebook_size,
    )
    return create_audiotree_dataset(
        tmpdir,
        batch_size=2,
        crop_length_seconds=2,
        input_configs=spec.input_configs,
        target_config=spec.target_tokens_config,
        seed=0,
    )


def test_pipeline_to_mx_array(tmp_path):
    spec = sft_mlx.TinyPOCSpecMLX()
    ds = _make_dataset(str(tmp_path), spec)
    source, target = to_source_target(
        next(iter(ds)), spec.target_tokens_config, asarray=mx.array,
    )
    assert source.shape == (2, 50, spec.input_num_channels)
    assert target.shape == (2, 50, spec.target_tokens_config.rvq_truncation_level)
    assert source.dtype == mx.int16
    assert target.dtype == mx.int16


def test_freeze_encoder_accounting(tmp_path):
    spec = sft_mlx.TinyPOCSpecMLX()
    model = sft_mlx.build_model(spec)
    n_total = sft_mlx.count_params(model.parameters())
    model.encoder.freeze()
    n_trainable = sft_mlx.count_params(model.trainable_parameters())
    assert n_trainable > 0
    assert n_trainable < n_total
    # Trainable subset is exactly total minus the encoder's contribution.
    n_enc = sft_mlx.count_params(model.encoder.parameters())
    assert n_trainable == n_total - n_enc


def test_trainstep_decreases_loss(tmp_path):
    spec = sft_mlx.TinyPOCSpecMLX()
    ds = _make_dataset(str(tmp_path), spec)
    config = SFTConfig(
        data_dir=str(tmp_path), batch_size=2, crop_length_seconds=2,
        total_steps=20, learning_rate=1e-3, freeze_encoder=True,
        warmup_steps=0,
    )
    model = sft_mlx.build_model(spec)
    model.encoder.freeze()
    optimizer = sft_mlx.build_optimizer(config)
    loss_fn = sft_mlx.make_loss_fn(model)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    it = iter(ds)
    for _ in range(config.total_steps):
        source, target = to_source_target(
            next(it), spec.target_tokens_config, asarray=mx.array,
        )
        loss, grads = loss_and_grad_fn(source, target)
        grads, _ = sft_mlx.clip_grad_norm(grads, config.max_grad_norm)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss))

    assert np.mean(losses[:5]) > np.mean(losses[-5:]), (
        f"loss did not decrease: first5={np.mean(losses[:5]):.3f} "
        f"last5={np.mean(losses[-5:]):.3f}"
    )


def test_gradient_accumulation_compiled_path(tmp_path):
    """train() with gradient_accumulation_steps>1 exercises the compiled
    accum_grad_fn / apply_grads_fn path: finite, decreasing loss."""
    spec = sft_mlx.TinyPOCSpecMLX()
    _make_dataset(str(tmp_path / "data"), spec)
    config = SFTConfig(
        data_dir=str(tmp_path / "data"), batch_size=2, crop_length_seconds=2,
        total_steps=12, learning_rate=1e-3, warmup_steps=0,
        lora_rank=4, lora_alpha=8.0, gradient_accumulation_steps=4,
        output_dir=str(tmp_path / "out"), log_every_steps=100,
    )
    losses = sft_mlx.train(config, spec)
    assert len(losses) == 12
    assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
    assert np.mean(losses[:4]) > np.mean(losses[-4:]), (
        f"loss did not decrease: {losses}"
    )


def test_save_and_load_round_trip(tmp_path):
    spec = sft_mlx.TinyPOCSpecMLX()
    model = sft_mlx.build_model(spec)

    # `to_logits.linear.weight` is initialized by `nn.Linear`'s truncated
    # normal, so it's non-zero — a clean target for the save/mutate/restore
    # invariant. (EinsumDense kernels are zero-initialized until weights
    # are loaded, which would make the "mutate then assert different"
    # check trivially false.)
    w = model.decoder.to_logits.linear.weight
    pre = mx.array(w)
    assert float(mx.abs(pre).max()) > 0, "expected non-zero init weight"

    path = sft_mlx.save_checkpoint(model, 1, str(tmp_path))
    assert os.path.exists(path)

    # Mutate, then reload — value must match the snapshot.
    model.decoder.to_logits.linear.weight = mx.zeros_like(w)
    assert not mx.allclose(model.decoder.to_logits.linear.weight, pre)
    model.load_weights(path)
    assert mx.allclose(model.decoder.to_logits.linear.weight, pre)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
