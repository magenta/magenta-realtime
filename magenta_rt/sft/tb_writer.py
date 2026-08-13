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

"""TensorFlow-free TensorBoard writer (`tensorboardX`) for the MLX trainer.

The NNX trainer logs through `clu.metric_writers`, which imports TensorFlow.
The MLX trainer can't take that path: TensorFlow aborts when co-resident with
Metal/MLX in the same process (see the note at the top of
`notebooks/sft/train_mlx.py`). `tensorboardX` writes the same `tfevents`
protobuf records — scalars *and* audio — with no TensorFlow dependency, so the
MLX trainer streams the *same* live curves and periodically-generated clips to a
browser as the NNX trainer:

    tensorboard --logdir <output_dir> --samples_per_plugin audio=200

This mirrors the slice of the `clu.metric_writers.MetricWriter` interface the
trainers call (`write_scalars` / `write_audios` / `write_hparams` / `flush` /
`close`), so it drops in alongside :class:`magenta_rt.sft.wandb_writer.WandbWriter`.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import numpy as np

try:
    from tensorboardX import SummaryWriter
    _TBX_AVAILABLE = True
except ImportError:
    _TBX_AVAILABLE = False


def _to_scalar(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(np.asarray(v).item())
    except (ValueError, AttributeError, TypeError):
        return None


def _to_mono_1d(arr: Any) -> np.ndarray:
    """Reduce an audio array to a mono ``[T]`` float vector for ``add_audio``.

    tensorboardX's ``add_audio`` takes a single-channel ``[T]`` clip. Accepts
    ``[T]``, ``[T, C]`` / ``[C, T]`` (downmixed), or a leading batch axis
    ``[B, ...]`` (first item). Time is taken as the longer of the last two axes.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 2:
        # Mean over the shorter (channel) axis; keep the longer (time) axis.
        a = a.mean(axis=0) if a.shape[0] < a.shape[1] else a.mean(axis=1)
    return np.ascontiguousarray(a.reshape(-1))


class TBWriter:
    """Minimal `clu`-compatible writer backed by `tensorboardX.SummaryWriter`."""

    def __init__(self, logdir: str):
        if not _TBX_AVAILABLE:
            raise ImportError(
                "tensorboardX is not installed. Install with: pip install tensorboardX"
            )
        os.makedirs(logdir, exist_ok=True)
        self._logdir = logdir
        self._w = SummaryWriter(logdir=str(logdir))

    @property
    def logdir(self) -> str:
        return self._logdir

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        for k, v in scalars.items():
            s = _to_scalar(v)
            if s is not None:
                self._w.add_scalar(k, s, global_step=step)

    def write_audios(
        self, step: int, audios: Mapping[str, np.ndarray], *, sample_rate: int
    ) -> None:
        for k, a in audios.items():
            self._w.add_audio(k, _to_mono_1d(a), global_step=step,
                              sample_rate=sample_rate)

    def write_hparams(self, hparams: Mapping[str, Any]) -> None:
        # add_hparams spawns a separate run dir; a text blob is friendlier and
        # keeps everything under the one logdir. Rendered in the TEXT tab.
        lines = "\n".join(f"{k}: {v}" for k, v in sorted(hparams.items()))
        self._w.add_text("config", "```\n" + lines + "\n```", global_step=0)

    def flush(self) -> None:
        self._w.flush()

    def close(self) -> None:
        self._w.close()


def resolve_tb_dir(config) -> str:
    """``config.tensorboard_dir`` or, if empty, ``<output_dir>/tb``."""
    return config.tensorboard_dir or os.path.join(config.output_dir, "tb")


def maybe_make_tb_writer(config) -> Optional["TBWriter"]:
    """Build a :class:`TBWriter` at ``resolve_tb_dir(config)``.

    Returns ``None`` (with a printed note) rather than raising when
    ``tensorboardX`` isn't installed — keeps the MLX trainer usable in minimal
    envs (it still writes WAVs to ``<output_dir>/samples`` and the loss curve).
    """
    if not _TBX_AVAILABLE:
        print("[sft] tensorboardX not installed — no TensorBoard scalars/audio "
              "(pip install tensorboardX). WAV samples + loss_curve.png still written.")
        return None
    return TBWriter(resolve_tb_dir(config))
