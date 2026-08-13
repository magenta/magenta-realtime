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

"""Optional Weights & Biases writer that plugs into `magenta_rt.metric_writers`.

Designed to run *alongside* the default TensorBoard writer via
`metric_writers.MultiWriter` — every scalar / audio / histogram landed in
TB is mirrored to W&B. Borrows the shape from the `WandbWriter` in
DBraun's JAX monorepo (sans the argbind framework dependency).

Usage:

    from magenta_rt.sft.wandb_writer import maybe_make_wandb_writer
    writer = metric_writers.MultiWriter([
        metric_writers.create_default_writer(logdir=tb_dir),
        *([maybe_make_wandb_writer(config)] if config.use_wandb else []),
    ])
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from magenta_rt import metric_writers


class WandbWriter(metric_writers.MetricWriter):
    """CLU-compatible `MetricWriter` that forwards to `wandb.log(...)`.

    All writes are step-keyed so they line up with the TB writer when both
    are wrapped in `metric_writers.MultiWriter`.
    """

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        logdir: Optional[str] = None,
        job_type: str = "train",
    ):
        if not _WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        # Derive project / name from logdir if not provided. Matches the
        # convention "<output_dir>/<run_name>" we use in the trainers.
        if logdir is not None:
            p = Path(logdir)
            if project is None:
                project = p.parent.name or "magenta-rt-sft"
            if name is None:
                name = p.name

        self._run = wandb.init(
            project=project,
            name=name,
            entity=entity or None,
            config=dict(config) if config else None,
            job_type=job_type,
            reinit="finish_previous",
        )

    # ---- scalar / array primitives --------------------------------------

    @staticmethod
    def _to_scalar(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(np.asarray(v).item())
        except (ValueError, AttributeError, TypeError):
            return None

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        log = {}
        for k, v in scalars.items():
            s = self._to_scalar(v)
            if s is not None:
                log[k] = s
        if log:
            wandb.log(log, step=step)

    def write_summaries(
        self,
        step: int,
        values: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.write_scalars(step, values)

    def write_audios(
        self,
        step: int,
        audios: Mapping[str, np.ndarray],
        *,
        sample_rate: int,
    ) -> None:
        log: dict[str, Any] = {}
        for k, a in audios.items():
            arr = np.asarray(a)
            if arr.ndim == 3:
                log[k] = [
                    wandb.Audio(arr[i], sample_rate=sample_rate, caption=f"{k}_{i}")
                    for i in range(min(arr.shape[0], 4))
                ]
            elif arr.ndim in (1, 2):
                log[k] = wandb.Audio(arr, sample_rate=sample_rate, caption=k)
        if log:
            wandb.log(log, step=step)

    def write_images(self, step: int, images: Mapping[str, np.ndarray]) -> None:
        log: dict[str, Any] = {}
        for k, img in images.items():
            arr = np.asarray(img)
            if arr.ndim == 4:
                log[k] = [wandb.Image(arr[i]) for i in range(min(arr.shape[0], 8))]
            elif arr.ndim == 3:
                log[k] = wandb.Image(arr)
        if log:
            wandb.log(log, step=step)

    def write_histograms(
        self,
        step: int,
        arrays: Mapping[str, np.ndarray],
        num_buckets: Optional[Mapping[str, int]] = None,
    ) -> None:
        log = {k: wandb.Histogram(np.asarray(a)) for k, a in arrays.items()}
        if log:
            wandb.log(log, step=step)

    def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
        log = {k: wandb.Html(str(v)) for k, v in texts.items()}
        if log:
            wandb.log(log, step=step)

    def write_videos(self, step: int, videos: Mapping[str, np.ndarray]) -> None:
        pass  # not needed; SFT doesn't generate video

    def write_hparams(self, hparams: Mapping[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(dict(hparams), allow_val_change=True)

    def flush(self) -> None:
        # wandb.log is synchronous from the client's perspective; nothing buffered.
        pass

    def close(self) -> None:
        if self._run is not None:
            wandb.finish()
            self._run = None


def maybe_make_wandb_writer(config) -> Optional["WandbWriter"]:
    """Build a WandbWriter from an SFTConfig if `use_wandb` is set; else None.

    Returns None (with a printed warning) rather than raising when wandb
    isn't installed — keeps the trainers usable in minimal envs.
    """
    if not getattr(config, "use_wandb", False):
        return None
    if not _WANDB_AVAILABLE:
        print("[sft] use_wandb=True but `wandb` is not installed — skipping.")
        return None
    return WandbWriter(
        config=dataclasses_to_dict(config),
        project=config.wandb_project or None,
        name=config.wandb_name or None,
        entity=config.wandb_entity or None,
        logdir=config.output_dir,
    )


def dataclasses_to_dict(config) -> dict:
    """Convert an SFTConfig (or any dataclass) to a wandb-friendly dict."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(config):
        v = getattr(config, f.name)
        if isinstance(v, (str, int, float, bool, type(None))):
            out[f.name] = v
        else:
            out[f.name] = str(v)
    return out
