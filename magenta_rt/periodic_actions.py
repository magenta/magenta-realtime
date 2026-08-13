# Copyright 2026 The CLU Authors.
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

# Adapted from CommonLoopUtils (https://github.com/google/CommonLoopUtils),
# clu/periodic_actions.py, with modifications by David Braun for magenta-rt:
# kept ReportProgress and PeriodicCallback (dropped the profiler/platform
# actions), report via absl logging, and use concurrent.futures for async.

"""Periodic actions executed from inside the training loop.

A ``PeriodicAction`` is created once and called after every training step; it
triggers on a fixed step/time cadence. This is a dependency-light port of
``clu.periodic_actions`` covering what magenta-rt uses (``ReportProgress`` and
``PeriodicCallback``), with no dependency on ``clu.platform``/``clu.profiler``/
TensorFlow. ``ReportProgress`` reports via absl logging (and, optionally, a
``MetricWriter``) instead of an XManager work unit.
"""

from __future__ import annotations

import abc
import collections
import contextlib
import time
from collections.abc import Callable, Iterable
from concurrent import futures

import jax
import jax.numpy as jnp
from absl import logging

from .metric_writers import MetricWriter


@jax.jit
def _squareit(x):
    """Minimal computation used to drain JAX's async dispatch queue."""
    return x**2


def _format_secs(secs: float) -> str:
    """Formats seconds like ``123456.7`` as strings like ``"1d10h17m"``."""
    s = ""
    days = int(secs / (3600 * 24))
    secs -= days * 3600 * 24
    if days:
        s += f"{days}d"
    hours = int(secs / 3600)
    secs -= hours * 3600
    if hours:
        s += f"{hours}h"
    mins = int(secs / 60)
    s += f"{mins}m"
    return s


class PeriodicAction(abc.ABC):
    """Base class for actions triggered periodically during training.

    Subclasses implement ``_apply``; the base class handles the trigger cadence
    and may be customized by overriding ``_should_trigger``.
    """

    def __init__(
        self,
        *,
        every_steps: int | None = None,
        every_secs: float | None = None,
        on_steps: Iterable[int] | None = None,
    ):
        """Creates an action that triggers periodically.

        Args:
            every_steps: Trigger when the current step is divisible by this.
            every_secs: Trigger when this many seconds elapsed since last trigger.
            on_steps: Trigger when the current step is in this set.
        """
        self._every_steps = every_steps
        self._every_secs = every_secs
        self._on_steps = set(on_steps or [])
        self._previous_step: int | None = None
        self._previous_time: float | None = None
        self._last_step: int | None = None

    def _init_and_check(self, step: int, t: float) -> None:
        """Initializes bookkeeping and checks the action is called every step."""
        if self._previous_step is None:
            self._previous_step = step
            self._previous_time = t
            self._last_step = step
        elif self._every_steps is not None and step - self._last_step != 1:
            raise ValueError(
                f"PeriodicAction must be called after every step once "
                f"(every_steps={self._every_steps}, "
                f"previous_step={self._previous_step}, step={step})."
            )
        else:
            self._last_step = step

    def _should_trigger(self, step: int, t: float) -> bool:
        if self._every_steps is not None and step % self._every_steps == 0:
            return True
        if self._every_secs is not None and t - self._previous_time > self._every_secs:
            return True
        if step in self._on_steps:
            return True
        return False

    def _after_apply(self, step: int, t: float) -> None:
        self._previous_step = step
        self._previous_time = t

    def __call__(self, step: int, t: float | None = None) -> bool:
        """Calls the action; returns whether it triggered (never on first call)."""
        if t is None:
            t = time.monotonic()
        self._init_and_check(step, t)
        if self._should_trigger(step, t):
            self._apply(step, t)
            self._after_apply(step, t)
            return True
        return False

    @abc.abstractmethod
    def _apply(self, step: int, t: float): ...


class ReportProgress(PeriodicAction):
    """Logs training progress and, optionally, ``steps_per_sec``/``uptime``.

    Also offers ``timed(name)`` to break the loop's wall-clock down by section.
    """

    def __init__(
        self,
        *,
        num_train_steps: int | None = None,
        writer: MetricWriter | None = None,
        every_steps: int | None = None,
        every_secs: float | None = 60.0,
        on_steps: Iterable[int] | None = None,
    ):
        """Creates a progress reporter.

        Args:
            num_train_steps: Total number of training steps (enables ETA / %).
            writer: Optional writer for ``steps_per_sec`` and ``uptime`` scalars.
            every_steps: How often to report, in steps.
            every_secs: How often to report, in seconds.
            on_steps: Additional steps on which to report.
        """
        on_steps = set(on_steps or [])
        if num_train_steps is not None:
            on_steps.add(num_train_steps)
        super().__init__(
            every_steps=every_steps, every_secs=every_secs, on_steps=on_steps
        )
        if num_train_steps is not None and num_train_steps < 0:
            num_train_steps = None
        self._num_train_steps = num_train_steps
        self._writer = writer
        self._time_per_part: dict[str, float] = collections.defaultdict(float)
        self._t0 = time.monotonic()
        # A single worker guarantees timed() measurements run sequentially.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._persistent_notes = ""

    def set_persistent_notes(self, message: str) -> None:
        """Sets a prefix message kept across reports."""
        self._persistent_notes = message

    def _should_trigger(self, step: int, t: float) -> bool:
        # step == previous_step only on the first call; never report then.
        return step != self._previous_step and super()._should_trigger(step, t)

    def _apply(self, step: int, t: float) -> None:
        steps_per_sec = (step - self._previous_step) / (t - self._previous_time)
        message = f"{steps_per_sec:.1f} steps/s"
        if self._num_train_steps:
            eta_seconds = (self._num_train_steps - step) / steps_per_sec
            message += (
                f", {100 * step / self._num_train_steps:.1f}% "
                f"({step}/{self._num_train_steps}), "
                f"ETA: {_format_secs(eta_seconds)}"
            )
        if self._time_per_part:
            total = time.monotonic() - self._t0
            message += " ({} : {})".format(
                _format_secs(total),
                ", ".join(
                    f"{100 * dt / total:.1f}% {name}"
                    for name, dt in sorted(self._time_per_part.items())
                ),
            )
        if self._persistent_notes:
            message = f"{self._persistent_notes}\n{message}"
        logging.info("Progress: %s", message)
        if self._writer is not None:
            self._writer.write_scalars(step, {"steps_per_sec": steps_per_sec})
            self._writer.write_scalars(step, {"uptime": time.monotonic() - self._t0})

    @contextlib.contextmanager
    def timed(self, name: str, wait_jax_async_dispatch: bool = True):
        """Measures time spent in a named section of the training loop.

        Reported progress breaks the total time down by section. When
        ``wait_jax_async_dispatch`` is ``True`` the JAX async dispatch queue is
        drained at the section's start and end so the measurement reflects the
        section's JAX computations rather than just the Python statements.

        Args:
            name: Section name to accumulate time under.
            wait_jax_async_dispatch: Drain JAX's dispatch queue around the block.
        """
        if not wait_jax_async_dispatch:
            start = time.monotonic()
            yield
            self._time_per_part[name] += time.monotonic() - start
            return

        def start_measurement(barrier: jax.Array) -> float:
            barrier.block_until_ready()
            return time.monotonic()

        def stop_measurement(
            start_future: futures.Future[float], barrier: jax.Array
        ) -> None:
            barrier.block_until_ready()
            self._time_per_part[name] += time.monotonic() - start_future.result()

        start_future = self._executor.submit(
            start_measurement, barrier=_squareit(jnp.array(0.0))
        )
        yield
        self._executor.submit(
            stop_measurement,
            start_future=start_future,
            barrier=_squareit(jnp.array(0.0)),
        )


class PeriodicCallback(PeriodicAction):
    """Calls a user callback each time it triggers."""

    def __init__(
        self,
        *,
        every_steps: int | None = None,
        every_secs: float | None = None,
        on_steps: Iterable[int] | None = None,
        callback_fn: Callable,
        execute_async: bool = False,
        pass_step_and_time: bool = True,
    ):
        """Creates a periodic callback.

        Args:
            every_steps: See ``PeriodicAction``.
            every_secs: See ``PeriodicAction``.
            on_steps: See ``PeriodicAction``.
            callback_fn: Callback receiving ``step`` and ``t`` by keyword (plus
                any extra kwargs passed to ``__call__``) when
                ``pass_step_and_time`` is ``True``.
            execute_async: If ``True``, run the callback on a background thread.
                Exceptions are surfaced on a subsequent call.
            pass_step_and_time: Whether to pass ``step``/``t`` to the callback.
        """
        super().__init__(
            every_steps=every_steps, every_secs=every_secs, on_steps=on_steps
        )
        self._cb_results = collections.deque(maxlen=1)
        self.pass_step_and_time = pass_step_and_time
        self._execute_async = execute_async
        if execute_async:
            logging.info(
                "Callback %s will be executed asynchronously; errors are raised "
                "when they become available.",
                getattr(callback_fn, "__name__", repr(callback_fn)),
            )
            self._executor = futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="PeriodicCallback"
            )
            self._pending: list[futures.Future] = []
        self._cb_fn = callback_fn

    def __call__(self, step: int, t: float | None = None, **kwargs) -> bool:
        if t is None:
            t = time.monotonic()
        self._init_and_check(step, t)
        if self._should_trigger(step, t):
            self._apply(step, t, **kwargs)
            self._after_apply(step, t)
            return True
        return False

    def get_last_callback_result(self):
        """Returns the most recent callback result."""
        return self._cb_results[0]

    def _run(self, step, t, **kwargs):
        if self.pass_step_and_time:
            return self._cb_fn(step=step, t=t, **kwargs)
        return self._cb_fn(**kwargs)

    def _apply(self, step, t, **kwargs):
        if self._execute_async:
            # Surface any prior background error before scheduling more work.
            for fut in [f for f in self._pending if f.done()]:
                fut.result()
            self._pending = [f for f in self._pending if not f.done()]
            self._pending.append(self._executor.submit(self._run, step, t, **kwargs))
        else:
            self._cb_results.append(self._run(step, t, **kwargs))
