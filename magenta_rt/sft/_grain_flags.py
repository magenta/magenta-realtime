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

"""Make grain's absl flags readable outside an ``absl.app`` entry point."""

from __future__ import annotations


def ensure_absl_flags_parsed() -> None:
    """Pin grain's absl flags to their defaults if nothing has parsed a command line.

    grain reads module-level absl flags from inside its worker machinery — e.g.
    ``--grain_enable_multiprocess_worker_profiling``, read on the thread that
    starts ``mp_prefetch`` workers. absl refuses to read a flag before a command
    line has been parsed (``UnparsedFlagAccessError``), and nothing here ever
    parses one: ``mrt`` is a click application and the trainers under
    ``notebooks/sft/`` are plain scripts, so neither goes through ``absl.app.run``.

    Marking the flag set parsed leaves every flag at its declared default, which
    is exactly what an absl entry point invoked without grain flags would have
    produced. It is a no-op when a command line *has* been parsed, so a caller
    that does use ``absl.app`` keeps its own flag values.

    Only the multiprocessing path needs this; a single-process grain pipeline
    (``worker_count == 0``) never reads the flag, which is why the unit tests —
    all single-process — never hit it.

    TODO(grain): remove this once grain 0.2.19 is on PyPI.
    https://github.com/google/grain/pull/1362 fixes the flag access upstream, so
    this helper (and its call sites in ``export.py`` / ``data.py``) becomes dead
    code — but the fix is not in a released wheel yet, so the workaround has to
    stay until the dependency floor can move.
    """
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
