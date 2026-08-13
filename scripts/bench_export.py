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

"""Sweep (batch_size, workers) to find the fastest codec-only SFT export config.

Builds the NNX codec once, pre-compiles each batch size, then for each config
times two export runs of different lengths and reports the **steady-state**
rate ``(N2 - N1) / (T2 - T1)`` — differencing cancels the per-run startup
(grain worker spawn + pipeline fill), so the numbers reflect sustained
throughput, not the one-off warmup. Run from the repo root inside the venv.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import click

from export_sft_from_playlist import _build_codec_nnx, _files_from_itunes_xml


def _time_export(codec, files, *, num_samples, duration, trim_frames,
                 batch_size, workers, saliency):
    from magenta_rt.sft.export import export_tree_dataset

    out = tempfile.mkdtemp(prefix="bench_export_")
    start = time.time()
    export_tree_dataset(
        None, out, codec=codec, style_model=None, transcriber=None,
        files=files, num_samples=num_samples, duration=duration,
        trim_frames=trim_frames, batch_size=batch_size, seed=0,
        excerpt=saliency, worker_count=workers,
        worker_buffer_size=1, save_embedding=False,
    )
    elapsed = time.time() - start
    shutil.rmtree(out, ignore_errors=True)
    return elapsed


@click.command()
@click.option("--itunes-xml", default="iTunes-Electronic-Playlist.xml")
@click.option("--duration", type=float, default=4.0)
@click.option("--trim-seconds", type=float, default=1.0)
@click.option("--batch-sizes", default="4,8,16",
              help="Comma-separated SpectroStream encode batch sizes.")
@click.option("--workers-list", default="4,8,12,18",
              help="Comma-separated grain worker counts.")
@click.option("--n-small", type=int, default=48)
@click.option("--n-large", type=int, default=168)
def main(itunes_xml, duration, trim_seconds, batch_sizes, workers_list,
         n_small, n_large):
    """Benchmark export throughput across a batch_size x workers grid."""
    from audiotree import ExcerptConfig

    from magenta_rt.sft.export import FRAME_RATE

    batch_sizes = [int(x) for x in batch_sizes.split(",")]
    workers_list = [int(x) for x in workers_list.split(",")]
    trim_frames = round(trim_seconds * FRAME_RATE)
    saliency = ExcerptConfig(strategy="loudest", num_tries=8, lufs_cutoff=-60.0)

    files = [f for f in _files_from_itunes_xml(itunes_xml)
             if os.path.isfile(f)]
    print(f"[bench] {len(files)} files; duration={duration}s "
          f"trim={trim_frames}f; steady rate = "
          f"({n_large}-{n_small})/(T_large - T_small)")
    print("[bench] building codec (once) ...")
    codec = _build_codec_nnx("mrt2_small", None)

    # Pre-compile the codec encode for every batch size so the timed runs are
    # compile-free (compile is per-process-per-shape and cached across calls).
    for bs in sorted(set(batch_sizes)):
        _time_export(codec, files, num_samples=bs, duration=duration,
                     trim_frames=trim_frames, batch_size=bs, workers=0,
                     saliency=saliency)
    print("[bench] warmup/compile done. sweeping ...\n")

    results = []
    for bs in batch_sizes:
        for w in workers_list:
            t_small = _time_export(codec, files, num_samples=n_small,
                                   duration=duration, trim_frames=trim_frames,
                                   batch_size=bs, workers=w, saliency=saliency)
            t_large = _time_export(codec, files, num_samples=n_large,
                                   duration=duration, trim_frames=trim_frames,
                                   batch_size=bs, workers=w, saliency=saliency)
            rate = (n_large - n_small) / max(t_large - t_small, 1e-6)
            results.append((bs, w, rate))
            print(f"  batch={bs:>3}  workers={w:>3}  ->  "
                  f"{rate:5.2f} ex/s   (T{n_small}={t_small:5.1f}s, "
                  f"T{n_large}={t_large:5.1f}s)")

    results.sort(key=lambda r: r[2], reverse=True)
    bs, w, rate = results[0]
    print(f"\n[bench] BEST: batch={bs}, workers={w}  ->  {rate:.2f} ex/s")
    for n in (20_000, 18_000, 2_000):
        print(f"        ~{n} samples ≈ {n / rate / 60:.0f} min "
              f"({n / rate / 3600:.1f} h)")


if __name__ == "__main__":
    main()
