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

"""Audio processing utils for :class:`audiotree.AudioTree`.

This module holds magenta-rt's audio *helper functions* (``compute_rms``,
``apply_gain``, ``peak_normalize``, ``concatenate``, ...).
The container itself comes straight from the
`audiotree <https://github.com/DBraun/audiotree>`_ package — import it as
``from audiotree import AudioTree`` (it is intentionally *not* re-exported
here). It is a ``flax.struct`` dataclass (a JAX pytree) with:

* ``waveform``: ``[batch, channels, samples]`` float32 audio (**note the
  channels-first layout** — librosa-native),
* ``codes``: optional neural-codec tokens; magenta_rt keeps them frame-major
  ``[batch, nframes, num_codebooks]``,
* ``metadata``: free-form per-example side data (a pytree node),
* ``sample_rate``: static (non-pytree) field.

magenta-rt previously shipped a minimal local reimplementation of AudioTree
(``Waveform``, with a ``[batch, samples, channels]`` layout and
``samples``/``tokens`` field names); that class — and later its alias —
have been removed.

Migration notes (old ``Waveform`` -> ``AudioTree``):

* ``.samples`` (audio array) -> ``.waveform`` — beware: ``AudioTree.samples``
  is the **sample count** (``waveform.shape[-1]``), not the array.
* ``.tokens`` -> ``.codes``; ``.num_samples`` -> ``.samples``.
* ``to_mono`` / ``to_stereo`` / ``resample`` / ``from_file`` / ``batch_fn`` /
  batch indexing & iteration are AudioTree methods.
* ``write`` is now ``AudioTree.write`` (e.g. ``tree[0].write(path)``);
  ``compute_rms`` / ``apply_gain`` / ``peak_normalize`` / ``concatenate`` are
  module-level functions here.
"""

import functools

import audiotree
import jax.numpy as jnp
import librosa
import numpy as np


def _xp(arr):
  """Returns the array module (numpy or jax.numpy) backing ``arr``."""
  return np if isinstance(arr, np.ndarray) else jnp


# ---------------------------------------------------------------------------
# librosa-domain analysis (single, materialized waveform)
# ---------------------------------------------------------------------------


def _librosa_samples(tree: audiotree.AudioTree) -> np.ndarray:
  """Returns audio in a librosa-compatible form.

  Librosa expects: np.ndarray [shape=(n,) or (nch, n)]. With the ``[B, C, T]``
  layout this is just ``waveform[0]`` (channel-major), squeezed for mono.
  Only valid for a single (unbatched) waveform.
  """
  assert tree.batch_size == 1, (
      f"librosa ops require a single waveform; got batch_size={tree.batch_size}"
  )
  w = np.asarray(tree.waveform)[0]  # [nch, n]
  return w[0] if w.shape[0] == 1 else w


def compute_rms(
    tree: audiotree.AudioTree,
    # NOTE: Librosa defaults
    hop_length_seconds: float = 512 / 22050,
    frame_length_seconds: float = 2048 / 22050,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes RMS amplitude over frames; returns ``(times, rms)``."""
  rms = librosa.feature.rms(
      y=_librosa_samples(tree),
      hop_length=round(hop_length_seconds * tree.sample_rate),
      frame_length=round(frame_length_seconds * tree.sample_rate),
  )[0]
  t = np.arange(rms.shape[0]) * hop_length_seconds
  return t, rms


def compute_peak_rms(tree: audiotree.AudioTree, *args, **kwargs) -> float:
  """Computes peak RMS amplitude over frames."""
  _, rms = compute_rms(tree, *args, **kwargs)
  peak_rms = np.max(rms)
  assert peak_rms >= 0
  return peak_rms


def peak_amplitude(tree: audiotree.AudioTree) -> float:
  """Peak absolute sample value across the whole (batched) waveform."""
  return float(np.abs(np.asarray(tree.waveform)).max())


# ---------------------------------------------------------------------------
# Transforms (immutable; numpy/jax-agnostic). ``codes`` are carried through
# unchanged via ``.replace`` — callers that mutate the audio should drop
# stale codes explicitly if needed.
# ---------------------------------------------------------------------------


def apply_gain(tree: audiotree.AudioTree, gain: float) -> audiotree.AudioTree:
  """Applies linear gain, returning a new AudioTree."""
  return tree.replace(waveform=tree.waveform * gain)


def peak_normalize(tree: audiotree.AudioTree, max_value: float = 1.0) -> audiotree.AudioTree:
  """Normalizes audio to a particular peak amplitude value."""
  peak = peak_amplitude(tree)
  gain = 1.0 if peak == 0 else max_value / peak
  return apply_gain(tree, gain)


def concatenate(trees: list[audiotree.AudioTree]) -> audiotree.AudioTree:
  """Concatenates a list of AudioTrees (and their codes) along the time axis."""
  if not trees:
    raise ValueError("No waveforms to concatenate.")
  all_sample_rates = set(t.sample_rate for t in trees)
  if len(all_sample_rates) != 1:
    raise ValueError("All waveforms must have the same sample rate.")
  sample_rate = all_sample_rates.pop()
  all_num_channels = set(t.num_channels for t in trees)
  if len(all_num_channels) != 1:
    raise ValueError("All waveforms must have the same number of channels.")
  all_batch_sizes = set(t.batch_size for t in trees)
  if len(all_batch_sizes) != 1:
    raise ValueError("All waveforms must have the same batch size.")
  # Concatenate along the time (samples) axis — the last axis in [B, C, T].
  xp = _xp(trees[0].waveform)
  result_waveform = xp.concatenate([t.waveform for t in trees], axis=-1)
  # Concatenate codes along their frame axis when every tree carries them.
  code_list = [t.codes for t in trees]
  if all(c is not None for c in code_list):
    result_codes = _xp(code_list[0]).concatenate(code_list, axis=1)
  else:
    result_codes = None
  return audiotree.AudioTree(result_waveform, sample_rate, codes=result_codes)


@functools.lru_cache(maxsize=1)
def crossfade_ramp(
    num_samples: int,
    style: str = "eqpower",
):
  """Returns a crossfade ramp for a given style and direction."""
  # Compute crossfade
  if style == "linear":
    # Linear crossfade
    ramp = np.linspace(
        0, 1, num_samples, endpoint=False, dtype=np.float32
    )
  elif style == "eqpower":
    # Equal power crossfade
    ramp = np.sin(
        np.linspace(
            0,
            np.pi / 2,
            num_samples,
            endpoint=False,
            dtype=np.float32,
        )
    )
  else:
    raise ValueError(f"Unsupported crossfade style: {style}")
  return ramp


def amp_to_db(amp: float, amp_ref: float = 1.0) -> float:
  if amp < 0:
    raise ValueError(f"Amplitude must be non-negative. Got {amp}.")
  if amp_ref <= 0:
    raise ValueError(f"Reference amplitude must be positive. Got {amp_ref}.")
  if amp == 0:
    return float("-inf")
  else:
    return float(20 * np.log10(amp / amp_ref))


def db_to_amp(db: float, amp_ref: float = 1.0) -> float:
  if amp_ref <= 0:
    raise ValueError(f"Reference amplitude must be positive. Got {amp_ref}.")
  return float(np.power(10, db / 20.0) * amp_ref)
