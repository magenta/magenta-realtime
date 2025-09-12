# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Generate music offline using Magenta RealTime!

Example usage:
  python -m magenta_rt.generate \
      --prompt "fast tempo gabber,./violin.wav" \
      --weights "2.0,1.0" \
      --output "output.wav"
"""

import pathlib

from absl import app
from absl import flags
import numpy as np
import tqdm

from . import audio
from . import system

_PROMPTS = flags.DEFINE_list(
    'prompt',
    None,
    'Prompt to generate.',
    required=True,
)
_WEIGHTS = flags.DEFINE_list(
    'weight',
    None,
    'Weight for each prompt.',
)
_DURATION = flags.DEFINE_float(
    'duration',
    30.0,
    'Duration of the output audio.',
)
_OUTPUT = flags.DEFINE_string(
    'output',
    'output.wav',
    'Path to the output audio file.',
)
_TAG = flags.DEFINE_string(
    'tag',
    'large',
    'Tag of the model to use.',
)
_DEVICE = flags.DEFINE_string(
    'device',
    'gpu',
    'Device to use.',
)


def main(unused_argv):
  # Parse prompts
  prompts = []
  for prompt_or_path in _PROMPTS.value:
    path = pathlib.Path(prompt_or_path)
    if path.exists():
      # Assume prompt is a path to an audio file
      prompt = audio.Waveform.from_file(str(path))
    else:
      # If not, assume prompt is a text prompt
      prompt = prompt_or_path
    prompts.append(prompt)

  # Parse weights
  weights = _WEIGHTS.value
  if weights is None:
    weights = [1.0] * len(prompts)
  else:
    try:
      weights = [float(w) for w in weights]
    except ValueError as e:
      raise ValueError(
          'Weights must be a comma-separated list of floats, but got'
          f' {weights}.'
      ) from e

  # Check that number of prompts and weights match
  if len(prompts) != len(weights):
    raise ValueError(
        'Number of prompts must match number of weights, but got'
        f' {len(prompts)} prompts and {len(weights)} weights.'
    )

  # Init system
  magenta_rt = system.MagentaRT(
      tag=_TAG.value, device=_DEVICE.value, lazy=False
  )

  # Blend styles
  styles = np.array([magenta_rt.embed_style(p) for p in prompts])
  weights = np.array(weights, dtype=np.float32)
  weights /= weights.sum()
  style = (weights[:, np.newaxis] * styles).mean(axis=0)

  # Generate and write output
  chunks = []
  state = None
  num_chunks = int(np.ceil(_DURATION.value / magenta_rt.config.chunk_length))
  num_samples = round(_DURATION.value * magenta_rt.sample_rate)
  for _ in tqdm.tqdm(range(num_chunks)):
    chunk, state = magenta_rt.generate_chunk(state=state, style=style)
    chunks.append(chunk)
  generated = audio.concatenate(
      chunks, crossfade_time=magenta_rt.config.crossfade_length
  )[:num_samples]
  generated.write(_OUTPUT.value)


if __name__ == '__main__':
  app.run(main)
