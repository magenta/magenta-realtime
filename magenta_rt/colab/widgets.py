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

"""Colab widgets for Magenta RT."""

import base64
import concurrent.futures
import functools
import importlib
from typing import Callable
import uuid
import ipywidgets as ipw
import numpy as np
import resampy
from . import prompt_types
from . import utils

colab = importlib.import_module('google.colab')


class Prompt:
  """Text prompt widget.

  This widget allows to input a text prompt, a slider value and a text value
  linked to the slider.
  """

  def __init__(self):
    self.slider = ipw.FloatSlider(
        value=0,
        min=0,
        max=2,
        step=0.001,
        readout=False,
        layout=ipw.Layout(
            display='flex',
            width='auto',
            flex='16 1 0%',
        ),
    )
    self.text = ipw.Text(
        value='',
        placeholder='Enter a style',
        layout=ipw.Layout(
            display='flex',
            width='auto',
            flex='16 1 0%',
        ),
    )
    self.label = ipw.FloatText(
        value=0,
        disabled=False,
        layout=ipw.Layout(
            display='flex',
            width='4em',
        ),
    )
    ipw.link((self.slider, 'value'), (self.label, 'value'))

  def get_widget(self):
    """Shows the widget in the current cell."""
    return ipw.HBox(
        children=[
            self.text,
            self.slider,
            self.label,
        ],
        layout=ipw.Layout(display='flex', width='50em'),
    )

  @property
  def prompt_value(self):
    return self.text


class AudioPrompt(Prompt):
  """Audio prompt widget.

  This widget allows to upload an audio file, a slider value and a text value
  linked to the slider.
  """

  def __init__(self):
    super().__init__()
    utils._load_js('static/js/upload_audio.js')  # pylint: disable=protected-access

    self.upload_button = ipw.Button(
        value='Upload',
        description='Upload audio file',
        layout=ipw.Layout(
            display='flex',
            width='auto',
            flex='16 1 0%',
        ),
    )
    callback_name = f'notebook.uploadAudio/{uuid.uuid4()}'

    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    colab.output.register_callback(
        callback_name,
        functools.partial(self.executor.submit, self.audio_callback),
    )

    def _on_click(*args, **kwargs):
      del args, kwargs
      utils._call_js('uploadAudio', callback_name)  # pylint: disable=protected-access

    self.upload_button.on_click(_on_click)
    self.value = None
    self.parameter_callback = None

  def observe(self, callback):
    self.parameter_callback = callback

  def audio_callback(
      self, filename: str, audio_data_b64: str, sample_rate: int, **kwargs
  ):
    """Callback for audio upload."""
    del kwargs
    x = np.frombuffer(base64.b64decode(audio_data_b64), dtype=np.float32)
    x = x.copy()

    self.upload_button.description = filename
    audio = resampy.resample(x, sample_rate, 16_000)
    if self.parameter_callback is None:
      return

    self.parameter_callback(
        dict(
            name='value',
            new=prompt_types.AudioPrompt(value=audio),
        )
    )

  def get_widget(self):
    """Shows the widget in the current cell."""
    return ipw.HBox(
        children=[
            self.upload_button,
            self.slider,
            self.label,
        ],
        layout=ipw.Layout(display='flex', width='50em'),
    )

  @property
  def prompt_value(self):
    return self


class LiveAudioPrompt(Prompt):
  """Live audio prompt widget."""

  def __init__(
      self,
      audio_embedding_fn: Callable[[np.ndarray], np.ndarray],
      sample_rate: int = 16_000,
      buffer_seconds: int = 10,
      trigger_embedding_every_n_seconds: int = 4,
  ):
    super().__init__()
    # A text box that looks like the text prompts, but with input disabled.
    self.text = ipw.Text(
        value=f'Input Audio (last {buffer_seconds}s)',
        layout=ipw.Layout(
            display='flex',
            width='auto',
            flex='16 1 0%',
        ),
        disabled=True,
    )

    self._parameter_callback = None
    self._audio_embedding_fn = audio_embedding_fn

    self._input_audio = np.zeros(
        (sample_rate * buffer_seconds,), dtype=np.float32
    )
    self._sample_rate = sample_rate
    self._trigger_embedding_every_n_seconds = trigger_embedding_every_n_seconds
    self._num_new_seconds = 0
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    self.value = None

  def observe(self, callback):
    self.parameter_callback = callback

  @property
  def prompt_value(self):
    return self

  def update_embedding(self):
    embedding = prompt_types.EmbeddingPrompt(
        value=self._audio_embedding_fn(self._input_audio)
    )
    self.parameter_callback(dict(name='value', new=embedding))

  def update_audio_input(self, audio: np.ndarray) -> None:
    assert audio.ndim == 1, 'Audio must be 1D.'
    self._input_audio = np.concatenate(
        [self._input_audio, audio],
        axis=-1,
    )[-len(self._input_audio) :]

    self._num_new_seconds += len(audio) / self._sample_rate
    if self._num_new_seconds >= self._trigger_embedding_every_n_seconds:
      self._num_new_seconds -= self._trigger_embedding_every_n_seconds
      self._executor.submit(self.update_embedding)


def area(name: str, *childrens: ipw.Widget) -> ipw.Widget:
  """Groups multiple widgets inside a box with an explicit label.

  Args:
    name: label to display
    *childrens: list of ipw.Widget to display

  Returns:
    An ipw.Widget containing all childrens.
  """
  return ipw.Box(
      children=[ipw.HTML(f'<h3>{name}</h3>')] + list(childrens),
      layout=ipw.Layout(
          border='solid 1px',
          padding='.2em',
          margin='.2em',
          display='flex',
          flex_flow='column',
      ),
  )
