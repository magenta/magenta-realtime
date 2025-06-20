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

"""Utilities to enable real-time audio in colab."""

import base64
import functools
import importlib
import json
import os
import pathlib
from typing import Any, Callable

import IPython.display as ipd
import ipywidgets as ipw
import numpy as np


# using importlib to avoid build issues
colab = importlib.import_module("google.colab")


class Parameters:
  """Stores and updates values coming from ipywidgets.widgets."""

  _VALUES = {}
  _UI_ELEMENTS = {}

  @classmethod
  def update_values(cls, **kwargs):
    cls._VALUES.update(kwargs)

  @classmethod
  def get_values(cls) -> dict[str, Any]:
    return cls._VALUES

  @classmethod
  def reset(cls):
    cls._UI_ELEMENTS = {}
    cls._VALUES = {}

  @classmethod
  def register_ui_elements(
      cls, display: bool = True, **elements: ipw.widgets.ValueWidget
  ):
    """Registers ipywidgets and handle value changes asynchronously."""
    cls._UI_ELEMENTS.update(elements)

    def _handler(name, change):
      if change["name"] != "value":
        return

      cls.update_values(**{name: change["new"]})

    for name, obj in elements.items():
      obj.observe(functools.partial(_handler, name))
      cls.update_values(**{name: obj.value})
      if display:
        obj.description = name
        ipd.display(obj)


def _array_to_wav_bytes(fn):
  """Wraps an ndarray compatible function to operate with javascript."""

  def wrapper(inputs: str, **kwargs):
    inputs = base64.b64decode(inputs)
    inputs = np.frombuffer(inputs, dtype=np.int16)
    inputs = inputs.astype(np.float32) / (2**15 - 1)

    outputs: np.ndarray = fn(inputs, **kwargs)

    if outputs.shape[0] != inputs.shape[0]:
      print(f"warning, inconsistent shapes ({inputs.shape}->{outputs.shape})")
      outputs = outputs[: inputs.shape[0]]

    output_dict = {}
    if isinstance(outputs, tuple):
      buffer, metadata = outputs
      output_dict["metadata"] = json.dumps(metadata)
    else:
      buffer = outputs

    buffer = np.clip(buffer, -1, 1) * (2**15 - 1)
    buffer = buffer.astype(np.int16).tobytes()

    buffer = base64.b64encode(buffer).decode()
    output_dict["audiob64"] = buffer

    return ipd.JSON(output_dict)

  return wrapper


def _call_js(fn_name: str, *args):
  def _format_arg(arg):
    if isinstance(arg, str):
      return f"'{arg}'"
    elif isinstance(arg, bool):
      return "true" if arg else "false"
    return str(arg)

  args = ",".join(map(_format_arg, args))
  ipd.display(ipd.Javascript(f"{fn_name}({args});"))


def _load_asset(path: str):
  root = pathlib.Path(__file__).parent
  with open(os.path.join(root, path), "r") as f:
    f = f.read()
  return f




def _load_html(path: str):
  html = _load_asset(path)
  ipd.display(ipd.HTML(html))


def _load_js(path: str):
  js = _load_asset(path)
  ipd.display(ipd.Javascript(js))


def _get_js_data_url(path: str):
  js = _load_asset(path)
  prefix = "data:application/javascript;base64,"
  return prefix + base64.b64encode(js.encode()).decode()


class AudioStreamer:
  """Calls a python audio callback at regular intervals using a webaudio backend."""

  def __init__(
      self,
      audio_callback: Callable[[np.ndarray], np.ndarray],
      rate: int,
      buffer_size: int,
      enable_input: bool = False,
      warmup: bool = True,
      raw_input_audio: bool = False,
      enable_automatic_gain_control_on_input: bool = False,
      num_output_channels: int = 1,
      additional_buffered_samples: int = 0,
      start_streaming_callback: Callable[[], None] | None = None,
  ):
    _load_html("static/html/ui.html")

    if warmup:
      audio_callback(np.zeros(buffer_size))  # warmup

    # Registering callbacks
    colab.output.register_callback(
        "notebook.audioCallback",
        _array_to_wav_bytes(audio_callback),
    )

    if start_streaming_callback is None:
      start_streaming_callback = lambda: None

    colab.output.register_callback(
        "notebook.startStreamingCallback",
        start_streaming_callback,
    )

    _load_js("static/js/ring_buffer_node.js")
    _load_js("static/js/streamer.js")

    _call_js(
        "audioContextInit",
        rate,
        buffer_size,
        enable_input,
        raw_input_audio,
        enable_automatic_gain_control_on_input,
        num_output_channels,
        additional_buffered_samples,
        _get_js_data_url("static/js/ring_buffer.js"),
    )

    self.audio_callback = audio_callback

  def reset_ring_buffer(self):
    _call_js("resetRingBuffer")
