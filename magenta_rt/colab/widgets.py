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

import ipywidgets as ipw


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
    self.value = ipw.FloatText(
        value=0,
        disabled=False,
        layout=ipw.Layout(
            display='flex',
            width='4em',
        ),
    )
    ipw.link((self.slider, 'value'), (self.value, 'value'))

  def get_widget(self):
    """Shows the widget in the current cell."""
    return ipw.HBox(
        children=[
            self.text,
            self.slider,
            self.value,
        ],
        layout=ipw.Layout(display='flex', width='50em'),
    )


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
