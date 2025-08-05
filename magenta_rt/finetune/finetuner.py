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

"""T5X InteractiveModel for finetuning."""

import os
import pathlib

import tqdm

from .. import asset
from ..depthformer import model as model_lib


class MagentaRTFinetuner:
  """Wrapper around T5X InteractiveModel for Magenta RT finetuning."""

  def __init__(
      self,
      checkpoint_dir: str | None = None,
      output_dir: str | None = None,
      tag: str = 'base',
      batch_size: int = 8,
  ):
    self.batch_size = batch_size
    self.tag = tag
    if checkpoint_dir is None:
      if self.tag == 'base':
        path = 'checkpoints/llm_base_x4286_c1860k.tar'
      else:
        path = 'checkpoints/llm_large_x3047_c1860k.tar'
      self.checkpoint_dir = asset.fetch(path, is_dir=True, extract_archive=True)
    else:
      self.checkpoint_dir = checkpoint_dir

    if output_dir is None:
      output_dir = str(pathlib.Path(pathlib.Path.cwd() / 'finetune'))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    _, _, self.interactive_model = model_lib.load_pretrained_model(
        checkpoint_dir=self.checkpoint_dir,
        size=self.tag,
        batch_size=self.batch_size,
        num_partitions=1,
        model_parallel_submesh=None,
        gin_overrides='',
        output_dir=output_dir,
    )

  def train(self, train_iter, num_steps, save_ckpt_period=1000):
    self.accuracy = []
    self.loss = []
    for step in tqdm.tqdm(range(num_steps)):
      self.interactive_model.train_step_from_batch_iterator(train_iter)
      self.accuracy.append(self.interactive_model.train_summary['accuracy'])
      self.loss.append(self.interactive_model.train_summary['loss'])
      if step % save_ckpt_period == 0:
        self.save_checkpoint()

  def save_checkpoint(self):
    self.interactive_model.save_checkpoint()

  @property
  def train_state(self):
    return self.interactive_model.train_state

  @property
  def train_summary(self):
    return self.interactive_model.train_summary
