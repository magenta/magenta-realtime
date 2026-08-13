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

"""SFT training utilities for Magenta-RT V2.

Framework-neutral pieces (configs, grain data pipeline, fake-data writer,
freeze helper, and the shared trainer glue in ``trainer_common``). The actual
training drivers live under ``notebooks/sft/`` (``train_nnx.py`` /
``train_mlx.py``) so hyperparameters can be tweaked without round-tripping
through the package surface.
"""

from .checkpoint import (
    export_nnx_to_linen_safetensors,
    load_nnx_depthformer_from_safetensors,
)
from .configs import SFTConfig, TinyPOCSpec
from .earlystop import EarlyStopper
from .freeze import Frozen, freeze_module
from .lora_nnx import (
    LoRAAdapter,
    MRTLoRAParam,
    all_linear_targets,
    default_targets,
    inject_lora,
    merge_lora_into_base,
)


_LAZY_DATA = {"create_audiotree_dataset"}
_LAZY_TRANSFORMS = {
    "AddFixedStyle",
    "augment_batch",
    "to_source_target",
    "EncodeWithCodec",
    "PrepareTarget",
    "PrepareSource",
    "PrepareCFG",
    "AudioTreeRandomCrop",
    "AudioTreeMusicCoCaSticky",
    "StyleEmbeddingJitter",
    "rvq_tokenize",
}


def __getattr__(name):
    # ``create_audiotree_dataset`` and the AudioTree transforms live in modules
    # that import the heavy ``grain`` native stack. Load them lazily so the lightweight
    # modules (``checkpoint``, ``freeze``, ``lora``, ``configs``) stay importable
    # in environments without ``grain`` (e.g. the non-SFT CI test process).
    if name in _LAZY_DATA:
        from . import data

        return getattr(data, name)
    if name in _LAZY_TRANSFORMS:
        from . import transforms

        return getattr(transforms, name)
    if name in ("export_tree_dataset", "mt3_transcriber"):
        from . import export

        return getattr(export, name)
    if name in ("save_lora_adapters", "load_lora_adapters", "read_lora_metadata"):
        from . import lora_io

        # `read_lora_metadata` is exposed under that name; it's `read_metadata`
        # inside the module.
        return lora_io.read_metadata if name == "read_lora_metadata" else getattr(
            lora_io, name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AddFixedStyle",
    "AudioTreeMusicCoCaSticky",
    "AudioTreeRandomCrop",
    "EarlyStopper",
    "EncodeWithCodec",
    "Frozen",
    "LoRAAdapter",
    "MRTLoRAParam",
    "PrepareCFG",
    "PrepareSource",
    "PrepareTarget",
    "SFTConfig",
    "StyleEmbeddingJitter",
    "TinyPOCSpec",
    "all_linear_targets",
    "augment_batch",
    "create_audiotree_dataset",
    "default_targets",
    "export_nnx_to_linen_safetensors",
    "export_tree_dataset",
    "freeze_module",
    "inject_lora",
    "load_lora_adapters",
    "load_nnx_depthformer_from_safetensors",
    "merge_lora_into_base",
    "mt3_transcriber",
    "read_lora_metadata",
    "save_lora_adapters",
    "rvq_tokenize",
    "to_source_target",
]
