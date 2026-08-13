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

"""Compute a MusicCoCa text-prompt embedding + RVQ tokens in ISOLATION.

The mlx_pure MusicCoCa is fully native (no TFLite): its text encoder is a pure
MLX ``TextEncoder``. But the *text* path additionally loads a **SentencePiece**
tokenizer (a C++ library) to turn the prompt into token ids; the *audio* path
does not. That C++ tokenizer runtime deadlocks/aborts ("mutex lock failed" /
abseil "Lock blocking") once ``grain``/``audiotree``/JAX are also live in the
process — and the SFT export needs grain for audio decoding. (This is why
per-clip AUDIO MusicCoCa coexists with grain fine — it never touches
SentencePiece — while the text embed does not.)

So the SFT export runs this module as a **separate process** to embed the fixed
``--style-prompt`` once, writing ``embedding`` (``[768]``) + ``tokens``
(``[rvq_levels]``) to an ``.npz``. The SentencePiece runtime lives and dies in
this child; the export process never loads it and feeds the arrays straight to
``export_tree_dataset(style_embedding=..., style_tokens=...)``.

Usage:
    python -m magenta_rt.sft.embed_prompt --prompt "electronic breakbeat" \\
        --backend mlx_pure --out /tmp/prompt.npz
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True, help="Text style prompt.")
    p.add_argument("--backend", choices=["tflite", "mlx_pure", "nnx"],
                   default="tflite")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--use-mapper", action="store_true", default=False,
                   help="Refine the text embedding into audio space via the "
                        "MusicCoCa mapper (matches embed_style's use_mapper).")
    args = p.parse_args()

    if args.backend == "tflite":
        # The shipped TFLite MusicCoCa (downloaded by `mrt models init`); its
        # tokens match `mrt {nnx,mlx_pure} generate` inference, which uses the
        # same TFLite model — so train/infer style conditioning agree.
        from magenta_rt.musiccoca import MusicCoCa
    elif args.backend == "mlx_pure":
        from magenta_rt.mlx_pure.musiccoca import MusicCoCa
    else:
        from magenta_rt.nnx.musiccoca import MusicCoCa

    style_model = MusicCoCa()
    embedding = np.asarray(
        style_model.embed_text(args.prompt, use_mapper=args.use_mapper),
        dtype=np.float32,
    )  # [embedding_dim]
    tokens = np.asarray(
        style_model.tokenize(embedding[None]), dtype=np.int32
    )[0]
    np.savez(args.out, embedding=embedding, tokens=tokens)
    print(f"[embed_prompt] {args.prompt!r} -> embedding{embedding.shape} "
          f"tokens{tokens.shape} -> {args.out}")


if __name__ == "__main__":
    main()
