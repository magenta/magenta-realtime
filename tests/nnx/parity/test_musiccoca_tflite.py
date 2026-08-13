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

"""Parity: nnx MusicCoCa vs the original TFLite models.

Requires the MusicCoCa TFLite resources plus the converted
``musiccoca_nnx.safetensors`` (``python -m magenta_rt.nnx.musiccoca.convert``),
so it is gated like the other real-weight tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from magenta_rt import paths

pytestmark = pytest.mark.checkpoint

_RESOURCE_DIR = paths.musiccoca_dir()
_REQUIRED = [
    "audio_preprocessor.tflite",
    "music_encoder.tflite",
    "text_encoder.tflite",
    "pretrained_vector_quantizer.tflite",
    "mapper.tflite",
    "spm.model",
    "musiccoca_nnx.safetensors",
]
if not all((_RESOURCE_DIR / f).exists() for f in _REQUIRED):
    pytest.skip(
        f"MusicCoCa resources missing in {_RESOURCE_DIR}",
        allow_module_level=True,
    )


def _run_tflite(name: str, *inputs: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(_RESOURCE_DIR / f"{name}.tflite"))
    interp.allocate_tensors()
    for detail, x in zip(interp.get_input_details(), inputs):
        interp.set_tensor(
            detail["index"],
            np.asarray(x, dtype=detail["dtype"]).reshape(detail["shape"]),
        )
    interp.invoke()
    (out,) = interp.get_output_details()
    return interp.get_tensor(out["index"])


@pytest.fixture(scope="module")
def module():
    from magenta_rt.nnx.musiccoca import from_safetensors

    return from_safetensors(_RESOURCE_DIR / "musiccoca_nnx.safetensors")


@pytest.fixture(scope="module")
def vocab():
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(_RESOURCE_DIR / "spm.model"))
    return sp


@pytest.fixture(scope="module")
def waveform():
    return (np.random.RandomState(0).randn(160000) * 0.1).astype(np.float32)


def test_frontend_parity(module, waveform):
    ref = _run_tflite("audio_preprocessor", waveform[None])
    got = np.asarray(module.frontend(waveform[None]))
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_music_encoder_parity(module, waveform):
    mel = _run_tflite("audio_preprocessor", waveform[None])
    ref = _run_tflite("music_encoder", mel)
    got = np.asarray(module.audio(mel))
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_audio_embed_end_to_end(module, waveform):
    mel = _run_tflite("audio_preprocessor", waveform[None])
    ref = _run_tflite("music_encoder", mel)
    got = np.asarray(module.encode_clips(waveform[None]))
    np.testing.assert_allclose(got, ref, atol=2e-4)


@pytest.mark.parametrize(
    "text",
    ["staccato funk", "lush orchestral strings with a hint of jazz", ""],
)
def test_text_encoder_parity(module, vocab, text):
    from magenta_rt.nnx.musiccoca import encode_text

    ids, paddings = encode_text(vocab, text)
    ref = _run_tflite("text_encoder", ids[None], paddings[None])
    got = np.asarray(module.encode_tokens(ids[None], paddings[None]))
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_quantizer_parity(module, vocab):
    from magenta_rt.nnx.musiccoca import encode_text

    ids, paddings = encode_text(vocab, "ambient piano")
    emb = _run_tflite("text_encoder", ids[None], paddings[None])
    ref = _run_tflite("pretrained_vector_quantizer", emb).flatten()
    got = np.asarray(module.tokenize(emb)).flatten()
    np.testing.assert_array_equal(got, ref)


def test_mapper_parity(module, vocab):
    from magenta_rt.nnx.musiccoca import encode_text

    ids, paddings = encode_text(vocab, "ambient piano")
    emb = _run_tflite("text_encoder", ids[None], paddings[None])
    noise = np.random.RandomState(0).randn(768).astype(np.float32)
    ref = _run_tflite("mapper", emb, noise[None])
    got = np.asarray(module.mapper(emb, noise[None]))
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_system_matches_tflite_system(vocab):
    """End-to-end: nnx MusicCoCa class vs the TFLite-backed MusicCoCa."""
    from magenta_rt import musiccoca as tflite_musiccoca
    from magenta_rt.nnx import musiccoca as nnx_musiccoca

    ref_model = tflite_musiccoca.MusicCoCa(resource_dir=_RESOURCE_DIR)
    got_model = nnx_musiccoca.MusicCoCa(resource_dir=_RESOURCE_DIR)

    prompt = "breakbeat with heavy bass"
    ref = ref_model.embed_text(prompt)
    got = got_model.embed_text(prompt)
    np.testing.assert_allclose(got, ref, atol=1e-4)

    np.testing.assert_array_equal(
        got_model.tokenize(got), ref_model.tokenize(ref.astype(np.float32))
    )

    ref_mapped = ref_model.embed_text(prompt, use_mapper=True, seed=3)
    got_mapped = got_model.embed_text(prompt, use_mapper=True, seed=3)
    np.testing.assert_allclose(got_mapped, ref_mapped, atol=1e-4)
