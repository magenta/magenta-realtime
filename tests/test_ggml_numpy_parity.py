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

"""Parity of the numpy depthformer (the ggml port's spec) against JAX fp32.

The numpy model in magenta_rt/ggml/numpy_model.py is the executable
specification the C++/ggml port is written from, so it must track JAX exactly.
Reference tensors come from scripts/generate_ggml_reference.py; run that first:

    python scripts/generate_ggml_reference.py --model=mrt2_small --num-steps=2
"""

import json
import unittest
from pathlib import Path

import numpy as np

from magenta_rt import paths
from magenta_rt.ggml.convert import _read_safetensors
from magenta_rt.ggml.numpy_model import Cache, DepthformerNumpy

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "mrt2_small"
REF_DIR = REPO_ROOT / "tests" / "testdata" / "ggml" / MODEL
LONG_REF_DIR = REPO_ROOT / "tests" / "testdata" / "ggml" / f"{MODEL}_long"
MAX_PAST_HORIZON = 41
CACHE_SLOTS = MAX_PAST_HORIZON + 1
_ATOL = 1e-3  # matches tests/test_bitlevel_parity.py


def _build(ref_dir=REF_DIR):
    manifest = json.loads((ref_dir / "manifest.json").read_text())["tensors"]

    def ref(name):
        entry = manifest[name]
        arr = np.fromfile(ref_dir / entry["file"], dtype=np.float32)
        return arr.reshape(entry["shape"])

    weights, _ = _read_safetensors(str(paths.resolve_checkpoint(f"{MODEL}.safetensors")))

    # Encoder channel layout, mirroring magenta_rt/*/model.py.
    from magenta_rt.jax import model as jax_model
    exp = jax_model.get_model_class(MODEL)()
    per_channel = []
    for cfg in exp.input_configs[1:]:
        per_channel += [cfg.per_rvq_vocab_size] * cfg.rvq_truncation_level
    config = dict(
        n_mulan=exp.input_configs[0].rvq_truncation_level,
        t_layers=12, d_layers=2, t_model_dims=1024,
        mulan_offsets=(np.arange(exp.input_configs[0].rvq_truncation_level)
                       * exp.input_configs[0].per_rvq_vocab_size).astype(np.int32),
        regular_offsets=np.concatenate(
            [[0], np.cumsum(per_channel[:-1])]).astype(np.int32),
    )
    return DepthformerNumpy(weights, config), ref


@unittest.skipUnless(REF_DIR.exists(), f"missing references in {REF_DIR}")
class TestNumpyParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model, ref = _build()
        cls.model = model
        cls.ref = staticmethod(ref)
        cls.tokens = np.fromfile(REF_DIR / "cond_tokens.i32.bin", dtype=np.int32)

    def test_encoder(self):
        got = self.model.encode(self.tokens)
        want = self.ref("step0.depthformer.encoder.body.step").ravel()
        np.testing.assert_allclose(got, want, atol=_ATOL, rtol=0)

    def test_temporal_outputs(self):
        enc = self.model.encode(self.tokens)
        x = np.mean([self.model.embed_token(0)] * 12, axis=0)
        caches = {
            "self": [Cache(CACHE_SLOTS, 8, 128) for _ in range(12)],
            "cross": [Cache(CACHE_SLOTS, 8, 128) for _ in range(12)],
        }
        got = self.model.temporal_step(x, enc, caches)
        want = self.ref("step0.depthformer.decoder.temporal_outputs").ravel()
        np.testing.assert_allclose(got, want, atol=_ATOL, rtol=0)

    def test_depth_logits_all_rvq(self):
        """Teacher-forced: JAX's sampled token is fed back at each RVQ step."""
        depth_input = self.ref("step0.depthformer.decoder.temporal_outputs").ravel()
        caches = [Cache(13, 6, 128) for _ in range(2)]
        for rvq in range(12):
            got = self.model.depth_step(depth_input, caches)
            want = self.ref(f"step0.depthformer.decoder.depth_logits.{rvq}").ravel()
            np.testing.assert_allclose(
                got, want, atol=_ATOL, rtol=0,
                err_msg=f"depth logits mismatch at rvq_index={rvq}")
            token = int(self.ref(f"step0.depthformer.decoder.depth_samples.{rvq}").ravel()[0])
            depth_input = self.model.embed_token(token)


@unittest.skipUnless(LONG_REF_DIR.exists(), f"missing references in {LONG_REF_DIR}")
class TestDeepStreamParity(unittest.TestCase):
    """Streams past the point where the KV cache first wraps.

    The cache must hold max_past_horizon + 1 frames. With only
    max_past_horizon slots the oldest frame is evicted one step early, which
    produces correct output for the first 41 steps and then diverges hard
    (max_abs ~0.4). A short parity run cannot catch this.
    """

    N_STEPS = 45

    @classmethod
    def setUpClass(cls):
        model, ref = _build(LONG_REF_DIR)
        cls.model = model
        cls.ref = staticmethod(ref)
        cls.tokens = np.fromfile(LONG_REF_DIR / "cond_tokens.i32.bin", dtype=np.int32)

    def test_streams_past_cache_wrap(self):
        enc = self.model.encode(self.tokens)
        caches = {
            "self": [Cache(CACHE_SLOTS, 8, 128) for _ in range(12)],
            "cross": [Cache(CACHE_SLOTS, 8, 128) for _ in range(12)],
        }
        prev = np.zeros(12, np.int32)
        for step in range(self.N_STEPS):
            x = np.mean([self.model.embed_token(int(t)) for t in prev], axis=0)
            got = self.model.temporal_step(x, enc, caches)
            want = self.ref(f"step{step}.depthformer.decoder.temporal_outputs").ravel()
            np.testing.assert_allclose(
                got, want, atol=_ATOL, rtol=0,
                err_msg=f"temporal output diverged at streaming step {step}")
            prev = np.array(
                [int(self.ref(f"step{step}.depthformer.decoder.depth_samples.{j}").ravel()[0])
                 for j in range(12)], np.int32)


if __name__ == "__main__":
    unittest.main()
