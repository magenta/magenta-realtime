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

"""Parity tests for ResidualVectorQuantizer."""

from __future__ import annotations

import mlx.core as mx
import pytest
import sequence_layers.mlx as sl

from magenta_rt.mlx.spectrostream import modeling as mrt_ss
from magenta_rt.mlx_pure.spectrostream import ResidualVectorQuantizer
from .conftest import assert_close, tol


def _seq(values: mx.array) -> sl.Sequence:
    return sl.Sequence(values, mx.ones(values.shape[:2], dtype=mx.bool_))


def _build_sl_rvq(num_quantizers, num_embeddings, embedding_dim, *,
                  use_unique_codes=False, truncation_level=None,
                  encoded_truncation_level=None):
    cfg = mrt_ss.ResidualVectorQuantizer.Config(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        use_unique_codes=use_unique_codes,
        beta=0.0,
        dynamic_masking=False,
        target_num_quantizers=(num_quantizers,),
        full_quantizer_dropout_rate=0.0,
        full_quantizer_commitment=False,
        truncation_level=truncation_level,
        encoded_truncation_level=encoded_truncation_level,
    )
    return cfg.make()


def test_codes_to_embeddings_parity_no_unique(rng_key):
    num_quantizers, num_embeddings, dim = 4, 16, 8
    sl_rvq = _build_sl_rvq(num_quantizers, num_embeddings, dim, use_unique_codes=False)
    sub = mx.random.split(rng_key)[0]
    sl_rvq.embedding = mx.random.normal(sl_rvq.embedding.shape, key=sub) * 0.1

    pure = ResidualVectorQuantizer(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=dim,
        use_unique_codes=False,
    )
    pure.embedding = sl_rvq.embedding

    codes = mx.random.randint(0, num_embeddings, (2, 5, num_quantizers), key=rng_key)
    sl_y = sl_rvq.codes_to_embeddings(_seq(codes)).values
    pure_y = pure.codes_to_embeddings(codes)
    a, r = tol(mx.float32, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="codes_to_emb")


def test_codes_to_embeddings_unique_codes_pure_only(rng_key):
    """sl's `use_unique_codes=True` path crashes (uses non-existent
    ``mx.mod``). Verify pure's path works by checking it equals the
    no-unique-codes result on the same effective indices.
    """
    num_quantizers, num_embeddings, dim = 4, 16, 8
    pure = ResidualVectorQuantizer(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=dim,
        use_unique_codes=True,
    )
    pure.embedding = mx.random.normal(pure.embedding.shape, key=rng_key) * 0.1

    pure2 = ResidualVectorQuantizer(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=dim,
        use_unique_codes=False,
    )
    pure2.embedding = pure.embedding

    raw_codes = mx.random.randint(0, num_embeddings, (2, 5, num_quantizers), key=mx.random.split(rng_key)[0])
    offsets = mx.arange(num_quantizers) * num_embeddings
    unique_codes = raw_codes + offsets

    y_unique = pure.codes_to_embeddings(unique_codes)
    y_plain = pure2.codes_to_embeddings(raw_codes)
    a, r = tol(mx.float32, "leaf")
    assert_close(y_unique, y_plain, atol=a, rtol=r, name="codes_to_emb_unique_internal")


def test_embeddings_to_codes_parity(rng_key):
    num_quantizers, num_embeddings, dim = 4, 16, 8
    sl_rvq = _build_sl_rvq(num_quantizers, num_embeddings, dim, use_unique_codes=False)
    sl_rvq.embedding = mx.random.normal(sl_rvq.embedding.shape, key=mx.random.split(rng_key)[0]) * 0.1

    pure = ResidualVectorQuantizer(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=dim,
    )
    pure.embedding = sl_rvq.embedding

    inputs = mx.random.normal((2, 5, dim), key=rng_key) * 0.1
    sl_codes = sl_rvq.embeddings_to_codes(_seq(inputs)).values
    pure_codes = pure.embeddings_to_codes(inputs)
    assert mx.array_equal(sl_codes, pure_codes).item(), (
        f"sl: {sl_codes.tolist()} pure: {pure_codes.tolist()}"
    )


def test_truncation_level_parity(rng_key):
    """codes_to_embeddings with truncation_level < num_quantizers."""
    num_quantizers, num_embeddings, dim = 4, 16, 8
    truncation = 2
    sl_rvq = _build_sl_rvq(
        num_quantizers, num_embeddings, dim,
        truncation_level=truncation,
    )
    sl_rvq.embedding = mx.random.normal(sl_rvq.embedding.shape, key=mx.random.split(rng_key)[0]) * 0.1

    pure = ResidualVectorQuantizer(
        num_quantizers=num_quantizers,
        num_embeddings=num_embeddings,
        embedding_dim=dim,
        truncation_level=truncation,
    )
    pure.embedding = sl_rvq.embedding

    # Use only `truncation` codebook columns.
    codes = mx.random.randint(0, num_embeddings, (2, 5, truncation), key=rng_key)
    sl_y = sl_rvq.codes_to_embeddings(_seq(codes)).values
    pure_y = pure.codes_to_embeddings(codes)
    a, r = tol(mx.float32, "leaf")
    assert_close(sl_y, pure_y, atol=a, rtol=r, name="rvq_truncated")
