"""Smoke tests for D3 Task 2 reranking (csai415.rerank).

The default-path tests run anywhere (no model download). The real cross-encoder / MMR
checks load models, so they are gated behind ``RUN_RERANK_MODEL=1`` (mirrors the
``D2_STACK_UP`` gating in ``tests/test_smoke.py``) to keep the default suite offline & fast.
"""

from __future__ import annotations

import os

import pytest

from csai415.rerank import get_reranker, rerank

# A tiny fixture pool: c1 is on-topic for the query, c2/c3 are not.
_QUERY = "what is the attention mechanism in transformers"
_TEXT = {
    "c1": "The attention mechanism lets a transformer weigh all tokens when encoding each one.",
    "c2": "Photosynthesis converts sunlight into chemical energy in plant cells.",
    "c3": "The mitochondrion is the powerhouse of the cell.",
}
_IDS = ["c2", "c3", "c1"]  # deliberately mis-ordered so a real reranker must move c1 up


def test_default_rerank_delegates_to_blessed(monkeypatch):
    # rerank() uses whatever _BLESSED_RERANKER names (production may be a real model,
    # exercised by the gated tests below). Force "none" for a hermetic, model-free check
    # of the delegation + contract shape.
    monkeypatch.setattr("csai415.rerank._BLESSED_RERANKER", "none")
    out = rerank(_QUERY, _IDS, _TEXT, top_n=2)
    assert out == ["c2", "c3"]
    assert len(out) <= 2 and set(out) <= set(_IDS)


def test_blessed_failure_falls_back_to_retrieval_order(monkeypatch):
    # A blessed model that can't load must degrade to retrieval order, never crash.
    monkeypatch.setattr("csai415.rerank._BLESSED_RERANKER", "bge")
    monkeypatch.setattr(
        "csai415.rerank._load_cross_encoder",
        lambda name: (_ for _ in ()).throw(RuntimeError("simulated outage")),
    )
    out = rerank(_QUERY, _IDS, _TEXT, top_n=2)
    assert out == ["c2", "c3"]  # original order kept, no exception


def test_none_backend_matches_default():
    assert get_reranker("none")(_QUERY, _IDS, _TEXT, top_n=2) == ["c2", "c3"]


def test_empty_pool_is_safe():
    assert rerank(_QUERY, [], {}, top_n=5) == []


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        get_reranker("bogus")


@pytest.mark.parametrize("kind", ["minilm", "bge", "mmr"])
def test_get_reranker_builds_without_loading(kind):
    # Building a reranker must not download a model — only calling it does.
    assert callable(get_reranker(kind))


@pytest.mark.skipif(
    os.environ.get("RUN_RERANK_MODEL") != "1",
    reason="set RUN_RERANK_MODEL=1 to run model-backed reranker checks (downloads models)",
)
@pytest.mark.parametrize("kind", ["minilm", "bge", "mmr"])
def test_real_reranker_surfaces_relevant_chunk(kind):
    out = get_reranker(kind)(_QUERY, _IDS, _TEXT, top_n=3)
    assert out[0] == "c1"  # the on-topic chunk should be ranked first
    assert sorted(out) == sorted(_IDS)  # a reordering, not a drop
