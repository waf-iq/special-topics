"""Smoke tests for D1.

Checks the package imports and the eval/runcard contracts haven't drifted.
NDCG correctness tests get added by Pair A in a follow-up commit.
"""

from __future__ import annotations

import inspect

import pytest


def test_package_imports():
    import csai415
    from csai415 import automl, eval as eval_mod, ingest, online, retrieve, runcard


def test_evaluate_signature():
    from csai415.eval import evaluate
    sig = inspect.signature(evaluate)
    params = list(sig.parameters)
    assert params[:3] == ["retriever_fn", "queries", "k"], (
        f"eval.evaluate signature changed. Got {params}"
    )


def test_retriever_config_fields():
    from csai415.retrieve import RetrieverConfig
    rc = RetrieverConfig()
    for field in ("metric", "svd_dim", "normalize", "hybrid_weight", "seed"):
        assert hasattr(rc, field), f"RetrieverConfig missing {field}"


def test_evaluate_with_fake_retriever():
    from csai415.eval import evaluate

    def fake_retriever(query, k, hybrid_weight):
        return ["c1", "c2", "c3", "c4", "c5"][:k]

    queries = [
        {"qid": "q1", "question": "test?", "relevant_chunk_ids": ["c1"], "topic": "x"},
        {"qid": "q2", "question": "test?", "relevant_chunk_ids": ["c99"], "topic": "x"},
    ]
    out = evaluate(fake_retriever, queries, k=5)
    assert set(out.keys()) == {"ndcg5", "recall5", "p95_latency_ms"}
    assert 0.0 <= out["ndcg5"] <= 1.0
    assert 0.0 <= out["recall5"] <= 1.0
    assert out["p95_latency_ms"] >= 0.0


def test_runcard_write(tmp_path):
    from csai415.runcard import write_runcard

    out = tmp_path / "card.yaml"
    write_runcard(
        best_params={"k": 10, "metric": "cosine", "svd_dim": 128, "normalize": True, "hybrid_weight": 0.6},
        best_value=0.72,
        n_trials=60,
        embedding_model="sentence-transformers/bge-small-en",
        chunks_parquet=tmp_path / "nope.parquet",
        gold_jsonl=tmp_path / "nope.jsonl",
        metrics={"ndcg5": 0.72, "recall5": 0.65, "p95_latency_ms": 180.0,
                 "baseline_ndcg5": 0.58, "baseline_recall5": 0.5, "baseline_p95_latency_ms": 140.0},
        out_path=out,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "best_params" in text
    assert "hybrid_weight" in text


@pytest.mark.xfail(reason="Ingest not yet implemented")
def test_ingest_produces_chunks():
    from pathlib import Path
    assert Path("data/processed/chunks.parquet").exists()
