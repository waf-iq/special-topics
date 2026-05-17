"""Smoke tests for D1.

Checks the package imports, the eval/runcard contracts haven't drifted,
and the NDCG@5 math is correct against hand-computed cases.
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


# NDCG@5 tests with hand-computed values (Q1 from my AI session)

def test_ndcg_known_partial_hit():
    # 3 relevant docs, retriever returns 1 of them at rank 2
    # NDCG@5 = (1/log2(3)) / (1 + 1/log2(3) + 1/log2(4)) ~ 0.296
    from csai415.eval import ndcg_at_k
    retrieved = ["other_1", "rel_a", "other_2", "other_3", "other_4"]
    relevant = {"rel_a", "rel_b", "rel_c"}
    assert abs(ndcg_at_k(retrieved, relevant, k=5) - 0.2960) < 0.001


def test_ndcg_perfect_top_ranking():
    from csai415.eval import ndcg_at_k
    retrieved = ["rel_a", "rel_b", "rel_c", "other_1", "other_2"]
    relevant = {"rel_a", "rel_b", "rel_c"}
    assert ndcg_at_k(retrieved, relevant, k=5) == 1.0


def test_ndcg_zero_relevance():
    from csai415.eval import ndcg_at_k
    assert ndcg_at_k(["a", "b", "c"], set(), k=5) == 0.0


def test_ndcg_relevant_docs_exceed_k():
    # 7 relevant, all 5 retrieved are relevant -> NDCG=1.0
    # if IDCG is computed over all 7 (no min cap), this fails
    from csai415.eval import ndcg_at_k
    retrieved = ["rel_1", "rel_2", "rel_3", "rel_4", "rel_5"]
    relevant = {"rel_1", "rel_2", "rel_3", "rel_4", "rel_5", "rel_6", "rel_7"}
    assert ndcg_at_k(retrieved, relevant, k=5) == 1.0


def test_ndcg_retrieved_shorter_than_k():
    # robustness: retriever returns fewer than k items
    from csai415.eval import ndcg_at_k
    retrieved = ["rel_a"]
    relevant = {"rel_a", "rel_b"}
    score = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 < score < 1.0


@pytest.mark.xfail(reason="Ingest not yet implemented")
def test_ingest_produces_chunks():
    from pathlib import Path
    assert Path("data/processed/chunks.parquet").exists()
