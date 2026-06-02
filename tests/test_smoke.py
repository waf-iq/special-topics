"""Smoke tests for D1.

Checks the package imports, the eval/runcard contracts haven't drifted,
and the NDCG@5 math is correct against hand-computed cases.
"""

from __future__ import annotations

import inspect
import os

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


def test_runcard_extended_fields(tmp_path):
    from csai415.runcard import write_runcard
    import yaml

    out = tmp_path / "card.yaml"
    write_runcard(
        best_params={"candidate_k": 10, "metric": "cosine"},
        best_value=0.72,
        n_trials=80,
        embedding_model="BAAI/bge-small-en-v1.5",
        chunks_parquet=tmp_path / "chunks.parquet",
        gold_jsonl=tmp_path / "gold.jsonl",
        metrics={"ndcg5_tune": 0.72, "ndcg5_holdout": 0.68},
        split={"strategy": "80_20", "split_seed": 42},
        sampler_config={"class": "TPESampler", "multivariate": True, "seed": 42},
        pruner_config={"class": "NopPruner"},
        study_storage="sqlite:///study.db",
        notes="single-seed study; 5-fold CV deferred to D2",
        out_path=out,
    )

    card = yaml.safe_load(out.read_text())
    assert card["schema_version"] == "3"   # bumped in D1-rework B3
    assert card["automl"]["sampler"]["multivariate"] is True
    assert card["automl"]["storage"] == "sqlite:///study.db"
    assert card["split"]["strategy"] == "80_20"
    assert card["notes"].startswith("single-seed")
    assert "env" in card and "code" in card
    assert card["dataset"]["chunks_sha256"] is None
    # v3 rework fields are optional and absent here (this test calls
    # write_runcard without the rework kwargs)
    assert "blessed_method" not in card["automl"]
    assert "comparison" not in card["automl"]


# --- D2-A4 live-stack integration smoke (gated) -------------------------

D2_STACK_UP = os.environ.get("D2_STACK_UP", "0") == "1"


@pytest.mark.skipif(
    not D2_STACK_UP,
    reason="D2 stack not running — set D2_STACK_UP=1 with docker compose up",
)
def test_d2_stack_smoke():
    """End-to-end: FastAPI /search + Mongo + Qdrant + Neo4j all reachable
    and populated. Run with `D2_STACK_UP=1 pytest tests/test_smoke.py -v`
    after `docker compose up -d` and seeding (Musab's D2-INT1 + D2-INT2 reseed).
    """
    import httpx
    from neo4j import GraphDatabase
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    # 1. FastAPI /healthz returns 200 (also pings Qdrant when CSAI415_USE_QDRANT=1)
    r = httpx.get("http://localhost:8000/healthz", timeout=5)
    assert r.status_code == 200, f"/healthz: {r.status_code} {r.text}"

    # 2. /search returns top-k with required fields and a plausible scifact hit
    r = httpx.post(
        "http://localhost:8000/search",
        json={"query": "diffusion tensor imaging of white matter", "k": 5},
        timeout=10,
    )
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 5
    required = {"chunk_id", "paper_id", "title", "text", "page_range", "score"}
    for h in hits:
        assert required <= set(h)

    # 3. /search source filter routes correctly — all hits must be scifact-sourced
    r = httpx.post(
        "http://localhost:8000/search",
        json={"query": "language model pretraining", "k": 5, "source": "scifact"},
        timeout=10,
    )
    assert r.status_code == 200
    assert all(h["chunk_id"].startswith("scifact:") for h in r.json())

    # 4. Mongo papers + chunks collections populated (counts from D2-INT1 verification)
    mc = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    db = mc.csai415
    assert db.papers.count_documents({}) > 5000, "papers undersized — re-seed?"
    assert db.chunks.count_documents({}) > 15000, "chunks undersized — re-seed?"
    mc.close()

    # 5. Qdrant has all three collections per D2-INT2
    qc = QdrantClient(url="http://localhost:6333")
    coll_names = {c.name for c in qc.get_collections().collections}
    for needed in ("chunks_bge384", "chunks_bge384_scifact", "chunks_bge384_arxiv"):
        assert needed in coll_names, f"missing collection {needed!r}"
    assert qc.get_collection("chunks_bge384").points_count > 15000

    # 6. Neo4j has Paper + Author + Topic nodes (from D2-C1 seed)
    drv = GraphDatabase.driver("bolt://localhost:7687", auth=None)
    with drv.session() as sess:
        counts = sess.run(
            "MATCH (p:Paper) WITH count(p) AS papers "
            "MATCH (a:Author) WITH papers, count(a) AS authors "
            "MATCH (t:Topic) RETURN papers, authors, count(t) AS topics"
        ).single()
    drv.close()
    assert counts["papers"] > 100, "Paper nodes undersized"
    assert counts["authors"] > 500, "Author nodes undersized"
    assert counts["topics"] >= 5, "Topic nodes undersized"
