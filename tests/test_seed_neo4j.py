"""Tests for the Neo4j graph loader (D2-C1).

Neo4j has no in-memory mode (unlike Qdrant's ``:memory:``), so the loader is
split into pure transform logic (normalize_paper / should_load / parquet
reader) and a thin ``seed_graph`` that drives a duck-typed session. The session
contract is tiny — just ``.run(query, **params)`` — so a fake recorder stands in
for a live database and these tests need no Docker, no Neo4j, no network.

The live-graph reseed + smoke is covered separately by D2-A4's gated stack test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The loader lives in scripts/ (not on the src pythonpath), so load it by path.
_SPEC = importlib.util.spec_from_file_location(
    "seed_neo4j", Path(__file__).resolve().parents[1] / "scripts" / "seed_neo4j.py"
)
seed_neo4j = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seed_neo4j)

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")


class FakeSession:
    """Records ``.run`` calls instead of touching a database."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.calls.append((query, params))


# --- pure transform ------------------------------------------------------


def test_normalize_paper_arxiv():
    raw = {
        "paper_id": "2605.15198v1",
        "title": "A Title",
        "authors": np.array(["Ziyu Guo", "Rain Liu", "Ziyu Guo"]),  # dup + numpy array
        "year": 2026.0,
        "topic": "cs.CL",
        "source": "arxiv-demo",
    }
    p = seed_neo4j.normalize_paper(raw)
    assert p["paper_id"] == "2605.15198v1"
    assert p["year"] == 2026 and isinstance(p["year"], int)
    assert p["source"] == "arxiv"  # arxiv-demo collapsed to the contract enum
    assert p["authors"] == ["Ziyu Guo", "Rain Liu"]  # deduped, order preserved
    assert p["topics"] == ["cs.CL"]  # scalar topic -> list


def test_normalize_paper_scifact_is_empty_and_skipped():
    raw = {
        "paper_id": "4983",
        "title": "Some SciFact paper",
        "authors": None,
        "year": float("nan"),
        "topic": None,
        "source": "scifact",
    }
    p = seed_neo4j.normalize_paper(raw)
    assert p["authors"] == [] and p["topics"] == [] and p["year"] is None
    assert seed_neo4j.should_load(p) is False


def test_topics_list_from_contract_shape():
    # Mongo contract carries "topics" as a list; loader accepts it directly.
    raw = {"paper_id": "x", "authors": ["A"], "topics": ["cs.CL", "cs.AI", "cs.CL"]}
    p = seed_neo4j.normalize_paper(raw)
    assert p["topics"] == ["cs.CL", "cs.AI"]


# --- graph load against a fake session -----------------------------------


def test_seed_graph_applies_constraints_and_merges():
    papers = [
        {"paper_id": "p1", "title": "T1", "authors": ["A", "B"], "year": 2026, "topic": "cs.CL", "source": "arxiv"},
        {"paper_id": "p2", "title": "T2", "authors": None, "year": None, "topic": None, "source": "scifact"},
    ]
    sess = FakeSession()
    stats = seed_neo4j.seed_graph(sess, papers)

    assert stats["papers_loaded"] == 1
    assert stats["papers_skipped"] == 1  # scifact, no authors
    assert stats["authors"] == 2
    assert stats["topics"] == 1
    assert stats["wrote_edges"] == 2
    assert stats["about_edges"] == 1

    # 3 constraint statements + 1 MERGE for the loaded paper; skipped paper writes nothing.
    constraint_calls = [c for c in sess.calls if c[0].startswith("CREATE CONSTRAINT")]
    merge_calls = [c for c in sess.calls if "MERGE (p:Paper" in c[0]]
    assert len(constraint_calls) == 3
    assert len(merge_calls) == 1
    assert merge_calls[0][1]["paper_id"] == "p1"
    assert merge_calls[0][1]["authors"] == ["A", "B"]


def test_seed_graph_idempotent_uses_merge_only():
    # Every write statement must be MERGE-based so reseeds don't duplicate.
    papers = [{"paper_id": "p1", "authors": ["A"], "topic": "cs.CL", "source": "arxiv"}]
    sess = FakeSession()
    seed_neo4j.seed_graph(sess, papers, apply_constraints=False)
    write = next(c for c in sess.calls if "Paper" in c[0])[0]
    assert "CREATE (" not in write
    assert write.count("MERGE") >= 3  # paper, author+edge, topic+edge


# --- dev parquet fallback ------------------------------------------------


def test_parquet_fallback_skips_scifact_and_dedups():
    if not CHUNKS_PARQUET.exists():
        pytest.skip("chunks.parquet not present — run ingest first")
    papers = list(seed_neo4j.iter_papers_from_parquet(CHUNKS_PARQUET, source="arxiv"))
    # 5 arxiv-demo + 2 arxiv unique papers in the current corpus; all have authors.
    assert len(papers) >= 5
    norm = [seed_neo4j.normalize_paper(p) for p in papers]
    assert all(p["source"] == "arxiv" for p in norm)
    assert all(seed_neo4j.should_load(p) for p in norm)

    # Loading them through a fake session must skip nothing.
    sess = FakeSession()
    stats = seed_neo4j.seed_graph(sess, papers)
    assert stats["papers_skipped"] == 0
    assert stats["papers_loaded"] == len(papers)
    assert stats["authors"] > 0
