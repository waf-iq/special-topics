"""Smoke tests for the Qdrant-backed dense backend (D2-B2).

Parity test: a HybridRetriever wired to QdrantDenseBackend must return the
same top-5 chunk_ids as a HybridRetriever wired to NumpyDenseBackend on a
shared corpus slice. This is the contract for the FastAPI app's dense backend
swap (D2-INT2) — if parity holds on 200 chunks with 5 SciFact queries, the
swap won't regress against D1's holdout metrics.

Uses ``QdrantClient(":memory:")`` so the test needs no Docker, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csai415.retrieve import HybridRetriever, RetrieverConfig


pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402

from csai415.qdrant_dense import QdrantDenseBackend, seed_collection_from_parquet  # noqa: E402

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
QA_JSONL = Path("data/gold/qa.jsonl")

# Blessed BOHB config from configs/winning_runcard.yaml. Drop SVD/normalize off
# (Qdrant backend doesn't support either; D2 production path skips them too).
BLESSED = RetrieverConfig(
    metric="l2",
    svd_dim=None,
    normalize=False,
    hybrid_weight=0.7765905795848593,
    candidate_k=27,
    bm25_k1=2.9199897374940935,
    bm25_b=0.34523676127036784,
)


@pytest.fixture(scope="module")
def corpus_slice() -> pd.DataFrame:
    if not CHUNKS_PARQUET.exists():
        pytest.skip("chunks.parquet not present — run ingest first")
    df = pd.read_parquet(CHUNKS_PARQUET)
    # 200 chunks mixing scifact + arxiv so source-filter behavior is exercised.
    scifact = df[df["source"] == "scifact"].head(150)
    arxiv = df[df["source"].isin({"arxiv", "arxiv-demo"})].head(50)
    out = pd.concat([scifact, arxiv], ignore_index=True)
    if len(out) < 50:
        pytest.skip(f"Not enough chunks to run parity test (got {len(out)})")
    return out


@pytest.fixture(scope="module")
def sample_queries() -> list[str]:
    if not QA_JSONL.exists():
        pytest.skip("qa.jsonl not present — run build_gold_from_scifact first")
    qs: list[str] = []
    with QA_JSONL.open(encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line)["question"])
            if len(qs) >= 5:
                break
    return qs


@pytest.fixture(scope="module")
def seeded_client(corpus_slice: pd.DataFrame, tmp_path_factory) -> QdrantClient:
    # Persist the slice to a temp parquet so seed_collection_from_parquet
    # reads from the same path shape it does in production.
    tmp = tmp_path_factory.mktemp("qdrant_parity")
    slice_path = tmp / "slice.parquet"
    corpus_slice.to_parquet(slice_path)
    client = QdrantClient(":memory:")
    n = seed_collection_from_parquet(client, slice_path)
    assert n == len(corpus_slice)
    return client


def test_qdrant_backend_smoke(seeded_client: QdrantClient, corpus_slice: pd.DataFrame):
    """Backend instantiation + top_k + scores_for return sane shapes."""
    backend = QdrantDenseBackend(seeded_client, metric="l2")
    assert backend.n_points == len(corpus_slice)

    q = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    hits = backend.top_k(q, k=5)
    assert len(hits) == 5 * 2  # over-fetch
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in hits)

    idx = np.array([i for i, _ in hits], dtype=np.int64)
    scores = backend.scores_for(q, idx)
    assert scores.shape == (len(idx),)


def test_qdrant_source_filter(seeded_client: QdrantClient):
    """source filter restricts the candidate pool."""
    backend_sf = QdrantDenseBackend(seeded_client, metric="l2", source_filter="scifact")
    q = np.random.default_rng(1).standard_normal(384).astype(np.float32)
    hits = backend_sf.top_k(q, k=5)
    assert len(hits) > 0
    # All returned points must be scifact — verified by retrieving payload
    points = seeded_client.retrieve(
        collection_name="chunks_bge384",
        ids=[i for i, _ in hits],
        with_payload=True,
    )
    assert all(p.payload["source"] == "scifact" for p in points)


def test_numpy_vs_qdrant_parity(
    seeded_client: QdrantClient,
    corpus_slice: pd.DataFrame,
    sample_queries: list[str],
):
    """Top-5 chunk_ids must match between numpy and Qdrant backends.

    The blessed L2 metric is enforced at fusion time in both paths, so the
    cosine-ANN candidate pool's only failure mode is missing an L2-best
    candidate. On 200 chunks with candidate_k=27 (over-fetch=54), the pool
    covers ~27% of the corpus — overlap is essentially guaranteed.
    """
    numpy_retriever = HybridRetriever(corpus_slice, BLESSED)

    qdrant_backend = QdrantDenseBackend(seeded_client, metric="l2")
    qdrant_retriever = HybridRetriever(corpus_slice, BLESSED, dense_backend=qdrant_backend)

    mismatches = []
    for q in sample_queries:
        numpy_top = numpy_retriever.search(q, k=5)
        qdrant_top = qdrant_retriever.search(q, k=5)
        if numpy_top != qdrant_top:
            mismatches.append({"query": q[:60], "numpy": numpy_top, "qdrant": qdrant_top})

    assert not mismatches, f"top-5 parity broke on {len(mismatches)} of {len(sample_queries)} queries: {mismatches}"
