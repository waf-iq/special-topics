"""Tests for the FastAPI /search service (D2-B1).

Uses a small fixture parquet (60 scifact + 40 arxiv chunks) so the app's
startup — which builds one BM25 + dense index per source — stays fast and
needs no Docker. The real BGE embedder still loads (same as the other smoke
tests), so the run downloads/uses the cached model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from csai415.api import create_app

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")

REQUIRED_FIELDS = {"chunk_id", "paper_id", "title", "text", "page_range", "score"}


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    if not CHUNKS_PARQUET.exists():
        pytest.skip("chunks.parquet not present — run ingest first")
    df = pd.read_parquet(CHUNKS_PARQUET)
    scifact = df[df["source"] == "scifact"].head(60)
    arxiv = df[df["source"].isin({"arxiv", "arxiv-demo"})].head(40)
    slice_df = pd.concat([scifact, arxiv], ignore_index=True)
    if len(scifact) < 30 or len(arxiv) < 30:
        pytest.skip(f"not enough chunks for fixture (scifact={len(scifact)}, arxiv={len(arxiv)})")

    fixture_path = tmp_path_factory.mktemp("api") / "api_chunks.parquet"
    slice_df.to_parquet(fixture_path)

    app = create_app(chunks_path=fixture_path)
    # `with` triggers the lifespan hook so the retrievers actually build.
    with TestClient(app) as c:
        yield c


def test_healthz_ok(client: TestClient):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_returns_topk(client: TestClient):
    resp = client.post("/search", json={"query": "diffusion tensor imaging of white matter", "k": 5})
    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 5
    for h in hits:
        assert REQUIRED_FIELDS <= set(h)
        assert isinstance(h["score"], float)


def test_source_filter_narrows(client: TestClient):
    resp = client.post(
        "/search",
        json={"query": "language model pretraining", "k": 5, "source": "scifact"},
    )
    assert resp.status_code == 200
    hits = resp.json()
    assert hits, "expected at least one scifact hit"
    # chunk_id format is "<source>:<paper_id>:<position>" — every hit must be scifact.
    assert all(h["chunk_id"].startswith("scifact:") for h in hits)


# --- D2-INT2 Qdrant-backed variant ----------------------------------------

pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402

from csai415.api import SOURCE_COLLECTIONS, SOURCE_GROUPS  # noqa: E402
from csai415.qdrant_dense import seed_collection_from_parquet  # noqa: E402


@pytest.fixture(scope="module")
def qdrant_client_seeded(tmp_path_factory) -> tuple[QdrantClient, Path]:
    """In-memory Qdrant pre-seeded with the same three collections the live
    stack carries (full / scifact / arxiv). Verifies the D2-INT2 wiring works
    without needing Docker.
    """
    if not CHUNKS_PARQUET.exists():
        pytest.skip("chunks.parquet not present — run ingest first")
    df = pd.read_parquet(CHUNKS_PARQUET)
    scifact = df[df["source"] == "scifact"].head(60).reset_index(drop=True)
    arxiv = df[df["source"].isin({"arxiv", "arxiv-demo"})].head(40).reset_index(drop=True)
    slice_df = pd.concat([scifact, arxiv], ignore_index=True)
    if len(scifact) < 30 or len(arxiv) < 30:
        pytest.skip(f"not enough chunks for Qdrant fixture (scifact={len(scifact)}, arxiv={len(arxiv)})")

    tmp = tmp_path_factory.mktemp("api_qdrant")
    full_path = tmp / "full.parquet"
    scifact_path = tmp / "scifact.parquet"
    arxiv_path = tmp / "arxiv.parquet"
    slice_df.to_parquet(full_path)
    scifact.to_parquet(scifact_path)
    arxiv.to_parquet(arxiv_path)

    client = QdrantClient(":memory:")
    seed_collection_from_parquet(client, full_path, collection=SOURCE_COLLECTIONS[None])
    seed_collection_from_parquet(client, scifact_path, collection=SOURCE_COLLECTIONS["scifact"])
    seed_collection_from_parquet(client, arxiv_path, collection=SOURCE_COLLECTIONS["arxiv"])
    return client, full_path


@pytest.fixture(scope="module")
def qdrant_client_app(qdrant_client_seeded) -> TestClient:
    client, fixture_path = qdrant_client_seeded
    app = create_app(chunks_path=fixture_path, qdrant_client=client)
    with TestClient(app) as c:
        yield c


def test_qdrant_healthz_ok(qdrant_client_app: TestClient):
    """Health check passes when Qdrant is reachable and the full collection exists."""
    resp = qdrant_client_app.get("/healthz")
    assert resp.status_code == 200


def test_qdrant_search_returns_topk(qdrant_client_app: TestClient):
    """Same /search behavior under the Qdrant backend as numpy — top-5, all fields present."""
    resp = qdrant_client_app.post(
        "/search", json={"query": "diffusion tensor imaging of white matter", "k": 5}
    )
    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 5
    for h in hits:
        assert REQUIRED_FIELDS <= set(h)
        assert isinstance(h["score"], float)


def test_qdrant_source_filter_narrows(qdrant_client_app: TestClient):
    """Per-source Qdrant collection routes correctly — scifact filter yields scifact-only hits."""
    resp = qdrant_client_app.post(
        "/search",
        json={"query": "language model pretraining", "k": 5, "source": "scifact"},
    )
    assert resp.status_code == 200
    hits = resp.json()
    assert hits, "expected at least one scifact hit"
    assert all(h["chunk_id"].startswith("scifact:") for h in hits)
