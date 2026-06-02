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
