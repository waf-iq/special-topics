"""Pair B — FastAPI ``/search`` service. See D2_TASKS.md §D2-B1, D2-INT2.

Exposes the D1 hybrid retriever (BM25 + dense) over HTTP. The blessed BOHB
config is loaded once at startup from ``configs/winning_runcard.yaml`` — never
from the request body. The request only carries ``query``, ``k`` and an
optional ``source`` filter.

Source filtering uses one retriever per source built at startup (``None`` =
whole corpus, ``"scifact"``, ``"arxiv"``). Filtering at candidate-generation
time — rather than post-filtering ``/search`` results — keeps SciFact eval
honest: arXiv chunks never enter the candidate pool, so they can't push real
SciFact answers out of the top-k (D2 brief, Risk #4).

Dense backend (D2-INT2): when ``CSAI415_USE_QDRANT=1`` each per-source
retriever is wired to a separate Qdrant collection (``chunks_bge384`` for the
full corpus, ``chunks_bge384_{scifact,arxiv}`` for the subsets). The
per-collection layout sidesteps a corpus_idx mismatch — each collection's
point IDs match the reset-indexed local df of the retriever consuming it.
**D3 will refactor this to one global collection + filter-at-query-time**
(plan B in the design notes); for D2 we keep three collections to avoid
touching the just-merged per-source routing logic.

Without the env flag the app falls back to the numpy in-memory backend (D1
default) so the existing test suite keeps working without Docker.

Runs under uvicorn as ``csai415.api:app`` (see Dockerfile / docker-compose).
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from csai415.retrieve import CHUNKS_PARQUET, HybridRetriever, RetrieverConfig, load_chunks

DEFAULT_RUNCARD = Path("configs/winning_runcard.yaml")

# A request's ``source`` value -> the parquet ``source`` labels it covers.
# arxiv-demo is folded into "arxiv" (same provenance, just the original 5 PDFs).
SOURCE_GROUPS: dict[str, set[str]] = {
    "scifact": {"scifact"},
    "arxiv": {"arxiv", "arxiv-demo"},
}

# Per-source Qdrant collection names. Each collection is seeded with that
# source's reset-indexed subset of chunks.parquet so point IDs match the
# retriever's local df. See module docstring for the D3 refactor note.
SOURCE_COLLECTIONS: dict[Optional[str], str] = {
    None: "chunks_bge384",
    "scifact": "chunks_bge384_scifact",
    "arxiv": "chunks_bge384_arxiv",
}


def load_blessed_config(path: Path | str = DEFAULT_RUNCARD) -> RetrieverConfig:
    """Read the frozen BOHB winner from the run-card into a ``RetrieverConfig``.

    The config is loaded at startup, not per request — clients can't re-tune
    the retriever through ``/search``.
    """
    with open(path, encoding="utf-8") as f:
        card = yaml.safe_load(f)
    bp = card["automl"]["best_params"]
    return RetrieverConfig(
        metric=bp["metric"],
        svd_dim=bp["svd_dim"],
        normalize=bp["normalize"],
        hybrid_weight=bp["hybrid_weight"],
        candidate_k=bp["candidate_k"],
        bm25_k1=bp["bm25_k1"],
        bm25_b=bp["bm25_b"],
        seed=card.get("seed", 42),
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query.")
    k: int = Field(5, ge=1, le=100, description="Number of hits to return.")
    source: Optional[str] = Field(
        None, description="Restrict to a corpus: 'scifact', 'arxiv', or null for all."
    )


class SearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    text: str
    page_range: Optional[str]
    score: float


def _page_range(row: pd.Series) -> str | None:
    """Format ``page_start``/``page_end`` as 'start-end' (or 'start' when equal).

    SciFact chunks carry no page info (NaN), so they return None.
    """
    ps = row.get("page_start")
    if ps is None or (isinstance(ps, float) and math.isnan(ps)):
        return None
    ps = int(ps)
    pe = row.get("page_end")
    pe = int(pe) if pe is not None and not (isinstance(pe, float) and math.isnan(pe)) else ps
    return str(ps) if pe == ps else f"{ps}-{pe}"


def _clean_str(value) -> str:
    """Coerce a possibly-NaN/None cell to a plain string for the response."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def create_app(
    chunks_path: Path | str | None = None,
    config_path: Path | str | None = None,
    qdrant_client=None,
) -> FastAPI:
    """Build the FastAPI app.

    Paths default to the production locations but can be overridden (used by the
    tests to point at a small fixture parquet so startup stays fast). Env vars
    ``CSAI415_CHUNKS_PARQUET`` and ``CSAI415_RUNCARD`` override the defaults.

    Dense backend (D2-INT2): if ``qdrant_client`` is passed in (test-time
    injection) or ``CSAI415_USE_QDRANT=1`` (production), each per-source
    retriever is wired to its Qdrant collection per ``SOURCE_COLLECTIONS``.
    Otherwise the numpy in-memory backend is used (D1 default).
    """
    chunks_path = Path(
        chunks_path or os.environ.get("CSAI415_CHUNKS_PARQUET", CHUNKS_PARQUET)
    )
    config_path = Path(config_path or os.environ.get("CSAI415_RUNCARD", DEFAULT_RUNCARD))

    use_qdrant_env = os.environ.get("CSAI415_USE_QDRANT", "0") == "1"
    injected_client = qdrant_client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = load_blessed_config(config_path)
        df = load_chunks(chunks_path)

        # Resolve dense-backend strategy. Explicit injection wins; otherwise
        # the env flag controls production behavior.
        client = injected_client
        if client is None and use_qdrant_env:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333")
            )

        def _build_retriever(sub_df: pd.DataFrame, source_key: Optional[str]) -> HybridRetriever:
            if client is None:
                return HybridRetriever(sub_df, cfg)
            from csai415.qdrant_dense import QdrantDenseBackend

            backend = QdrantDenseBackend(
                client,
                collection=SOURCE_COLLECTIONS[source_key],
                metric=cfg.metric,
            )
            return HybridRetriever(sub_df, cfg, dense_backend=backend)

        # One retriever over the whole corpus (source=None) plus one per source.
        # Each per-source retriever is built over a clean subset, so BOTH its
        # BM25 and dense candidate pools are already restricted to that source.
        retrievers: dict[Optional[str], HybridRetriever] = {
            None: _build_retriever(df, None)
        }
        for src, labels in SOURCE_GROUPS.items():
            sub = df[df["source"].isin(labels)].reset_index(drop=True)
            if len(sub):
                retrievers[src] = _build_retriever(sub, src)

        app.state.retrievers = retrievers
        # chunk_id -> row, over the FULL corpus, for joining metadata onto hits.
        app.state.lookup = df.set_index("chunk_id")
        app.state.config = cfg
        app.state.qdrant_client = client
        yield

    app = FastAPI(title="CSAI415 PDF-Papers /search", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        """200 once the retriever is loaded and (when wired) Qdrant is reachable.

        Mongo / Neo4j reachability isn't checked here because /search doesn't
        depend on them at request time — only ingest + seed do.
        """
        if not getattr(app.state, "retrievers", None):
            raise HTTPException(status_code=503, detail="retriever not loaded")
        client = getattr(app.state, "qdrant_client", None)
        if client is not None:
            try:
                client.get_collection(SOURCE_COLLECTIONS[None])
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"qdrant unreachable: {exc!s}",
                ) from exc
        return {"status": "ok"}

    @app.post("/search", response_model=list[SearchHit])
    def search(req: SearchRequest):
        retrievers: dict = app.state.retrievers
        if req.source is not None and req.source not in retrievers:
            raise HTTPException(
                status_code=400,
                detail=f"unknown source {req.source!r}; expected one of "
                f"{sorted(s for s in retrievers if s)} or null",
            )
        retriever: HybridRetriever = retrievers[req.source]
        lookup: pd.DataFrame = app.state.lookup

        hits = retriever.search_with_scores(req.query, req.k)
        out: list[SearchHit] = []
        for chunk_id, score in hits:
            row = lookup.loc[chunk_id]
            out.append(
                SearchHit(
                    chunk_id=chunk_id,
                    paper_id=_clean_str(row["paper_id"]),
                    title=_clean_str(row["title"]),
                    text=_clean_str(row["text"]),
                    page_range=_page_range(row),
                    score=score,
                )
            )
        return out

    return app


# Module-level app for uvicorn / Docker (`uvicorn csai415.api:app`).
app = create_app()
