"""Pair B — FastAPI ``/search`` service. See D2_TASKS.md §D2-B1, D2-INT2.

Exposes the D1 hybrid retriever (BM25 + dense) over HTTP. The blessed BOHB
config is loaded once at startup from ``configs/winning_runcard.yaml`` — never
from the request body. The request only carries ``query``, ``k`` and an
optional ``source`` filter.

Source filtering (D3 refactor) uses ONE retriever over the whole corpus plus a
query-time ``source`` filter passed down to both backends. Filtering at
candidate-generation time — rather than post-filtering ``/search`` results —
keeps SciFact eval honest: arXiv chunks never enter the candidate pool, so they
can't push real SciFact answers out of the top-k (D2 brief, Risk #4).

Dense backend (D2-INT2 / D3): when ``CSAI415_USE_QDRANT=1`` the retriever is
wired to a single Qdrant collection (``chunks_bge384``, seeded from the full
parquet so point IDs are the global corpus_idx). The ``source`` filter is applied
at query time via the payload ``source`` field — collapsing D2's three
per-source collections into one (the locked D3 "1 collection + filter-at-query-
time" decision). The BM25 side mirrors the filter in-process.

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

from csai415.graphrag import GraphRAGExecutor
from csai415.qdrant_dense import QDRANT_COLLECTION
from csai415.retrieve import CHUNKS_PARQUET, HybridRetriever, RetrieverConfig, load_chunks

DEFAULT_RUNCARD = Path("configs/winning_runcard.yaml")

# A request's ``source`` value -> the parquet ``source`` labels it covers.
# arxiv-demo is folded into "arxiv" (same provenance, just the original 5 PDFs).
# Passed as ``source_labels`` to the retriever for query-time filtering.
SOURCE_GROUPS: dict[str, set[str]] = {
    "scifact": {"scifact"},
    "arxiv": {"arxiv", "arxiv-demo"},
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


# --- D3: GraphRAG /ask contract (see briefs/D3_TASKS.md, Hour 0) -----------------------
class CitationModel(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    page_range: Optional[str]
    score: float


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question.")
    k: int = Field(5, ge=1, le=50, description="Number of cited sources to ground on.")
    mode: str = Field("hybrid", description="Retrieval mode: 'vector', 'graph', or 'hybrid'.")
    rerank: bool = Field(True, description="Apply the cross-encoder rerank stage.")


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    contexts: list[str]
    mode: str
    rerank: bool
    latency_ms: float
    steps: list[dict]


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

    Dense backend (D3): if ``qdrant_client`` is passed in (test-time injection)
    or ``CSAI415_USE_QDRANT=1`` (production), the single retriever is wired to the
    one ``chunks_bge384`` collection; ``source`` filtering happens at query time.
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

        # D3: ONE retriever over the whole corpus. Source restriction is a query-time
        # filter (see /search), so point IDs stay the global corpus_idx and a single
        # Qdrant collection backs everything.
        if client is None:
            retriever = HybridRetriever(df, cfg)
        else:
            from csai415.qdrant_dense import QdrantDenseBackend

            backend = QdrantDenseBackend(client, collection=QDRANT_COLLECTION, metric=cfg.metric)
            retriever = HybridRetriever(df, cfg, dense_backend=backend)

        app.state.retriever = retriever
        # chunk_id -> row, over the FULL corpus, for joining metadata onto hits.
        app.state.lookup = df.set_index("chunk_id")
        app.state.config = cfg
        app.state.qdrant_client = client

        # D3 (Task 7): wire the live Neo4j driver into the executor so graph/hybrid
        # modes can query the subgraph. Gated by CSAI415_USE_NEO4J so the test suite
        # and offline runs stay graph-less (select_subgraph degrades to fallback when
        # neo4j_driver is None). Empty password ⇒ auth=None (dev containers).
        neo4j_driver = None
        if os.environ.get("CSAI415_USE_NEO4J", "0") == "1":
            from neo4j import GraphDatabase

            url = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            pwd = os.environ.get("NEO4J_PASSWORD", "")
            auth = (user, pwd) if pwd else None
            neo4j_driver = GraphDatabase.driver(url, auth=auth)
        app.state.neo4j_driver = neo4j_driver
        app.state.executor = GraphRAGExecutor(retriever, app.state.lookup, neo4j_driver=neo4j_driver)
        yield
        if neo4j_driver is not None:
            neo4j_driver.close()

    app = FastAPI(title="CSAI415 PDF-Papers /search", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        """200 once the retriever is loaded and (when wired) Qdrant is reachable.

        Mongo / Neo4j reachability isn't checked here because /search doesn't
        depend on them at request time — only ingest + seed do.
        """
        if not getattr(app.state, "retriever", None):
            raise HTTPException(status_code=503, detail="retriever not loaded")
        client = getattr(app.state, "qdrant_client", None)
        if client is not None:
            try:
                client.get_collection(QDRANT_COLLECTION)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"qdrant collection {QDRANT_COLLECTION!r} unreachable: {exc!s}",
                ) from exc
        return {"status": "ok"}

    @app.post("/search", response_model=list[SearchHit])
    def search(req: SearchRequest):
        if req.source is not None and req.source not in SOURCE_GROUPS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown source {req.source!r}; expected one of "
                f"{sorted(SOURCE_GROUPS)} or null",
            )
        retriever: HybridRetriever = app.state.retriever
        lookup: pd.DataFrame = app.state.lookup

        source_labels = SOURCE_GROUPS[req.source] if req.source is not None else None
        hits = retriever.search_with_scores(req.query, req.k, source_labels=source_labels)
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

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        """GraphRAG executor endpoint (D3). Returns a grounded answer with [n] citations.

        The blessed retrieval config + mode policy live server-side; the request only
        carries the question, k, mode, and the rerank toggle.
        """
        executor: GraphRAGExecutor = getattr(app.state, "executor", None)
        if executor is None:
            raise HTTPException(status_code=503, detail="executor not loaded")
        if req.mode not in ("vector", "graph", "hybrid"):
            raise HTTPException(
                status_code=400, detail=f"unknown mode {req.mode!r}; expected vector|graph|hybrid"
            )
        res = executor.answer(req.query, k=req.k, mode=req.mode, rerank=req.rerank)
        return AskResponse(
            answer=res.answer,
            citations=[CitationModel(**vars(c)) for c in res.citations],
            contexts=res.contexts,
            mode=res.mode,
            rerank=res.rerank,
            latency_ms=res.latency_ms,
            steps=res.steps,
        )

    return app


# Module-level app for uvicorn / Docker (`uvicorn csai415.api:app`).
app = create_app()
