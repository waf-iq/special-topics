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

import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from csai415.graphrag import GraphRAGExecutor
from csai415.qdrant_dense import QDRANT_COLLECTION
from csai415.retrieve import CHUNKS_PARQUET, HybridRetriever, RetrieverConfig, load_chunks

DEFAULT_RUNCARD = Path("configs/winning_runcard.yaml")
STATIC_DIR = Path(__file__).parent / "static"  # 1-page demo UI served at GET /

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


class AnswererResult(BaseModel):
    answerer: str
    answer: str
    gen_latency_ms: float
    error: Optional[str] = None


class CompareResponse(BaseModel):
    """Same retrieved evidence, answered by each configured answerer (Qwen vs Groq)."""

    query: str
    mode: str
    rerank: bool
    citations: list[CitationModel]
    contexts: list[str]
    steps: list[dict]
    retrieve_latency_ms: float
    results: list[AnswererResult]


class JudgeRequest(BaseModel):
    """A streamed answer + its evidence, to be scored by the RAGAS/Groq judge (T4)."""

    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)


class JudgeResponse(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    available: bool = True
    detail: Optional[str] = None


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

    @app.get("/", response_class=HTMLResponse)
    def index():
        """The 1-page GraphRAG demo UI (vanilla HTML/JS calling /ask)."""
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="UI not found")
        return page.read_text(encoding="utf-8")

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
        from csai415.answer import current_backend

        return {"status": "ok", "answerer": current_backend()}

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

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        """Server-Sent-Events twin of ``/ask`` — emits the pipeline live for the demo UI.

        Event stream (each ``event:``/``data:`` frame is JSON):
          ``step``     one per retrieval stage as it completes (retrieve→subgraph→graph_filter
                       →rerank), carrying the funnel counts + graph entities the UI animates.
          ``blocked``  the T6 injection guard rejected the query *before* any model ran — this
                       is the live "break-it" moment; no further events follow.
          ``evidence`` the final citations + contexts (so the cite panel fills before generation).
          ``token``    a generated text delta (token-by-token answer reveal).
          ``done``     end-of-stream: total latency + the parsed ``[n]`` citations actually used.
          ``error``    an exception bubbled up mid-stream (e.g. Ollama down).
        Generation runs through the same ``CSAI415_ANSWERER`` seam as ``/ask``, so base / tuned /
        Groq all stream through here unchanged.
        """
        executor: GraphRAGExecutor = getattr(app.state, "executor", None)
        if executor is None:
            raise HTTPException(status_code=503, detail="executor not loaded")
        if req.mode not in ("vector", "graph", "hybrid"):
            raise HTTPException(
                status_code=400, detail=f"unknown mode {req.mode!r}; expected vector|graph|hybrid"
            )

        from csai415.answer import _parse_citations, stream_answer
        from csai415.safety import detect_injection_regex

        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def event_stream():
            t0 = time.perf_counter()
            try:
                # T6 safety gate first — a blocked query never reaches retrieval or the model.
                blocked, reason = detect_injection_regex(req.query)
                if blocked:
                    yield _sse("blocked", {"reason": reason})
                    return

                pipe = executor.run_pipeline(req.query, k=req.k, mode=req.mode, rerank=req.rerank)
                citations, contexts = [], []
                try:
                    while True:
                        yield _sse("step", next(pipe))
                except StopIteration as stop:
                    citations, contexts, _ = stop.value

                yield _sse(
                    "evidence",
                    {
                        "citations": [vars(c) for c in citations],
                        "contexts": contexts,
                    },
                )

                acc = ""
                for delta in stream_answer(req.query, citations, contexts):
                    acc += delta
                    yield _sse("token", {"t": delta})

                yield _sse(
                    "done",
                    {
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "cited": _parse_citations(acc, len(contexts)),
                    },
                )
            except Exception as exc:  # surface any mid-stream failure to the UI
                yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/judge", response_model=JudgeResponse)
    def judge(req: JudgeRequest):
        """Score one streamed answer with the real RAGAS faithfulness/relevance judge (T4).

        Deliberately *off* the hot path: the UI calls this after the answer finishes streaming
        and shows a live badge, so the pipeline stays fast. The judge needs ``GROQ_API_KEY`` +
        the ragas deps; when either is missing (or the answer is a refusal) it returns
        ``available=false`` with a reason rather than failing — the badge just shows "unavailable".
        """
        from csai415.answer import REFUSAL

        if not req.contexts or req.answer.strip() == REFUSAL:
            return JudgeResponse(available=False, detail="no answer/context to judge")
        if not os.environ.get("GROQ_API_KEY"):
            return JudgeResponse(available=False, detail="GROQ_API_KEY not set")
        try:
            from csai415.ragas_groq import ragas_score

            scores = ragas_score([req.query], [req.answer], [list(req.contexts)])
            return JudgeResponse(
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
            )
        except Exception as exc:  # ragas/dep/network failure → graceful "unavailable"
            return JudgeResponse(available=False, detail=f"{type(exc).__name__}: {exc}")

    @app.post("/compare", response_model=CompareResponse)
    def compare(req: AskRequest):
        """Retrieve ONCE, then answer the same evidence with each configured answerer
        (``CSAI415_COMPARE``, default ``qwen2.5:3b-instruct,groq``) — the zero-shot /
        ceiling side-by-side. Generation latency is per-model on identical contexts.
        """
        executor: GraphRAGExecutor = getattr(app.state, "executor", None)
        if executor is None:
            raise HTTPException(status_code=503, detail="executor not loaded")
        if req.mode not in ("vector", "graph", "hybrid"):
            raise HTTPException(
                status_code=400, detail=f"unknown mode {req.mode!r}; expected vector|graph|hybrid"
            )
        from csai415.answer import generate_answer

        answerers = [
            a.strip()
            for a in os.environ.get("CSAI415_COMPARE", "qwen2.5:3b-instruct,groq").split(",")
            if a.strip()
        ]
        orig = os.environ.get("CSAI415_ANSWERER")
        results: list[AnswererResult] = []
        try:
            # Base pass with NO answerer ⇒ fast extractive gen, so latency ≈ pure retrieval.
            os.environ.pop("CSAI415_ANSWERER", None)
            base = executor.answer(req.query, k=req.k, mode=req.mode, rerank=req.rerank)
            for a in answerers:
                os.environ["CSAI415_ANSWERER"] = a
                t0 = time.perf_counter()
                try:
                    text, _ = generate_answer(req.query, base.citations, base.contexts)
                    err = None
                except Exception as exc:  # e.g. Groq key missing, Ollama down
                    text, err = "", f"{type(exc).__name__}: {exc}"
                results.append(
                    AnswererResult(
                        answerer=a, answer=text,
                        gen_latency_ms=(time.perf_counter() - t0) * 1000.0, error=err,
                    )
                )
        finally:
            if orig is None:
                os.environ.pop("CSAI415_ANSWERER", None)
            else:
                os.environ["CSAI415_ANSWERER"] = orig

        return CompareResponse(
            query=req.query, mode=base.mode, rerank=base.rerank,
            citations=[CitationModel(**vars(c)) for c in base.citations],
            contexts=base.contexts, steps=base.steps,
            retrieve_latency_ms=base.latency_ms, results=results,
        )

    return app


# Module-level app for uvicorn / Docker (`uvicorn csai415.api:app`).
app = create_app()
