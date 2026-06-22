"""D3-2 — GraphRAG executor (the backbone). See briefs/D3_TASKS.md (Task D3-2).

Owner: Task D3-2. Orchestrates the pipeline the rubric grades (GraphRAG pipeline, 8%):

    subgraph (Cypher)  →  expand to supporting chunks  →  hybrid blend
        →  optional rerank  →  answer with [n] citations + page ranges

**The CONTRACTS frozen at H0** are the ``Citation`` and ``AnswerResult`` dataclasses and
``GraphRAGExecutor.answer(query, k, mode, rerank) -> AnswerResult``. Every other task
plugs in through an injected seam function:

    D3-1  select_fn(query, neo4j_driver=...) -> SubgraphResult     (graph_select.py)
    D3-3  rerank_fn(query, candidate_ids, chunk_text, top_n) -> ids (rerank.py)
    D3-4  answer_fn(query, citations, contexts) -> (text, used_idxs) (answer.py)

The H0 body is a working baseline: ``mode="vector"`` is pure dense, ``"hybrid"`` uses the
blessed fusion weight, ``"graph"`` applies *graph-as-filter* (keep only candidates whose
paper is in the subgraph, with fallback if that empties the pool). D3-2 owner extends the
graph stage into the filter/booster/expansion comparison from the task doc.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from csai415.answer import generate_answer as _default_answer
from csai415.graph_select import select_subgraph as _default_select
from csai415.rerank import rerank as _default_rerank

Mode = Literal["vector", "graph", "hybrid"]

# Per-mode hybrid_weight passed to the retriever. None ⇒ use the blessed config default.
_MODE_HYBRID_WEIGHT: dict[str, float | None] = {"vector": 1.0, "hybrid": None, "graph": None}


@dataclass
class Citation:
    """One cited source. Mirrors the /search ``SearchHit`` shape so the UI is consistent."""

    chunk_id: str
    paper_id: str
    title: str
    page_range: str | None
    score: float


@dataclass
class AnswerResult:
    """The executor's output. ``contexts[i]`` is the chunk text for ``citations[i]``."""

    answer: str
    citations: list[Citation]
    contexts: list[str]
    mode: str
    rerank: bool
    latency_ms: float
    steps: list[dict] = field(default_factory=list)


def _page_range(row: pd.Series) -> str | None:
    """Format ``page_start``/``page_end`` as 'start-end' (or 'start'); None when absent."""
    ps = row.get("page_start")
    if ps is None or (isinstance(ps, float) and math.isnan(ps)):
        return None
    ps = int(ps)
    pe = row.get("page_end")
    pe = int(pe) if pe is not None and not (isinstance(pe, float) and math.isnan(pe)) else ps
    return str(ps) if pe == ps else f"{ps}-{pe}"


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


class GraphRAGExecutor:
    """Holds the retriever + chunk lookup + the three pluggable seam functions.

    ``retriever`` must expose ``search_with_scores(query, k, hybrid_weight) -> [(chunk_id, score)]``
    (the D2 ``HybridRetriever`` does). ``lookup_df`` is the full corpus indexed by ``chunk_id``
    (carries ``paper_id``, ``title``, ``text``, ``page_start``/``page_end``).
    """

    def __init__(
        self,
        retriever,
        lookup_df: pd.DataFrame,
        *,
        select_fn: Callable = _default_select,
        rerank_fn: Callable = _default_rerank,
        answer_fn: Callable = _default_answer,
        neo4j_driver=None,
        candidate_k: int = 20,
    ):
        self.retriever = retriever
        self.lookup = lookup_df if lookup_df.index.name == "chunk_id" else lookup_df.set_index("chunk_id")
        self.select_fn = select_fn
        self.rerank_fn = rerank_fn
        self.answer_fn = answer_fn
        self.neo4j_driver = neo4j_driver
        self.candidate_k = candidate_k

    def _text(self, chunk_id: str) -> str:
        try:
            return _clean(self.lookup.loc[chunk_id]["text"])
        except KeyError:
            return ""

    def _citation(self, chunk_id: str, score: float) -> Citation:
        row = self.lookup.loc[chunk_id]
        return Citation(
            chunk_id=chunk_id,
            paper_id=_clean(row.get("paper_id")),
            title=_clean(row.get("title")),
            page_range=_page_range(row),
            score=float(score),
        )

    def run_pipeline(self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = True):
        """Retrieval pipeline as a generator — the single source of truth for stages 1–4.

        ``yield``s one step dict per stage as it completes (so a streaming endpoint can light
        up the UI in real time), then ``return``s ``(citations, contexts, score_by_id)`` via the
        generator's ``StopIteration.value``. ``answer()`` drains this and adds the generate stage;
        the SSE ``/ask/stream`` endpoint drains it and then streams generation token-by-token.
        Generation itself is deliberately *not* here — it's the one stage with two shapes
        (blocking vs. streamed), so each caller owns it.
        """
        hw = _MODE_HYBRID_WEIGHT.get(mode)

        # 1. Candidate retrieval — pool ≥ k so the reranker has room to reorder.
        pool = self.retriever.search_with_scores(query, self.candidate_k, hybrid_weight=hw)
        candidates = [cid for cid, _ in pool]
        score_by_id = {cid: float(s) for cid, s in pool}
        yield {
            "stage": "retrieve", "mode": mode, "hybrid_weight": hw,
            "n_candidates": len(candidates), "pool": list(candidates),
        }

        # 2. Graph guidance (graph/hybrid modes). H0 default = graph-as-filter with fallback.
        if mode in ("graph", "hybrid"):
            sub = self.select_fn(query, neo4j_driver=self.neo4j_driver)
            yield {
                "stage": "subgraph", "method": sub.method, "cypher": sub.cypher_used,
                "n_papers": len(sub.paper_ids),
                # reporting extras for the UI (entities the graph linked + the papers it pulled)
                "entities": [{"name": e.name, "label": e.label, "score": round(e.score, 3)} for e in sub.linked],
                "paper_ids": list(sub.paper_ids)[:12],
            }
            if sub.paper_ids:
                wanted = set(sub.paper_ids)
                in_sub = [cid for cid in candidates if self._paper_of(cid) in wanted]
                if mode == "graph" and in_sub:  # graph mode hard-filters; hybrid leaves the pool for D3-2 to tune
                    dropped = [c for c in candidates if c not in set(in_sub)]
                    candidates = in_sub
                    yield {"stage": "graph_filter", "kept": len(candidates), "dropped": dropped}

        # 3. Optional rerank → final top-k.
        if rerank:
            chunk_text = {cid: self._text(cid) for cid in candidates}
            before = list(candidates)
            candidates = self.rerank_fn(query, candidates, chunk_text, top_n=k)
            yield {"stage": "rerank", "top_n": k, "before": before[:k], "after": list(candidates)}
        else:
            candidates = candidates[:k]

        # 4. Build citations + aligned contexts (the generator's return value).
        citations = [self._citation(cid, score_by_id.get(cid, 0.0)) for cid in candidates]
        contexts = [self._text(cid) for cid in candidates]
        return citations, contexts, score_by_id

    def answer(self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = True) -> AnswerResult:
        t0 = time.perf_counter()
        steps: list[dict] = []

        # Drain the shared pipeline generator, collecting each stage as it completes.
        pipe = self.run_pipeline(query, k, mode, rerank)
        try:
            while True:
                steps.append(next(pipe))
        except StopIteration as stop:
            citations, contexts, _ = stop.value

        # 5. Generate the grounded answer.
        answer_text, used = self.answer_fn(query, citations, contexts)
        steps.append({"stage": "generate", "cited": used})

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            contexts=contexts,
            mode=mode,
            rerank=rerank,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            steps=steps,
        )

    def _paper_of(self, chunk_id: str) -> str | None:
        try:
            return self.lookup.loc[chunk_id]["paper_id"]
        except KeyError:
            return None


# --- Module-level convenience: a default executor over the full parquet ----------------
# Lazily built (loads BGE + builds indexes on first call) so importing this module is cheap.
# D3-2 owner: wire the Neo4j driver + single Qdrant collection here.

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
_default_executor: GraphRAGExecutor | None = None


def get_default_executor() -> GraphRAGExecutor:
    global _default_executor
    if _default_executor is None:
        from csai415.api import load_blessed_config
        from csai415.retrieve import HybridRetriever, load_chunks

        df = load_chunks(CHUNKS_PARQUET)
        cfg = load_blessed_config()
        retriever = HybridRetriever(df, cfg)
        _default_executor = GraphRAGExecutor(retriever, df.set_index("chunk_id"))
    return _default_executor


def answer(query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = True) -> AnswerResult:
    """Convenience wrapper over the default executor (builds it on first call)."""
    return get_default_executor().answer(query, k, mode, rerank)
