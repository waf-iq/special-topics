"""H0 contract smoke test for the D3 GraphRAG scaffold. See briefs/D3_TASKS.md (Hour 0).

Verifies the FROZEN contracts compose end-to-end using a FAKE retriever — no BGE model,
no Groq, no Neo4j, no Docker. It asserts the *shape* of the executor's output and that
``/ask`` is wired; it does NOT assert any member's algorithm (the seam stubs are theirs to
replace). When an owner swaps in their real implementation, this test should still pass.
"""

from __future__ import annotations

import pandas as pd

from csai415.graphrag import AnswerResult, Citation, GraphRAGExecutor


class _FakeRetriever:
    """Duck-types HybridRetriever.search_with_scores so the contract runs without BGE."""

    def __init__(self, ranking: list[tuple[str, float]]):
        self._ranking = ranking

    def search_with_scores(self, query: str, k: int, hybrid_weight=None):
        return self._ranking[:k]


def _fixture_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"chunk_id": "arxiv:p1:0", "paper_id": "p1", "title": "Paper One",
             "text": "Transformers use attention. A second sentence.", "page_start": 1, "page_end": 2},
            {"chunk_id": "arxiv:p2:0", "paper_id": "p2", "title": "Paper Two",
             "text": "BERT is bidirectional. Another sentence.", "page_start": 3, "page_end": 3},
            {"chunk_id": "arxiv:p3:0", "paper_id": "p3", "title": "Paper Three",
             "text": "Something unrelated entirely.", "page_start": None, "page_end": None},
        ]
    ).set_index("chunk_id")


def test_executor_answer_contract():
    df = _fixture_lookup()
    ranking = [("arxiv:p1:0", 0.9), ("arxiv:p2:0", 0.7), ("arxiv:p3:0", 0.5)]
    ex = GraphRAGExecutor(_FakeRetriever(ranking), df, candidate_k=3)

    res = ex.answer("what is attention", k=2, mode="hybrid", rerank=True)

    assert isinstance(res, AnswerResult)
    assert res.mode == "hybrid" and res.rerank is True
    assert len(res.citations) == len(res.contexts) <= 2          # contexts aligned with citations
    assert all(isinstance(c, Citation) for c in res.citations)

    top = res.citations[0]
    assert top.chunk_id == "arxiv:p1:0"
    assert top.page_range == "1-2"                                # page range formatted from lookup
    assert top.title == "Paper One"

    stages = [s["stage"] for s in res.steps]
    assert stages and stages[0] == "retrieve" and "generate" in stages
    assert res.latency_ms >= 0
    assert "[1]" in res.answer                                    # stub answerer cites its sources


def test_executor_modes_and_rerank_toggle():
    df = _fixture_lookup()
    ranking = [("arxiv:p1:0", 0.9), ("arxiv:p2:0", 0.7), ("arxiv:p3:0", 0.5)]
    ex = GraphRAGExecutor(_FakeRetriever(ranking), df, candidate_k=3)

    res_vec = ex.answer("x", k=3, mode="vector", rerank=False)
    assert res_vec.mode == "vector"
    assert len(res_vec.citations) == 3                            # no rerank → top-k of the pool

    # graph mode with the H0 fallback (empty subgraph) must not crash; degrades gracefully.
    res_graph = ex.answer("x", k=2, mode="graph", rerank=True)
    assert res_graph.mode == "graph" and len(res_graph.citations) <= 2


def test_ask_route_registered():
    """/ask is wired into the app without running the (BGE-loading) lifespan."""
    from csai415.api import create_app

    app = create_app()
    assert any(getattr(r, "path", None) == "/ask" for r in app.routes)
