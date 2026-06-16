"""D3-3 — Reranking. See briefs/D3_TASKS.md (Task D3-3).

Owner: Task D3-3. Reorder a candidate pool by joint (query, chunk) relevance.

**The CONTRACT is frozen at H0** — the ``rerank`` signature. The body is the
identity (keep retrieval order, take top_n) until the owner wires a real reranker
and compares: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` vs ``BAAI/bge-reranker-base``
vs MMR-diversity vs none, plus a candidate-pool sweep (doctor's Week 07 §5). Load any
model lazily inside the function so importing this module stays cheap.
"""

from __future__ import annotations


def rerank(
    query: str,
    candidate_ids: list[str],
    chunk_text: dict[str, str],
    top_n: int = 5,
) -> list[str]:
    """Return the top_n ``candidate_ids`` reordered by relevance to ``query``.

    ``chunk_text`` maps chunk_id → text (so the reranker can score the pair without a
    DB round-trip). H0 STUB: identity — first ``top_n`` in the order given.
    """
    return candidate_ids[:top_n]
