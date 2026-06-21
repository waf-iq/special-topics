"""D3 Task 2 — Reranking. Owner: Ahmed Soliman. See briefs/D3_D4_TASKS.md (Task 2).

Second-stage reranking. The D2 ``HybridRetriever`` (BM25 + dense fusion, ``search_with_scores``)
scores query and chunk *independently* and is cheap enough to shortlist a candidate pool over
the whole corpus. A reranker then reads each (query, chunk) pair *jointly* and reorders that
small pool — more accurate, too slow to run corpus-wide, so it only touches the shortlist.
The executor (``graphrag.py``) calls ``rerank()`` between candidate retrieval and answer
generation; whatever it returns becomes the cited sources.

**The CONTRACT is frozen at H0** — the ``rerank(query, candidate_ids, chunk_text, top_n)``
signature. The default body delegates to the *blessed* backend named in ``_BLESSED_RERANKER``.
Until the team picks a winner together from ``reports/D3/d3_rerank.md`` it stays ``"none"``
(identity), which keeps the H0 contract test green in a model-less environment.

Backends compared in ``scripts/eval_rerank.py`` (build any via :func:`get_reranker`):
  - ``"none"``   identity — keep retrieval order, take ``top_n`` (the no-rerank baseline)
  - ``"minilm"`` cross-encoder ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (fast)
  - ``"bge"``    cross-encoder ``BAAI/bge-reranker-base`` (stronger, heavier)
  - ``"mmr"``    Maximal Marginal Relevance over ``bge-small-en-v1.5`` embeddings (diversity)

Models load lazily inside each backend (cached) so importing this module stays cheap and the
contract test never downloads a model.
"""

from __future__ import annotations

import logging
from typing import Callable

_log = logging.getLogger(__name__)

# A reranker: (query, candidate_ids, chunk_id->text, top_n) -> reordered ids (<= top_n).
RerankFn = Callable[[str, list, dict, int], list]

# Cross-encoder model ids, keyed by the short backend name.
_MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}
_MMR_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # same family as the corpus embedder
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The comparison winner (reports/D3/d3_rerank.md). Chosen: "bge" (BAAI/bge-reranker-base) —
# best NDCG@5 (+9.3pp over no-rerank at pool 20). Pair with executor candidate_k=20 (the pool
# sweet spot — see the note to the Task 7 integrator in briefs/D3_D4_TASKS.md). A bge load
# failure degrades to retrieval order via _with_fallback, so /ask never crashes.
_BLESSED_RERANKER = "bge"

# Lazy model cache: importing this module loads nothing.
_CACHE: dict = {}


def rerank(
    query: str,
    candidate_ids: list[str],
    chunk_text: dict[str, str],
    top_n: int = 5,
) -> list[str]:
    """Frozen-contract entry point used by the executor.

    Returns the ``top_n`` ``candidate_ids`` reordered by relevance to ``query``.
    ``chunk_text`` maps chunk_id -> text so the reranker scores the pair without a DB
    round-trip. Delegates to the blessed backend (``_BLESSED_RERANKER``).
    """
    return get_reranker(_BLESSED_RERANKER)(query, candidate_ids, chunk_text, top_n)


def get_reranker(kind: str, **kwargs) -> RerankFn:
    """Build a reranker callable for ``kind`` in {none, minilm, bge, mmr}.

    The callable has the same signature as :func:`rerank`. Models are loaded lazily on
    first call and cached, so building a reranker is free. Every model-backed reranker is
    wrapped in :func:`_with_fallback`, so a model failure (or empty pool) degrades to the
    original retrieval order instead of raising.
    """
    if kind == "none":
        return _identity
    if kind in _MODELS:
        return _with_fallback(_make_cross_encoder(_MODELS[kind]), kind)
    if kind == "mmr":
        return _with_fallback(_make_mmr(**kwargs), "mmr")
    raise ValueError(f"unknown reranker kind {kind!r}; expected none|minilm|bge|mmr")


def _with_fallback(fn: RerankFn, name: str) -> RerankFn:
    """Wrap a reranker so any failure degrades to retrieval order — defined once, used
    by every model backend. Reranking is a quality boost over an already-good candidate
    pool, so a model outage must never break ``/ask``: we keep the original order instead
    of raising. Also short-circuits an empty pool.
    """

    def safe(query, candidate_ids, chunk_text, top_n=5):
        if not candidate_ids:
            return []
        try:
            return fn(query, candidate_ids, chunk_text, top_n)
        except Exception:  # noqa: BLE001 — resilience boundary: degrade, never crash the caller
            _log.warning("reranker %s failed; keeping retrieval order", name, exc_info=True)
            return list(candidate_ids[:top_n])

    return safe


def _identity(query, candidate_ids, chunk_text, top_n=5):
    return list(candidate_ids[:top_n])


def _make_cross_encoder(model_name: str) -> RerankFn:
    # Pure logic — failure handling is centralized in _with_fallback (see get_reranker).
    def fn(query, candidate_ids, chunk_text, top_n=5):
        model = _load_cross_encoder(model_name)
        pairs = [(query, chunk_text.get(cid, "")) for cid in candidate_ids]
        scores = model.predict(pairs)
        order = sorted(range(len(candidate_ids)), key=lambda i: float(scores[i]), reverse=True)
        return [candidate_ids[i] for i in order[:top_n]]

    return fn


def _make_mmr(lambda_: float = 0.7, embed_model: str = _MMR_EMBED_MODEL) -> RerankFn:
    """MMR: greedily pick chunks that are relevant to the query yet diverse from
    those already picked. ``lambda_`` trades relevance (1.0) against diversity (0.0)."""

    # Pure logic — failure handling is centralized in _with_fallback (see get_reranker).
    def fn(query, candidate_ids, chunk_text, top_n=5):
        import numpy as np

        model = _load_embedder(embed_model)
        q_emb = model.encode(
            [_BGE_QUERY_PREFIX + query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        texts = [chunk_text.get(cid, "") for cid in candidate_ids]
        d_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        rel = d_emb @ q_emb  # cosine similarity (vectors are normalized)
        sim = d_emb @ d_emb.T  # pairwise candidate similarity

        selected: list[int] = []
        remaining = list(range(len(candidate_ids)))
        while remaining and len(selected) < top_n:
            if not selected:
                i = max(remaining, key=lambda j: rel[j])
            else:
                i = max(
                    remaining,
                    key=lambda j: lambda_ * rel[j]
                    - (1 - lambda_) * max(sim[j][s] for s in selected),
                )
            selected.append(i)
            remaining.remove(i)
        return [candidate_ids[i] for i in selected]

    return fn


def _load_cross_encoder(model_name: str):
    key = ("ce", model_name)
    if key not in _CACHE:
        from sentence_transformers import CrossEncoder

        _CACHE[key] = CrossEncoder(model_name)
    return _CACHE[key]


def _load_embedder(model_name: str):
    key = ("emb", model_name)
    if key not in _CACHE:
        from sentence_transformers import SentenceTransformer

        _CACHE[key] = SentenceTransformer(model_name)
    return _CACHE[key]
