"""Pair A — Evaluation harness. See MEMBER_BRIEF.md §6.A.

Contract: evaluate(retriever_fn, queries, k=5) -> {ndcg5, recall5, p95_latency_ms}.
Both Pair B's AutoML objective and Pair C's prequential loop call this.
"""

from __future__ import annotations

import time
from math import log2
from typing import Callable

import numpy as np


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG@k. With multi-doc relevance per SciFact claim this does not degenerate to MRR."""
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant:
            dcg += 1.0 / log2(i + 2)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def evaluate(
    retriever_fn: Callable[[str, int, float], list[str]],
    queries: list[dict],
    k: int = 5,
    hybrid_weight: float = 0.5,
) -> dict:
    """Returns {ndcg5, recall5, p95_latency_ms}.

    queries: list of {qid, question, relevant_chunk_ids, topic}.
    """
    ndcgs, recalls, latencies_ms = [], [], []
    for q in queries:
        relevant = set(q["relevant_chunk_ids"])
        t0 = time.perf_counter()
        retrieved = retriever_fn(q["question"], k, hybrid_weight)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
    return {
        "ndcg5": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall5": float(np.mean(recalls)) if recalls else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }
