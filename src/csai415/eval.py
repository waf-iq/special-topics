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


# --- D3-5: answer-level evaluation (faithfulness, answer-relevance, latency) ------------
# See briefs/D3_TASKS.md (Task D3-5). Owner: Task D3-5.
#
# CONTRACT frozen at H0: evaluate_answers(answer_fn, gold) -> the dict below.
# answer_fn(question) returns an AnswerResult-like object with .answer (str),
# .citations (each with .chunk_id), .contexts (list[str]), .latency_ms (float).
# gold rows come from data/gold/qa_answers.jsonl (question + reference_answer +
# relevant_chunk_ids + page_range).
#
# H0 STUB for faithfulness/answer_relevance: a cheap lexical-overlap PROXY so the harness
# (and D3-6's HPO) runs end-to-end today. D3-5 owner REPLACES these two with real RAGAS
# metrics judged by Groq llama-3.3-70b (via langchain-groq) — see the task doc. recall5
# and p95 latency are already real.


def _overlap(a: str, b: str) -> float:
    import re

    aw = set(re.findall(r"[a-z]{3,}", (a or "").lower()))
    bw = set(re.findall(r"[a-z]{3,}", (b or "").lower()))
    return len(aw & bw) / len(aw) if aw else 0.0


def evaluate_answers(answer_fn: Callable, gold: list[dict]) -> dict:
    """Returns {faithfulness, answer_relevance, recall5, p95_latency_ms, n}.

    faithfulness/answer_relevance are H0 lexical PROXIES (answer↔contexts, answer↔question)
    — D3-5 swaps in RAGAS+Groq. recall5 = fraction of gold relevant_chunk_ids that appear
    in the answer's citations. Latency is measured per call (p95).
    """
    faith, relev, recalls, latencies_ms = [], [], [], []
    for q in gold:
        t0 = time.perf_counter()
        res = answer_fn(q["question"])
        latencies_ms.append(getattr(res, "latency_ms", None) or (time.perf_counter() - t0) * 1000)

        contexts = " ".join(getattr(res, "contexts", []) or [])
        faith.append(_overlap(res.answer, contexts))          # PROXY — replace with RAGAS faithfulness
        relev.append(_overlap(res.answer, q["question"]))     # PROXY — replace with RAGAS answer_relevance

        cited = {c.chunk_id for c in getattr(res, "citations", [])}
        relevant = set(q.get("relevant_chunk_ids", []))
        recalls.append(len(cited & relevant) / len(relevant) if relevant else 0.0)

    return {
        "faithfulness": float(np.mean(faith)) if faith else 0.0,
        "answer_relevance": float(np.mean(relev)) if relev else 0.0,
        "recall5": float(np.mean(recalls)) if recalls else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "n": len(gold),
    }
