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


def _assert_no_leakage(gold: list[dict], train_path: str = "data/train/qa_train.jsonl") -> None:
    """Assert no question/chunk overlap between gold eval set and QLoRA train set."""
    import os
    if not os.path.exists(train_path):
        return
    import json as _json
    with open(train_path) as f:
        train_rows = [_json.loads(l) for l in f]
    gold_q = {r["question"].lower().strip() for r in gold}
    train_q = {r["question"].lower().strip() for r in train_rows}
    overlap = gold_q & train_q
    assert not overlap, f"Train/eval leakage detected — {len(overlap)} shared question(s): {list(overlap)[:3]}"


def evaluate_answers(answer_fn: Callable, gold: list[dict]) -> dict:
    """Returns {faithfulness, answer_relevance, recall5, p95_latency_ms, n}.

    faithfulness and answer_relevance are scored by RAGAS via Groq llama-3.3-70b.
    recall5 = fraction of gold relevant_chunk_ids cited by the answer.
    """
    from csai415.ragas_groq import ragas_score

    _assert_no_leakage(gold)

    questions, answers, contexts_list, ground_truths, recalls, latencies_ms = [], [], [], [], [], []

    for q in gold:
        t0 = time.perf_counter()
        res = answer_fn(q["question"])
        latencies_ms.append(getattr(res, "latency_ms", None) or (time.perf_counter() - t0) * 1000)

        questions.append(q["question"])
        answers.append(res.answer)
        contexts_list.append(list(getattr(res, "contexts", []) or []))
        ground_truths.append(q.get("reference_answer", ""))

        cited = {c.chunk_id for c in getattr(res, "citations", [])}
        relevant = set(q.get("relevant_chunk_ids", []))
        recalls.append(len(cited & relevant) / len(relevant) if relevant else 0.0)

    ragas = ragas_score(questions, answers, contexts_list, ground_truths)

    return {
        "faithfulness": ragas["faithfulness"],
        "answer_relevance": ragas["answer_relevancy"],
        "recall5": float(np.mean(recalls)) if recalls else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "n": len(gold),
    }
