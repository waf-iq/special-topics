"""Pair C — River online learner with ADWIN drift handling. See MEMBER_BRIEF.md §6.C.

Task: adapt the hybrid_weight in real time from binary click-helpful feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from river import drift, linear_model, preprocessing

from .eval import ndcg_at_k

SPLIT_INDICES_PATH = Path("configs/d1_split_indices.json")
GOLD_JSONL = Path("data/gold/qa.jsonl")


@dataclass
class OnlineLearnerState:
    """Tracks the current adaptive hybrid_weight and a stream of prequential NDCG@5."""
    current_weight: float = 0.5
    prequential_ndcg5: list[float] = field(default_factory=list)
    drift_events: list[int] = field(default_factory=list)


def build_learner():
    """Build the River model. Abdurlahman's slice — fills in his ContextualBanditLearner here."""
    raise NotImplementedError("Pair C (Abdurlahman) — see §6.C question 1 for model class rationale.")


def build_drift_detector(delta: float = 0.002) -> drift.ADWIN:
    """ADWIN detector. See §6.C question 3 for delta sizing."""
    return drift.ADWIN(delta=delta)


def _load_holdout_queries() -> list[dict]:
    """Pull the 60 holdout queries (not the tune set) so the static baseline isn't inflated
    by queries Pair B's AutoML already optimized on. Reads the split written by Pair B's
    run_and_record(), so seeds match.
    """
    with GOLD_JSONL.open(encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f]
    with SPLIT_INDICES_PATH.open() as f:
        split = json.load(f)
    return [all_queries[i] for i in split["holdout"]]


def simulate_feedback_stream(
    queries: list[dict] | None = None,
    n_events: int = 200,
    drift_at: int = 100,
    seed: int = 42,
):
    """Yield (event_idx, query_dict, drift_indicator). Query-length shift at drift_at.

    Sort holdout queries by question length. Before drift_at: cycle the SHORT half;
    after drift_at: cycle the LONG half. BM25 favors short queries, dense favors long,
    so this shift maximally motivates the adaptive learner to change hybrid_weight.
    drift_indicator is 0 before drift_at, 1 after — used by the plot to mark the line.
    """
    if queries is None:
        queries = _load_holdout_queries()

    sorted_by_len = sorted(queries, key=lambda q: len(q["question"]))
    midpoint = len(sorted_by_len) // 2
    short_pool = sorted_by_len[:midpoint]
    long_pool = sorted_by_len[midpoint:]

    for event_idx in range(n_events):
        if event_idx < drift_at:
            q = short_pool[event_idx % len(short_pool)]
            drift_indicator = 0
        else:
            q = long_pool[(event_idx - drift_at) % len(long_pool)]
            drift_indicator = 1
        yield event_idx, q, drift_indicator


def run_prequential(
    retriever_fn: Callable[[str, int, float], list[str]],
    queries: list[dict] | None = None,
    n_events: int = 200,
    drift_at: int = 100,
) -> OnlineLearnerState:
    """Run the prequential evaluation loop and return populated state for plotting.

    Per event: learner picks hybrid_weight from query text, we retrieve top-5, compute
    NDCG@5 for the plot AND a binary hit reward for ADWIN + learner update. When ADWIN
    fires we record the event and call learner.reset_to_baseline().
    """
    learner = build_learner()
    detector = build_drift_detector()
    state = OnlineLearnerState()

    for event_idx, query, _drift_indicator in simulate_feedback_stream(queries, n_events, drift_at):
        question = query["question"]
        relevant = set(query["relevant_chunk_ids"])

        weight = learner.predict_action(question)
        retrieved = retriever_fn(question, 5, weight)
        ndcg = ndcg_at_k(retrieved, relevant, k=5)
        reward = 1.0 if len(set(retrieved) & relevant) > 0 else 0.0

        state.prequential_ndcg5.append(ndcg)
        state.current_weight = weight

        learner.update(question, weight, reward)
        detector.update(reward)
        if detector.drift_detected:
            state.drift_events.append(event_idx)
            learner.reset_to_baseline()

    return state