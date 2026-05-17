"""Pair C — River online learner with ADWIN drift handling. See MEMBER_BRIEF.md §6.C.

Task: adapt the hybrid_weight in real time from binary click-helpful feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from river import drift, linear_model, preprocessing


@dataclass
class OnlineLearnerState:
    """Tracks the current adaptive hybrid_weight and a stream of prequential NDCG@5."""
    current_weight: float = 0.5
    prequential_ndcg5: list[float] = field(default_factory=list)
    drift_events: list[int] = field(default_factory=list)


def build_learner():
    """Build the River model. See §6.C question 1 for model class rationale."""
    raise NotImplementedError("Pair C — propose either contextual bandit (§6.C Q2) or regression on rolling feedback.")


def build_drift_detector(delta: float = 0.002) -> drift.ADWIN:
    """ADWIN detector. See §6.C question 3 for delta sizing."""
    return drift.ADWIN(delta=delta)


def simulate_feedback_stream(queries: list[dict], n_events: int = 200, drift_at: int = 100, seed: int = 42):
    """Yield (event_idx, query, true_label) with a topic shift at drift_at. See §6.C question 4."""
    raise NotImplementedError("Pair C — choose drift type (topic shift recommended) and document choice.")


def run_prequential(
    retriever_fn: Callable[[str, int, float], list[str]],
    queries: list[dict],
    n_events: int = 200,
    drift_at: int = 100,
) -> OnlineLearnerState:
    """Run the prequential evaluation loop and return populated state for plotting."""
    raise NotImplementedError("Pair C — see §6.C question 5 for sliding-window vs cumulative choice.")
