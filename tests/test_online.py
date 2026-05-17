"""Pair C — unit tests for the online learner. Fast: fake retriever, no model.

Covers the §6.C acceptance bar: a River contextual-bandit learner that maps
(query_features, current_weight, feedback in {0,1}) -> next weight, cold-starts
deterministically from the AutoML weight, resets to that baseline on drift, and
a prequential loop that produces a populated state plus a same-stream static
baseline.
"""

from __future__ import annotations

from csai415.online import (
    WEIGHT_GRID,
    FALLBACK_WEIGHT,
    ContextualBanditLearner,
    OnlineLearnerState,
    build_learner,
    load_automl_weight,
    query_features,
    run_prequential,
    simulate_feedback_stream,
    _reward,
)

QUERIES = [
    {"qid": f"q{i}", "question": " ".join(["protein"] * (3 + i)),
     "relevant_chunk_ids": [f"c{i}"], "topic": None}
    for i in range(12)
]


def _fake_retriever(query, k, hybrid_weight):
    # Hit only on the dense side — gives the bandit a real signal so the
    # learning / reset paths are exercised, not just shapes.
    base = ["zzz"] * k
    if hybrid_weight >= 0.5:
        base[0] = "c0"
    return base[:k]


def test_load_automl_weight_falls_back_when_runcard_absent():
    assert load_automl_weight(path="configs/__nope__.yaml") == FALLBACK_WEIGHT


def test_build_learner_cold_starts_from_winning_weight():
    lrn = build_learner(winning_weight=0.7)
    assert isinstance(lrn, ContextualBanditLearner)
    assert lrn.winning_weight == 0.7
    # deterministic cold start: no feedback yet -> exactly the AutoML weight
    assert lrn.predict_action("any query at all") == 0.7
    assert lrn.predict_action("another different query") == 0.7


def test_query_features_triad():
    f = query_features("the quick brown fox")
    assert set(f) == {"intercept", "norm_length", "word_count"}
    assert f["intercept"] == 1.0
    assert 0.0 <= f["norm_length"] <= 1.0
    assert 0.0 <= f["word_count"] <= 1.0


def test_predict_action_returns_grid_weight_after_init():
    lrn = build_learner(winning_weight=0.5, seed=1)
    lrn.update("seed query", 0.5, 1.0)            # flips initialized -> True
    for _ in range(20):
        w = lrn.predict_action("some query text here")
        assert w in WEIGHT_GRID
        assert 0.0 <= w <= 1.0


def test_update_trains_only_chosen_action_and_reset_restores_coldstart():
    lrn = build_learner(winning_weight=0.5)
    for _ in range(5):
        w = lrn.predict_action("a protein claim about kinase signalling")
        lrn.update("a protein claim about kinase signalling", w, 1.0)
    assert sum(lrn.counts) == 5
    assert lrn.initialized is True
    lrn.reset_to_baseline()
    assert sum(lrn.counts) == 0
    assert lrn.initialized is False
    # back to deterministic cold start at the safe AutoML weight
    assert lrn.predict_action("x") == lrn.winning_weight


def test_on_drift_is_alias_for_reset_to_baseline():
    lrn = build_learner(winning_weight=0.25)
    lrn.update("q", 0.25, 1.0)
    lrn.on_drift()
    assert lrn.initialized is False
    assert lrn.predict_action("q") == 0.25


def test_reward_is_binary_top_k():
    assert _reward(["a", "b", "c"], {"b"}, k=5) == 1.0
    assert _reward(["a", "b", "c"], {"z"}, k=5) == 0.0
    assert _reward(["a", "b", "c"], {"c"}, k=2) == 0.0  # c is at rank 3, k=2


def test_simulate_stream_query_style_drift_transforms_post():
    claims = [{"qid": f"q{i}",
               "question": "the role of protein kinase signalling in tumour growth regulation",
               "relevant_chunk_ids": [f"c{i}"], "topic": None} for i in range(6)]
    stream = simulate_feedback_stream(claims, n_events=20, drift_at=10, seed=42)
    assert len(stream) == 20
    pre_q = {q["question"] for _, q, _ in stream[:10]}
    post_q = {q["question"] for _, q, _ in stream[10:]}
    assert pre_q == {claims[0]["question"]}            # pre: untouched claim
    assert all(len(q.split()) <= 4 for q in post_q)    # post: keyworded
    assert post_q != pre_q
    assert all(rel for _, _, rel in stream)            # relevance ids unchanged


def test_simulate_stream_length_mode_still_available():
    stream = simulate_feedback_stream(QUERIES, n_events=20, drift_at=10, seed=42,
                                      drift_kind="length")
    pre_len = sum(len(q["question"].split()) for _, q, _ in stream[:10]) / 10
    post_len = sum(len(q["question"].split()) for _, q, _ in stream[10:]) / 10
    assert post_len > pre_len


def test_run_prequential_populates_state():
    state = run_prequential(_fake_retriever, QUERIES, n_events=40, drift_at=20)
    assert isinstance(state, OnlineLearnerState)
    assert len(state.prequential_ndcg5) == 40
    assert len(state.baseline_ndcg5) == 40
    assert len(state.chosen_weights) == 40
    assert len(state.rewards) == 40
    # Grid weights once initialized; the deterministic cold-start (and the
    # post-reset re-cold-start) legitimately emits the exact AutoML weight,
    # which need not be a grid point (real run-card -> ~0.81).
    cold = state.baseline_weight
    assert all(w in WEIGHT_GRID or w == cold for w in state.chosen_weights)
    assert all(0.0 <= n <= 1.0 for n in state.prequential_ndcg5)
    assert all(r in (0.0, 1.0) for r in state.rewards)   # strictly binary
    assert state.drift_at == 20


def test_run_prequential_is_reproducible():
    s1 = run_prequential(_fake_retriever, QUERIES, n_events=30, drift_at=15, seed=7)
    s2 = run_prequential(_fake_retriever, QUERIES, n_events=30, drift_at=15, seed=7)
    assert s1.prequential_ndcg5 == s2.prequential_ndcg5
    assert s1.chosen_weights == s2.chosen_weights
