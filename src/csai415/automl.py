"""Pair B — Optuna study tuning the hybrid retriever. See MEMBER_BRIEF.md §6.B.

Objective: maximize NDCG@5 on the gold Q/A set, optionally penalized by p95 latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna

from .eval import evaluate
from .retrieve import HybridRetriever, RetrieverConfig, load_chunks, make_retriever_fn

GOLD_JSONL = Path("data/gold/qa.jsonl")


def load_queries(path: Path = GOLD_JSONL) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_objective(chunks_df, queries, k: int = 5, latency_penalty_ms: float = 0.0):
    """Return an Optuna objective closure. See §6.B question 2 for penalty rationale."""

    def objective(trial: optuna.Trial) -> float:
        config = RetrieverConfig(
            metric=trial.suggest_categorical("metric", ["cosine", "l2", "dot"]),
            svd_dim=trial.suggest_categorical("svd_dim", [None, 64, 128, 256]),
            normalize=trial.suggest_categorical("normalize", [True, False]),
            hybrid_weight=trial.suggest_float("hybrid_weight", 0.0, 1.0),
            seed=42,
        )
        k_param = trial.suggest_int("k", 1, 50)
        retriever = HybridRetriever(chunks_df, config)
        fn = make_retriever_fn(retriever)
        metrics = evaluate(fn, queries, k=max(k_param, k))
        score = metrics["ndcg5"]
        if latency_penalty_ms > 0:
            score -= (metrics["p95_latency_ms"] / latency_penalty_ms) * 0.01
        return score

    return objective


def run_study(
    n_trials: int = 60,
    study_name: str = "csai415-d1-knn",
    storage: str | None = None,
    callbacks: list | None = None,
) -> optuna.Study:
    """Run the Optuna study. See §6.B question 1 for search-space sizing.

    Pass MLflow callback from csai415.mlflow_tracking.make_mlflow_callback()
    via `callbacks=[cb]` to enable experiment tracking (Musab's slice §6.D).
    """
    chunks_df = load_chunks()
    queries = load_queries()
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    objective = build_objective(chunks_df, queries)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=callbacks or [])
    return study


def best_config(study: optuna.Study) -> dict[str, Any]:
    return {"value": study.best_value, "params": study.best_params}
