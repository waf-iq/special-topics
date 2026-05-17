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
    """Return an Optuna objective closure.

    For D1 we optimize raw NDCG@5 and record p95 latency as a user_attr so the
    notebook can show the latency/quality tradeoff without folding it into the
    objective. latency_penalty_ms is kept in the signature for D2 (where multi-
    objective or constraints_func will handle latency properly) — don't delete it.
    """

    def objective(trial: optuna.Trial) -> float:
        config = RetrieverConfig(
            metric=trial.suggest_categorical("metric", ["cosine", "l2", "dot"]),
            svd_dim=trial.suggest_categorical("svd_dim", [None, 64, 128, 256]),
            normalize=trial.suggest_categorical("normalize", [True, False]),
            hybrid_weight=trial.suggest_float("hybrid_weight", 0.0, 1.0),
            candidate_k=trial.suggest_int("candidate_k", 5, 50),
            seed=42,
        )
        retriever = HybridRetriever(chunks_df, config)
        fn = make_retriever_fn(retriever)
        metrics = evaluate(fn, queries, k=k, hybrid_weight=config.hybrid_weight)
        trial.set_user_attr("ndcg5", metrics["ndcg5"])
        trial.set_user_attr("recall5", metrics["recall5"])
        trial.set_user_attr("p95_latency_ms", metrics["p95_latency_ms"])
        return metrics["ndcg5"]

    return objective


def run_study(
    n_trials: int = 80,
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
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20, multivariate=True)
    pruner = optuna.pruners.NopPruner()
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
