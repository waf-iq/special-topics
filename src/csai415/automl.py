"""Pair B — Optuna study tuning the hybrid retriever. See MEMBER_BRIEF.md §6.B.

Objective: maximize NDCG@5 on the gold Q/A set, optionally penalized by p95 latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna

from .eval import evaluate
from .ingest import EMBED_MODEL
from .retrieve import (
    CHUNKS_PARQUET,
    HybridRetriever,
    RetrieverConfig,
    load_chunks,
    make_retriever_fn,
)
from .runcard import RUNCARD_PATH, write_runcard

GOLD_JSONL = Path("data/gold/qa.jsonl")
SPLIT_INDICES_PATH = Path("configs/d1_split_indices.json")
STUDY_STORAGE_DIR = Path("studies")


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
            bm25_k1=trial.suggest_float("bm25_k1", 0.5, 3.0),
            bm25_b=trial.suggest_float("bm25_b", 0.0, 1.0),
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
    chunks_df=None,
    queries=None,
) -> optuna.Study:
    """Run the Optuna study. See §6.B question 1 for search-space sizing.

    Pass MLflow callback from csai415.mlflow_tracking.make_mlflow_callback()
    via `callbacks=[cb]` to enable experiment tracking (Musab's slice §6.D).
    """
    if chunks_df is None:
        chunks_df = load_chunks()
    if queries is None:
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


def _split_queries(queries, holdout_frac: float = 0.20, split_seed: int = 42):
    """80/20 split stratified by whether the claim has multiple relevant docs.

    Writes the indices to SPLIT_INDICES_PATH so Pair C uses the same split.
    Returns (tune_queries, holdout_queries, split_meta_dict).
    """
    from sklearn.model_selection import train_test_split

    indices = list(range(len(queries)))
    stratify = [len(q["relevant_chunk_ids"]) > 1 for q in queries]

    tune_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_frac,
        random_state=split_seed,
        stratify=stratify,
    )

    tune_queries = [queries[i] for i in tune_idx]
    holdout_queries = [queries[i] for i in holdout_idx]

    SPLIT_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_INDICES_PATH.open("w") as f:
        json.dump({"tune": sorted(tune_idx), "holdout": sorted(holdout_idx)}, f)

    split_meta = {
        "strategy": "stratified_by_n_relevant_bool",
        "split_seed": split_seed,
        "n_tune": len(tune_idx),
        "n_holdout": len(holdout_idx),
        "indices_path": str(SPLIT_INDICES_PATH),
    }
    return tune_queries, holdout_queries, split_meta


def run_and_record(
    n_trials: int = 80,
    study_name: str = "csai415-d1-knn",
    holdout_frac: float = 0.20,
    split_seed: int = 42,
    callbacks: list | None = None,
    out_path: Path = RUNCARD_PATH,
) -> Path:
    """End-to-end D1 entry point: split queries, tune on 240, eval winner + baseline
    on the held-out 60, write the runcard.
    """
    chunks_df = load_chunks()
    queries = load_queries()
    tune_queries, holdout_queries, split_meta = _split_queries(
        queries, holdout_frac=holdout_frac, split_seed=split_seed
    )

    STUDY_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = STUDY_STORAGE_DIR / f"{study_name}.db"
    storage = f"sqlite:///{storage_path.as_posix()}"

    study = run_study(
        n_trials=n_trials,
        study_name=study_name,
        storage=storage,
        callbacks=callbacks,
        chunks_df=chunks_df,
        queries=tune_queries,
    )

    # Winning config (everything except seed comes from the study)
    winning = RetrieverConfig(
        metric=study.best_params["metric"],
        svd_dim=study.best_params["svd_dim"],
        normalize=study.best_params["normalize"],
        hybrid_weight=study.best_params["hybrid_weight"],
        candidate_k=study.best_params["candidate_k"],
        bm25_k1=study.best_params["bm25_k1"],
        bm25_b=study.best_params["bm25_b"],
        seed=42,
    )
    winner_fn = make_retriever_fn(HybridRetriever(chunks_df, winning))
    tune_w = evaluate(winner_fn, tune_queries, k=5, hybrid_weight=winning.hybrid_weight)
    holdout_w = evaluate(winner_fn, holdout_queries, k=5, hybrid_weight=winning.hybrid_weight)

    # Baselines for the report table. All use cosine, no SVD, normalize=True, candidate_k=10
    # (RetrieverConfig defaults). Only hybrid_weight varies — same retriever reused since
    # hybrid_weight is a per-call override.
    baseline = RetrieverConfig()
    baseline_fn = make_retriever_fn(HybridRetriever(chunks_df, baseline))
    bm25_only = evaluate(baseline_fn, holdout_queries, k=5, hybrid_weight=0.0)
    dense_only = evaluate(baseline_fn, holdout_queries, k=5, hybrid_weight=1.0)
    default_hybrid = evaluate(baseline_fn, holdout_queries, k=5, hybrid_weight=0.5)

    metrics = {
        "winner_tune": tune_w,
        "winner_holdout": holdout_w,
        "baselines_holdout": {
            "bm25_only": bm25_only,
            "dense_only": dense_only,
            "default_hybrid": default_hybrid,
        },
    }

    sampler_config = {
        "class": "TPESampler",
        "seed": 42,
        "n_startup_trials": 20,
        "multivariate": True,
    }
    pruner_config = {"class": "NopPruner"}
    notes = (
        f"single-seed study; holdout n={len(holdout_queries)} so NDCG@5 CI is wide (~±0.05). "
        "5-fold CV and multi-objective deferred to D2."
    )

    return write_runcard(
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials=n_trials,
        embedding_model=EMBED_MODEL,
        chunks_parquet=CHUNKS_PARQUET,
        gold_jsonl=GOLD_JSONL,
        metrics=metrics,
        split=split_meta,
        sampler_config=sampler_config,
        pruner_config=pruner_config,
        study_storage=storage,
        notes=notes,
        out_path=out_path,
    )
