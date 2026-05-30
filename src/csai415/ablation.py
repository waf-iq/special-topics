"""Pair B — Task B2: search-space ablation. See D1_REWORK_TASKS.md.

The marker scored AutoML 20/50 ("just tried one optimiser ... tick the box?").
B2's answer is evidence: we expanded the search space (added BM25 k1/b on top of
the original 5 dims) and prove here that each dimension earned its place.

Method: take the AutoML winner from configs/winning_runcard.yaml, hold it fixed,
drop ONE dimension at a time back to its RetrieverConfig default, and re-evaluate
on the same 60 held-out queries Optuna never saw. The drop in NDCG@5 vs the full
winner is the "did AutoML actually need this dimension" signal:
    large negative delta  -> the optimizer's choice mattered
    delta ~ 0             -> the default was already fine (often because the
                            winner *chose* the default for that dim)

This must run AFTER run_and_record() has re-tuned over the expanded 7-dim space,
otherwise bm25_k1/bm25_b sit at their defaults in the winner and dropping them is
a no-op.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import yaml

from .automl import _split_queries, load_queries
from .eval import evaluate
from .retrieve import (
    HybridRetriever,
    RetrieverConfig,
    load_chunks,
    make_retriever_fn,
)
from .runcard import RUNCARD_PATH

ABLATION_CSV = Path("reports/search_space_ablation.csv")

# Every dimension Optuna searches over (seed is fixed, so it is not a search dim).
SEARCH_DIMS = [
    "metric",
    "svd_dim",
    "normalize",
    "hybrid_weight",
    "candidate_k",
    "bm25_k1",
    "bm25_b",
]


def winner_from_runcard(path: Path = RUNCARD_PATH) -> RetrieverConfig:
    """Rebuild the winning RetrieverConfig from the run-card's best_params.

    Falls back to RetrieverConfig defaults for any missing key so a stale (pre-B2)
    run-card still loads — but warns, because the ablation is only meaningful once
    the run-card carries tuned bm25_k1/bm25_b.
    """
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    bp = card["automl"]["best_params"]
    d = RetrieverConfig()
    if "bm25_k1" not in bp or "bm25_b" not in bp:
        print(
            f"WARNING: {path} has no bm25_k1/bm25_b in best_params — re-run "
            "run_and_record() over the expanded space first, or the BM25 rows "
            "will show a meaningless zero delta."
        )
    return RetrieverConfig(
        metric=bp.get("metric", d.metric),
        svd_dim=bp.get("svd_dim", d.svd_dim),
        normalize=bp.get("normalize", d.normalize),
        hybrid_weight=bp.get("hybrid_weight", d.hybrid_weight),
        candidate_k=bp.get("candidate_k", d.candidate_k),
        bm25_k1=bp.get("bm25_k1", d.bm25_k1),
        bm25_b=bp.get("bm25_b", d.bm25_b),
        seed=42,
    )


def _score(config: RetrieverConfig, chunks_df, holdout) -> dict:
    """Build a retriever for `config` and evaluate it on the holdout queries.

    hybrid_weight is passed through to evaluate() because search() treats it as a
    per-call override — so resetting that dimension only takes effect if we hand
    the config's value to evaluate(), not just set it on the config.
    """
    fn = make_retriever_fn(HybridRetriever(chunks_df, config))
    return evaluate(fn, holdout, k=5, hybrid_weight=config.hybrid_weight)


def run_ablation(
    runcard_path: Path = RUNCARD_PATH,
    out_path: Path = ABLATION_CSV,
) -> Path:
    """Hold the winner fixed, drop each dim to its default, write the CSV."""
    chunks_df = load_chunks()
    # Same seed/split as the study, so we score on the identical 60 holdout queries.
    _, holdout, _ = _split_queries(load_queries())

    winner = winner_from_runcard(runcard_path)
    defaults = RetrieverConfig()

    # Baseline row: the full winner, nothing dropped.
    full = _score(winner, chunks_df, holdout)
    base_ndcg = full["ndcg5"]
    rows = [
        {
            "dropped_dim": "none (winner)",
            "winner_value": "-",
            "default_value": "-",
            "ndcg5": round(full["ndcg5"], 4),
            "recall5": round(full["recall5"], 4),
            "p95_latency_ms": round(full["p95_latency_ms"], 2),
            "delta_ndcg5": 0.0,
        }
    ]

    # One row per dimension reset to its default.
    for dim in SEARCH_DIMS:
        ablated = dataclasses.replace(winner, **{dim: getattr(defaults, dim)})
        m = _score(ablated, chunks_df, holdout)
        rows.append(
            {
                "dropped_dim": dim,
                "winner_value": getattr(winner, dim),
                "default_value": getattr(defaults, dim),
                "ndcg5": round(m["ndcg5"], 4),
                "recall5": round(m["recall5"], 4),
                "p95_latency_ms": round(m["p95_latency_ms"], 2),
                "delta_ndcg5": round(m["ndcg5"] - base_ndcg, 4),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = run_ablation()
    print(f"wrote {path}\n")
    print(pd.read_csv(path).to_string(index=False))
