"""Smoke tests for the multi-method HPO comparison (D1 rework task B1).

We don't run the full 5×80-trial comparison here — that takes ~hours. We just
verify the API surface, the RunResult dataclass shape, that the Grid runner
works end-to-end on a tiny grid, and that the multi-fidelity wiring (trial.report
+ pruning) doesn't break under Hyperband. Random/TPE/BOHB share scaffolding
with Grid/Hyperband, so they're covered transitively.

Tests skip if the corpus or gold files are absent so a fresh clone still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
GOLD_JSONL = Path("data/gold/qa.jsonl")


def _have_data() -> bool:
    return CHUNKS_PARQUET.exists() and GOLD_JSONL.exists()


def test_runresult_fields_match_lab():
    """RunResult mirrors the prof's HPO-tutorial RunResult plus our extras."""
    from csai415.hpo_methods import RunResult
    fields = set(RunResult.__dataclass_fields__)
    # Lab fields (must be present so the comparison table reads the same way)
    assert {"method", "best_params", "best_val_score", "test_score",
            "elapsed_s", "n_evals", "notes"} <= fields
    # Project extras (rubric: Recall@5, p95 latency, multi-fidelity stats)
    assert {"holdout_recall5", "holdout_p95_ms", "n_pruned"} <= fields


def test_methods_registry_matches_lab():
    """The 5 methods are the same 5 the prof's notebook compares (Grid, Random,
    Bayesian, Hyperband, BOHB) — order matters for notebook rendering."""
    from csai415.hpo_methods import METHODS
    assert METHODS == ("grid", "random", "tpe_bayesian", "hyperband", "bohb")


def test_grid_search_space_is_finite_and_small():
    """Grid Search budget should be comparable to a single random batch, not
    explode. We discretize for the comparison to be fair on wall-clock."""
    from csai415.hpo_methods import GRID_SEARCH_SPACE
    n = 1
    for v in GRID_SEARCH_SPACE.values():
        n *= len(v)
    assert 8 <= n <= 64, f"grid has {n} cells — adjust GRID_SEARCH_SPACE"


@pytest.mark.skipif(not _have_data(), reason="corpus/gold not present")
def test_grid_runner_end_to_end():
    """Grid + tiny 4-cell grid: just verifies the runner wires up Optuna,
    holdout eval, and RunResult assembly without errors."""
    from csai415.hpo_methods import RunResult, run_grid
    from csai415.retrieve import load_chunks

    chunks = load_chunks().head(200)  # small corpus -> fast BM25/dense build
    with GOLD_JSONL.open(encoding="utf-8") as f:
        queries = [json.loads(line) for line in f][:10]
    tune, holdout = queries[:8], queries[8:]

    tiny_space = {
        "metric":        ["cosine"],
        "svd_dim":       [None],
        "normalize":     [True],
        "hybrid_weight": [0.0, 0.5, 1.0],     # 3 cells
        "candidate_k":   [10],
    }
    result = run_grid(chunks, tune, holdout,
                      search_space=tiny_space, storage_dir=None)

    assert isinstance(result, RunResult)
    assert result.method == "grid"
    assert result.n_evals == 3                  # all 3 grid cells ran
    assert result.n_pruned == 0                 # Grid uses NopPruner
    assert 0.0 <= result.best_val_score <= 1.0
    assert 0.0 <= result.test_score <= 1.0


@pytest.mark.skipif(not _have_data(), reason="corpus/gold not present")
def test_multifidelity_wiring_via_hyperband():
    """Hyperband exercises the multi-fidelity code path: build_objective with
    multifidelity=True, trial.report at each rung, and the pruning machinery.
    With only 2 trials Optuna won't actually prune (not enough history) — the
    test just verifies the code runs and reports look sane."""
    from csai415.hpo_methods import RunResult, run_hyperband
    from csai415.retrieve import load_chunks

    chunks = load_chunks().head(200)
    with GOLD_JSONL.open(encoding="utf-8") as f:
        queries = [json.loads(line) for line in f][:20]
    tune, holdout = queries[:15], queries[15:]   # ladder caps at 15 < 240

    result = run_hyperband(chunks, tune, holdout, n_trials=2, storage_dir=None)

    assert isinstance(result, RunResult)
    assert result.method == "hyperband"
    # n_evals counts COMPLETE trials, n_pruned counts pruned. Both possible
    # outcomes are valid here — sum should equal the trials we asked for or
    # less (Optuna may complete fewer if it hits a duplicate suggestion).
    assert result.n_evals + result.n_pruned <= 2
    assert result.n_evals >= 1                   # at least one trial ran
    assert "ladder" in result.notes              # multi-fidelity attribution

    # The RunResult should carry holdout metrics
    assert 0.0 <= result.holdout_recall5 <= 1.0
    assert result.holdout_p95_ms >= 0.0
