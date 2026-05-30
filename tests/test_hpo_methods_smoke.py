"""Smoke tests for the multi-method HPO comparison (D1 rework tasks B1 + B3).

We don't run the full 5×80-trial comparison here — that takes ~hours. We just
verify the API surface, the RunResult dataclass shape, that the Grid runner
works end-to-end on a tiny grid, and that the multi-fidelity wiring (trial.report
+ pruning) doesn't break under Hyperband. Random/TPE/BOHB share scaffolding
with Grid/Hyperband, so they're covered transitively.

B3 tests verify write_blessed_runcard() against the actual B1 artifacts on
disk — they skip if those artifacts haven't been generated yet (fresh clone).

Tests skip if the corpus or gold files are absent so a fresh clone still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
GOLD_JSONL = Path("data/gold/qa.jsonl")
COMPARISON_CSV = Path("reports/sampler_comparison.csv")
BOHB_STUDY_DB  = Path("studies/csai415-d1-bohb.db")


def _have_data() -> bool:
    return CHUNKS_PARQUET.exists() and GOLD_JSONL.exists()


def _have_b1_artifacts() -> bool:
    return COMPARISON_CSV.exists() and BOHB_STUDY_DB.exists()


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


# --------------------------------------------------------------------------- #
# B3 — blessed-runcard writer
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (_have_data() and _have_b1_artifacts()),
    reason="needs B1 artifacts on disk (run `python -m csai415.hpo_methods` first)",
)
def test_write_blessed_runcard_from_b1_artifacts(tmp_path):
    """End-to-end: read the actual BOHB study + comparison CSV that B1 produced,
    write a v3 runcard to tmp_path, parse it back, and check the blessed-method
    fields are populated.

    Skips baseline recompute (`recompute_baselines=False`) so the test stays
    fast — that path is separately covered by `test_baselines_recompute_runs`.
    """
    import yaml
    from csai415.hpo_methods import write_blessed_runcard

    out = tmp_path / "rework_runcard.yaml"
    written = write_blessed_runcard(
        blessed_method="bohb",
        out_path=out,
        recompute_baselines=False,
    )
    assert written == out
    card = yaml.safe_load(out.read_text(encoding="utf-8"))

    # Schema + rework-specific fields
    assert card["schema_version"] == "3"
    assert card["automl"]["blessed_method"] == "bohb"
    assert "comparison" in card["automl"]
    assert "comparison_csv" in card["automl"]

    # The comparison embeds 5 method rows in lab order (CSV preserves it).
    methods_in_table = [row["method"] for row in card["automl"]["comparison"]]
    assert len(methods_in_table) == 5
    assert set(methods_in_table) == {"grid", "random", "tpe_bayesian", "hyperband", "bohb"}

    # best_params must be a real dict (not a string), and must match the
    # blessed row's params — that's the schema's whole point.
    blessed_row = next(r for r in card["automl"]["comparison"] if r["method"] == "bohb")
    assert isinstance(blessed_row["best_params"], dict)
    assert card["automl"]["best_params"] == blessed_row["best_params"]

    # Sampler/pruner config records *intent* (not just Optuna's serialization).
    assert card["automl"]["sampler"]["class"] == "TPESampler"
    assert card["automl"]["pruner"]["class"] == "HyperbandPruner"
    assert card["automl"]["pruner"]["min_resource"] == 1
    assert card["automl"]["pruner"]["reduction_factor"] == 2

    # Winner metrics come from the CSV row, not a fresh eval
    assert 0.0 <= card["metrics"]["winner_tune"]["ndcg5"] <= 1.0
    assert 0.0 <= card["metrics"]["winner_holdout"]["ndcg5"] <= 1.0
    # Baselines skipped this test (see recompute_baselines=False above)
    assert "baselines_holdout" not in card["metrics"]


@pytest.mark.skipif(
    not (_have_data() and _have_b1_artifacts()),
    reason="needs B1 artifacts on disk (run `python -m csai415.hpo_methods` first)",
)
def test_write_blessed_runcard_baselines_recompute(tmp_path):
    """Slower path: recompute_baselines=True actually evaluates the three
    legacy baselines (BM25 / dense / default-hybrid) on the holdout. The
    v2-shaped `metrics.baselines_holdout` keys must appear so Pair C and Pair A
    don't have to change their readers."""
    import yaml
    from csai415.hpo_methods import write_blessed_runcard

    out = tmp_path / "rework_runcard.yaml"
    write_blessed_runcard(blessed_method="bohb", out_path=out, recompute_baselines=True)
    card = yaml.safe_load(out.read_text(encoding="utf-8"))

    baselines = card["metrics"]["baselines_holdout"]
    assert set(baselines.keys()) == {"bm25_only", "dense_only", "default_hybrid"}
    for name, m in baselines.items():
        assert set(m.keys()) == {"ndcg5", "recall5", "p95_latency_ms"}
        assert 0.0 <= m["ndcg5"] <= 1.0, f"{name}: {m}"


def test_blessed_method_lookup_table_covers_registry():
    """The sampler/pruner config table must have an entry per METHODS — if a
    new method is added to the registry without updating _METHOD_SAMPLER_PRUNER,
    write_blessed_runcard will KeyError. Catch that here."""
    from csai415.hpo_methods import METHODS, _METHOD_SAMPLER_PRUNER
    assert set(_METHOD_SAMPLER_PRUNER.keys()) == set(METHODS)
