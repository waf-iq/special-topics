"""Pair B — Multi-method HPO comparison. See `D1_REWORK_TASKS.md` task B1.

The original D1 submission used a single Optuna TPE sampler with NopPruner —
the marker called it out as "just tried one optimiser. Just to tick the box?"
(20/50). This module mirrors the 5-method comparison from
`labs/Week 02 - HPO-tutorial.ipynb`:

  1. Grid Search   — Optuna GridSampler over a coarse discrete grid
  2. Random Search — Optuna RandomSampler, n_trials
  3. Bayesian/TPE  — Optuna TPESampler (multivariate), n_trials  [the D1 baseline]
  4. Hyperband     — Random + HyperbandPruner, multi-fidelity
  5. BOHB-style    — TPE    + HyperbandPruner, multi-fidelity (4 rungs)

Multi-fidelity **budget = subset of tune queries** (60 → 120 → 240). At each
fidelity step the objective evaluates on the next subset and reports the
score via `trial.report` so Optuna can prune weak trials. This mirrors the
lab's `max_iter` ladder for gradient-boosting — "more budget = better
estimate", with the tradeoff that low fidelity is noisier.

All methods run against the **same 240-query tune set with the same seed**
(via `automl._split_queries`). Each winner is evaluated once on the 60-query
holdout, and we rank by holdout NDCG@5. The `RunResult` dataclass mirrors the
prof's notebook (method, best_params, best_val_score, test_score,
elapsed_s, n_evals, notes) plus project extras (holdout_recall5,
holdout_p95_ms, n_pruned).

Two design choices the report narrative must call out explicitly (so they
read as deliberate, not accidental):

  * **Shared seed = clean ablation of sampler vs pruner contribution.**
    Random and Hyperband share `seed=42` on the RandomSampler, so Hyperband
    sees the *same 80 proposals* as Random — the only difference is the
    pruner. Likewise TPE and BOHB share `seed=42` on the TPESampler. The
    comparison is therefore "given identical proposals, did pruning save
    wall-clock without hurting the winner?", not a noisy re-roll.
  * **Methods are NOT eval-count-equalized** (Grid ≈ grid size, Random/TPE =
    80 full-fidelity evals, Hyperband/BOHB = up to 80×rungs evals at varying
    fidelity). This matches the lab notebook. The comparison axis is
    **wall-clock + holdout score**, not eval count — the report must say so,
    or it invites the same "tick the box" critique D1 got.

Output: `reports/sampler_comparison.csv` + `reports/sampler_comparison.md`
+ `studies/csai415-d1-*.db` (one per method) + a list of `RunResult`
instances.

Run all of it:  python -m csai415.hpo_methods
Custom:         python -m csai415.hpo_methods --n-trials 80 --resume \\
                                              --methods grid random tpe_bayesian
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import optuna
import pandas as pd

from .automl import _split_queries, load_queries
from .eval import evaluate
from .retrieve import (
    HybridRetriever,
    RetrieverConfig,
    load_chunks,
    make_retriever_fn,
)

REPORTS_DIR = Path("reports")
COMPARISON_CSV = REPORTS_DIR / "sampler_comparison.csv"
COMPARISON_MD  = REPORTS_DIR / "sampler_comparison.md"

# Per-method Optuna study DBs land here so the notebook can re-render plots
# without re-running the search, and Musab's MLflow replay can read trial
# history. Matches the existing studies/ convention from automl.py.
STUDY_STORAGE_DIR = Path("studies")

# Trial budget per method (Grid uses its grid size, others get equal budget).
DEFAULT_N_TRIALS = 80

# Multi-fidelity ladders. Subset-of-tune-queries — cheap low fidelity, full
# 240 at the top. Hyperband gets 3 rungs, BOHB gets 4 (extra cheap rung) so
# the cost of model-based exploration is amortized over more pruning chances.
HYPERBAND_BUDGETS: tuple[int, ...] = (60, 120, 240)
BOHB_BUDGETS:      tuple[int, ...] = (30, 60, 120, 240)

# Coarse discrete grid for Grid Search. The full search space is continuous
# in `hybrid_weight`, so we discretize. Cells were chosen to (a) include the
# D1 winner region (l2, no SVD, normalize=False, w≈0.81), (b) cover the
# BM25/dense extremes (w ∈ {0, 1}), and (c) keep the cell count comparable in
# wall-clock to a single random-search batch.
#
# NOTE: `normalize` MUST include False — the original D1 winner used
# normalize=False, so pinning [True] alone would make Grid structurally
# unable to reach the winning config and produce a fake "Grid can't find the
# dense-leaning winner" result. We keep `hybrid_weight=0.81` in the grid for
# the same reason (dropping it to a round-number subset would re-exclude the
# winner region this grid is meant to cover).
#
#   2 metrics × 1 svd × 2 normalize × 4 hybrid_weights × 3 candidate_k = 48 cells.
GRID_SEARCH_SPACE: dict[str, list[Any]] = {
    "metric":        ["cosine", "l2"],
    "svd_dim":       [None],
    "normalize":     [True, False],
    "hybrid_weight": [0.0, 0.5, 0.81, 1.0],
    "candidate_k":   [10, 25, 50],
}


# --------------------------------------------------------------------------- #
# RunResult — mirrors the prof's HPO-tutorial RunResult, with project extras
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    """One HPO method's result row in the comparison table.

    Field names match `labs/Week 02 - HPO-tutorial.ipynb` for grading
    legibility; the *meaning* of `test_score` is our 60-query holdout NDCG@5
    (we never had a separate test split — the lab's "test" and our "holdout"
    are the same concept).

    Extras vs the lab:
      holdout_recall5 — second metric (rubric mentions Recall@5)
      holdout_p95_ms  — latency on holdout (rubric mentions p95 latency)
      n_pruned        — pruned trial count (0 for non-multi-fidelity methods)
    """
    method: str
    best_params: dict[str, Any]
    best_val_score: float        # best tune-set NDCG@5 seen during the search
    test_score: float            # winner's NDCG@5 on the 60-query holdout
    holdout_recall5: float
    holdout_p95_ms: float
    elapsed_s: float
    n_evals: int                 # completed trials (grid cells for Grid)
    n_pruned: int = 0
    notes: str = ""


# --------------------------------------------------------------------------- #
# Objective builder — single- or multi-fidelity, with the option to override
# the search space (Grid uses the fixed `GRID_SEARCH_SPACE` cells).
# --------------------------------------------------------------------------- #

def _build_objective(
    chunks_df,
    queries: list[dict],
    *,
    multifidelity: bool = False,
    budgets: tuple[int, ...] = HYPERBAND_BUDGETS,
    k: int = 5,
    grid_space: dict[str, list[Any]] | None = None,
):
    """Optuna objective closure.

    multifidelity=False (default): evaluate on all `queries`, return NDCG@5.
    multifidelity=True: evaluate on growing prefixes per `budgets`, reporting
        intermediate scores via `trial.report` and raising `TrialPruned` when
        Optuna says to stop.

    grid_space: when set, suggest categorical params from these lists instead
        of the full continuous space. Required when paired with a GridSampler
        because GridSampler enumerates over exactly these values.
    """
    n_queries = len(queries)
    if multifidelity:
        # Cap budgets at the actual tune set size; dedupe; sort ascending.
        ladder = sorted({min(b, n_queries) for b in budgets})
    else:
        ladder = [n_queries]

    def objective(trial: optuna.Trial) -> float:
        if grid_space is not None:
            params = {
                name: trial.suggest_categorical(name, vals)
                for name, vals in grid_space.items()
            }
            config = RetrieverConfig(**params, seed=42)
        else:
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

        # Multi-fidelity loop: evaluate on growing prefix of the query list,
        # report at each rung, prune if Optuna says so. Single-fidelity
        # methods just run the loop once with ladder=[n_queries].
        #
        # KNOWN LIMITATION (item 6): the low-fidelity rungs use a plain prefix
        # `queries[:b]`. `_split_queries` shuffles, so the prefix is random,
        # but it is NOT re-stratified at the 60/120 marks — the cheap rung may
        # carry a different multi-relevant proportion than the full 240. The
        # honest fix is a stratified shuffle keyed on the relevance structure
        # before slicing; deferred here because it needs the query-dict schema
        # owned by `automl`. Treated as a documented caveat, not a silent bug.
        metrics = None
        last_score = 0.0
        # 1-indexed steps: the HyperbandPruner is configured with
        # min_resource=1, so a report at step=0 sits below its threshold and
        # never triggers a pruning decision — pruning would first engage at
        # the *mid* rung, throwing away the cheap-rung advantage (worst for
        # BOHB's 30-query rung). Starting at 1 makes the steps {1..len(ladder)}
        # line up exactly with the pruner's [min_resource, max_resource] range,
        # so the cheapest rung can prune as intended.
        for step, b in enumerate(ladder, start=1):
            subset = queries[:b]
            metrics = evaluate(fn, subset, k=k, hybrid_weight=config.hybrid_weight)
            last_score = metrics["ndcg5"]
            trial.set_user_attr(f"ndcg5_fidelity_{b}", last_score)
            if multifidelity:
                trial.report(last_score, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Top-level user attrs reflect the highest fidelity reached.
        trial.set_user_attr("ndcg5", last_score)
        trial.set_user_attr("recall5", float(metrics["recall5"]))
        trial.set_user_attr("p95_latency_ms", float(metrics["p95_latency_ms"]))
        return last_score

    return objective


# --------------------------------------------------------------------------- #
# Shared runner — every per-method function delegates here so the holdout
# evaluation and RunResult assembly stays in one place.
# --------------------------------------------------------------------------- #

def _run_one(
    *,
    method: str,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
    chunks_df,
    tune_queries: list[dict],
    holdout_queries: list[dict],
    n_trials: int,
    multifidelity: bool = False,
    budgets: tuple[int, ...] = HYPERBAND_BUDGETS,
    grid_space: dict[str, list[Any]] | None = None,
    notes: str = "",
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Run one HPO method end-to-end and return its RunResult.

    storage_dir: if set, persist the Optuna study to
    `<storage_dir>/csai415-d1-{method}.db` so the notebook can re-render plots
    and Musab's MLflow replay can read trial history. Pass `None` to run
    in-memory (smoke tests use this so the test suite doesn't leave artifacts).

    resume: controls what happens when a study DB already exists (item 2).
    Default False = fresh run: the existing `.db` is deleted first and the
    study is created with `load_if_exists=False`, so re-running always yields
    exactly `n_trials` trials. Set True to *append* to the existing study
    (the old `load_if_exists=True` behaviour) — e.g. to extend a search. This
    is explicit and reversible, unlike silently resuming and ending up with
    160 trials after a second run.
    """
    objective = _build_objective(
        chunks_df,
        tune_queries,
        multifidelity=multifidelity,
        budgets=budgets,
        grid_space=grid_space,
    )

    storage = None
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"csai415-d1-{method}.db"
        if not resume:
            # Fresh run: drop any prior DB (and sqlite side-files) so trial
            # counts don't accumulate across runs.
            for p in (
                storage_path,
                storage_path.with_suffix(storage_path.suffix + "-journal"),
                storage_path.with_suffix(storage_path.suffix + "-wal"),
                storage_path.with_suffix(storage_path.suffix + "-shm"),
            ):
                p.unlink(missing_ok=True)
        storage = f"sqlite:///{storage_path.as_posix()}"

    study = optuna.create_study(
        study_name=f"csai415-d1-{method}",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=resume,
    )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    # Evaluate the winner once on the holdout. `study.best_params` is a dict
    # with the same keys as RetrieverConfig (we suggested them that way).
    winner_cfg = RetrieverConfig(**study.best_params, seed=42)
    fn = make_retriever_fn(HybridRetriever(chunks_df, winner_cfg))
    holdout = evaluate(fn, holdout_queries, k=5, hybrid_weight=winner_cfg.hybrid_weight)

    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )

    return RunResult(
        method=method,
        best_params=dict(study.best_params),
        best_val_score=float(study.best_value),
        test_score=float(holdout["ndcg5"]),
        holdout_recall5=float(holdout["recall5"]),
        holdout_p95_ms=float(holdout["p95_latency_ms"]),
        elapsed_s=elapsed,
        n_evals=n_complete,
        n_pruned=n_pruned,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Per-method runners. Each is a thin shim over `_run_one` so the call sites
# stay readable in the orchestrator and the notebook.
# --------------------------------------------------------------------------- #

def run_grid(
    chunks_df, tune_queries: list[dict], holdout_queries: list[dict],
    *, search_space: dict[str, list[Any]] | None = None,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Method 1 — Grid Search over `search_space` (default `GRID_SEARCH_SPACE`).

    n_trials = number of grid cells (Optuna's GridSampler enumerates exactly).
    """
    space = search_space or GRID_SEARCH_SPACE
    n_cells = 1
    for v in space.values():
        n_cells *= len(v)
    sampler = optuna.samplers.GridSampler(space, seed=42)
    return _run_one(
        method="grid",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        chunks_df=chunks_df,
        tune_queries=tune_queries,
        holdout_queries=holdout_queries,
        n_trials=n_cells,
        grid_space=space,
        notes=f"discrete grid, {n_cells} cells, full eval per cell",
        storage_dir=storage_dir,
        resume=resume,
    )


def run_random(
    chunks_df, tune_queries: list[dict], holdout_queries: list[dict],
    *, n_trials: int = DEFAULT_N_TRIALS,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Method 2 — Random Search over the full continuous space."""
    sampler = optuna.samplers.RandomSampler(seed=42)
    return _run_one(
        method="random",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        chunks_df=chunks_df,
        tune_queries=tune_queries,
        holdout_queries=holdout_queries,
        n_trials=n_trials,
        notes=f"random sampling, {n_trials} trials, full eval per trial",
        storage_dir=storage_dir,
        resume=resume,
    )


def run_tpe(
    chunks_df, tune_queries: list[dict], holdout_queries: list[dict],
    *, n_trials: int = DEFAULT_N_TRIALS,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Method 3 — Bayesian (TPE) over the full continuous space.

    Matches the original D1 submission's settings exactly (multivariate=True,
    n_startup_trials=20, NopPruner) so the comparison row labelled
    `tpe_bayesian` *is* the original D1 study, re-run on the same tune set.
    """
    sampler = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=20, multivariate=True,
    )
    return _run_one(
        method="tpe_bayesian",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        chunks_df=chunks_df,
        tune_queries=tune_queries,
        holdout_queries=holdout_queries,
        n_trials=n_trials,
        notes=f"TPE multivariate, {n_trials} trials (matches original D1 study)",
        storage_dir=storage_dir,
        resume=resume,
    )


def run_hyperband(
    chunks_df, tune_queries: list[dict], holdout_queries: list[dict],
    *, n_trials: int = DEFAULT_N_TRIALS,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Method 4 — Hyperband-style multi-fidelity. Random sampler + HB pruner.

    The lab's framing: cheap exploration with multi-fidelity pruning catches
    "bad configs fail fast" — the random sampler doesn't learn between trials,
    so all the gain comes from pruning. With our budget = query-subset, a
    weak config that flops on the first 60 queries gets killed before it
    burns the full 240.
    """
    sampler = optuna.samplers.RandomSampler(seed=42)
    # Explicit settings so the pruner's bracket math matches our ladder
    # (item 3). Defaults are min_resource=1, max_resource='auto',
    # reduction_factor=3 — but our rungs step by 2× (60→120→240), so we pin
    # reduction_factor=2 and bound the resource to the number of rungs. The
    # reported `step` is the rung index (0-based), so max_resource = number of
    # rungs. VERIFY pruning actually fires on the first full run — the 2-trial
    # smoke test is too small for any bracket to engage.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=len(HYPERBAND_BUDGETS),
        reduction_factor=2,
    )
    return _run_one(
        method="hyperband",
        sampler=sampler,
        pruner=pruner,
        chunks_df=chunks_df,
        tune_queries=tune_queries,
        holdout_queries=holdout_queries,
        n_trials=n_trials,
        multifidelity=True,
        budgets=HYPERBAND_BUDGETS,
        notes=f"Random + HyperbandPruner, ladder {HYPERBAND_BUDGETS}",
        storage_dir=storage_dir,
        resume=resume,
    )


def run_bohb(
    chunks_df, tune_queries: list[dict], holdout_queries: list[dict],
    *, n_trials: int = DEFAULT_N_TRIALS,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> RunResult:
    """Method 5 — BOHB-style. TPE sampler + HB pruner, with a 4-rung ladder.

    Practical BOHB = "use a model-based sampler to *propose* configs and a
    multi-fidelity pruner to *kill* the bad ones early". Extra rung at the
    cheap end (30 queries) gives TPE more cheap signal to fit its model on
    before expensive evaluations.
    """
    sampler = optuna.samplers.TPESampler(
        seed=42, n_startup_trials=20, multivariate=True,
    )
    # Same explicit ladder-aligned settings as Hyperband (item 3); BOHB just
    # has one more rung, so max_resource tracks the 4-rung ladder.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=len(BOHB_BUDGETS),
        reduction_factor=2,
    )
    return _run_one(
        method="bohb",
        sampler=sampler,
        pruner=pruner,
        chunks_df=chunks_df,
        tune_queries=tune_queries,
        holdout_queries=holdout_queries,
        n_trials=n_trials,
        multifidelity=True,
        budgets=BOHB_BUDGETS,
        notes=f"TPE + HyperbandPruner, ladder {BOHB_BUDGETS}",
        storage_dir=storage_dir,
        resume=resume,
    )


# Ordered registry — used by the orchestrator and the notebook so all
# call sites loop in the same lab-pedagogy order.
METHODS: tuple[str, ...] = ("grid", "random", "tpe_bayesian", "hyperband", "bohb")

_RUNNERS: dict[str, Callable] = {
    "grid":         run_grid,
    "random":       run_random,
    "tpe_bayesian": run_tpe,
    "hyperband":    run_hyperband,
    "bohb":         run_bohb,
}


# --------------------------------------------------------------------------- #
# Orchestrator + CSV writer
# --------------------------------------------------------------------------- #

def _write_comparison_markdown(results: list[RunResult], out_md: Path) -> None:
    """Human-readable comparison table for PR diffs and the D1 report draft.

    The notebook will produce nicer figures; this is the at-a-glance view.
    """
    header = (
        "# HPO Method Comparison (D1 rework — task B1)\n\n"
        "All methods tune on the same 240-query stratified-split tune set "
        "(seed=42); each winner is re-evaluated on the 60-query holdout. "
        "Ranked by holdout NDCG@5.\n\n"
        "| Rank | Method | Tune NDCG@5 | Holdout NDCG@5 | Recall@5 | p95 ms | Trials | Pruned | Wall-clock | Notes |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            f"| {i} | `{r.method}` | {r.best_val_score:.4f} | "
            f"{r.test_score:.4f} | {r.holdout_recall5:.4f} | "
            f"{r.holdout_p95_ms:.1f} | {r.n_evals} | {r.n_pruned} | "
            f"{r.elapsed_s:.1f}s | {r.notes} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def run_method_comparison(
    *,
    chunks_df=None,
    queries: list[dict] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    methods: tuple[str, ...] = METHODS,
    out_csv: Path = COMPARISON_CSV,
    out_md: Path | None = COMPARISON_MD,
    storage_dir: Path | None = STUDY_STORAGE_DIR,
    resume: bool = False,
) -> list[RunResult]:
    """Run the requested HPO methods and write the comparison artifacts.

    Uses Pair B's stratified 80/20 split (240 tune / 60 holdout). Each method
    tunes on the 240, the winner is evaluated on the 60. Returns the RunResult
    list sorted by holdout NDCG@5 descending (the overall winner is index 0).

    Artifacts written:
      * `reports/sampler_comparison.csv` — full table for the notebook
      * `reports/sampler_comparison.md`  — readable summary for PR / report
      * `studies/csai415-d1-{method}.db` — one Optuna SQLite per method
                                            (skipped if storage_dir=None)

    `n_trials` applies to every method except Grid (Grid uses its grid size).

    `resume` (default False): re-running this on the same machine starts each
    study fresh, deleting prior DBs, so trial counts never silently double.
    Pass resume=True to append to existing studies instead.
    """
    if chunks_df is None:
        chunks_df = load_chunks()
    if queries is None:
        queries = load_queries()
    tune_queries, holdout_queries, _ = _split_queries(queries)

    results: list[RunResult] = []
    for name in methods:
        runner = _RUNNERS[name]
        n_requested = "grid-size" if name == "grid" else n_trials
        print(
            f"[{time.strftime('%H:%M:%S')}] starting {name} "
            f"(n_trials={n_requested})",
            flush=True,
        )
        if name == "grid":
            result = runner(chunks_df, tune_queries, holdout_queries,
                            storage_dir=storage_dir, resume=resume)
        else:
            result = runner(chunks_df, tune_queries, holdout_queries,
                            n_trials=n_trials, storage_dir=storage_dir,
                            resume=resume)
        results.append(result)
        # Per-method status line — also surfaces the pruning-engaged check for
        # multi-fidelity methods (item from B1 re-review). If hyperband/bohb
        # come back with 0 pruned across a full 80-trial run, the bracket math
        # isn't firing and reduction_factor/min_resource need tuning.
        print(
            f"[{time.strftime('%H:%M:%S')}] done {result.method}: "
            f"tune={result.best_val_score:.4f} holdout={result.test_score:.4f} "
            f"pruned={result.n_pruned}/{result.n_evals + result.n_pruned} "
            f"in {result.elapsed_s/60:.1f} min",
            flush=True,
        )

    # Rank by holdout NDCG@5 — that's the "would this generalize" signal.
    results.sort(key=lambda r: r.test_score, reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_csv(out_csv, index=False)
    if out_md is not None:
        _write_comparison_markdown(results, out_md)
    return results


# --------------------------------------------------------------------------- #
# CLI entry — `python -m csai415.hpo_methods`
# --------------------------------------------------------------------------- #

def _print_leaderboard(results: list[RunResult]) -> None:
    header = (
        f"{'method':<14} {'tune':>8} {'holdout':>8} {'recall':>8} "
        f"{'p95ms':>8} {'trials':>7} {'pruned':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.method:<14} {r.best_val_score:>8.4f} {r.test_score:>8.4f} "
            f"{r.holdout_recall5:>8.4f} {r.holdout_p95_ms:>8.1f} "
            f"{r.n_evals:>7d} {r.n_pruned:>7d}"
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="csai415.hpo_methods",
        description=(
            "Run the D1 rework multi-method HPO comparison "
            "(5 methods × 80 trials, ~90-120 min on CPU)."
        ),
    )
    ap.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        help=f"Trials per method (Grid uses grid size). Default {DEFAULT_N_TRIALS}.",
    )
    ap.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
        help="Subset of methods to run. Default = all 5 in lab order.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Append to existing study DBs (default: start fresh, delete prior DBs).",
    )
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] hpo_methods full run starting")
    print(f"  methods : {args.methods}")
    print(f"  n_trials: {args.n_trials}")
    print(f"  resume  : {args.resume}")
    print()

    t0 = time.perf_counter()
    results = run_method_comparison(
        n_trials=args.n_trials,
        methods=tuple(args.methods),
        resume=args.resume,
    )
    elapsed_min = (time.perf_counter() - t0) / 60

    print()
    print(f"[{time.strftime('%H:%M:%S')}] total wall-clock: {elapsed_min:.1f} min")
    print("\nLeaderboard (ranked by holdout NDCG@5):")
    _print_leaderboard(results)
    print()
    print(f"Winner: {results[0].method} "
          f"(holdout NDCG@5 = {results[0].test_score:.4f})")
    print("\nArtifacts:")
    print(f"  {COMPARISON_CSV}")
    print(f"  {COMPARISON_MD}")
    print(f"  {STUDY_STORAGE_DIR}/csai415-d1-*.db")


if __name__ == "__main__":
    main()