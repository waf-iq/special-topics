"""Task 5 — Ablation + multi-objective HPO over GraphRAG knobs. Owner: Yehia Noureldin.

This module is a RUNNER, not a config picker. It is built against the frozen
H0 contracts only:

    GraphRAGExecutor.answer(query, k, mode, rerank) -> AnswerResult
    evaluate_answers(answer_fn, gold) -> {faithfulness, answer_relevance,
                                           recall5, p95_latency_ms, n}

I never import or edit graph_select.py, rerank.py, or answer.py directly.
I only call the executor's public .answer() method, so whatever Tasks 1/2/3
ship underneath is invisible to this file — that's the point of contracts.

Two deliverables live here:
    1. run_ablation_grid()  -> the 3-mode x 2-rerank factorial table.
    2. run_nsga2_search()   -> Optuna NSGA-II Pareto search over continuous/
                               categorical GraphRAG knobs, on top of whichever
                               mode the ablation table recommends.

WHY OPTUNA + NSGA-II OVER THE ALTERNATIVES (see ai_logs/yehia_d3.md for the
full comparison table):
    - Grid search: combinatorial explosion, and each evaluation calls a live
      Groq judge (RAGAS) under a 30 rpm free-tier cap. Not affordable here.
    - Random search / single-objective Bayesian opt: both require collapsing
      faithfulness and latency into one scalar BEFORE search, which silently
      bakes in a trade-off weight the team never agreed on. The brief asks
      for a Pareto front, not a pre-weighted scalar.
    - pymoo NSGA-II: handles continuous knobs well but is awkward with mixed
      categorical knobs (retrieval_mode, rerank on/off) and would introduce a
      second HPO framework alongside D1's Optuna/BOHB-based work.
    - Optuna's NSGAIISampler: native categorical+continuous search space,
      same API the rest of the team already uses (D1 runcard), built-in
      study.best_trials Pareto front output, built-in plot_pareto_front().
      => Chosen.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import optuna
from optuna.samplers import NSGAIISampler

RetrievalMode = Literal["vector", "graph", "hybrid"]


# --------------------------------------------------------------------------- #
# Part 1 — Ablation grid (factorial, not search)
# --------------------------------------------------------------------------- #

@dataclass
class AblationRow:
    mode: RetrievalMode
    rerank: bool
    faithfulness: float
    answer_relevance: float
    recall5: float
    p95_latency_ms: float
    n: int


def run_ablation_grid(
    executor,                       # GraphRAGExecutor instance (Task 7 wires this in)
    evaluate_answers: Callable,     # the frozen eval.evaluate_answers contract
    gold: list[dict],               # qa_answers.jsonl rows (Ahmad's Task 4 output)
    k: int = 5,
) -> list[AblationRow]:
    """Run the full 3 (mode) x 2 (rerank) factorial grid.

    A factorial grid, not a sampled search, on purpose: with only 6 cells we
    can afford to run every cell exactly once and see interaction effects
    (does rerank help hybrid as much as it helps vector-only?) that a sampler
    would only show probabilistically.
    """
    modes: list[RetrievalMode] = ["vector", "graph", "hybrid"]
    rerank_options = [False, True]
    rows: list[AblationRow] = []

    for mode, rerank in itertools.product(modes, rerank_options):
        print(f"[ablation] mode={mode} rerank={rerank} ...")

        def answer_fn(query: str, _mode=mode, _rerank=rerank):
            # eval.py's evaluate_answers expects answer_fn to return the raw
            # AnswerResult object (it does res.answer internally), not an
            # already-unwrapped string. Confirmed from the real traceback:
            # eval.py line 107 does `answers.append(res.answer)`.
            return executor.answer(query, k=k, mode=_mode, rerank=_rerank)

        metrics = evaluate_answers(answer_fn=answer_fn, gold=gold)

        rows.append(AblationRow(
            mode=mode,
            rerank=rerank,
            faithfulness=metrics["faithfulness"],
            answer_relevance=metrics["answer_relevance"],
            recall5=metrics["recall5"],
            p95_latency_ms=metrics["p95_latency_ms"],
            n=metrics["n"],
        ))

    return rows


def ablation_rows_to_markdown(rows: list[AblationRow]) -> str:
    """Render the ablation table as markdown for the report."""
    header = "| mode | rerank | faithfulness | answer_relevance | recall@5 | p95_latency_ms | n |"
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for r in sorted(rows, key=lambda r: -r.faithfulness):
        lines.append(
            f"| {r.mode} | {r.rerank} | {r.faithfulness:.3f} | "
            f"{r.answer_relevance:.3f} | {r.recall5:.3f} | "
            f"{r.p95_latency_ms:.0f} | {r.n} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Part 2 — NSGA-II multi-objective search
# --------------------------------------------------------------------------- #

@dataclass
class HPOResult:
    study: optuna.Study
    best_trials: list[optuna.trial.FrozenTrial]
    knee_point: optuna.trial.FrozenTrial | None = None


def _suggest_knobs(trial: optuna.Trial, fixed_mode: RetrievalMode) -> dict:
    """Define the search space. fixed_mode is locked to whatever
    run_ablation_grid recommended — NSGA-II then tunes the knobs WITHIN
    that mode rather than re-searching mode (already settled by the grid).
    """
    return {
        "mode": fixed_mode,
        "rerank": trial.suggest_categorical("rerank", [True, False]),
        # candidate_k: how many chunks survive vector/graph retrieval
        # before rerank/fusion sees them. Bigger = slower, maybe more faithful.
        "candidate_k": trial.suggest_int("candidate_k", 5, 40, step=5),
        # top_n: how many of those candidates actually get reranked.
        # Only meaningful when rerank=True; harmless no-op otherwise.
        "rerank_top_n": trial.suggest_int("rerank_top_n", 3, 15),
        # k: final number of citations passed to the answerer.
        "k": trial.suggest_int("k", 3, 8),
    }


def run_nsga2_search(
    executor,
    evaluate_answers: Callable,
    gold: list[dict],
    fixed_mode: RetrievalMode,
    n_trials: int = 40,
    population_size: int = 10,
    seed: int = 42,
) -> HPOResult:
    """Optuna NSGA-II search over (rerank, candidate_k, rerank_top_n, k).

    Two objectives, BOTH directions declared explicitly so Optuna doesn't
    guess: maximize faithfulness, minimize p95 latency.
    """

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        knobs = _suggest_knobs(trial, fixed_mode)

        def answer_fn(query: str, _knobs=knobs):
            # Same fix as run_ablation_grid: return the raw AnswerResult,
            # eval.py unwraps .answer itself.
            return executor.answer(
                query,
                k=_knobs["k"],
                mode=_knobs["mode"],
                rerank=_knobs["rerank"],
            )

        t0 = time.perf_counter()
        metrics = evaluate_answers(answer_fn=answer_fn, gold=gold)
        wall_ms = (time.perf_counter() - t0) * 1000

        # Prefer the harness's own p95 if it computed one across the gold
        # set; fall back to this trial's wall-clock as a proxy otherwise.
        latency = metrics.get("p95_latency_ms", wall_ms)

        trial.set_user_attr("faithfulness", metrics["faithfulness"])
        trial.set_user_attr("answer_relevance", metrics["answer_relevance"])
        trial.set_user_attr("p95_latency_ms", latency)

        return metrics["faithfulness"], latency

    sampler = NSGAIISampler(population_size=population_size, seed=seed)
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # faithfulness up, latency down
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    best_trials = study.best_trials  # the Pareto front
    knee = _pick_knee_point(best_trials)

    return HPOResult(study=study, best_trials=best_trials, knee_point=knee)


def _pick_knee_point(
    pareto_trials: list[optuna.trial.FrozenTrial],
) -> optuna.trial.FrozenTrial | None:
    """Pick the 'knee' of the Pareto front: the point with the steepest
    marginal trade-off, i.e. where giving up a little latency stops buying
    much faithfulness.

    Method: normalize both objectives to [0,1], then pick the point closest
    to the ideal corner (max faithfulness, min latency) by Euclidean distance.
    This is the simplest defensible knee-finder (doctor's §7 lecture uses the
    same min-distance-to-utopia-point method) — fancier curvature-based knee
    detectors (menger curvature, kneedle algorithm) exist but need a smooth,
    densely-sampled front; with ~10-40 NSGA-II trials our front is too sparse
    for curvature estimation to be stable, so min-distance-to-utopia is both
    simpler AND more robust at this sample size.
    """
    if not pareto_trials:
        return None

    faiths = [t.values[0] for t in pareto_trials]
    lats = [t.values[1] for t in pareto_trials]

    f_min, f_max = min(faiths), max(faiths)
    l_min, l_max = min(lats), max(lats)

    def norm(v, lo, hi):
        return 0.5 if hi == lo else (v - lo) / (hi - lo)

    best_trial, best_dist = None, float("inf")
    for t in pareto_trials:
        nf = 1 - norm(t.values[0], f_min, f_max)   # distance from max faithfulness
        nl = norm(t.values[1], l_min, l_max)        # distance from min latency
        dist = (nf ** 2 + nl ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_trial = dist, t

    return best_trial


def pareto_to_markdown(result: HPOResult) -> str:
    """Render the Pareto front + knee-point recommendation as markdown."""
    lines = [
        "| trial | rerank | candidate_k | rerank_top_n | k | faithfulness | p95_latency_ms | knee? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    knee_num = result.knee_point.number if result.knee_point else None
    for t in sorted(result.best_trials, key=lambda t: t.values[1]):
        marker = "⭐" if t.number == knee_num else ""
        lines.append(
            f"| {t.number} | {t.params['rerank']} | {t.params['candidate_k']} | "
            f"{t.params['rerank_top_n']} | {t.params['k']} | "
            f"{t.values[0]:.3f} | {t.values[1]:.0f} | {marker} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Real run — data/gold/qa_answers.jsonl now exists (Ahmad, Task 4, 40 real
# hand-curated arXiv cs.CL questions). This block runs the actual ablation
# grid against the real GraphRAGExecutor + real evaluate_answers and prints
# the real table. The notebook (04_graphrag_ablation.ipynb) does the same
# thing but also renders the NSGA-II Pareto plot — run this file directly
# for a quick terminal check, run the notebook for the full report artifact.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    from pathlib import Path

    from csai415.graphrag import get_default_executor
    from csai415.eval import evaluate_answers

    gold_path = Path("data/gold/qa_answers.jsonl")
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(gold)} real gold rows from {gold_path}")

    executor = get_default_executor()

    print("\n=== Running real ablation grid (6 cells: 3 modes x rerank on/off) ===")
    print("This calls Groq (RAGAS judge) per cell — expect this to take a few minutes")
    print("under the 30 rpm free-tier cap.\n")

    rows = run_ablation_grid(
        executor=executor,
        evaluate_answers=evaluate_answers,
        gold=gold,
        k=5,
    )

    print("\n=== Real ablation results ===")
    print(ablation_rows_to_markdown(rows))

    # Save the real table to disk so it can be pasted into the D3 report
    # without retyping it.
    out_path = Path("reports/D3/ablation_table.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# T5 — GraphRAG Ablation Grid (real run)\n\n" + ablation_rows_to_markdown(rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {out_path}")

    print(
        "\nNSGA-II Pareto search NOT run from this script — run "
        "notebooks/04_graphrag_ablation.ipynb for the full Pareto front + plot, "
        "since it also needs WINNING_MODE set from the table above first."
    )
