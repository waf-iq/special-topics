# D1 Rework — Team Tasks

We received **20/50 on AutoML** and **30/50 on Online learning** for D1. This document is the rework plan to close the gap before D2 starts. The original submission still ships; we are layering corrections on top of it on a new branch.

**Order matters.** Wave 1 tasks can be done in parallel. Wave 2 tasks depend on Wave 1 outputs. Wave 3 (report) depends on everything else.

---

## What the marker said (decoded)

| Rubric item | Score | Marker note | What it means in code |
|---|---|---|---|
| AutoML design (6%) | 20/50 | "just tried one optimiser. Just to tick the box?" | `automl.py` only uses `TPESampler`. Need ≥2–3 samplers (or Optuna vs FLAML) to show a real choice was made. |
| Online learning (6%) | 30/50 | "One model, no warmup, too little data to test, 200 events!" | `online.py` has one ε-greedy contextual bandit, deterministic cold start (no random-exploration warmup), `n_events=200` over 60 holdout queries (~3 reuses per query — statistically thin). |

Report quality (3%) was not specifically called out, but it must be rewritten to cover the new evidence.

---

## Wave 1 — start in parallel (no cross-dependencies)

### Pair B — AutoML (WAFIQ + Ahmed Soliman)

#### Task B1 — Multi-method HPO comparison ✅ DONE (2026-05-30)
**Owner:** WAFIQ

Recalibrated up from "≥3 samplers" to the prof's Week 02 HPO lab framing: **five methods** (Grid, Random, Bayesian/TPE, Hyperband multi-fidelity, BOHB) per `labs/Week 02 - HPO-tutorial.ipynb`. Implemented in `src/csai415/hpo_methods.py` (not `automl.py`) to keep the old TPE-only path intact for backwards compat.

**Outcome:** All five methods converge to the same configuration neighborhood (`metric=l2, svd_dim=None, normalize=False, hybrid_weight ∈ [0.73, 0.83], candidate_k ∈ [24, 34]`). Paired bootstrap on the 60 holdout queries showed Grid and BOHB produce **identical NDCG@5 on every single holdout query** (the 1.5pp `hybrid_weight` difference doesn't flip top-5 ordering), and TPE is within the same bootstrap CI.

**Blessed method (Option C):** BOHB — best lab-narrative pick (53/80 trials pruned, continuous-space search), tied with Grid empirically. Reported alongside Grid and TPE as "statistically indistinguishable" in the rework report.

**Outputs produced:**
- `reports/sampler_comparison.csv` + `.md` — 5-row leaderboard
- `studies/csai415-d1-{grid,random,tpe_bayesian,hyperband,bohb}.db` — one Optuna SQLite per method
- `tests/test_hpo_methods_smoke.py` — 5 tests, all green

Run with `python -m csai415.hpo_methods`. See [[project-b1-blessed-winner]] (in memory) for the bootstrap detail.

#### Task B2 — Search-space ablation (REFRAMED from "expand + ablate")
**Owner:** Ahmed Soliman

Originally scoped as "add a new dimension to the search space + ablate". After B1 showed every method lands at the same answer, the case for **adding** a new dimension in D1-rework is weak — defer that to D2. Keep the **ablation half** of B2, which directly addresses the rubric line "clear search space" by showing each existing dim actually contributes.

- Hold the BOHB-blessed config fixed: `metric=l2, svd_dim=None, normalize=False, hybrid_weight=0.825, candidate_k=25`.
- For each dim in `RetrieverConfig`, drop it back to its `RetrieverConfig` default (cosine / None / True / 0.5 / 10) one at a time and re-evaluate on the 60-query holdout.
- Output `reports/search_space_ablation.csv`: rows = dims, cols = `dim_name`, `value_blessed`, `value_default`, `holdout_ndcg5_blessed`, `holdout_ndcg5_default`, `delta`.
- Defensible new-dim work (BM25 `k1`/`b`, query-prefix toggle) moves to D2 with a one-line note in the D1 report.
- **Does not block B3** — can run in parallel since B3 only consumes B1 output.

### Pair C — Online learning (Abdurlahman + Yehia)

#### Task C1 — Multiple River learners + warmup
**Owner:** Abdurlahman Alali

- Address the "one model" + "no warmup" critiques. Implement at least **3 learner variants** behind a `build_learner(kind=...)` switch in `src/csai415/online.py`:
  1. `eps_greedy_contextual` — current bandit (keep as-is, this is the baseline now).
  2. `eps_greedy_noncontext` — per-action running means only, no features. This is the control: does the contextual signal actually help, or is the win just exploration?
  3. `logistic_bandit` — `river.linear_model.LogisticRegression` per action (binary reward → log-likelihood is the natural fit).
- **Warmup phase**: configurable `warmup_events: int` (default e.g. 100). During warmup the learner picks actions **uniformly at random** and updates normally — no cold-start deterministic action, no ε-greedy. After warmup, switch to the variant's normal policy. Document why warmup exists (without it the per-action regressors are biased toward the AutoML-winning action because that's the only one that gets samples).
- Add a smoke test per variant.

#### Task C2 — Scale the event stream + multi-drift
**Owner:** Yehia Noureldin

- Address "too little data — 200 events!". In `simulate_feedback_stream`:
  - Bump default `n_events` to **≥2000** (200 → 2000 is 10×; that's ~33 reuses per holdout query, which is acceptable for a prequential study).
  - Change `drift_at: int` to `drift_points: list[int]` — plant **two** drifts (e.g., events 800 and 1500) so the recovery cycle is observed twice. The current single-drift schedule is too thin to make a statistical claim.
  - Re-sweep ADWIN `delta` against the new stream length and rewrite the `build_drift_detector` docstring with the new sweep table. With 2000 events tighter deltas should fire properly.
- Update `OnlineLearnerState` to track per-window pre/post-drift NDCG@5 for each drift point.

### Solo — MLflow

#### Task M1 — MLflow supports multiple studies
**Owner:** Musab

- Restructure `src/csai415/mlflow_tracking.py` so each of the 5 method study DBs from B1 (`studies/csai415-d1-{grid,random,tpe_bayesian,hyperband,bohb}.db`) gets its own parent MLflow run, with each method's trials as child runs.
- `reports/mlflow_top5.md` becomes per-method (top-5 per method + an overall blessed run pointing at the BOHB winner).
- Compare-runs screenshot should now show all 5 methods (Grid / Random / TPE / Hyperband / BOHB) side by side — that's the new headline for Musab's slice.
- **Important:** the old single-study constants (`STUDY_NAME = "csai415-d1-knn"`, `STUDY_STORAGE = "sqlite:///studies/csai415-d1-knn.db"`) are dead code paths now. Either delete or update to read from the rework studies.

---

## Wave 2 — runs after Wave 1 lands

### Pair B

#### Task B3 — Regenerate the runcard
**Owner:** WAFIQ
**Depends on:** B1 (done) — **no longer blocked by B2** since search space is unchanged

Inputs are already on disk from B1; no recompute needed:
- `studies/csai415-d1-bohb.db` (blessed method's full trial history)
- `reports/sampler_comparison.csv` (5-method leaderboard for the comparison section)

Tasks:
- Extend `csai415.runcard.write_runcard()` to accept a `comparison` table + `blessed_method` name. Schema bump to v3.
- Add `csai415.hpo_methods.write_blessed_runcard()` — reads the BOHB study + comparison CSV, calls `write_runcard()` with the right kwargs. Wire it into the CLI via `--no-runcard` flag (default = write).
- Notebook regeneration (`optimization_history.png`, `param_importances.png`, `winner_vs_baselines.png`, new `sampler_comparison.png`) — handled in `notebooks/01_automl.ipynb` as a follow-up, NOT part of the runcard-write code path.
- New `configs/winning_runcard.yaml` overwrites the original. Old version recoverable from git SHA `a5662cc`.

### Pair C

#### Task C3 — Multi-model prequential figure
**Owner:** Yehia + Abdurlahman (jointly)
**Depends on:** B3, C1, C2

- Uses the updated runcard from B3 (so the static baseline weight is correct).
- Run all 3 variants from C1 against the 2000-event stream from C2, plus the static-AutoML-weight baseline.
- New `reports/prequential.png` overlays all 4 curves.
- New `reports/online_learning_results.csv`: rows = {static, eps_greedy_contextual, eps_greedy_noncontext, logistic_bandit}, cols = pre-drift NDCG@5, post-drift-1 NDCG@5, post-drift-2 NDCG@5, ADWIN firings count.
- **Headline claim must be quantitatively defensible**: either ≥5% post-drift lift vs static (the brief's bar), or a documented why-not with evidence. If contextual beats non-contextual by ≥5%, that's also a publishable finding — it justifies the features.

### Solo

#### Task M2 — Replay the new studies into MLflow
**Owner:** Musab
**Depends on:** B3

- Replay all sampler studies, tag the blessed run, regenerate `mlflow_top5.md` and `mlflow_parallel_coords.png`.

---

## Wave 3 — report rewrite (runs last)

### Pair A (Ahmad Fraij + Yousef Alsakkaf)

#### Task A1 — Rewrite `reports/D1_report.md`
**Owner:** Yousef Alsakkaf
**Depends on:** B3, C3, M2

- Still 2 pages max. New sections to add / rewrite:
  - **AutoML results:** sampler comparison table from B1 + one paragraph defending the chosen sampler. Search-space ablation table from B2.
  - **Online learning:** multi-model comparison table from C3. Explain the warmup phase (1 sentence). Updated quantitative claim.
  - **Decisions/pitfalls:** honestly note that the original submission used one sampler and one learner; explain what the comparison revealed.
- Replace `prequential.png` and `winner_vs_baselines.png` references with the regenerated versions.

#### Task A2 — Smoke tests stay green
**Owner:** Ahmad Fraij
**Depends on:** B3, C3

- Add a multi-sampler smoke test (1 trial each, just verifies the registry wires up).
- Add a multi-learner smoke test (verifies each `kind=` builds and accepts one update).
- `pytest tests/test_smoke.py` must remain green after all of Wave 1–2.

---

## Bonus — not explicitly flagged by the marker but worth fixing

The AutoML rubric for Excellent says "sensible objective (**NDCG/Recall + latency**)". The current objective is raw NDCG@5 with latency recorded as a `user_attr` only. If B1 has spare cycles, run one extra study with a latency-penalized objective (e.g., `NDCG@5 - 1e-4 * p95_latency_ms`) and add it as a row in the sampler comparison. Cheap, defensible, and closes the "+latency" gap in the rubric.

---

## Branching and PR convention

- One feature branch per task: `rework/b1-multi-sampler`, `rework/c1-multi-learner-warmup`, etc.
- PR into `main`. Each PR needs review from one teammate outside the pair (same rule as the original D1 brief).
- Every member must keep ≥2 commits under their own GitHub identity across the rework — same as the original.
- **AI logs:** each member should keep their AI chat history for the rework in `ai_logs/<name>_rework.md` (separate from the original D1 log).
