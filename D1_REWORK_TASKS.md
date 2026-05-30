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

#### Task B1 — Multi-sampler comparison
**Owner:** WAFIQ

- In `src/csai415/automl.py`, add a `SAMPLERS` registry and a `run_multi_sampler_studies()` driver that runs the same search space, same 240-query tune set, same 80 trials under at least **3 samplers**: `TPESampler` (current), `RandomSampler`, `CmaEsSampler`. Optional fourth: a separate FLAML study (the brief literally says "Optuna/FLAML").
- Each sampler writes its own SQLite DB in `studies/`.
- Output `reports/sampler_comparison.csv`: rows = samplers, cols = `best_tune_ndcg5`, `holdout_ndcg5`, `holdout_recall5`, `holdout_p95_ms`, `wall_clock_sec`.
- Winner is picked on **holdout** NDCG@5 (not tune — that's how we got burned on overfit last time).
- Update `runcard.yaml` schema to record `sampler.winner` and the full comparison.

#### Task B2 — Expand search space + ablation
**Owner:** Ahmed Soliman

- The rubric Excellent line is "**clear** search space". Currently the space is 5 dims. Add at least one defensible new dimension — e.g., BM25 `k1`/`b`, or query-prefix toggle on/off (BGE has a prompt for asymmetric retrieval).
- Surface the new params on `RetrieverConfig` in `src/csai415/retrieve.py` and on the Optuna trial in `automl.py`.
- Produce `reports/search_space_ablation.csv`: hold the winner fixed, drop each dim back to its default, re-evaluate on holdout. This is the "did the AutoML actually need this dimension" evidence.

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

- Restructure `src/csai415/mlflow_tracking.py` so each sampler from B1 gets its own parent MLflow run, with the 80 trials as child runs.
- `reports/mlflow_top5.md` becomes per-sampler (top-5 per sampler + an overall blessed run).
- Compare-runs screenshot should now show TPE vs Random vs CMA-ES side by side — that's the new headline for Musab's slice.

---

## Wave 2 — runs after Wave 1 lands

### Pair B

#### Task B3 — Regenerate the runcard
**Owner:** WAFIQ
**Depends on:** B1, B2

- After B1 + B2 are merged, run `run_and_record()` end-to-end against the chosen sampler + expanded space.
- New `configs/winning_runcard.yaml` committed.
- `notebooks/01_automl.ipynb` regenerates `optimization_history.png`, `param_importances.png`, `winner_vs_baselines.png`, plus a new `sampler_comparison.png` (bar chart of holdout NDCG@5 per sampler).

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
