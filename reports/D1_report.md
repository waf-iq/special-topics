# CSAI415 D1 — Hybrid Retrieval + AutoML + Online Learning over SciFact

**Team Devflexi**: Ahmed Soliman, Ahmad Fraij, Abdurlahman Alali, Musab, WAFIQ Akram ABO DAKEN, Yehia Noureldin, Yousef Alsakkaf.

## Methods

Corpus is SciFact (BEIR, 5,183 abstracts) loaded via `ir_datasets`, plus 5 arXiv cs.CL PDFs ingested end-to-end as a PDF-pipeline demo (357 chunks, not used in evaluation) for a total of 6,020 chunks. Long abstracts (>512 tokens, ~9%) are split with 50-token overlap; everything else is one chunk per abstract. The gold set is SciFact's 300 manually-judged test claims with multi-doc relevance, stratified-split 80/20 (240 tune / 60 holdout) by whether a claim has multiple relevant documents. Embedding is `BAAI/bge-small-en-v1.5` (384-dim) with BGE's asymmetric query prefix applied only at query time. The retriever is a hybrid BM25 + dense kNN over the corpus matrix (brute-force numpy at this scale), with optional TruncatedSVD; per-query min-max-scaled scores are blended by a tunable `hybrid_weight`. AutoML is an Optuna study of 80 trials with multivariate TPE (`n_startup_trials=20`) and `NopPruner`, optimizing NDCG@5 on the 240-query tune set. The 60-query holdout is evaluated once on the winner for the generalization check. Online learning is an ε-greedy contextual bandit over a 5-action discretization of `hybrid_weight`, cold-started at the AutoML-winning value; ADWIN monitors the static probe's reward and resets the bandit on detected drift.

## Corpus and evaluation — Pair A

The ingest pipeline materialized **6,020 chunks** from SciFact (5,663) and the 5 demo arXiv PDFs (357). Of the 5,183 SciFact abstracts, **455 (≈8.8%)** exceeded the embedder's 512-token window and were split with 50-token overlap; the rest are one-chunk-per-abstract. The 300-claim test split has multi-document relevance — most claims have a single gold chunk but a meaningful tail of multi-relevant claims sets the evaluation regime (NDCG@5 with binary relevance, not MRR):

| Relevant chunks per claim | 1   | 2  | 3+ |
|---------------------------|----:|---:|---:|
| Count                     | 232 | 53 | 15 |

![SciFact claim relevance distribution](qa_relevance_distribution.png)

The eval harness (`evaluate()`) returns `{ndcg5, recall5, p95_latency_ms}` per call and is shared verbatim by Pair B's Optuna objective and Pair C's prequential loop — single source of truth across the three slices. Corpus and gold files are versioned in the repo with SHA-256 captured in the runcard.

## Results — Pair B (AutoML)

Baseline-vs-AutoML, all measured on the 60-query holdout (none of these queries were seen by TPE):

| Config             | NDCG@5 | Recall@5 | p95 latency (ms) |
|--------------------|-------:|---------:|-----------------:|
| BM25 only          |  0.416 |    0.465 |            103.1 |
| Dense only         |  0.563 |    0.649 |             97.4 |
| Default hybrid     |  0.534 |    0.593 |             92.0 |
| **AutoML winner**  |**0.565**|   0.632 |            112.0 |

Winning configuration: `metric=l2, svd_dim=None, normalize=False, hybrid_weight=0.810, candidate_k=24`.

![Winner vs baselines](winner_vs_baselines.png)

**Honest overfit reading**: tune NDCG@5 = 0.717 vs holdout NDCG@5 = 0.565, a 15-point gap. The 80/20 stratified holdout caught what a single split would have hidden — without the blind evaluation the study would have reported the inflated tune number.

## Results — Pair C (Online learning)

Prequential evaluation on 200 events drawn from the 60 holdout queries, with a query-style drift planted at event 100 (natural-language claims → 2-token keyword queries). The static baseline replays the same stream at the AutoML-winning `hybrid_weight=0.81`; the adaptive learner reacts.

| Window      | Adaptive NDCG@5 | Static NDCG@5 |
|-------------|----------------:|--------------:|
| Pre-drift   |          0.5488 |        0.6000 |
| Post-drift  |          0.3317 |        0.3011 |

Post-drift delta (adaptive − static): **+3.06%**, below the 5% bar set in §6.C. ADWIN fired at event 159 — a 59-event lag from the planted drift, because the binary reward variance over a 100-event window swamps the Δ≈0.26 hit-rate shift produced by query-style drift against a well-tuned baseline (tighter `delta` values never fired at all in the empirical sweep). The +3% improvement is real but the lag eats most of the post-drift window; we report this honestly rather than tuning the bar to match.

![Prequential NDCG@5 — adaptive vs static](prequential.png)

## Experiment tracking — Solo (Musab)

The 80-trial Optuna study was replayed into MLflow (`csai415-d1-automl` experiment, SQLite backend) for compare-runs visualization and artifact tracking. The winning trial is tagged `csai415.blessed=true` with the runcard YAML and the three Pair B figures attached as artifacts. Top-5 trials by NDCG@5 (full table in `reports/mlflow_top5.md`):

| Run ID  | NDCG@5 | candidate_k | Metric | Hybrid Wt. |
|---------|-------:|------------:|--------|-----------:|
| 2016ed1 | 0.7166 |          24 | l2     |      0.810 |
| 3dd4d36 | 0.7164 |          23 | dot    |      0.834 |
| 0fbf9a0 | 0.7163 |          18 | dot    |      0.838 |
| fda2415 | 0.7152 |          47 | l2     |      0.802 |
| b24ef1c | 0.7149 |          38 | l2     |      0.798 |

All top-5 use heavily dense-leaning weights (>= 0.80) with no SVD and `l2`/`dot` distance — TPE converged to a stable region rather than wandering. The Parallel Coordinates view (below) makes this concrete: high-NDCG trials cluster on dense-heavy weights with `svd_dim=None` and `metric ∈ {l2, dot}`.

![Optuna Parallel Coordinates over the 80-trial study](mlflow_parallel_coords.png)

## Decisions and pitfalls

- **Held-out evaluation is the headline.** The 80/20 stratified split converted "winner NDCG@5 = 0.72" (overfit) into "winner NDCG@5 = 0.57 on unseen queries" — and exposed that the AutoML winner barely beats pure dense (0.565 vs 0.563) on SciFact with `bge-small-en`. The hybrid signal here is genuinely weak.
- **Naïve 50/50 hybrid (0.534) underperforms pure dense (0.563).** Default blending hurts on this corpus — dense at `bge-small` quality dominates, and BM25 adds noise at the median weight.
- **Online learning missed the 5% bar by 1.94 points.** Root cause is ADWIN lag against a binary reward signal on a strong baseline: smaller `delta` values produced zero false positives but zero true positives either; `delta=0.5` was the smallest setting that fired. Sliding-window mean over 20 events still smooths the post-drift recovery curve enough to show the adaptive learner climbing while the static curve stays flat.
- **Single-seed study.** Multi-seed stability (Optuna seed variance) is the cleanest D2 follow-up; the 60-query holdout's NDCG@5 standard error is roughly ±0.04, so a 3-seed sweep would either confirm the winner or expose noise.
- **Reproducibility.** `configs/winning_runcard.yaml` captures the git SHA, the dirty flag, python + package versions, dataset SHA-256s of `chunks.parquet` and `qa.jsonl`, sampler/pruner config, study storage path, the 240/60 split seed + indices path, and both tune and holdout metrics for the winner plus all three baselines. The same `configs/d1_split_indices.json` is read by Pair C so the adaptive learner runs on queries TPE never saw.

## Reproducibility

One-command setup is in the README. Full run: `python -c "from csai415.automl import run_and_record; run_and_record()"` regenerates the runcard; `notebooks/01_automl.ipynb` and `notebooks/02_online_learning.ipynb` regenerate the figures. SciFact is loaded from `ir_datasets`; the corpus chunks and gold qrels are versioned in the repo so a fresh clone reproduces these exact numbers.

## Licensing

Code: MIT (see `LICENSE`). Corpus: SciFact (CC BY-NC 2.0 via BEIR). Demo arXiv PDFs are open-access cs.CL submissions.
