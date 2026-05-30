# HPO Method Comparison (D1 rework — task B1)

All methods tune on the same 240-query stratified-split tune set (seed=42); each winner is re-evaluated on the 60-query holdout. Ranked by holdout NDCG@5.

| Rank | Method | Tune NDCG@5 | Holdout NDCG@5 | Recall@5 | p95 ms | Trials | Pruned | Wall-clock | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `grid` | 0.7165 | 0.5649 | 0.6489 | 106.6 | 48 | 0 | 791.0s | discrete grid, 48 cells, full eval per cell |
| 2 | `bohb` | 0.7170 | 0.5649 | 0.6489 | 128.5 | 27 | 53 | 1637.7s | TPE + HyperbandPruner, ladder (30, 60, 120, 240) |
| 3 | `tpe_bayesian` | 0.7166 | 0.5646 | 0.6322 | 102.0 | 80 | 0 | 1349.0s | TPE multivariate, 80 trials (matches original D1 study) |
| 4 | `random` | 0.7141 | 0.5610 | 0.6267 | 105.1 | 80 | 0 | 1435.8s | random sampling, 80 trials, full eval per trial |
| 5 | `hyperband` | 0.7141 | 0.5610 | 0.6267 | 102.3 | 18 | 62 | 1111.1s | Random + HyperbandPruner, ladder (60, 120, 240) |
