# D3 Task-1 — Guidance-policy comparison

Probe set: 30 entity-anchored content questions (22 specified, 8 underspecified; seed=42); candidate pool k=30. **Hit@k** = a chunk of the target paper reached the top-k; **cand_recall** = a target chunk is anywhere in the reshaped candidate list (the signal a downstream reranker sees — this is where expansion pays off); **subgraph_prec@5** = fraction of the top-5 inside the selected subgraph. Policies apply to the hybrid pool; vector/hybrid carry no graph guidance.

## Overall

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_prec@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |   0.733 |    0.733 |         0.767 |             0.587 |              0     |             0.002 |
| hybrid    |   0.767 |    0.767 |         0.767 |             0.627 |              0.034 |             0     |
| filter    |   0.767 |    0.767 |         0.767 |             0.767 |              0.034 |             0.013 |
| booster   |   0.767 |    0.767 |         0.767 |             0.74  |              0.034 |             0.01  |
| expansion |   0.767 |    0.767 |         1     |             0.627 |              0.034 |             0.006 |

## By regime

*Specified* queries already retrieve well (hit@5 near ceiling) → graph adds **precision**. *Underspecified* queries miss the paper under vector search → graph adds **recall** (expansion injects the missed subgraph chunks; cand_recall rises).

### Regime: specified

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_prec@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |       1 |        1 |             1 |             0.8   |                  0 |             0.002 |
| hybrid    |       1 |        1 |             1 |             0.845 |                  0 |             0     |
| filter    |       1 |        1 |             1 |             1     |                  0 |             0.013 |
| booster   |       1 |        1 |             1 |             0.982 |                  0 |             0.011 |
| expansion |       1 |        1 |             1 |             0.845 |                  0 |             0.006 |

### Regime: underspecified

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_prec@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |   0     |    0     |         0.125 |             0     |              0     |             0.002 |
| hybrid    |   0.125 |    0.125 |         0.125 |             0.025 |              0.125 |             0     |
| filter    |   0.125 |    0.125 |         0.125 |             0.125 |              0.125 |             0.011 |
| booster   |   0.125 |    0.125 |         0.125 |             0.075 |              0.125 |             0.01  |
| expansion |   0.125 |    0.125 |         1     |             0.025 |              0.125 |             0.006 |
