# D3 Task-1 — Guidance-policy comparison

Probe set: 46 entity-anchored content questions (26 specified, 20 underspecified; seed=42); candidate pool k=30. **Hit@k** = a chunk of the target paper reached the top-k; **cand_recall** = a target chunk is anywhere in the reshaped candidate list (the signal a downstream cross-encoder reranker sees); **subgraph_conc@5** = fraction of the top-5 whose paper is in the selected subgraph (concentration, *not* relevance). Policies apply to the hybrid pool; vector/hybrid carry no graph guidance.

## Overall

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_conc@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |   0.543 |    0.587 |         0.652 |             0.448 |              0     |             0.002 |
| hybrid    |   0.565 |    0.587 |         0.696 |             0.47  |              0.022 |             0     |
| filter    |   0.696 |    0.696 |         0.696 |             0.674 |              0.153 |             0.014 |
| booster   |   0.696 |    0.696 |         0.696 |             0.565 |              0.153 |             0.01  |
| expansion |   0.565 |    0.587 |         1     |             0.47  |              0.022 |             0.006 |

## By regime

*Specified* queries already retrieve well (hit@5 near ceiling) → graph adds **precision**: filter pushes subgraph_conc@5 to ~1.0 with no recall loss; booster is the recall-safe middle ground. *Underspecified* queries miss the paper under vector search → graph adds **recall**: expansion raises cand_recall by injecting subgraph chunks the vector pool missed.

**Honesty note on expansion.** For solo-author underspecified probes the target paper *is* the linked subgraph, so a correct link + expansion injects it by construction — `cand_recall→1.0` shows the mechanism and its ceiling (bounded by link precision), not a free recovery of arbitrary chunks. The non-tautological half is `subgraph_conc@5`: expansion's injected chunks land at the tail, so top-5 precision is unchanged — expansion buys candidate recall *without* polluting the top-k the reranker orders. cs.CL (dominant-topic) probes are included to show filter's limit: when the subgraph is 113/155 papers, filtering barely narrows the field.

### Regime: specified

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_conc@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |   0.962 |        1 |             1 |             0.792 |              0     |             0.002 |
| hybrid    |   1     |        1 |             1 |             0.831 |              0.038 |             0     |
| filter    |   1     |        1 |             1 |             0.962 |              0.038 |             0.014 |
| booster   |   1     |        1 |             1 |             0.946 |              0.038 |             0.011 |
| expansion |   1     |        1 |             1 |             0.831 |              0.038 |             0.006 |

### Regime: underspecified

| policy    |   hit@5 |   hit@10 |   cand_recall |   subgraph_conc@5 |   Δhit@5_vs_vector |   guidance_ms_med |
|:----------|--------:|---------:|--------------:|------------------:|-------------------:|------------------:|
| vector    |     0   |     0.05 |           0.2 |              0    |                0   |             0.002 |
| hybrid    |     0   |     0.05 |           0.3 |              0    |                0   |             0     |
| filter    |     0.3 |     0.3  |           0.3 |              0.3  |                0.3 |             0.012 |
| booster   |     0.3 |     0.3  |           0.3 |              0.07 |                0.3 |             0.009 |
| expansion |     0   |     0.05 |           1   |              0    |                0   |             0.006 |
