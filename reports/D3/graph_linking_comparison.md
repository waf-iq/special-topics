# D3 Task-1 — Entity-linking comparison

Probe set: 27 labelled queries derived from the live graph (seed=42); precision/recall over (label, node) pairs; `fp_rate_neg` = fraction of no-entity queries that wrongly linked something.

| method   | precision   | recall   | f1    | fp_rate_neg   | median_ms   | p95_ms           |
|:---------|:------------|:---------|:------|:--------------|:------------|:-----------------|
| fuzzy    | 1.0         | 1.0      | 1.0   | 0.0           | 0.6         | 1.3              |
| spacy    | 0.96        | 0.889    | 0.923 | 0.0           | 3.4         | 4.5              |
| llm      | n/a         | n/a      | n/a   | n/a           | n/a         | n/a (no backend) |
