# D3 Task-1 — Entity-linking comparison

Probe set: 36 labelled queries derived from the live graph (seed=42); precision/recall/F1 over (label, node) pairs on the *linkable* probes (author plain/prose/typo-1/typo-3, in-vocabulary topics, held-out topic paraphrases, author+topic). Diagnostics: **fp_rate_neg** = fraction of non-linkable queries (generic questions + plausible *absent* author names) that wrongly linked something; **heldout_topic_recall** = recall on topic paraphrases deliberately NOT in the synonym table.

| method   | precision   | recall   | f1    | fp_rate_neg   | heldout_topic_recall   | median_ms   | p95_ms           |
|:---------|:------------|:---------|:------|:--------------|:-----------------------|:------------|:-----------------|
| fuzzy    | 1.0         | 0.844    | 0.915 | 0.0           | 0.0                    | 0.5         | 1.1              |
| spacy    | 0.913       | 0.656    | 0.764 | 0.125         | 0.0                    | 3.2         | 4.1              |
| llm      | n/a         | n/a      | n/a   | n/a           | n/a                    | n/a         | n/a (no backend) |

**Reading it.** Fuzzy leads on precision, recall and latency; spaCy trails because PERSON NER misfires on typo'd/unusual author spans. The low `heldout_topic_recall` for *both* offline linkers is the honest limitation: topic linking is bounded by the hand-built synonym table, so a paraphrase it has never seen does not link. Embedding- or LLM-based topic linking is the future-work fix.
