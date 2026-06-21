# Yousef Alsakkaf — D3 Task 1 (Graph-guided retrieval) — AI log

**Share-link:** _<paste your Claude Code session share-link here>_
**Deliverable:** D3 Task 1 — `src/csai415/graph_select.py` (D3 GraphRAG, 8%).

This session built the real `select_subgraph` against the live Neo4j graph, plus the
comparison harness and tables. Below is what I compared, the decisions, and where I pushed
back on the AI / iterated (the full reasoning is in the linked transcript).

## Setup
- Brought up `docker compose up -d neo4j`, seeded from the parquet fallback (155 papers /
  764 authors / 13 topics), and confirmed `select_subgraph(query, neo4j_driver=…)` runs
  against the live driver (`check_graph.py`). Python 3.12 venv (3.14 too new for the torch stack).

## Approaches compared
**1. Entity linking — fuzzy vs spaCy vs LLM-NER.**
- fuzzy (rapidfuzz over Capitalized spans + a topic-synonym table): P/R/F1 = 1.0 / 0.844 /
  0.915 @ 0.5 ms, 0 false positives. **Chosen as default.**
- spaCy PERSON-NER → fuzzy author link: 0.913 / 0.656 / 0.764 @ 3.2 ms, fp_rate 0.125
  (misfires on typo'd names + one absent name). Worse and slower.
- LLM-NER (same `CSAI415_ANSWERER` backend as the answerer): implemented; `select_subgraph`
  degrades to fuzzy when no backend is reachable. Marked n/a without a GPU/key.

**2. Cypher routing — which of the D2 templates fires.** Routed by linked node type:
author∩topic → collaboration(02) → two-topic(05) → author(01) → topic(04). Found the
year-window template (03) is degenerate (corpus is all 2026), so it is intentionally unused.

**3. Guidance policy — graph-as-filter vs booster vs expansion vs vector-only.** Measured on
two regimes: *specified* (graph adds precision: filter → subgraph_conc@5 ≈ 1.0, no recall
loss) and *underspecified* (graph adds recall: filter/booster lift hit@5 0.0→0.3, expansion
lifts cand_recall 0.2→1.0). Overall filter/booster lift hit@5 +15.3pp vs vector-only.
**Recommended default: filter** (matches the executor's graph mode), booster when link
confidence is lower, expansion to feed the reranker on entity-only queries.

## Where I pushed back / iterated
- First guidance probe set was **too easy** — title-keyphrase queries trivially retrieved
  their own paper (hit@5 saturated at 1.0), hiding any recall effect. Added an
  *under-specified* regime so vector-only has headroom; that surfaced the real +15pp lift.
- Ran an **adversarial multi-agent review** (4 lenses → independent verification per finding;
  21 confirmed). Acted on the methodology critiques rather than accepting the clean numbers:
  - **Circularity:** the first topic probes used the same phrases as the linker's synonym
    table → guaranteed 1.0 recall. Added held-out paraphrases NOT in the table; this dropped
    fuzzy recall to an honest 0.844 and exposed that topic linking is synonym-table-bound
    (logged as future work: embedding-based topic linking).
  - **Tautology:** expansion's cand_recall 0.2→1.0 is true by construction for solo-author
    probes. Reframed the claim around the non-tautological `subgraph_conc@5` (renamed from
    `_prec@5`) and documented the ceiling (bounded by link precision).
  - **Honesty:** added plausible absent author names (precision test) + harder 3-char typos;
    raised the under-specified n 8→20; added a cs.CL dominant-topic control showing filter's
    limit. Plus a correctness fix (fallback method attribution) and a cache hardening
    (WeakKeyDictionary), and +13 regression tests.

## Verification
- 49 unit tests for graph_select + 3 frozen H0 contract tests, all green, no Docker needed.
- Live behaviour confirmed via `check_graph.py` and the eval harness against Neo4j.
