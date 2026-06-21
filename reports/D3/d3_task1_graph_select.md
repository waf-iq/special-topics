# D3 Task 1 — Graph-guided retrieval (report)

**Owner:** Yousef Alsakkaf · **Rubric:** D3 GraphRAG (8%) · **Code:** `src/csai415/graph_select.py`
· **Harness:** `scripts/eval_graph_select.py` · **Tables:** `graph_linking_comparison.md`,
`graph_guidance_comparison.md` (regenerate with `.venv/bin/python scripts/eval_graph_select.py`).

## 1. What it does

```
query → entity linking → Cypher routing → subgraph (paper_ids) → guidance reshapes retrieval
        fuzzy|spacy|llm   which template     SubgraphResult        filter|booster|expansion
```

`select_subgraph(query, neo4j_driver=…)` links the query to **Author/Topic** nodes in the live
Neo4j graph (155 papers / 764 authors / 13 topics), routes to one of the D2 Cypher templates,
and returns the `paper_ids` of the selected subgraph. When nothing links — or there is no
driver, no graph hit, or a graph error — it returns a `fallback` with empty `paper_ids` and the
executor degrades to pure vector/hybrid retrieval. `apply_guidance(...)` then reshapes a
candidate ranking under the chosen policy; it lives in this module so the integrator can wire any
policy into the executor without editing `graphrag.py`.

## 2. Entity linking — fuzzy vs spaCy vs LLM-NER

36 labelled probes derived from the live graph (seed=42): author plain/prose/typo-1/typo-3,
in-vocabulary topics, **held-out** topic paraphrases (not in the synonym table), author+topic,
and non-linkable negatives (generic questions + plausible *absent* author names).

| method | precision | recall | F1 | fp_rate_neg | heldout_topic_recall | median ms |
|---|---|---|---|---|---|---|
| **fuzzy** | **1.000** | **0.844** | **0.915** | **0.000** | 0.000 | **0.5** |
| spacy | 0.913 | 0.656 | 0.764 | 0.125 | 0.000 | 3.2 |
| llm | n/a (no backend) | | | | | |

**Verdict: fuzzy (rapidfuzz) is the shipped default** — it wins on precision, recall *and*
latency. spaCy trails because PERSON-NER misfires on typo'd/unusual author spans and even
false-links a plausible absent name (fp_rate 0.125). LLM-NER is implemented (`link_llm`, same
`CSAI415_ANSWERER` backend as the answerer) and `select_subgraph` degrades to fuzzy when no
backend is reachable; the row is `n/a` on a laptop with no GPU/key.

**Honest limitation:** `heldout_topic_recall = 0.0` for both offline linkers — topic linking is
bounded by the hand-built synonym table, so a paraphrase it has never seen does not link.
Embedding-/LLM-based topic linking is the future-work fix.

## 3. Cypher routing — which template fires

Routing is by linked node type, priority high→low: author∩topic intersection →
collaboration (`02`) → two-topic (`05`) → author (`01`) → topic (`04`). The corpus is entirely
year 2026, so the year-window template (`03`) is degenerate and intentionally unused.
`cypher_used` records the template that fired (verified live in `check_graph.py`).

## 4. Guidance policy — filter vs booster vs expansion vs vector-only

46 entity-anchored content questions (26 specified, 20 underspecified; seed=42), candidate
pool k=30. `Hit@k` = a chunk of the target paper reached top-k; `cand_recall` = a target chunk
is anywhere in the reshaped candidate list (what the reranker sees); `subgraph_conc@5` =
fraction of top-5 inside the subgraph (concentration, not relevance).

**Overall**

| policy | hit@5 | cand_recall | subgraph_conc@5 | Δhit@5 vs vector |
|---|---|---|---|---|
| vector | 0.543 | 0.652 | 0.448 | 0.000 |
| hybrid | 0.565 | 0.696 | 0.470 | +0.022 |
| **filter** | **0.696** | 0.696 | **0.674** | **+0.153** |
| booster | 0.696 | 0.696 | 0.565 | +0.153 |
| expansion | 0.565 | **1.000** | 0.470 | +0.022 |

- **Specified queries** (dense already retrieves well, hit@5≈1.0): graph adds **precision** —
  `filter` lifts subgraph_conc@5 to **0.962** with no recall loss; `booster` is the
  recall-safe middle ground.
- **Underspecified queries** (entity-only; vector hit@5 = 0.0): graph adds **recall** —
  `filter`/`booster` lift hit@5 **0.0 → 0.3** by promoting in-pool subgraph chunks, and
  `expansion` lifts `cand_recall` **0.2 → 1.0** by injecting subgraph chunks the vector pool
  missed (which the cross-encoder reranker then orders).

**Honesty note on expansion:** for solo-author underspecified probes the target paper *is* the
linked subgraph, so a correct link + expansion injects it by construction — `cand_recall→1.0`
demonstrates the mechanism and its ceiling (bounded by link precision), not free recovery. The
non-tautological half is `subgraph_conc@5`: expansion's injected chunks land at the tail, so
top-5 precision is unchanged — it buys candidate recall *without* polluting the top-k.

## 5. Recommendation (the winner) + integration

- **Linker:** `fuzzy` (`method="auto"` resolves to it; override with `CSAI415_GRAPH_LINK`).
- **Default guidance:** **`filter`** — best hit@5 (+15.3pp) *and* highest concentration; matches
  the executor's current `mode="graph"` default, with a graceful empty-pool fallback so it is
  never worse than vector-only. Its safety depends on link precision, which fuzzy keeps at 1.0.
- **Use `booster`** when link confidence is lower (recall-safe), **`expansion`** to feed the
  reranker on under-specified, entity-only queries.
- **Task 7 wiring:** `select_subgraph` already matches the executor seam; `apply_guidance(ids,
  subgraph, paper_of=…, policy=…, scores=…, chunks_by_paper=…)` consumes the executor's existing
  candidate/score data, so adding booster/expansion is a one-call change in `graphrag.py`.

## 6. Validity & limitations

Probes are derived programmatically from the live graph (the hand-curated arXiv gold is Task 4's
deliverable, not yet committed), so gold labels are reliable by construction but the test
distribution is synthetic. The under-specified regime is n=20 (interpret its deltas as
directional). Topic linking is synonym-table-bound (§2). All numbers are reproducible at seed=42.
Findings here were hardened by an adversarial multi-agent review (21 confirmed issues fixed:
de-circularized topic probes, added absent-author/held-out stress cases, reframed expansion's
metric, corrected the fallback method attribution, +13 regression tests).
