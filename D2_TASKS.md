# D2 — Team Tasks (1-day plan)

D2 = Week 7, **15% of project** = Ingest & storage (3%) + Hybrid retrieval (5%) + Graph build (5%) + Engineering (2%). Full rubric in `CSAI415_Project_Brief.pdf`.

**Corpus decision (locked):** Option X. Expand the existing 5 arXiv demo PDFs → **~150 arXiv cs.CL PDFs** (real authors / year / primary_category for free). SciFact stays as the retrieval-eval corpus because that's where the 300 qrels live; arXiv becomes the graph-build + `/search` demo corpus. Single Qdrant collection with a `source` field used as an eval-time filter.

**Pairs (current as of D1-rework — not the original D1 layout):**
- Pair A: Ahmad Fraij + Yousef Alsakkaf
- Pair B: WAFIQ Akram ABO DAKEN + Ahmed Soliman
- Pair C: Abdurlahman Alali + Yehia Noureldin
- Solo: Musab

**Branching:** one feature branch per task — `d2-a1-ingest-mongo-qdrant`, `d2-b1-search-api`, etc. PR into `main`, review from one teammate outside the pair.

---

## What D2 must hand in (rubric → artifacts)

| Rubric line | Weight | Artifact in repo |
|---|---|---|
| Ingest & storage | 3% | `seed_mongo.py` + `seed_qdrant.py`; Mongo `papers`+`chunks` collections with provenance; Qdrant `chunks_bge384` HNSW index; schema doc |
| Hybrid retrieval | 5% | `POST /search` (FastAPI) backed by Qdrant + BM25 with blessed BOHB config; `reports/d2_search_metrics.csv` (Recall@5, Recall@10, p95 latency) + 5 top-k example queries with citations |
| Graph build | 5% | Neo4j seeded with Authors/Papers/Topics; `cypher/` directory with 3–5 documented queries; dataflow diagram (`reports/d2_dataflow.{mmd,png}`) |
| Engineering | 2% | `docker-compose.yml` (Mongo + Qdrant + Neo4j + FastAPI), `.env.example`, healthchecks, `make seed`, one-command README quickstart, smoke test |

---

## Hour 0 — contracts (whole team, 30 min, no exceptions)

Without these locked, parallel work collides at integration. Everyone leaves H0 with their slice and the contracts below.

**Mongo (`csai415` DB)**
```
papers:  { _id: paper_id, title, authors: [str], year: int|null,
           venue: str|null, topics: [str], source: "scifact"|"arxiv",
           pdf_path: str|null, ingested_at }
chunks:  { _id: chunk_id, paper_id, text, position: int,
           page_start: int|null, page_end: int|null, source }
```
**Topics:** `papers.topics` is the **full arXiv categories list** (e.g. `["cs.CL", "cs.AI"]`), not just the primary. Sourced from `data/raw_pdfs/arxiv_batch_meta.json` (arxiv) and `arxiv_meta.json` (arxiv-demo). SciFact papers have `topics: []` (BEIR carries no topic metadata). Using the full list avoids a degenerate single-Topic-node graph.

**Qdrant**
- Collection `chunks_bge384`, vector size **384**, distance **Cosine** (BGE is trained for cosine).
- Payload: `{ chunk_id, paper_id, source, page_start, page_end, title, text }`. `text` is in the payload so `/search` is one-store (no Mongo round-trip per hit). ~15 MB total payload for ~10–13k chunks — fine.
- Eval-time queries use a payload filter `source == "scifact"` so SciFact qrels remain valid.
- **Metric note:** the blessed BOHB winner uses `metric="l2"`. Qdrant ANN runs cosine for candidate generation only — the backend over-fetches `candidate_k * 2` candidates, then `scores_for()` fetches vectors via `client.retrieve()` and computes L2 in numpy for fusion. Net: blessed L2 metric is preserved at fusion-time rescoring; expected drift from D1 runcard <0.5pp NDCG@5.

**Neo4j**
- Nodes: `(:Paper {paper_id, title, year, source})`, `(:Author {name})`, `(:Topic {name})`, `(:Venue {name})` if present.
- Edges: `(:Author)-[:WROTE]->(:Paper)`, `(:Paper)-[:ABOUT]->(:Topic)`, `(:Paper)-[:PUBLISHED_IN]->(:Venue)`.
- Constraints: unique on `Paper.paper_id`, `Author.name`, `Topic.name`. Loader uses `MERGE`, not `CREATE` (idempotent reseed).

**FastAPI**
```
POST /search
  body:     { query: str, k: int = 5, source: str|null = null }
  response: [ { chunk_id, paper_id, title, text, page_range, score } ]
GET /healthz   → 200 when Mongo + Qdrant + Neo4j all reachable
```
Blessed BOHB config (`hybrid_weight=0.777, candidate_k=27, metric=l2, bm25_k1=2.92, bm25_b=0.345`) loads from `configs/winning_runcard.yaml` at app startup — **never** in the request body.

---

## Wave 1 — parallel (everyone starts at H0, no cross-deps)

### Pair A — Corpus expansion + storage seeding

#### Task D2-A1 — Expand corpus to ~150 arXiv cs.CL PDFs ✅
**Nominal owner:** Ahmad Fraij. **Actual execution:** WAFIQ (covering — Ahmad unavailable at kickoff; commit `e8ecf3a`). 148/149 papers ingested, 9,644 new chunks. `ingest_arxiv_batch()` added to `src/csai415/ingest.py` + `scripts/ingest_arxiv_batch.py` runner.

- Pre-pick ~150 cs.CL arXiv IDs (e.g., `arxiv.Search(query="cat:cs.CL", max_results=150, sort_by=SubmittedDate)`).
- Download PDFs to `data/raw_pdfs/`, parse with PyMuPDF, chunk using the existing `_token_chunks` helper, embed with BGE-small-en-v1.5, append to `data/processed/chunks.parquet` with `source="arxiv"` and **populated `authors`, `year`, `topics=[primary_category]`**.
- Resumable: skip arXiv IDs already in the parquet by `paper_id`. Log + skip parse failures, don't crash the batch. Target ≥120 successful papers (brief allows 100–300).
- Adds 5s sleep between downloads (be polite to arXiv).
- **Output:** updated `chunks.parquet` with ~120–150 arXiv papers plus the existing 5,663 SciFact chunks. New row count target: ~10–13k chunks.

#### Task D2-A2 — Mongo + Qdrant seed scripts ✅
**Owner:** Yousef Alsakkaf (commits `245fed2` + `1b3ee4b` fix). Final re-run handled by Musab during D2-INT1.

- `scripts/seed_mongo.py`: read `chunks.parquet` → write to `papers` (one row per `paper_id`, deduped on title/authors/year/source) + `chunks`. `papers.topics` is sourced from `data/raw_pdfs/arxiv_batch_meta.json` + `arxiv_meta.json` (joined on `paper_id`), defaulting to `[]` for SciFact. Upsert by `_id` (idempotent). Add an index on `papers.authors`, `papers.year`, `chunks.paper_id`.
- `scripts/seed_qdrant.py`: read `chunks.parquet` → upsert points into `chunks_bge384` with the locked payload (including `text`). Recreate collection if vector size mismatches (one-shot dev safety, not for prod).
- Both scripts read connection info from env (`MONGO_URL`, `QDRANT_URL`) and accept `--source scifact|arxiv|all` so reseeds can be partial.
- **Output:** populated stores + a one-page `docs/d2_schemas.md` documenting the Mongo + Qdrant payload schemas with example records.

### Pair B — Retrieval API

#### Task D2-B1 — FastAPI app + `/search` skeleton wired to in-memory retriever ✅
**Owner:** Ahmed Soliman (commit `0f1df77` via PR #7). Shipped per-source retriever routing in `src/csai415/api.py` so D2-INT2's Qdrant injection slotted in cleanly. 3 pytest cases green.

- `src/csai415/api.py`: FastAPI app, `POST /search`, `GET /healthz`, lifespan hook that loads the blessed config from `configs/winning_runcard.yaml`.
- For Wave 1, dense backend is the in-memory `HybridRetriever` from D1 (so the endpoint is fully testable before D2-B2 lands).
- Request/response models in Pydantic per the locked contract.
- 3 pytest cases: healthz returns 200, `/search` returns top-k with required fields, `source="scifact"` filter narrows results.

#### Task D2-B2 — Qdrant-backed dense backend ✅
**Owner:** WAFIQ (commit `aaffaac`). `src/csai415/qdrant_dense.py` + 3 parity tests verifying numpy ↔ Qdrant top-5 match on SciFact queries with the L2-via-cosine-ANN workaround.

- New `src/csai415/qdrant_dense.py`: `QdrantDenseBackend` class with two methods — `top_k(query_vec, k)` (cosine ANN candidate generation; **over-fetches `k * 2`** to insure against cosine/L2 ordering differences) and `scores_for(query_vec, corpus_indices)` (fetches vectors via `client.retrieve()` and computes the configured metric — L2 for blessed config — in numpy).
- Modify `HybridRetriever` so the dense backend is pluggable via constructor injection — keep numpy brute force for tests, swap in Qdrant for the FastAPI app. Default `dense_backend=None` builds the numpy backend internally so D1 callers (automl, hpo_methods, online, tests) keep working.
- Add a startup-time sanity check: collection exists, vector size = 384, point count > 0. Fail fast and log clearly if not.
- Smoke test against a 100-vector fixture collection (in-memory Qdrant via `QdrantClient(":memory:")` — no Docker needed for tests).

### Pair C — Graph build

#### Task D2-C1 — Neo4j loader ✅
**Owner:** Abdurlahman Alali (commits `48defcc` + `7645d2d` via PR #2). `scripts/seed_neo4j.py` with MERGE-based idempotent writes, parquet dev fallback, 6 tests covering normalization/dedup/source filter.

- `scripts/seed_neo4j.py`: reads from Mongo `papers` (NOT directly from parquet — this is the only piece that needs Mongo). For each paper: `MERGE` the Paper node, MERGE each Author and create `(:Author)-[:WROTE]->(:Paper)`, MERGE each Topic and create `(:Paper)-[:ABOUT]->(:Topic)`. Idempotent.
- Apply uniqueness constraints at startup (no-op if already there).
- Skip papers with empty authors (i.e., SciFact rows — they don't have author metadata). Log the count of skipped papers.
- **Output:** seed script + a `docs/d2_graph_schema.md` with the node/edge spec.

#### Task D2-C2 — Cypher query library ✅
**Owner:** Yehia Noureldin (commit `1ce456c` via PR #6). 5 parameterized `.cypher` files in `cypher/` with intent comments.
**Develops against the 5-paper fixture graph — re-runs at H6 against the real seeded graph for the report.**

- `cypher/` directory, one `.cypher` file per query, each with a comment header explaining intent + expected output shape. Pick 5:
  1. Papers by a given author (`WROTE` traversal).
  2. Top 10 co-authors of a given author (2-hop `WROTE-Paper-WROTE`).
  3. Top 10 topics by paper count in a given year window.
  4. Papers about topic T and authors who wrote them.
  5. Authors who write on BOTH topic T1 and topic T2 (intersection).
- Each query gets a one-paragraph "why this is useful for the agent" note (D3 GraphRAG will consume these).
- After H6 reseed, capture actual outputs into `reports/d2_cypher_examples.md`.

### Solo + Pair C support — Engineering / infra

#### Task D2-M1 — Docker Compose + diagram + README ✅
**Owner:** Musab (commit `6b3c5c5` via PR #4). `docker-compose.yml` (mongo+qdrant+neo4j+api with healthchecks), `Dockerfile`, `Makefile` with `up`/`seed`/`logs`/`clean` targets, `reports/d2_dataflow.mmd` + rendered PNG, `.env.example`, README quickstart. Abdurlahman ultimately stayed focused on D2-C1 — Musab landed M1 solo.

- `docker-compose.yml`: services for `mongo` (port 27017), `qdrant` (6333), `neo4j` (7474+7687, `NEO4J_AUTH=none` for dev), `api` (the FastAPI app, built from `Dockerfile`). Named volumes for each store. Healthchecks for all 4.
- `.env.example` with `MONGO_URL`, `QDRANT_URL`, `NEO4J_URL`, `API_PORT`.
- `scripts/seed_all.sh` (or `Makefile` target `seed`) that runs D2-A2's two seed scripts + D2-C1's seed in order.
- `reports/d2_dataflow.mmd` (mermaid) + rendered `.png`: shows arXiv PDFs + SciFact → ingest → (Mongo + Qdrant + Neo4j) → FastAPI `/search` → user. Include the GraphRAG path as dashed lines labelled "D3".
- README quickstart: `docker compose up -d && make seed && curl localhost:8000/search ...`. One command should bring the stack up; second command should seed.

---

## Wave 2 — runs at H5 after Wave 1 lands (~30 min total)

### Task D2-INT1 — Reseed against expanded corpus ✅
**Owner:** Musab (commit `6d7ba08`). Brought up full stack, ran the three seeders against the expanded parquet. Mongo: 5,338 papers + 15,760 chunks. Qdrant: 15,760 vectors. Neo4j: 155 papers / 764 authors / 22 topics. Verification at `reports/d2_int1_verification.md`.

### Task D2-INT2 — Swap `/search` to Qdrant-backed dense ✅
**Owner:** WAFIQ (code: commits `1eff7fa` per-source backends + `5ba6473` seed_qdrant collection-name fix) + Musab (deployment: commits `2b5553d` live re-seed + `584521f` all-collection /healthz). Verification at `reports/d2_int2_verification.md`. p95 latency 438ms, well under 2s SLA.

**Approach taken (plan A): three Qdrant collections.** Each per-source `HybridRetriever` in `api.py` is wired to its own Qdrant collection — `chunks_bge384` for the full corpus, `chunks_bge384_scifact`, `chunks_bge384_arxiv` for the subsets. The per-collection layout sidesteps a corpus_idx mismatch between Qdrant's global point IDs and each retriever's reset-indexed local df. Gated on `CSAI415_USE_QDRANT=1` env var so the numpy in-memory fallback keeps the existing test suite working without Docker.

**Live re-seed required (Musab, ~5 min):** D2-INT1 only created `chunks_bge384`. After this PR merges, on the docker-compose host:
```bash
python scripts/seed_qdrant.py --source scifact   # creates chunks_bge384_scifact
python scripts/seed_qdrant.py --source arxiv     # creates chunks_bge384_arxiv
```
Then set `CSAI415_USE_QDRANT=1` in the API service's env and restart. `GET /healthz` should still return 200; it now also pings Qdrant for the full collection.

**Known trade-off (must land in D2-A3 report):** three collections must be reseeded any time the corpus changes — operational pain. The architecturally cleaner fix (one collection + source-filter-at-query-time inside `HybridRetriever.search()`) was deferred to D3 to avoid touching Ahmed's just-merged per-source routing logic under time pressure. D3's GraphRAG executor will benefit from the refactor anyway since the executor naturally wants one retriever, not three.

---

## Wave 3 — runs at H7 after Wave 2 lands (~2 hours)

### Task D2-B3 — Metrics table against `/search` ✅
**Nominal owner:** Ahmed Soliman. **Actual execution:** WAFIQ (covering — Ahmed offline; commit `ed3668f`). `scripts/eval_search_metrics.py` + `reports/d2_search_metrics.csv` + `reports/d2_topk_examples.md`. Hybrid_blessed lands within 0.5pp of D1 runcard holdout; honest framing required in report — `dense_only` beats `hybrid_blessed` on Recall@10 (0.778 vs 0.731), a real corpus characteristic.
**Depends on:** D2-INT2

- Run `csai415.eval.evaluate(retriever_fn=lambda q,k,hw: call_search_api(q, k, source="scifact"), queries=load_qa(), k=5)`.
- Repeat at `k=10`.
- Produce `reports/d2_search_metrics.csv`: rows = `{bm25_only, dense_only, hybrid_blessed}`, cols = `recall@5, recall@10, ndcg@5, p95_latency_ms`. Compare to D1 numbers as a sanity check (should be within ±1pp of the runcard's holdout metrics).
- Pick 5 example queries (mix of easy / hard / a couple from arXiv side without qrels) → `reports/d2_topk_examples.md` showing query + top-5 hits with `paper title, page_range, score`.

### Task D2-C3 — Capture real Cypher outputs ✅
**Owner:** Yehia Noureldin (commit `2426fac`). `reports/d2_cypher_examples.md` + `scripts/capture_cypher_outputs.py` runner. Captured against the live graph (155 papers / 764 authors / 13 topics).
**Depends on:** D2-INT1

- Re-run the 5 Cypher queries from D2-C2 against the real seeded graph (now with ~120 arxiv papers, ~300+ authors, ~1 topic = cs.CL plus any sub-categories like cs.CL+cs.LG).
- Capture outputs (truncated to 10 rows each) into `reports/d2_cypher_examples.md`.
- If `cs.CL` is the only topic (because we filtered to `cat:cs.CL`), pick query 3 as a year-window count instead so the result is non-trivial.

### Task D2-A3 — 2-page D2 report ✅
**Owners:** Ahmad Fraij + WAFIQ Akram ABO DAKEN (co-authored). Ahmad shipped the scaffold (commits `aaedfb1` + `3ac9879`); WAFIQ did the rework — embedded the dataflow diagram, pulled in real numbers from `d2_int1_verification.md` / `d2_int2_verification.md` / `d2_search_metrics.csv` / `d2_cypher_examples.md`, replaced the pitfalls section with the (a)–(f) list.

- `reports/D2_report.md` (compiled to PDF, same toolchain as D1).
- Sections: (1) one-paragraph architecture, (2) dataflow diagram from D2-M1, (3) ingest stats (papers + chunks per source), (4) `/search` metrics table from D2-B3, (5) top-k example queries with citations, (6) 5 Cypher queries + sample outputs, (7) decisions/pitfalls.
- Cite the blessed BOHB config from D1 — D2 inherits it, doesn't re-tune.
- **Pitfalls to call out explicitly:** (a) graph has no `CITES` edges — arXiv API doesn't expose citation data; deferred to D3 (brief permits this — "add CITES if time"); (b) Qdrant uses cosine ANN for candidate generation while blessed metric is L2 — L2 preserved at fusion-time rescoring (<0.5pp drift); (c) Author dedup is name-based, so two distinct people who share a name collapse into one node; (d) D2 retrieval reuses D1's blessed retriever — quality numbers are essentially D1's, the D2 work was moving the stack to production infra (Mongo + Qdrant + Neo4j behind FastAPI); (e) **Qdrant uses three per-source collections** (`chunks_bge384`, `_scifact`, `_arxiv`) to sidestep a corpus_idx mismatch between global Qdrant IDs and the per-source `HybridRetriever`'s reset-indexed local df — operationally clunky (reseed all three on any corpus change) and **scheduled to be refactored in D3** to a single collection with source-filter-at-query-time inside `HybridRetriever.search()`; D3's GraphRAG executor wants one retriever, not three, so the refactor pays for itself.

### Task D2-A4 — Smoke test stays green ✅
**Owner:** Ahmad Fraij (commit `2ac1bad` via PR #8). `tests/test_smoke.py::test_d2_stack_smoke` gated on `D2_STACK_UP=1`; default run shows `11 passed, 1 skipped`. With live stack: `12 passed`.
**Owner:** Ahmad Fraij
**Depends on:** D2-INT2

- Extend `tests/test_smoke.py` with a `test_d2_stack_smoke` that:
  1. spins up the FastAPI app in TestClient (requires Mongo + Qdrant reachable — gated on `D2_STACK_UP=1` env var so D1 CI doesn't break);
  2. hits `/search` with one query, asserts response shape;
  3. hits Neo4j, asserts at least one `Paper` node exists.
- `pytest tests/test_smoke.py` must remain green for the D1 path even when the D2 stack is not running.

---

## Critical path (where slack lives)

```
D2-A1 (arXiv expansion, ~3-4h, mostly unattended)
    │
    ▼
D2-INT1 (reseed, ~30min) ─────────┬─────▶ D2-C3 (capture cypher, 30min) ─┐
                                │                                  │
                                ▼                                  ▼
                          D2-INT2 (swap dense backend, 15min) ──▶ D2-B3 (metrics, 1h) ──▶ D2-A3 (report, 1.5h)
                                                                                       │
                                                                                       ▼
                                                                                  D2-A4 + final commit
```

Only **Yehia (D2-C3)** is genuinely blocked early-ish (needs D2-INT1's seeded graph at H6). Everyone else has Wave 1 work that fills H0–H5 against fixtures. **Slack lives entirely in Wave 1** — if D2-A1 finishes earlier than expected, the whole timeline compresses.

If D2-A1 slips past H6, the fallback is to seed Mongo + Qdrant + Neo4j with just the existing 5 arxiv-demo + SciFact and ship a smaller graph. The retrieval metrics table is unaffected because it uses SciFact qrels.

---

## Hour-by-hour cheatsheet

| Hour | Who does what |
|---|---|
| **H0** (0:30) | Whole team: lock contracts in this doc, split work, every member starts their Wave 1 task. |
| **H0.5–H5** | D2-A1 runs in background. Everyone else: build against fixtures / D1 parquet. |
| **H5** (0:30) | D2-INT1: Musab brings up `docker compose up -d` and runs all three seed scripts (Mongo + Qdrant + Neo4j). |
| **H5.5** (0:15) | D2-INT2: WAFIQ swaps `/search` dense backend to Qdrant. |
| **H6** (1:00) | D2-B3 metrics + D2-C3 cypher capture in parallel. Musab finalizes README + compose polish. |
| **H7** (1:30) | D2-A3 drafts the 2-page report. D2-A4 adds the D2 smoke test. |
| **H8.5** (0:30) | Every member updates `ai_logs/<name>_d2.md` with their AI share-link + 2-sentence summary. |
| **H9** | Final smoke test green → tag `d2-submitted` → done. |

---

## Risks to manage actively

1. **arXiv rate limits / flaky downloads (D2-A1).** Ahmad — use the `arxiv` lib's built-in backoff, pre-fetch the ID list once and persist, restart skips IDs already in the parquet.
2. **PyMuPDF parse failures (D2-A1).** Don't block the batch on a single bad PDF. Log to `data/raw_pdfs/parse_errors.json` and move on. Hitting 120/150 success is fine.
3. **Neo4j auth (D2-M1).** Disable for dev via `NEO4J_AUTH=none`. Document in README that prod would set a password.
4. **Source filter forgotten on eval (D2-B3).** Without `source="scifact"` on the eval, arxiv chunks contaminate top-5 and Recall@5 drops vs D1's number. Pair B writes a quick assertion in the eval script: `assert all(r["source"] == "scifact" for r in hits)` for one sample query.
5. **AI logs (everyone).** Same rule as D1 — empty `ai_logs/<name>_d2.md` = missing evidence. Don't share sessions, don't paste the final code in and ask for an explanation.

---

## Definition of done (D2 rubric checklist)

- [x] `data/processed/chunks.parquet` includes ≥120 arXiv cs.CL papers with real authors/year/topics, plus the existing SciFact corpus. *(150 arXiv papers + 5,663 SciFact = 15,760 total chunks)*
- [x] `docker compose up -d` brings Mongo + Qdrant + Neo4j + FastAPI cleanly. `GET /healthz` returns 200 within 30s of startup. *(Musab D2-INT1 + D2-INT2)*
- [x] `make seed` populates all three stores idempotently. Re-running is safe. *(WAFIQ fix `463c7f6` adds per-source Qdrant)*
- [x] `POST /search {"query": "...", "k": 5}` returns 5 hits with `chunk_id, paper_id, title, text, page_range, score`. *(verified in `d2_int2_verification.md`)*
- [x] `reports/d2_search_metrics.csv` exists with Recall@5/@10 + p95 latency for bm25/dense/hybrid on SciFact holdout. *(D2-B3, `ed3668f`)*
- [x] `reports/d2_topk_examples.md` has 5 example queries with citations. *(D2-B3 — 3 SciFact + 2 arXiv)*
- [x] `cypher/` has 5 documented queries; `reports/d2_cypher_examples.md` has their real outputs against the seeded graph. *(Yehia D2-C2 + D2-C3)*
- [x] `reports/d2_dataflow.png` (rendered from `.mmd`) committed. *(Musab D2-M1)*
- [x] `reports/D2_report.pdf` ≤ 2 pages, covers all rubric lines. *(Ahmad + WAFIQ co-authored)*
- [x] `pytest tests/test_smoke.py` green in both modes: default run is `11 passed, 1 skipped`; with `D2_STACK_UP=1` and live stack up, run is `12 passed`. *(D2-A4)*
- [ ] Each member has ≥2 commits under their own GitHub identity on D2 branches. *(Ahmed Soliman currently at 1 — needs a follow-up commit, or co-author trailer in D2-B3-related commits)*
- [ ] Each member has a populated `ai_logs/<name>_d2.{md,txt}` with share-link + summary. *(handled offline)*
