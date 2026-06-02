# CSAI415 — Deliverable 2 (D2) Report

**Authors:** Ahmad Fraij, WAFIQ Akram ABO DAKEN (co-authored, Pair A + Pair B).
**Team contributors:** Pair A — Ahmad Fraij + Yousef Alsakkaf · Pair B — WAFIQ Akram ABO DAKEN + Ahmed Soliman · Pair C — Abdurlahman Alali + Yehia Noureldin · Solo — Musab Kamberi.

## 1. Objective

D2 productionises the D1 retriever onto a service stack: ingestion → MongoDB + Qdrant + Neo4j → FastAPI `/search`, all behind Docker Compose. D2 is graded on engineering (Ingest 3% · Hybrid retrieval 5% · Graph build 5% · Engineering 2%); retrieval quality remains anchored to the D1 BOHB blessed config (`hybrid_weight=0.777, candidate_k=27, metric=l2, bm25_k1=2.92, bm25_b=0.345`) — D2 *moves* the retriever, it does not re-tune it.

## 2. Architecture

![D2 dataflow](d2_dataflow.png)

Source `reports/d2_dataflow.mmd`. Flow: SciFact (via `ir_datasets`) + 150 arXiv cs.CL PDFs → `ingest.py` chunk+embed → `chunks.parquet` → seeded into Mongo (paper/chunk provenance), Qdrant (384-d BGE vectors), Neo4j (Author/Paper/Topic graph) → FastAPI `/search` over BM25 + Qdrant fusion. The `make seed` target reseeds all three stores idempotently; `docker compose up -d` brings the stack up in ~30 s.

## 3. Ingest & storage (rubric 3%)

| Store | Schema highlights | Counts |
|---|---|---|
| Mongo `csai415` | `papers`: `_id=paper_id, title, authors[], year, topics[], source`; `chunks`: `_id=chunk_id, paper_id, text, position, page_start/end, source`. Indexes on `papers.authors`, `papers.year`, `chunks.paper_id`. | 5,338 papers, 15,760 chunks |
| Qdrant | Three collections (`chunks_bge384`, `_scifact`, `_arxiv`), 384-d cosine; payload `{chunk_id, paper_id, source, page_start, page_end, title, text}` so `/search` is one-store. | 15,760 / 5,663 / 10,097 vectors |
| Neo4j | `(:Paper)`, `(:Author)`, `(:Topic)`; `[:WROTE]`, `[:ABOUT]` edges. Uniqueness constraints; `MERGE` semantics (idempotent reseed). SciFact rows skipped (no authors in BEIR). | 155 papers / 764 authors / 13 topics |

Verification reports: `reports/d2_int1_verification.md` (initial seed) and `reports/d2_int2_verification.md` (post per-source Qdrant). Schemas at `docs/d2_schemas.md` + `docs/d2_graph_schema.md`. Corpus splits: SciFact = 5,663 chunks across 3,793 abstracts (BEIR test split, no page info); arXiv = 9,740 chunks across 150 papers (PyMuPDF page-mapped); arxiv-demo = 357 chunks across 5 D1 demo PDFs.

## 4. Hybrid retrieval (rubric 5%)

`POST /search` (Pydantic-validated, see `src/csai415/api.py`) returns top-k hits with `chunk_id, paper_id, title, text, page_range, score`. Per-source routing pre-builds one `HybridRetriever` per source so BM25 + dense candidate pools never include cross-source chunks. Qdrant dense runs cosine ANN over-fetching `candidate_k × 2`; the blessed L2 metric is preserved at fusion-time rescoring via `client.retrieve(with_vectors=True)`.

Metrics on the **60-query SciFact holdout** (same split as D1 runcard for direct comparability):

| Config | NDCG@5 | Recall@5 | Recall@10 | p95 latency |
|---|---|---|---|---|
| `bm25_only` (w=0.0) | 0.4172 | 0.4733 | 0.5317 | 103.7 ms |
| `dense_only` (w=1.0) | 0.5630 | 0.6489 | **0.7778** | 109.7 ms |
| `hybrid_blessed` (w=0.777) | **0.5611** | 0.6489 | 0.7306 | 106.0 ms |

Source: `reports/d2_search_metrics.csv`. D1 runcard `winner_holdout` reference: NDCG@5=0.561, Recall@5=0.649, p95=103 ms — drift on `hybrid_blessed` < 0.5pp on every metric, confirming the production stack preserves D1's blessed retrieval. **Honest finding:** `dense_only` beats `hybrid_blessed` on Recall@10 (0.778 vs 0.731). BM25's top-rank contribution helps NDCG@5 but crowds out relevant dense candidates lower in the ranking. D3 may revisit the fusion weight against deeper-k objectives.

Live-stack p95 (over HTTP, see `reports/d2_int2_verification.md`): 438 ms; p100 = 2.59 s on the very first request (Qdrant HNSW cold-cache). Subsequent requests sub-500 ms — passes the brief's ≤ 2 s SLA on p95. Top-k example queries (3 SciFact holdout + 2 qualitative arXiv) live in `reports/d2_topk_examples.md`.

## 5. Graph build (rubric 5%)

Neo4j seeded from Mongo via `scripts/seed_neo4j.py` (with parquet dev-fallback `make seed-dev`). Five Cypher queries in `cypher/`, captured outputs in `reports/d2_cypher_examples.md`:

1. **Papers by author** — `MATCH (a:Author {name:"Jun Wang"})-[:WROTE]->(p)` — returns 2 papers.
2. **Top co-authors** — 2-hop `WROTE-Paper-WROTE` — returns 5 co-authors of "Jun Wang".
3. **Top topics by year** (2025–2026) — `cs.CL=113, cs.LG=10, cs.CV=10, cs.CR=5, cs.AI=4`. *Demonstrates the full-categories list from D2-A2's fix — a single-primary-category ingest would have collapsed to one row.*
4. **Papers & authors by topic** (`cs.CL`) — first 10 paper/author rows.
5. **Authors on two topics** (`cs.CL ∩ cs.CV`) — intersection query, returns 1 author with 1 paper each side.

These map directly onto D3's GraphRAG executor surfaces: query 1 powers "context for paper X", query 2 powers community detection, queries 3/4 power topic-bounded retrieval, query 5 powers cross-topic agent reasoning.

## 6. Engineering (rubric 2%)

Stack: `docker-compose.yml` (Mongo 7, Qdrant 1.12.4, Neo4j 5, FastAPI app), `Dockerfile`, `Makefile` (`up`/`seed`/`seed-dev`/`logs`/`clean`/`diagram`), `.env.example`, mounted volumes per service. Health: `GET /healthz` 200 iff retrievers loaded **and** all three Qdrant collections reachable. Smoke: `tests/test_smoke.py::test_d2_stack_smoke` (gated on `D2_STACK_UP=1`) drives the live stack end-to-end (FastAPI → Mongo + Qdrant + Neo4j); the default suite stays green without Docker (46 passed, 1 skipped, 1 xpassed).

## 7. Decisions & pitfalls

(a) **No `CITES` edges.** The arXiv API doesn't expose citations. Deferred to D3 per the brief's "add CITES if time" allowance — a Semantic Scholar lookup is the natural D3 extension. (b) **Cosine ANN + L2 fusion rescore.** Qdrant collections use cosine for fast candidate generation; `QdrantDenseBackend.scores_for()` fetches vectors and recomputes L2 in numpy so the blessed metric is preserved. Empirical drift from D1 runcard < 0.5pp NDCG@5. (c) **Author dedup is name-based.** `MERGE (a:Author {name})` collapses two real "Jane Doe"s; at our scale (~750 authors) we expect 1-5 collisions — flagged here, not fixed. (d) **D2 reuses D1's blessed retrieval.** The retrieval quality numbers above are the D1 holdout numbers; D2's contribution is *engineering*. (e) **Three Qdrant collections.** A corpus_idx mismatch between Qdrant's global point IDs and the per-source HybridRetriever's reset-indexed local df forced a three-collection layout (`chunks_bge384`, `_scifact`, `_arxiv`); operationally requires a triple reseed on any corpus change. **Scheduled for D3 refactor** to one collection + source-filter-at-query-time inside `HybridRetriever.search()` — D3's GraphRAG executor naturally wants one retriever, not three, so the refactor pays for itself. (f) **Cold-start latency outlier.** p100 = 2.59 s on the first request after API restart (Qdrant HNSW index load); p95 = 438 ms over the wire. Production mitigation: a startup-time warm-up query.

## 8. Repo references

`src/csai415/api.py`, `src/csai415/qdrant_dense.py`, `scripts/{ingest_arxiv_batch,seed_mongo,seed_qdrant,seed_neo4j,eval_search_metrics,capture_cypher_outputs}.py`, `cypher/01..05_*.cypher`, `docker-compose.yml`, `Makefile`, `configs/winning_runcard.yaml`, `reports/{d2_dataflow,d2_int1_verification,d2_int2_verification,d2_search_metrics,d2_topk_examples,d2_cypher_examples}`. AI logs per member at `ai_logs/<name>_d2.{md,txt}`.
