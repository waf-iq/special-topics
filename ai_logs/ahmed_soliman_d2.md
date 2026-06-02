# AI Log — Ahmed Soliman — D2

**Task:** D2-B1 — FastAPI `/search` skeleton wired to the in-memory D1 retriever (Pair B).
**Date:** 2026-06-01
**Tool:** Claude Code (Opus 4.8)
**Session export:** `ai_logs/ahmed_soliman_d2_transcript.jsonl` (full session transcript)

---

## 2-sentence summary

I used the assistant to map the D2 architecture onto the actual codebase, then to plan and build the `POST /search` + `GET /healthz` FastAPI service backed by the D1 `HybridRetriever`, loading the blessed BOHB config from `winning_runcard.yaml` at startup. The two design calls I made — exposing the fused score via an additive `search_with_scores()` method, and doing source filtering with pre-built per-source retrievers (filter at candidate-generation, not post-hoc) — are mine; the assistant laid out the trade-offs and I picked.

---

## What I worked through

1. **Understood D2 in context.** D1 *found* the blessed retriever; D2 *productionizes* it (Mongo + Qdrant + Neo4j behind FastAPI + Docker). Retrieval quality should match D1 within ~1pp — D2 is an infra move, not a quality change.

2. **The two-corpus design.** One Qdrant collection, `source` as a payload filter:
   - **SciFact** (5,663 chunks) = the *eval* corpus — it's the only one with qrels (`data/gold/qa.jsonl`).
   - **arXiv** (9,740 + 357 demo) = the *graph + demo* corpus — has real authors/year/topics but no qrels.
   - **Risk #4:** if eval forgets `source="scifact"`, arXiv chunks contaminate the candidate pool and Recall@5 collapses — looks like D2 broke retrieval when it's really eval contamination.

3. **My two design decisions (the forks):**
   - **Fork 1 — exposing `score`:** `HybridRetriever.search()` returns only `chunk_id`s. I chose to add an **additive `search_with_scores()`** method so the fusion math stays in one place (`retrieve.py`) and the API stays a thin presentation layer. Additive → D1 callers are untouched. (Rejected: re-deriving fusion in the API = duplication/drift; fake rank-based score = weak for the B3 report.)
   - **Fork 2 — source filtering:** I chose **pre-built per-source retrievers** (`None`/`scifact`/`arxiv`) over a single masked retriever. Each per-source retriever is built over a clean subset, so *both* BM25 and dense candidate pools are naturally restricted — filtering happens at candidate-generation, which is what kills Risk #4. It's also forward-compatible with D2-INT2: "one retriever per source" maps onto "one `QdrantDenseBackend(source_filter=...)` per source," so the API code won't change when Qdrant swaps in.

## What I built

- **`src/csai415/retrieve.py`** — refactored fusion into a shared `_ranked_candidates()` helper; `search()` became a thin wrapper (identical ranking, guarded by the existing Qdrant parity smoke test); added `search_with_scores()`.
- **`src/csai415/api.py`** — `load_blessed_config()` (YAML→`RetrieverConfig`); `create_app()` factory with a lifespan that builds the per-source retrievers + a `chunk_id→row` lookup; Pydantic `SearchRequest`/`SearchHit`; `POST /search`, `GET /healthz`; module-level `app` for `uvicorn csai415.api:app`.
- **`tests/test_api.py`** — 3 cases on a small fixture parquet: healthz=200, `/search` top-k shape, `source="scifact"` narrows.
- **`requirements.txt`** — added `httpx` (FastAPI `TestClient` backend).

## Gotchas I hit

- **Python 3.9 + Pydantic:** `str | None` union annotations fail at runtime in Pydantic model fields on 3.9 (the rest of the repo's `|` unions are fine — they're dataclasses/function sigs kept as strings by `from __future__ import annotations`). Fixed by using `typing.Optional` in the Pydantic models only.
- **`page_range` is `None` for SciFact** (no page metadata) — handled the NaN explicitly rather than crashing the response model.

## Verification

- Existing Qdrant parity smoke test still green → the `search()` refactor preserved D1 ranking exactly.
- Full suite: **41 passed, 2 skipped, 1 xpassed**.
- Live boot against the real 15,760-chunk parquet: healthz 200, `/search` returned scored hits with page ranges, `source="scifact"` returned only scifact chunks, unknown source → 400.

## Next (D2-B3, Wave 3)

Run `csai415.eval.evaluate()` against the live `/search` with `source="scifact"` → `reports/d2_search_metrics.csv` + `reports/d2_topk_examples.md`, sanity-checked against D1's runcard numbers.
