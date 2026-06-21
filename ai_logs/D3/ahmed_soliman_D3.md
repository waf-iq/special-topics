# AI Log — D3 Task 2 (Reranking) — Ahmed Soliman

**Deliverable:** D3 · GraphRAG pipeline (8%)
**Task:** Task 2 — Reranking (`src/csai415/rerank.py`)
**Member:** Ahmed Soliman (`Montasers12`)
**Branch:** `task-2-rerank`
**AI session evidence:** `ai_logs/D3/ahmed_soliman_D3_transcript.txt` (exported via `/export`)

---

## 2-sentence summary

I used Claude Code to build a real second-stage reranker behind the frozen
`rerank(query, candidate_ids, chunk_text, top_n)` contract and to compare four approaches
(no-rerank baseline, MiniLM cross-encoder, BGE cross-encoder, MMR) on the 60-query SciFact
holdout for both quality (NDCG@5/Recall@5) and end-to-end latency (p95), with a candidate-pool
sweep. Based on the measured table we chose **BGE at pool 20** as the blessed reranker (+9.3pp
NDCG@5 over no-rerank), wired behind a single generic fail-safe wrapper so a model outage
degrades to retrieval order instead of crashing `/ask`.

---

## What I built

- `src/csai415/rerank.py` — real backends via `get_reranker(kind)`:
  `none` (identity), `minilm` (`cross-encoder/ms-marco-MiniLM-L-6-v2`),
  `bge` (`BAAI/bge-reranker-base`), `mmr` (MMR over `bge-small-en` embeddings). Models load
  lazily + cached. A single `_with_fallback` wrapper protects every model backend.
- `scripts/eval_rerank.py` — comparison harness; reuses `csai415.eval.evaluate` on the
  60-query SciFact holdout (same split as the D1 runcard / D2-B3), writes
  `reports/D3/d3_rerank.md` + `reports/D3/d3_rerank_comparison.csv`.
- `tests/test_rerank_smoke.py` — hermetic default-path + fallback tests; model-backed checks
  gated behind `RUN_RERANK_MODEL=1`.

## Approaches compared (the core of the task)

| reranker | best NDCG@5 | lift vs none | p95 latency | verdict |
|---|---|---|---|---|
| none (baseline) | 0.5611 | — | 72 ms | reference |
| minilm @ pool 20 | 0.6082 | +4.7pp | 332 ms | best quality-per-ms |
| **bge @ pool 20** | **0.6537** | **+9.3pp** | 1145 ms | **chosen — best quality** |
| mmr | 0.4774 | −8.4pp | 926 ms | rejected — regresses |

**Candidate-pool sweep finding:** pool = 20 is the sweet spot for both cross-encoders;
deeper pools (30/50) *lower* NDCG@5 and add latency (more distractors for the reranker).

**Why MMR lost:** SciFact claims usually have a single relevant chunk, so MMR's diversity
term penalizes the near-duplicate relevant chunks it should be surfacing.

## Design decisions discussed with the AI

- **`_BLESSED_RERANKER` constant** names the production reranker; kept at `"none"` during
  development (hermetic/deterministic tests) and flipped to `"bge"` once exactly, after the
  comparison — a single auditable one-line change.
- **Single generic fallback** (`_with_fallback`) instead of per-backend try/except (DRY):
  any reranker failure (or empty pool) degrades to the original retrieval order, so a model
  outage never 500s `/ask`. Verified with a simulated outage + a hermetic test.
- **Coordination note to the Task 7 integrator** (Abdulrahman) added in
  `briefs/D3_D4_TASKS.md`: keep executor `candidate_k=20` to match the pool sweet spot.

## Verification

- `pytest tests/test_graphrag_contract.py tests/test_rerank_smoke.py` → 11 passed, 3 skipped
  (model-gated). Shared H0 contract test stays green with `bge` as default.
- `RUN_RERANK_MODEL=1 pytest tests/test_rerank_smoke.py` → real minilm/bge/mmr each surface
  the relevant chunk first.
- `python scripts/eval_rerank.py` → regenerates the report + CSV.

## Artifacts

- `src/csai415/rerank.py`, `scripts/eval_rerank.py`, `tests/test_rerank_smoke.py`
- `reports/D3/d3_rerank.md`, `reports/D3/d3_rerank_comparison.csv`
- Brief note: `briefs/D3_D4_TASKS.md` (Task 7)
