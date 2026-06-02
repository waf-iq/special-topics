# D2-INT2 — Per-Source Qdrant Re-Seed Verification Report

**Owner:** Musab Kamberi  
**Date:** 2026-06-02  
**Task:** D2-INT2 follow-up — seed per-source Qdrant collections, wire API to Qdrant, verify live stack

---

## Task Description

After D2-INT2 (commit `1eff7fa`) wired `/search` to per-source Qdrant collections, the live stack
needed three collections seeded so the `?source=scifact|arxiv` route works at production:

- `chunks_bge384` — full corpus (already seeded in D2-INT1, restored after bug)
- `chunks_bge384_scifact` — SciFact-only subset (new)
- `chunks_bge384_arxiv` — arXiv-only subset (new)

Bug note: commit `5ba6473` fixed `seed_qdrant.py` where `--source` writes always went to the same
`chunks_bge384` collection. After pulling the fix and restoring the clobbered full-corpus collection,
all three were seeded correctly.

`docker-compose.yml` updated: `CSAI415_USE_QDRANT=1` added to the `api` service environment, and
`./data:/app/data:ro` volume mount added so the container can read `chunks.parquet` for BM25 index
construction at startup.

---

## Qdrant Collections — Point Counts

| Collection | Source filter | Points |
|---|---|---|
| `chunks_bge384` | all (full corpus) | 15,760 |
| `chunks_bge384_scifact` | scifact only | 5,663 |
| `chunks_bge384_arxiv` | arxiv + arxiv-demo | 10,097 |

---

## Docker Stack Status

| Service | Image | Status |
|---|---|---|
| `csai415_mongo` | `mongo:7` | healthy |
| `csai415_qdrant` | `qdrant/qdrant:v1.12.4` | healthy |
| `csai415_neo4j` | `neo4j:5` | healthy |
| `csai415_api` | `special-topics-api` | healthy |

---

## /healthz Check

```
GET http://localhost:8000/healthz
→ 200 OK

{"status":"ok"}
```

---

## Search Sanity Check

```
POST http://localhost:8000/search
{"query": "language model pretraining", "k": 5, "source": "scifact"}
```

**Response (top 5 hits, all from SciFact corpus):**

| # | chunk_id | title | score |
|---|---|---|---|
| 1 | `scifact:60206680:0` | R: A Language for Data Analysis and Graphics | 0.867 |
| 2 | `scifact:11943989:0` | Baby hands that move to the rhythm of language... | 0.855 |
| 3 | `scifact:3095620:1` | Distinct Parietal and Temporal Pathways... | 0.832 |
| 4 | `scifact:18379855:0` | The Natural Statistics of Audiovisual Speech | 0.777 |
| 5 | `scifact:83707680:0` | A forkhead-domain gene is mutated in a severe speech and language disorder | 0.748 |

All returned `chunk_id` values are prefixed `scifact:` — confirming the per-source routing is working
correctly (no arXiv chunks leaked into the SciFact result set).

---

## Search Check — source=arxiv

```
POST http://localhost:8000/search
{"query": "language model pretraining", "k": 5, "source": "arxiv"}
```

| # | chunk_id | title | score |
|---|---|---|---|
| 1 | `arxiv:2605.31164v1:40` | D³: Dynamic Directional Graph-Constrained Data Scheduling for LLM Training | 0.894 |
| 2 | `arxiv:2605.31494v1:29` | Consolidating Rewarded Perturbations for LLM Post-Training | 0.887 |
| 3 | `arxiv:2605.30348v1:39` | LLMSurgeon: Diagnosing Data Mixture of Large Language Models | 0.875 |
| 4 | `arxiv:2605.30717v1:32` | Neuron-Level Interventions for Gendered and Gender-Neutral Generation in LMs | 0.828 |
| 5 | `arxiv:2605.30348v1:38` | LLMSurgeon: Diagnosing Data Mixture of Large Language Models | 0.794 |

All `chunk_id` values prefixed `arxiv:` — no SciFact chunks in the arXiv pool. ✓

---

## Search Check — source=null (full corpus)

```
POST http://localhost:8000/search
{"query": "language model pretraining", "k": 5}
```

| # | chunk_id | title | score |
|---|---|---|---|
| 1 | `arxiv:2605.31164v1:40` | D³: Dynamic Directional Graph-Constrained Data Scheduling for LLM Training | 0.892 |
| 2 | `arxiv:2605.31494v1:29` | Consolidating Rewarded Perturbations for LLM Post-Training | 0.888 |
| 3 | `arxiv:2605.30348v1:39` | LLMSurgeon: Diagnosing Data Mixture of Large Language Models | 0.877 |
| 4 | `arxiv:2605.30717v1:32` | Neuron-Level Interventions for Gendered and Gender-Neutral Generation in LMs | 0.836 |
| 5 | `arxiv:2605.30348v1:38` | LLMSurgeon: Diagnosing Data Mixture of Large Language Models | 0.807 |

Full-corpus path works — critical for D3 GraphRAG flow. ✓

---

## Latency Measurement (20 requests, source=scifact)

20 sequential POST /search requests, response times in seconds (sorted ascending):

```
0.126, 0.145, 0.155, 0.156, 0.156, 0.166, 0.178, 0.195, 0.203, 0.207,
0.208, 0.214, 0.217, 0.234, 0.250, 0.277, 0.329, 0.388, 0.438, 2.591
```

| Percentile | Latency |
|---|---|
| p50 | 207ms |
| p95 | **438ms** |
| p100 (max) | 2591ms *(cold-cache outlier on first Qdrant query)* |

**p95 = 438ms — passes the ≤ 2s SLA.** The 2.59s outlier is Qdrant's warm-up on the first
request after container restart; subsequent requests are consistently sub-500ms.

---

## /healthz Fix (api.py)

Per teammate review: `/healthz` previously only pinged `chunks_bge384` (full corpus). A missing
per-source collection would pass health silently while `/search?source=scifact` would 500.

Fixed in this commit: healthz now iterates all three `SOURCE_COLLECTIONS` values and raises 503
if any collection is unreachable.

---

## Notes

- `CSAI415_USE_QDRANT=1` is set in the `api` service environment in `docker-compose.yml`.
- `./data:/app/data:ro` volume mount was added to `docker-compose.yml` — the API loads `chunks.parquet`
  at startup to build BM25 indexes even when Qdrant handles dense retrieval.
- Qdrant client/server version mismatch warning (client 1.18.0 vs server 1.12.4) is non-blocking.
