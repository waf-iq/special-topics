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

## Notes

- `CSAI415_USE_QDRANT=1` is set in the `api` service environment in `docker-compose.yml`.
- `./data:/app/data:ro` volume mount was added to `docker-compose.yml` — the API loads `chunks.parquet`
  at startup to build BM25 indexes even when Qdrant handles dense retrieval.
- Qdrant client/server version mismatch warning (client 1.18.0 vs server 1.12.4) is non-blocking.
