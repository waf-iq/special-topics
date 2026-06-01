# D2-INT1 — Reseed Verification Report

**Owner:** Musab Kamberi  
**Date:** 2026-06-02  
**Task:** D2-INT1 — Reseed all three stores against the expanded corpus

---

## Task Description

Full-chain reseed of Mongo + Qdrant + Neo4j after all Wave 1 tasks landed (D2-A1 ✅, D2-A2 ✅, D2-C1 ✅, D2-M1 ✅).

Concretely:
1. Bring up the Docker stack (`docker compose up -d`)
2. Run `scripts/seed_mongo.py` — seed MongoDB `papers` + `chunks` collections
3. Run `scripts/seed_qdrant.py` — seed Qdrant `chunks_bge384` vector collection
4. Run `scripts/seed_neo4j.py` — seed Neo4j author/paper/topic graph
5. Verify all three stores populated by spot-checking row/point/node counts

---

## Results

### Docker Stack

| Service | Image | Status |
|---|---|---|
| `csai415_mongo` | `mongo:7` | healthy |
| `csai415_qdrant` | `qdrant/qdrant:v1.12.4` | healthy |
| `csai415_neo4j` | `neo4j:5` | healthy |

### MongoDB (`csai415` DB)

| Collection | Count |
|---|---|
| `papers` | 5,338 upserted |
| `chunks` | 15,760 upserted |

### Qdrant

| Collection | Vectors | Dimensions | Distance |
|---|---|---|---|
| `chunks_bge384` | 15,760 | 384 | Cosine |

### Neo4j

| Metric | Count |
|---|---|
| Papers loaded | 155 (arXiv only) |
| Papers skipped | 5,183 (SciFact — no author metadata) |
| Unique authors | 764 |
| Unique topics | 22 |
| WROTE edges | 789 |
| ABOUT edges | 306 |

---

## Notes

- SciFact papers are correctly skipped in Neo4j — BEIR ships no author metadata for them.
- Qdrant version mismatch warning (client 1.18.0 vs server 1.12.4) is non-blocking; seeding completed successfully.
- All seeds are idempotent and safe to re-run (`UPSERT` / `MERGE` throughout).
