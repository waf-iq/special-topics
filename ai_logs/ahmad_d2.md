# AI Log — Ahmad Fraij — D2

**Task:** D2-A4 — live-stack smoke test gated by `D2_STACK_UP=1`.
**Date:** 2026-06-02
**AI share-link:** [D2 A4 stack smoke](16957cf9-3fdf-47d4-8103-a01afea689ca)

## 2-sentence summary

I used the assistant to implement a gated integration smoke test that checks reachability across FastAPI, Mongo, Qdrant, and Neo4j through direct localhost calls, while keeping default D1 CI green by skipping unless `D2_STACK_UP=1`. We discussed why this test intentionally avoids retriever-quality assertions, uses loose "seeded vs not seeded" thresholds, and explicitly checks per-source Qdrant collections so missing `chunks_bge384_scifact` fails loudly instead of surfacing as an opaque `/search` runtime error.
