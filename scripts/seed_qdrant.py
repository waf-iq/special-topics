"""D3 — Seed the single Qdrant collection from chunks.parquet.

The D3 refactor collapses D2's three per-source collections into ONE
(``chunks_bge384``) seeded from the full parquet, with ``source`` in the payload
so the api can filter at query time:

    python scripts/seed_qdrant.py                # (re)build chunks_bge384 from the full parquet
    python scripts/seed_qdrant.py --no-recreate  # upsert into the existing collection
    python scripts/seed_qdrant.py --keep-legacy  # don't drop the old per-source collections

Point IDs are the parquet row index (the global corpus_idx the retriever uses).
Reads QDRANT_URL from env (default: http://localhost:6333). Idempotent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from csai415.qdrant_dense import QDRANT_COLLECTION, seed_collection_from_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PARQUET = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"

# D2 per-source collections, now superseded by the single QDRANT_COLLECTION.
# Dropped on (re)seed unless --keep-legacy is passed, so the live store doesn't
# carry stale, divergent copies after the refactor.
LEGACY_COLLECTIONS = ("chunks_bge384_scifact", "chunks_bge384_arxiv")


def _drop_legacy(client: QdrantClient) -> list[str]:
    """Delete the old per-source collections if present. Returns the ones dropped."""
    existing = {c.name for c in client.get_collections().collections}
    dropped = []
    for name in LEGACY_COLLECTIONS:
        if name in existing:
            client.delete_collection(name)
            dropped.append(name)
    return dropped


def seed_qdrant(qdrant_url: str, recreate: bool = True, keep_legacy: bool = False) -> dict:
    """Seed the single ``chunks_bge384`` collection from the full parquet."""
    client = QdrantClient(url=qdrant_url)

    dropped = [] if keep_legacy else _drop_legacy(client)
    if dropped:
        print(f"seed_qdrant: dropped legacy collections {dropped}")

    n = seed_collection_from_parquet(
        client, CHUNKS_PARQUET, collection=QDRANT_COLLECTION, recreate=recreate
    )
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"seed_qdrant: {n} points into {QDRANT_COLLECTION} "
          f"(size={info.config.params.vectors.size}, dist={info.config.params.vectors.distance})")
    return {"points": n, "collection": QDRANT_COLLECTION, "dropped_legacy": dropped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-recreate", action="store_true",
                    help="upsert into the existing collection instead of recreating")
    ap.add_argument("--keep-legacy", action="store_true",
                    help="do not drop the old per-source collections")
    args = ap.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    recreate = not args.no_recreate

    print(f"seed_qdrant: url={qdrant_url} collection={QDRANT_COLLECTION} recreate={recreate}")
    seed_qdrant(qdrant_url, recreate=recreate, keep_legacy=args.keep_legacy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
