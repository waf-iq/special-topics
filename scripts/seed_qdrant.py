"""D2-A2 / D2-INT2 — Seed Qdrant collections from chunks.parquet.

Each ``--source`` writes to its OWN collection so the D2-INT2 per-source
retriever design works (each per-source HybridRetriever consumes a
collection whose point IDs match its reset-indexed local df):

    python scripts/seed_qdrant.py                     # all  -> chunks_bge384
    python scripts/seed_qdrant.py --source scifact    # scifact-only -> chunks_bge384_scifact
    python scripts/seed_qdrant.py --source arxiv      # arxiv + arxiv-demo -> chunks_bge384_arxiv
    python scripts/seed_qdrant.py --no-recreate       # upsert into existing collection

Reads QDRANT_URL from env (default: http://localhost:6333).
Idempotent: upserts by point ID (parquet row index).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

from csai415.qdrant_dense import QDRANT_COLLECTION, seed_collection_from_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PARQUET = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"

SOURCE_FILTERS = {
    "scifact": {"scifact"},
    "arxiv": {"arxiv", "arxiv-demo"},
}

# Per-source -> collection name. MUST match SOURCE_COLLECTIONS in
# src/csai415/api.py so the live api wires each per-source retriever to
# the right Qdrant collection.
SOURCE_COLLECTIONS = {
    None: QDRANT_COLLECTION,            # "chunks_bge384"  (full corpus)
    "scifact": "chunks_bge384_scifact",
    "arxiv": "chunks_bge384_arxiv",
}


def _prepare_parquet(source_filter: str | None) -> tuple[Path, bool]:
    """Return (parquet_path, is_temp). Filters to a temp file when needed."""
    if source_filter is None:
        return CHUNKS_PARQUET, False

    df = pd.read_parquet(CHUNKS_PARQUET)
    df = df[df["source"].isin(SOURCE_FILTERS[source_filter])]
    if df.empty:
        print(f"seed_qdrant: no rows for source={source_filter!r}")
        return Path(), True  # caller checks point count

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    df.reset_index(drop=True).to_parquet(tmp_path)
    return tmp_path, True


def seed_qdrant(qdrant_url: str, source_filter: str | None = None, recreate: bool = True) -> dict:
    """Seed the source's Qdrant collection from parquet. Returns stats dict.

    Collection name is derived from ``source_filter`` via
    :data:`SOURCE_COLLECTIONS` — different sources go to different collections
    so a re-seed of one source can't clobber another.
    """
    parquet_path, is_temp = _prepare_parquet(source_filter)
    collection = SOURCE_COLLECTIONS[source_filter]

    try:
        client = QdrantClient(url=qdrant_url)
        n = seed_collection_from_parquet(client, parquet_path, collection=collection, recreate=recreate)
    finally:
        if is_temp:
            parquet_path.unlink(missing_ok=True)

    info = client.get_collection(collection)
    print(f"seed_qdrant: {n} points into {collection} "
          f"(size={info.config.params.vectors.size}, dist={info.config.params.vectors.distance})")
    return {"points": n, "collection": collection}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["scifact", "arxiv", "all"], default="all")
    ap.add_argument("--no-recreate", action="store_true",
                    help="upsert into existing collection instead of recreating")
    args = ap.parse_args()

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    source = None if args.source == "all" else args.source
    recreate = not args.no_recreate

    print(f"seed_qdrant: url={qdrant_url} source={args.source} recreate={recreate}")
    seed_qdrant(qdrant_url, source_filter=source, recreate=recreate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
