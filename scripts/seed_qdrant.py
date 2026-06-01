"""D2-A2 — Seed Qdrant chunks_bge384 collection from chunks.parquet.

Usage:
    python scripts/seed_qdrant.py                     # seed all sources
    python scripts/seed_qdrant.py --source scifact    # only SciFact rows
    python scripts/seed_qdrant.py --source arxiv      # only arxiv + arxiv-demo rows
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
    """Seed Qdrant collection from parquet. Returns stats dict."""
    parquet_path, is_temp = _prepare_parquet(source_filter)

    try:
        client = QdrantClient(url=qdrant_url)
        n = seed_collection_from_parquet(client, parquet_path, recreate=recreate)
    finally:
        if is_temp:
            parquet_path.unlink(missing_ok=True)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"seed_qdrant: {n} points into {QDRANT_COLLECTION} "
          f"(size={info.config.params.vectors.size}, dist={info.config.params.vectors.distance})")
    return {"points": n}


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
