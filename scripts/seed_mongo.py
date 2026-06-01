"""D2-A2 — Seed MongoDB papers + chunks collections from chunks.parquet.

Usage:
    python scripts/seed_mongo.py                     # seed all sources
    python scripts/seed_mongo.py --source scifact    # only SciFact rows
    python scripts/seed_mongo.py --source arxiv      # only arxiv + arxiv-demo rows

Reads MONGO_URL from env (default: mongodb://localhost:27017).
Idempotent: upserts by _id, safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import MongoClient, ReplaceOne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PARQUET = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"

ARXIV_META_PATHS = [
    PROJECT_ROOT / "data" / "raw_pdfs" / "arxiv_batch_meta.json",
    PROJECT_ROOT / "data" / "raw_pdfs" / "arxiv_meta.json",
]

DB_NAME = "csai415"
BULK_BATCH = 1000

SOURCE_FILTERS = {
    "scifact": {"scifact"},
    "arxiv": {"arxiv", "arxiv-demo"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(v):
    """Coerce pandas NaN / None to Python None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _to_int(v):
    v = _clean(v)
    if v is None:
        return None
    return int(v)


def _to_list(v):
    """Coerce None / NaN / numpy array / scalar into a clean list of strings.

    Parquet stores authors as a numpy ndarray, not a Python list, so the old
    ``isinstance(v, list)`` check dropped every author. Handle ndarrays (via
    ``tolist``) and bare scalars too.
    """
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    if isinstance(v, str):
        return [v]
    if hasattr(v, "tolist"):  # numpy array / pandas array
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    return []


def _load_topics_map() -> dict[str, list[str]]:
    """paper_id -> full arXiv categories list. SciFact papers won't appear here."""
    topics: dict[str, list[str]] = {}
    for path in ARXIV_META_PATHS:
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            cats = entry.get("categories") or ([entry["topic"]] if entry.get("topic") else ["cs.CL"])
            topics[entry["paper_id"]] = cats
    return topics


def _filter_df(df: pd.DataFrame, source_filter: str | None) -> pd.DataFrame:
    if source_filter is None:
        return df
    allowed = SOURCE_FILTERS[source_filter]
    return df[df["source"].isin(allowed)]


# ---------------------------------------------------------------------------
# Paper / chunk document builders
# ---------------------------------------------------------------------------

def _build_paper_doc(row, topics_map: dict, now: datetime) -> dict:
    pid = row["paper_id"]
    source = row["source"]
    is_arxiv = source in ("arxiv", "arxiv-demo")
    # Prefer the full categories list from the arXiv meta JSON; fall back to the
    # single `topic` column in the parquet when that meta file isn't present
    # (otherwise arXiv papers land with no Topic nodes and the topic-based
    # Cypher queries return nothing).
    topics = topics_map.get(pid)
    if not topics:
        t = _clean(row.get("topic"))
        topics = [t] if t else []
    return {
        "_id": pid,
        "title": _clean(row.get("title")) or "",
        "authors": _to_list(row.get("authors")),
        "year": _to_int(row.get("year")),
        "venue": None,
        "topics": topics,
        "source": source,
        "pdf_path": f"data/raw_pdfs/{pid}.pdf" if is_arxiv else None,
        "ingested_at": now,
    }


def _build_chunk_doc(row, position: int) -> dict:
    return {
        "_id": row["chunk_id"],
        "paper_id": row["paper_id"],
        "text": row["text"],
        "position": position,
        "page_start": _to_int(row.get("page_start")),
        "page_end": _to_int(row.get("page_end")),
        "source": row["source"],
    }


# ---------------------------------------------------------------------------
# Bulk upsert
# ---------------------------------------------------------------------------

def _bulk_upsert(collection, docs: list[dict]) -> tuple[int, int]:
    """Upsert docs in batches. Returns (upserted, modified) counts."""
    ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs]
    upserted, modified = 0, 0
    for i in range(0, len(ops), BULK_BATCH):
        result = collection.bulk_write(ops[i : i + BULK_BATCH], ordered=False)
        upserted += result.upserted_count
        modified += result.modified_count
    return upserted, modified


# ---------------------------------------------------------------------------
# Main seed logic
# ---------------------------------------------------------------------------

def seed_mongo(mongo_url: str, source_filter: str | None = None) -> dict:
    """Seed papers + chunks collections. Returns stats dict."""
    df = _filter_df(pd.read_parquet(CHUNKS_PARQUET), source_filter)
    if df.empty:
        print(f"seed_mongo: no rows for source={source_filter!r}")
        return {"papers": 0, "chunks": 0}

    db = MongoClient(mongo_url)[DB_NAME]
    topics_map = _load_topics_map()
    now = datetime.now(timezone.utc)

    # Deduplicate papers (one doc per paper_id)
    seen: set[str] = set()
    paper_docs = []
    for _, row in df.iterrows():
        if row["paper_id"] in seen:
            continue
        seen.add(row["paper_id"])
        paper_docs.append(_build_paper_doc(row, topics_map, now))

    chunk_docs = [_build_chunk_doc(row, i) for i, (_, row) in enumerate(df.iterrows())]

    u, m = _bulk_upsert(db["papers"], paper_docs)
    print(f"seed_mongo: papers — {u} upserted, {m} modified")

    u, m = _bulk_upsert(db["chunks"], chunk_docs)
    print(f"seed_mongo: chunks — {u} upserted, {m} modified")

    db["papers"].create_index("authors")
    db["papers"].create_index("year")
    db["chunks"].create_index("paper_id")

    stats = {"papers": len(paper_docs), "chunks": len(chunk_docs)}
    print(f"seed_mongo: done — {stats}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["scifact", "arxiv", "all"], default="all")
    args = ap.parse_args()

    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    source = None if args.source == "all" else args.source
    print(f"seed_mongo: url={mongo_url} source={args.source}")
    seed_mongo(mongo_url, source_filter=source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
