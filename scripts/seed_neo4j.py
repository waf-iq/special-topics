"""Pair C — Neo4j graph loader. See D2_TASKS.md §D2-C1.

Reads paper metadata and MERGEs an idempotent property graph:

    (:Paper {paper_id, title, year, source})
    (:Author {name})
    (:Topic  {name})
    (:Author)-[:WROTE]->(:Paper)
    (:Paper)-[:ABOUT]->(:Topic)

Source of truth is **Mongo `papers`** (the H0 contract) — NOT the parquet.
The Neo4j build is the one piece that needs Mongo: it consumes the deduped,
metadata-joined `papers` collection that D2-A2's seed produces. A `--from-parquet`
dev escape hatch reads `chunks.parquet` directly so this loader is runnable and
testable before D2-A2 lands; it is dev-only and mirrors what D2-A2 will store.

SciFact papers carry no author metadata (BEIR ships none), so they have empty
`authors` and are skipped — loading them would create a degenerate authorless
graph. The skipped count is logged.

All writes use MERGE, so re-running is idempotent (safe reseed). Uniqueness
constraints on Paper.paper_id / Author.name / Topic.name are applied at startup
(no-op if already present).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

# --- Cypher --------------------------------------------------------------

CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
)

# One paper + its authors/topics in a single statement. UNWIND keeps it to one
# round-trip per paper and is fully idempotent (MERGE everywhere).
MERGE_PAPER = """
MERGE (p:Paper {paper_id: $paper_id})
SET p.title = $title, p.year = $year, p.source = $source
WITH p
UNWIND $authors AS author_name
  MERGE (a:Author {name: author_name})
  MERGE (a)-[:WROTE]->(p)
WITH DISTINCT p
UNWIND $topics AS topic_name
  MERGE (t:Topic {name: topic_name})
  MERGE (p)-[:ABOUT]->(t)
"""


# --- Paper normalization (pure, unit-tested) -----------------------------

# Parquet uses "arxiv-demo" / "arxiv"; the Mongo contract collapses both to the
# "arxiv" source enum. Normalize so Paper.source matches what D2-A2 stores.
_SOURCE_ALIASES = {"arxiv-demo": "arxiv", "arxiv": "arxiv", "scifact": "scifact"}


def _clean_year(value) -> Optional[int]:
    """Coerce pandas/Mongo null, NaN, or float year into int|None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_list(value) -> list[str]:
    """Coerce None / numpy array / scalar into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        value = [value]
    elif hasattr(value, "tolist"):  # numpy array
        value = value.tolist()
    out = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def normalize_paper(raw: dict) -> dict:
    """Map a raw paper record (Mongo doc or parquet-derived) to the graph schema.

    Returns a dict with keys: paper_id, title, year, source, authors, topics.
    Deduplicates authors/topics while preserving order.
    """
    authors = list(dict.fromkeys(_clean_list(raw.get("authors"))))
    # Accept both the contract's "topics" (list) and parquet's "topic" (scalar).
    topics_raw = raw.get("topics", raw.get("topic"))
    topics = list(dict.fromkeys(_clean_list(topics_raw)))
    source = _SOURCE_ALIASES.get(raw.get("source"), raw.get("source"))
    title = raw.get("title")
    return {
        "paper_id": str(raw["paper_id"]),
        "title": str(title) if title is not None else "",
        "year": _clean_year(raw.get("year")),
        "source": source,
        "authors": authors,
        "topics": topics,
    }


def should_load(paper: dict) -> bool:
    """A paper joins the graph only if it has at least one author.

    SciFact rows have no author metadata; loading them yields authorless nodes
    with no WROTE edges, which is the degenerate graph we want to avoid.
    """
    return bool(paper.get("authors"))


# --- Paper sources -------------------------------------------------------


def iter_papers_from_mongo(
    mongo_url: str, db: str = "csai415", source: str = "all"
) -> Iterator[dict]:
    """Primary path: stream `papers` from Mongo (the H0 contract collection)."""
    from pymongo import MongoClient

    client = MongoClient(mongo_url)
    query: dict = {}
    if source != "all":
        query["source"] = source
    try:
        for doc in client[db]["papers"].find(query):
            yield doc
    finally:
        client.close()


def iter_papers_from_parquet(parquet_path: Path, source: str = "all") -> Iterator[dict]:
    """Dev fallback: derive paper records from `chunks.parquet`.

    Dedups chunk rows to one record per paper_id. Mirrors the shape D2-A2's
    Mongo seed produces (source collapsed to scifact|arxiv) so the graph is
    identical whether sourced from Mongo or this fallback. Dev-only.
    """
    import pandas as pd

    cols = ["paper_id", "title", "authors", "year", "topic", "source"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df.drop_duplicates("paper_id", keep="first")
    for row in df.to_dict("records"):
        norm_source = _SOURCE_ALIASES.get(row.get("source"), row.get("source"))
        if source != "all" and norm_source != source:
            continue
        yield row


# --- Graph load ----------------------------------------------------------


def seed_graph(session, papers: Iterable[dict], apply_constraints: bool = True) -> dict:
    """MERGE papers into the graph through ``session`` (anything with ``.run``).

    ``session`` is duck-typed to the neo4j driver session — also satisfied by a
    fake recorder in tests, so this needs no live Neo4j. Returns load stats.
    """
    if apply_constraints:
        for stmt in CONSTRAINTS:
            session.run(stmt)

    stats = {
        "papers_loaded": 0,
        "papers_skipped": 0,
        "authors": set(),
        "topics": set(),
        "wrote_edges": 0,
        "about_edges": 0,
    }
    for raw in papers:
        paper = normalize_paper(raw)
        if not should_load(paper):
            stats["papers_skipped"] += 1
            continue
        session.run(
            MERGE_PAPER,
            paper_id=paper["paper_id"],
            title=paper["title"],
            year=paper["year"],
            source=paper["source"],
            authors=paper["authors"],
            topics=paper["topics"],
        )
        stats["papers_loaded"] += 1
        stats["authors"].update(paper["authors"])
        stats["topics"].update(paper["topics"])
        stats["wrote_edges"] += len(paper["authors"])
        stats["about_edges"] += len(paper["topics"])

    # Collapse sets to counts for a clean summary.
    stats["authors"] = len(stats["authors"])
    stats["topics"] = len(stats["topics"])
    return stats


# --- CLI -----------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed Neo4j with the paper/author/topic graph (D2-C1).")
    p.add_argument(
        "--source",
        choices=["scifact", "arxiv", "all"],
        default="all",
        help="Restrict to one source (default: all). scifact rows are skipped anyway (no authors).",
    )
    p.add_argument(
        "--from-parquet",
        type=Path,
        default=None,
        metavar="PATH",
        help="DEV ONLY: read papers from chunks.parquet instead of Mongo. "
        "Use before D2-A2's Mongo seed lands.",
    )
    p.add_argument("--mongo-url", default=os.environ.get("MONGO_URL", "mongodb://localhost:27017"))
    p.add_argument("--mongo-db", default="csai415")
    p.add_argument("--neo4j-url", default=os.environ.get("NEO4J_URL", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Omit for NEO4J_AUTH=none dev containers.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Read + normalize papers and print stats without touching Neo4j.",
    )
    return p.parse_args(argv)


def _papers_iter(args) -> Iterator[dict]:
    if args.from_parquet is not None:
        print(f"[seed_neo4j] DEV mode: reading papers from {args.from_parquet} (--from-parquet)")
        return iter_papers_from_parquet(args.from_parquet, source=args.source)
    print(f"[seed_neo4j] reading papers from Mongo {args.mongo_url} db={args.mongo_db}")
    return iter_papers_from_mongo(args.mongo_url, db=args.mongo_db, source=args.source)


class _DryRunSession:
    """Stand-in session that counts statements without a database."""

    def __init__(self):
        self.calls = 0

    def run(self, *_a, **_k):
        self.calls += 1


def main(argv=None) -> None:
    args = _parse_args(argv)
    papers = _papers_iter(args)

    if args.dry_run:
        stats = seed_graph(_DryRunSession(), papers, apply_constraints=False)
        _print_stats(stats, dry_run=True)
        return

    from neo4j import GraphDatabase

    auth = (args.neo4j_user, args.neo4j_password) if args.neo4j_password else None
    driver = GraphDatabase.driver(args.neo4j_url, auth=auth)
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            stats = seed_graph(session, papers)
    finally:
        driver.close()
    _print_stats(stats, dry_run=False)


def _print_stats(stats: dict, dry_run: bool) -> None:
    tag = "[seed_neo4j][dry-run]" if dry_run else "[seed_neo4j]"
    print(f"{tag} papers loaded : {stats['papers_loaded']}")
    print(f"{tag} papers skipped: {stats['papers_skipped']} (no authors — e.g. SciFact)")
    print(f"{tag} unique authors: {stats['authors']}")
    print(f"{tag} unique topics : {stats['topics']}")
    print(f"{tag} WROTE edges   : {stats['wrote_edges']}")
    print(f"{tag} ABOUT edges   : {stats['about_edges']}")


if __name__ == "__main__":
    main()
