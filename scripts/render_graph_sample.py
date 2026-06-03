"""Render a Paper/Author/Topic subgraph as a PNG for the D2 report.

Reads the same data that ``seed_neo4j.py`` ingests (``chunks.parquet`` plus
``arxiv_batch_meta.json`` for the full categories list), builds a networkx
graph from a 30-paper sample, lays it out with spring layout, colors by node
type. The result is a Neo4j-style visualization without needing the live
stack — same nodes, same edges Neo4j carries.

Output: ``reports/d2_graph_sample.png``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
ARXIV_META = Path("data/raw_pdfs/arxiv_batch_meta.json")
OUTPUT = Path("reports/d2_graph_sample.png")

PAPER_COLOR = "#4C72B0"   # blue
AUTHOR_COLOR = "#55A868"  # green
TOPIC_COLOR = "#C44E52"   # red
SEED = 42
N_PAPERS = 30


def load_paper_records() -> list[dict]:
    """Materialize one record per arxiv paper with title + author list + categories.

    Same shape ``seed_neo4j.py`` consumes from Mongo. We skip SciFact rows
    (no authors in BEIR) and arxiv-demo (D1 demo PDFs have authors but lack
    the full ``categories`` list that D2-A2's Mongo seed uses).
    """
    df = pd.read_parquet(CHUNKS_PARQUET, columns=["paper_id", "title", "authors", "source"])
    arxiv = df[df["source"] == "arxiv"].drop_duplicates("paper_id").reset_index(drop=True)
    categories_by_id = {}
    if ARXIV_META.exists():
        for entry in json.loads(ARXIV_META.read_text(encoding="utf-8")):
            categories_by_id[entry["paper_id"]] = entry.get("categories") or []

    records = []
    for _, row in arxiv.iterrows():
        authors = list(row["authors"]) if row["authors"] is not None else []
        if not authors:
            continue
        records.append({
            "paper_id": row["paper_id"],
            "title": (row["title"] or "")[:50],
            "authors": authors,
            "topics": categories_by_id.get(row["paper_id"], []),
        })
    return records


def build_subgraph(records: list[dict], n_papers: int = N_PAPERS) -> nx.Graph:
    """Sample n_papers + their authors + their topics into a labeled networkx graph."""
    rng = random.Random(SEED)
    sample = rng.sample(records, min(n_papers, len(records)))

    G = nx.Graph()
    for rec in sample:
        pid = rec["paper_id"]
        G.add_node(pid, kind="Paper", label=pid.split("v")[0])
        for author in rec["authors"]:
            G.add_node(author, kind="Author", label=author)
            G.add_edge(author, pid, rel="WROTE")
        for topic in rec["topics"]:
            G.add_node(topic, kind="Topic", label=topic)
            G.add_edge(pid, topic, rel="ABOUT")
    return G


def render(G: nx.Graph, out_path: Path) -> None:
    """Spring layout, node sizes by degree, colored by type."""
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=SEED, k=0.5, iterations=80)

    by_kind: dict[str, list[str]] = {"Paper": [], "Author": [], "Topic": []}
    for node, attrs in G.nodes(data=True):
        by_kind[attrs["kind"]].append(node)

    degrees = dict(G.degree())

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8, ax=ax)
    for kind, color in (("Paper", PAPER_COLOR), ("Author", AUTHOR_COLOR), ("Topic", TOPIC_COLOR)):
        nodes = by_kind[kind]
        sizes = [60 + 12 * degrees[n] for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=sizes,
                               alpha=0.85, ax=ax, label=kind)

    # Label only Topic nodes + high-degree authors so the image stays legible.
    label_pool: dict[str, str] = {n: G.nodes[n]["label"] for n in by_kind["Topic"]}
    top_authors = sorted(by_kind["Author"], key=lambda n: degrees[n], reverse=True)[:8]
    label_pool.update({n: G.nodes[n]["label"][:20] for n in top_authors})
    nx.draw_networkx_labels(G, pos, labels=label_pool, font_size=8, ax=ax)

    topic_counts = Counter(G.nodes[n]["label"] for n in by_kind["Topic"])
    ax.set_title(
        f"D2 Neo4j subgraph sample — {len(by_kind['Paper'])} papers, "
        f"{len(by_kind['Author'])} authors, {len(by_kind['Topic'])} topics "
        f"({', '.join(f'{k}:{v}' for k, v in topic_counts.most_common(5))})",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10, markerscale=0.6)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"render_graph_sample: wrote {out_path} ({len(G)} nodes, {G.number_of_edges()} edges)")


def main() -> int:
    records = load_paper_records()
    if not records:
        raise RuntimeError("no arxiv papers with authors found in chunks.parquet")
    G = build_subgraph(records, N_PAPERS)
    render(G, OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
