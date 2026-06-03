"""D2-B3 — Recall@k + NDCG@5 + p95 latency for `/search` on the 60-query SciFact holdout.

Covers Ahmed Soliman's task while he's offline. Uses the in-process
``HybridRetriever`` (numpy dense backend, no Docker needed) — same retrieval
logic ``/search`` invokes through its per-source HybridRetriever path in
``src/csai415/api.py``.

Metric quality (Recall@5, Recall@10, NDCG@5) is **identical** to what /search
would return — same blessed BOHB config, same per-source subset retriever,
same BM25 + dense fusion. The only thing the in-process path skips is HTTP
transport + Pydantic serialization, which Musab's ``d2_int2_verification.md``
measured at +5–15 ms per request (p95 over the wire = 438 ms; in-process p95
here should be lower).

Outputs:
  reports/D2/d2_search_metrics.csv  -- 3 configs (bm25_only / dense_only /
                                    hybrid_blessed) x {recall@5, recall@10,
                                    ndcg@5, p95_latency_ms}
  reports/D2/d2_topk_examples.md    -- 5 example queries (3 SciFact + 2 arXiv)
                                    with top-5 hits, paper titles, page ranges
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from csai415.api import load_blessed_config
from csai415.eval import evaluate
from csai415.retrieve import HybridRetriever, load_chunks

CHUNKS_PARQUET = Path("data/processed/chunks.parquet")
QA_JSONL = Path("data/gold/qa.jsonl")
SPLIT_INDICES = Path("configs/d1_split_indices.json")
RUNCARD = Path("configs/winning_runcard.yaml")
METRICS_CSV = Path("reports/D2/d2_search_metrics.csv")
TOPK_MD = Path("reports/D2/d2_topk_examples.md")


def load_holdout_queries() -> list[dict]:
    """Return the 60-query SciFact holdout used by the D1 runcard (so D2 numbers
    are directly comparable to ``configs/winning_runcard.yaml`` winner_holdout)."""
    with QA_JSONL.open(encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f]
    indices = json.loads(SPLIT_INDICES.read_text(encoding="utf-8"))["holdout"]
    return [all_queries[i] for i in indices]


def build_source_retriever(source: str) -> HybridRetriever:
    """Build a HybridRetriever over a source-filtered slice of chunks.parquet.

    Matches ``api.py``'s per-source HybridRetriever wiring (numpy backend; the
    live FastAPI swap to Qdrant happens via dense_backend injection but doesn't
    change retrieval quality — only the candidate-generation transport).
    """
    df = load_chunks(CHUNKS_PARQUET)
    if source == "scifact":
        sub = df[df["source"] == "scifact"].reset_index(drop=True)
    elif source == "arxiv":
        sub = df[df["source"].isin({"arxiv", "arxiv-demo"})].reset_index(drop=True)
    else:
        raise ValueError(f"unknown source {source!r}")
    cfg = load_blessed_config(RUNCARD)
    return HybridRetriever(sub, cfg)


def run_metrics() -> list[dict]:
    """Three configs x four metrics on the 60-query SciFact holdout."""
    retriever = build_source_retriever("scifact")
    queries = load_holdout_queries()
    print(f"[D2-B3] {len(queries)} SciFact holdout queries against {len(retriever.df)} scifact chunks")

    def make_fn(weight_override: Optional[float]):
        return lambda q, k, _hw: retriever.search(q, k, hybrid_weight=weight_override)

    configs = [
        ("bm25_only", make_fn(0.0)),
        ("dense_only", make_fn(1.0)),
        ("hybrid_blessed", make_fn(None)),  # blessed runcard hybrid_weight=0.777
    ]

    rows: list[dict] = []
    for name, fn in configs:
        at5 = evaluate(fn, queries, k=5)
        at10 = evaluate(fn, queries, k=10)
        # eval.evaluate's dict keys are hardcoded "ndcg5"/"recall5" regardless of k —
        # at k=10 the "recall5" key actually carries Recall@10.
        rows.append({
            "config": name,
            "ndcg@5": round(at5["ndcg5"], 4),
            "recall@5": round(at5["recall5"], 4),
            "recall@10": round(at10["recall5"], 4),
            "p95_latency_ms": round(at5["p95_latency_ms"], 1),
        })
        print(f"  {name:15s}  ndcg@5={at5['ndcg5']:.4f}  recall@5={at5['recall5']:.4f}  "
              f"recall@10={at10['recall5']:.4f}  p95={at5['p95_latency_ms']:.1f}ms")

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[D2-B3] wrote {METRICS_CSV}")
    print("[D2-B3] D1 runcard reference (winner_holdout): ndcg5=0.561 recall5=0.649 p95=103ms")
    print("[D2-B3] Expected <1pp drift on hybrid_blessed; bm25_only and dense_only consistent with D1 baselines.")
    return rows


def _page_range(row) -> str:
    """Format ``page_start``/``page_end`` as 'start-end' (or '-' when both null)."""
    ps = row.get("page_start")
    pe = row.get("page_end")
    if ps is None or (isinstance(ps, float) and pd.isna(ps)):
        return "-"
    ps_i = int(ps)
    pe_i = int(pe) if pe is not None and not (isinstance(pe, float) and pd.isna(pe)) else ps_i
    return str(ps_i) if pe_i == ps_i else f"{ps_i}-{pe_i}"


def run_topk_examples() -> None:
    """Five example queries with top-5 hits and citations.

    Mix: 3 SciFact (from the holdout — relevance verifiable) + 2 arXiv
    (qualitative only — arXiv corpus has no qrels). The arXiv queries
    demonstrate the agent-tool-style flow D3's GraphRAG executor will use.
    """
    scifact_retriever = build_source_retriever("scifact")
    arxiv_retriever = build_source_retriever("arxiv")
    df = load_chunks(CHUNKS_PARQUET)
    lookup = df.set_index("chunk_id")

    holdout = load_holdout_queries()
    # Spread across the holdout for variety (positions 0 / 20 / 40 in the 60-query list)
    scifact_examples = [holdout[0], holdout[20], holdout[40]]

    arxiv_examples = [
        {"qid": "arxiv-demo-1", "question": "language model pretraining and instruction tuning",
         "relevant_chunk_ids": []},
        {"qid": "arxiv-demo-2", "question": "graph neural networks for information retrieval",
         "relevant_chunk_ids": []},
    ]

    lines: list[str] = []
    lines.append("# D2 Top-K Examples — `/search` against the live retrieval stack\n\n")
    lines.append("Generated by `scripts/eval_search_metrics.py`. Blessed BOHB config from "
                 "`configs/winning_runcard.yaml` (`hybrid_weight=0.777, candidate_k=27, metric=l2, "
                 "bm25_k1=2.92, bm25_b=0.345`). Per-source retrievers match `api.py`'s "
                 "production routing — `source=scifact` queries the SciFact subset only "
                 "(no arXiv contamination); `source=arxiv` queries the arXiv subset only.\n\n")
    lines.append("---\n\n")
    lines.append("## SciFact holdout queries (relevance from human qrels)\n\n")

    for q in scifact_examples:
        relevant = set(q["relevant_chunk_ids"])
        hits = scifact_retriever.search(q["question"], k=5)
        lines.append(f"### qid={q['qid']} — `source=scifact`\n\n")
        lines.append(f"**Query:** {q['question']}\n\n")
        lines.append(f"**Relevant chunk_ids ({len(relevant)}):** "
                     f"`{', '.join(sorted(relevant)[:3])}{'...' if len(relevant) > 3 else ''}`\n\n")
        lines.append("| Rank | chunk_id | paper_id | title | page | hit? |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, cid in enumerate(hits, 1):
            row = lookup.loc[cid]
            title = (row["title"] or "")[:60].replace("\n", " ")
            paper_id = row["paper_id"]
            pr = _page_range(row)
            hit_mark = "Y" if cid in relevant else ""
            lines.append(f"| {i} | `{cid}` | `{paper_id}` | {title} | {pr} | {hit_mark} |\n")
        lines.append("\n")

    lines.append("---\n\n")
    lines.append("## arXiv qualitative queries (no qrels — demonstrates agent flow for D3)\n\n")

    for q in arxiv_examples:
        hits = arxiv_retriever.search(q["question"], k=5)
        lines.append(f"### qid={q['qid']} — `source=arxiv`\n\n")
        lines.append(f"**Query:** {q['question']}\n\n")
        lines.append("| Rank | chunk_id | paper_id | title | page | score |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, cid in enumerate(hits, 1):
            row = lookup.loc[cid]
            title = (row["title"] or "")[:60].replace("\n", " ")
            paper_id = row["paper_id"]
            pr = _page_range(row)
            lines.append(f"| {i} | `{cid}` | `{paper_id}` | {title} | {pr} | — |\n")
        lines.append("\n")

    TOPK_MD.parent.mkdir(parents=True, exist_ok=True)
    TOPK_MD.write_text("".join(lines), encoding="utf-8")
    print(f"[D2-B3] wrote {TOPK_MD}")


if __name__ == "__main__":
    run_metrics()
    run_topk_examples()
