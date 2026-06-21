"""D3 Task 2 — reranker comparison on the 60-query SciFact holdout. Owner: Ahmed Soliman.

Reorders the candidate pool from the blessed D2 ``HybridRetriever`` with each reranker
(``csai415.rerank.get_reranker``) and measures quality + end-to-end latency via
``csai415.eval.evaluate`` (the same harness the D1 AutoML objective and D2-B3 used, so the
numbers are directly comparable). SciFact is the only corpus with relevance labels, so it is
the gradeable target for an NDCG@5 *lift* figure; arXiv has no qrels.

``p95_latency_ms`` is end-to-end (retrieval + rerank), so the ``none`` row is pure retrieval
and every other row's extra latency is the rerank cost — exactly the quality/latency trade-off.

Outputs:
  reports/D3/d3_rerank_comparison.csv   -- one row per (reranker, pool)
  reports/D3/d3_rerank.md               -- comparison table + pool sweep + trade-off notes

The winner is left **TBD** — chosen by the team after reviewing the report.

Run:  .venv/bin/python -m scripts.eval_rerank      (from the repo root)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from csai415.eval import evaluate
from csai415.rerank import get_reranker
from csai415.retrieve import HybridRetriever, RetrieverConfig, load_chunks

CHUNKS = Path("data/processed/chunks.parquet")
QA = Path("data/gold/qa.jsonl")
SPLIT = Path("configs/d1_split_indices.json")
RUNCARD = Path("configs/winning_runcard.yaml")
OUT_CSV = Path("reports/D3/d3_rerank_comparison.csv")
OUT_MD = Path("reports/D3/d3_rerank.md")

RERANKERS = ["none", "minilm", "bge", "mmr"]  # main comparison
MAIN_POOL = 30  # candidates fed to the reranker in the main table
POOL_SWEEP = [10, 20, 30, 50]  # candidate-pool sweep
SWEEP_KINDS = ["minilm", "bge"]  # cross-encoders are the pool-sensitive ones
TOP_K = 5


def load_cfg() -> RetrieverConfig:
    card = yaml.safe_load(RUNCARD.read_text(encoding="utf-8"))
    bp = card["automl"]["best_params"]
    return RetrieverConfig(
        metric=bp["metric"],
        svd_dim=bp["svd_dim"],
        normalize=bp["normalize"],
        hybrid_weight=bp["hybrid_weight"],
        candidate_k=bp["candidate_k"],
        bm25_k1=bp["bm25_k1"],
        bm25_b=bp["bm25_b"],
        seed=card.get("seed", 42),
    )


def load_holdout() -> list[dict]:
    """The 60-query SciFact holdout (same split as the D1 runcard / D2-B3)."""
    qs = [json.loads(line) for line in QA.open(encoding="utf-8")]
    idx = json.loads(SPLIT.read_text(encoding="utf-8"))["holdout"]
    return [qs[i] for i in idx]


def build_scifact_retriever(cfg: RetrieverConfig):
    df = load_chunks(CHUNKS)
    sub = df[df["source"] == "scifact"].reset_index(drop=True)
    return HybridRetriever(sub, cfg), sub.set_index("chunk_id")


def make_fn(retriever, lookup, reranker, pool):
    """Wrap retrieve-pool-then-rerank into eval.evaluate's retriever_fn signature."""

    def fn(query, k, _hw):
        pool_ids = retriever.search(query, pool)
        ctext = {cid: str(lookup.loc[cid]["text"]) for cid in pool_ids}
        return reranker(query, pool_ids, ctext, top_n=k)

    return fn


def _measure(retriever, lookup, kind, pool, queries):
    """Run one config; warm up first so model-load time never enters the latency stats."""
    reranker = get_reranker(kind)
    fn = make_fn(retriever, lookup, reranker, pool)
    try:
        fn(queries[0]["question"], TOP_K, None)  # warmup: load embedder + reranker model
        return evaluate(fn, queries, k=TOP_K)
    except Exception as exc:  # model download/runtime failure — record, don't crash the run
        print(f"  !! {kind} pool={pool} failed: {type(exc).__name__}: {exc}")
        return None


def run() -> None:
    cfg = load_cfg()
    retriever, lookup = build_scifact_retriever(cfg)
    queries = load_holdout()
    print(f"[rerank] {len(queries)} SciFact holdout queries x {len(retriever.df)} scifact chunks")
    print(f"[rerank] blessed config: metric={cfg.metric} hybrid_weight={cfg.hybrid_weight:.3f} "
          f"candidate_k={cfg.candidate_k}")

    # Baseline (no rerank) NDCG@5, for the lift column.
    base = _measure(retriever, lookup, "none", MAIN_POOL, queries)
    base_ndcg = base["ndcg5"] if base else 0.0

    main_rows: list[dict] = []
    for kind in RERANKERS:
        m = base if kind == "none" else _measure(retriever, lookup, kind, MAIN_POOL, queries)
        if m is None:
            main_rows.append({"reranker": kind, "pool": MAIN_POOL, "ndcg@5": "ERR",
                              "recall@5": "ERR", "ndcg@5_lift": "ERR", "p95_latency_ms": "ERR"})
            continue
        main_rows.append({
            "reranker": kind, "pool": MAIN_POOL,
            "ndcg@5": round(m["ndcg5"], 4), "recall@5": round(m["recall5"], 4),
            "ndcg@5_lift": round(m["ndcg5"] - base_ndcg, 4),
            "p95_latency_ms": round(m["p95_latency_ms"], 1),
        })
        print(f"  {kind:8s} pool={MAIN_POOL}  ndcg@5={m['ndcg5']:.4f}  "
              f"lift={m['ndcg5'] - base_ndcg:+.4f}  recall@5={m['recall5']:.4f}  "
              f"p95={m['p95_latency_ms']:.1f}ms")

    sweep_rows: list[dict] = []
    for kind in SWEEP_KINDS:
        for pool in POOL_SWEEP:
            m = _measure(retriever, lookup, kind, pool, queries)
            if m is None:
                continue
            sweep_rows.append({
                "reranker": kind, "pool": pool,
                "ndcg@5": round(m["ndcg5"], 4), "recall@5": round(m["recall5"], 4),
                "p95_latency_ms": round(m["p95_latency_ms"], 1),
            })
            print(f"  sweep {kind:8s} pool={pool:3d}  ndcg@5={m['ndcg5']:.4f}  "
                  f"p95={m['p95_latency_ms']:.1f}ms")

    _write_csv(main_rows, sweep_rows)
    _write_md(main_rows, sweep_rows, cfg, len(queries), len(retriever.df), base_ndcg)
    print(f"[rerank] wrote {OUT_CSV} and {OUT_MD}")


def _write_csv(main_rows, sweep_rows) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["section", "reranker", "pool", "ndcg@5", "recall@5", "ndcg@5_lift", "p95_latency_ms"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in main_rows:
            w.writerow({"section": "main", **r})
        for r in sweep_rows:
            w.writerow({"section": "pool_sweep", **r})


def _md_table(rows, cols) -> str:
    head = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    return head + sep + body


def _write_md(main_rows, sweep_rows, cfg, n_q, n_chunks, base_ndcg) -> None:
    lines = []
    lines.append("# D3 Task 2 — Reranker comparison (SciFact 60-query holdout)\n\n")
    lines.append(f"Generated by `scripts/eval_rerank.py`. {n_q} SciFact holdout queries over "
                 f"{n_chunks} SciFact chunks. Candidate pool from the blessed D2 retriever "
                 f"(`metric={cfg.metric}`, `hybrid_weight={cfg.hybrid_weight:.3f}`, "
                 f"`candidate_k={cfg.candidate_k}`); each reranker reorders the pool down to "
                 f"top-{TOP_K}. `p95_latency_ms` is end-to-end (retrieval + rerank), so the "
                 f"`none` row is pure retrieval and the extra latency elsewhere is the rerank cost.\n\n")
    lines.append("**Backends:** `none` (identity baseline) · `minilm` "
                 "(cross-encoder/ms-marco-MiniLM-L-6-v2) · `bge` (BAAI/bge-reranker-base) · "
                 "`mmr` (Maximal Marginal Relevance over bge-small-en embeddings).\n\n")
    lines.append(f"## Main comparison (pool = {MAIN_POOL})\n\n")
    lines.append("`ndcg@5_lift` is vs the `none` baseline "
                 f"(NDCG@5 = {round(base_ndcg, 4)}).\n\n")
    lines.append(_md_table(main_rows,
                 ["reranker", "pool", "ndcg@5", "recall@5", "ndcg@5_lift", "p95_latency_ms"]))
    lines.append("\n## Candidate-pool sweep\n\n")
    lines.append("Does a deeper pool give the cross-encoder more to work with, and at what "
                 "latency cost?\n\n")
    lines.append(_md_table(sweep_rows, ["reranker", "pool", "ndcg@5", "recall@5", "p95_latency_ms"]))
    lines.append("\n## Reading the results / how to pick a winner\n\n")
    lines.append("- **Quality:** highest `ndcg@5` (and its `lift` over `none`).\n")
    lines.append("- **Latency:** `p95_latency_ms` — the cross-encoders add a transformer pass "
                 "per candidate, so they cost more than `none`/`mmr`.\n")
    lines.append("- **Trade-off:** prefer the backend with the best lift-per-millisecond unless "
                 "a higher-latency backend clears a quality bar we care about.\n")
    lines.append("- Note: SciFact claims usually have a single relevant chunk, so `mmr`'s "
                 "diversity term has little to reward and may not beat a pure cross-encoder here.\n\n")
    lines.append("**Winner: _TBD — chosen by the team after review._** "
                 "Once picked, set `_BLESSED_RERANKER` in `src/csai415/rerank.py` so `/ask` "
                 "uses it by default.\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
