"""D3 Task-1 evaluation harness — graph-guided retrieval comparison tables.

Produces the two deliverable tables (briefs/D3_D4_TASKS.md, Task 1):

  1. Entity-linking comparison  — fuzzy vs spaCy vs LLM-NER: precision / recall / F1
     over a labelled probe set, false-positive rate on no-entity queries, and latency.
  2. Guidance-policy comparison — vector-only vs hybrid vs graph-{filter,booster,expansion}:
     Hit@5 / Hit@10 / subgraph-precision@5 / latency, on entity-anchored content questions.

Both probe sets are derived **programmatically from the live graph + chunks.parquet** (seeded,
reproducible) because the hand-curated arXiv gold (data/gold/qa_answers.jsonl) is Task 4's
deliverable and not yet committed. The gold labels here are reliable by construction: a query
is built around a real Author/Topic/Paper node, so the correct link and the relevant paper
are known.

Run (Neo4j up + seeded, see check_graph.py):
    .venv/bin/python scripts/eval_graph_select.py
Writes reports/D3/graph_linking_comparison.{md,csv} and graph_guidance_comparison.{md,csv}.

Notes on methodology
--------------------
* Graph guidance only reshapes retrieval when the query names a graph entity; on a no-entity
  query select_subgraph returns fallback and every policy is a no-op (== vector-only). So the
  guidance probes are entity-anchored by design — that is the regime where graph helps, and
  the comparison quantifies *how* each policy trades precision for recall there.
* LLM-NER rows are emitted only if CSAI415_ANSWERER resolves to a reachable backend; otherwise
  the row is marked n/a (no backend) so the table is honest on a laptop with no GPU/key.
"""

from __future__ import annotations

import os
import random
import statistics as st
import time
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from csai415.api import load_blessed_config
from csai415.graph_select import (
    GraphIndex,
    apply_guidance,
    get_index,
    link_entities,
    select_subgraph,
)
from csai415.retrieve import HybridRetriever, load_chunks

SEED = 42
NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OUT = Path("reports/D3")
CANDIDATE_K = 30  # pool size pulled before guidance reshapes it

# Topic code → a natural phrase a user would type (drives topic-anchored probes).
TOPIC_PHRASE = {
    "cs.CL": "natural language processing",
    "cs.CV": "computer vision",
    "cs.LG": "machine learning",
    "cs.IR": "information retrieval",
    "cs.CR": "security",
    "cs.SE": "software engineering",
    "cs.AI": "artificial intelligence",
}


# --- probe construction ----------------------------------------------------------------
def _title_keyphrase(title: str) -> str:
    """First chunk of a title before punctuation — a distinctive content phrase."""
    head = title.split(":")[0].split("?")[0].split("—")[0].strip()
    return " ".join(head.split()[:8])


# Held-out topic paraphrases — deliberately NOT in graph_select.TOPIC_SYNONYMS, so they
# test whether topic linking GENERALIZES beyond its hard-coded table (it does not; this
# exposes the synonym-table-bound limitation rather than inflating recall with phrases the
# linker is wired to recognize — see the methodology note in the report).
HELDOUT_TOPIC_PARAPHRASE = {
    "cs.CL": "computational semantics",
    "cs.CV": "scene understanding",
    "cs.LG": "statistical learning theory",
    "cs.IR": "document ranking",
    "cs.CR": "threat modeling",
}

# Plausible but ABSENT author names (famous, not in this 2026 arXiv graph). Gold = no link;
# a linker that maps these to a similarly-spelled real node is making a precision error.
ABSENT_AUTHORS = ["Geoffrey Hinton", "Yoshua Bengio", "Fei-Fei Li", "Christopher Manning"]


def build_linking_probes(driver, index: GraphIndex, df: pd.DataFrame, rng: random.Random):
    """Labelled query → expected (authors, topics).

    Linkable probes (count toward precision/recall/F1): author plain/prose/typo-1/typo-3,
    in-vocabulary topics, held-out topic paraphrases, and author+topic. Non-linkable probes
    (count toward fp_rate only, gold empty): generic questions + plausible absent authors.
    Held-out paraphrases and absent authors are the honest stress cases the first probe set
    lacked — they keep fuzzy's score from being a self-fulfilling 1.0.
    """
    probes: list[dict] = []

    with driver.session() as s:
        authors = [r["n"] for r in s.run(
            "MATCH (a:Author)-[:WROTE]->(p) WITH a.name AS n, count(p) AS c "
            "WHERE size(split(n,' ')) >= 2 RETURN n ORDER BY c DESC, n LIMIT 60")]
    rng.shuffle(authors)
    sample_authors = authors[:12]

    def _drop_chars(name: str, k: int) -> str:
        sur = name.split()[-1]
        return name.replace(sur, sur[:-k]) if len(sur) > k + 2 else name

    for i, a in enumerate(sample_authors):
        if i % 4 == 0:
            q = f"What has {a} published?"
        elif i % 4 == 1:
            q = f"Tell me about the research contributions of {a}."
        elif i % 4 == 2:  # mild typo (drop 1 char) — should still link
            q = f"papers written by {_drop_chars(a, 1)}"
        else:  # harder typo (drop 3 chars) — stresses the fuzzy threshold
            q = f"papers written by {_drop_chars(a, 3)}"
        probes.append({"query": q, "authors": [a], "topics": [], "kind": "author"})

    # In-vocabulary topics (legitimate: users do type these).
    for code, phrase in TOPIC_PHRASE.items():
        if code in index.topics:
            probes.append({"query": f"recent advances in {phrase}", "authors": [], "topics": [code],
                           "kind": "topic"})

    # Held-out topic paraphrases (generalization test; expected to mostly miss).
    for code, phrase in HELDOUT_TOPIC_PARAPHRASE.items():
        if code in index.topics:
            probes.append({"query": f"recent advances in {phrase}", "authors": [], "topics": [code],
                           "kind": "topic-heldout"})

    # Author + topic.
    for a in sample_authors[:4]:
        code = rng.choice([c for c in TOPIC_PHRASE if c in index.topics])
        probes.append({"query": f"{a}'s work on {TOPIC_PHRASE[code]}",
                       "authors": [a], "topics": [code], "kind": "author+topic"})

    # Negatives — plausible absent authors (precision) + generic no-entity questions.
    for a in ABSENT_AUTHORS:
        probes.append({"query": f"What has {a} published?", "authors": [], "topics": [],
                       "kind": "negative"})
    for q in ["How does attention scale with sequence length?",
              "What is the capital of France?",
              "Explain gradient descent in simple terms.",
              "Summarize the main idea of this paragraph."]:
        probes.append({"query": q, "authors": [], "topics": [], "kind": "negative"})

    return probes


def build_guidance_probes(driver, df: pd.DataFrame, rng: random.Random):
    """Entity-anchored content questions → (target_paper_id, relevant_chunk_ids).

    Two regimes, on purpose, so the table shows BOTH faces of graph guidance:

    * ``specified`` — the query carries the paper's title keyphrase, so dense retrieval
      already surfaces the paper (hit@5 near-ceiling). Here graph's value is *precision*:
      filter/booster concentrate the top-k on the relevant subgraph.
    * ``underspecified`` — the query names only the entity ("What does {author} propose?"),
      so dense retrieval has little to lock onto and often misses the paper. Here graph's
      value is *recall*: expansion injects the subgraph's chunks the vector pool missed.

    Honesty note (re: expansion's cand_recall): for solo-author underspecified probes the
    target paper IS the linked subgraph, so a correct link + expansion injects it *by
    construction* — cand_recall→1.0 demonstrates the MECHANISM and its ceiling (it is bounded
    by link precision, not a free recovery of arbitrary missed chunks). The non-tautological
    counterpart is subgraph_concentration@5: expansion's stays at the vector baseline (the
    injected chunks land at the tail, not the top-5), i.e. expansion adds candidate recall
    *without* polluting top-k precision. That trade-off is the real, reportable result.

    Underspecified author probes use single-paper authors so the target is unambiguous.
    Topic-anchored probes span minority topics AND a few cs.CL probes (the dominant topic,
    where filtering barely narrows the field — included so the table shows filter's limit).
    """
    ar = df[df["source"].isin(["arxiv", "arxiv-demo"])].copy()
    chunks_by_paper = ar.groupby("paper_id")["chunk_id"].apply(list).to_dict()
    probes: list[dict] = []

    with driver.session() as s:
        rows = s.run(
            "MATCH (a:Author)-[:WROTE]->(p:Paper) "
            "WITH p, collect(a.name) AS names RETURN p.paper_id AS pid, names[0] AS author, "
            "p.title AS title LIMIT 200").data()
        # authors with exactly one paper → unambiguous target for underspecified queries
        solo = {r["n"]: r["pid"] for r in s.run(
            "MATCH (a:Author)-[:WROTE]->(p) WITH a.name AS n, count(p) AS c, collect(p.paper_id)[0] AS pid "
            "WHERE c = 1 AND size(split(n,' ')) >= 2 RETURN n, pid")}
    rng.shuffle(rows)

    # Author-anchored, SPECIFIED (title keyphrase present).
    n_spec = 0
    for r in rows:
        if r["pid"] not in chunks_by_paper or not r["title"]:
            continue
        kp = _title_keyphrase(r["title"])
        if len(kp.split()) < 3:
            continue
        probes.append({
            "query": f"What does {r['author']} propose about {kp}?",
            "target_paper": r["pid"], "relevant": chunks_by_paper[r["pid"]],
            "kind": "author", "regime": "specified",
        })
        n_spec += 1
        if n_spec >= 10:
            break

    # Author-anchored, UNDERSPECIFIED (entity only, no keyphrase). Up to 20 for less noise.
    solo_authors = list(solo.items())
    rng.shuffle(solo_authors)
    for author, pid in solo_authors[:20]:
        if pid in chunks_by_paper:
            probes.append({
                "query": f"What does {author} propose in their recent work?",
                "target_paper": pid, "relevant": chunks_by_paper[pid],
                "kind": "author", "regime": "underspecified",
            })

    # Topic-anchored, SPECIFIED — minority topics (filtering away cs.CL helps) + a few cs.CL
    # probes (the dominant topic, where the filter barely narrows: 113/155 papers).
    def _add_topic_probes(codes, cap):
        with driver.session() as s:
            trows = s.run(
                "MATCH (p:Paper)-[:ABOUT]->(t:Topic) WHERE t.name IN $codes "
                "RETURN p.paper_id AS pid, p.title AS title, t.name AS code", codes=codes).data()
        rng.shuffle(trows)
        n = 0
        for r in trows:
            if r["pid"] not in chunks_by_paper or not r["title"]:
                continue
            kp = _title_keyphrase(r["title"])
            if len(kp.split()) < 3:
                continue
            probes.append({
                "query": f"In {TOPIC_PHRASE.get(r['code'], r['code'])} research, {kp}?",
                "target_paper": r["pid"], "relevant": chunks_by_paper[r["pid"]],
                "kind": "topic", "regime": "specified", "topic": r["code"],
            })
            n += 1
            if n >= cap:
                break

    _add_topic_probes(["cs.CV", "cs.LG", "cs.CR", "cs.IR", "cs.SE", "cs.AI"], 12)
    _add_topic_probes(["cs.CL"], 4)  # dominant-topic control: filter can't narrow much here

    return probes, chunks_by_paper


# --- metric helpers --------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _backend_ok() -> bool:
    try:
        from csai415.answer import resolve_backend
        b = resolve_backend()
    except Exception:
        return False
    if b is None:
        return False
    client, model, _ = b
    try:  # one cheap reachability probe
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=1)
        return True
    except Exception:
        return False


# --- table 1: linking ------------------------------------------------------------------
def eval_linking(driver, index, probes, methods):
    rows = []
    n_neg = sum(1 for p in probes if p["kind"] == "negative")
    n_heldout = sum(1 for p in probes if p["kind"] == "topic-heldout")
    for method in methods:
        tp = fp = fn = 0
        fp_neg = 0       # any link emitted on a negative query (precision on absent/no entity)
        heldout_hit = 0  # held-out topic paraphrases correctly linked (generalization)
        lats = []
        for p in probes:
            gold = {("Author", a) for a in p["authors"]} | {("Topic", t) for t in p["topics"]}
            t0 = time.perf_counter()
            try:
                linked = link_entities(p["query"], index, method)
            except Exception:
                linked = []
            lats.append((time.perf_counter() - t0) * 1000.0)
            got = {(e.label, e.name) for e in linked}
            tp += len(gold & got)
            fp += len(got - gold)
            fn += len(gold - got)
            if p["kind"] == "negative" and got:
                fp_neg += 1
            if p["kind"] == "topic-heldout" and (gold & got):
                heldout_hit += 1
        prec, rec, f1 = _prf(tp, fp, fn)
        rows.append({
            "method": method, "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "fp_rate_neg": round(fp_neg / n_neg, 3) if n_neg else 0.0,
            "heldout_topic_recall": round(heldout_hit / n_heldout, 3) if n_heldout else 0.0,
            "median_ms": round(st.median(lats), 1),
            "p95_ms": round(sorted(lats)[max(0, int(0.95 * len(lats)) - 1)], 1),
        })
    return pd.DataFrame(rows)


# --- table 2: guidance -----------------------------------------------------------------
def _topk_ids(pool, k):
    return [cid for cid, _ in pool][:k]


MODES = ["vector", "hybrid", "filter", "booster", "expansion"]


def _score_probe(p, driver, retriever, paper_of, chunks_by_paper):
    """Per-probe per-mode raw metrics. Returns {mode: {hit5,hit10,cand_recall,subprec5,lat}}."""
    relevant = set(p["relevant"])
    cfg = retriever.config
    sub = select_subgraph(p["query"], neo4j_driver=driver, method="fuzzy")
    wanted = set(sub.paper_ids)

    pool_vec = retriever.search_with_scores(p["query"], CANDIDATE_K, hybrid_weight=1.0)
    pool_hyb = retriever.search_with_scores(p["query"], CANDIDATE_K, hybrid_weight=cfg.hybrid_weight)
    ids_hyb = [cid for cid, _ in pool_hyb]
    scores_hyb = {cid: s for cid, s in pool_hyb}

    out = {}
    for mode in MODES:
        t0 = time.perf_counter()
        if mode == "vector":
            ranked = [cid for cid, _ in pool_vec]
        elif mode == "hybrid":
            ranked = ids_hyb
        else:
            ranked = apply_guidance(ids_hyb, sub, paper_of=paper_of, policy=mode,
                                    scores=scores_hyb, chunks_by_paper=chunks_by_paper, max_expand=20)
        lat = (time.perf_counter() - t0) * 1000.0
        top5 = ranked[:5]
        in_sub = sum(1 for c in top5 if paper_of(c) in wanted) if wanted else 0
        out[mode] = {
            "hit5": 1.0 if relevant & set(top5) else 0.0,
            "hit10": 1.0 if relevant & set(ranked[:10]) else 0.0,
            "cand_recall": 1.0 if relevant & set(ranked) else 0.0,  # target chunk anywhere in the reshaped list
            "subprec5": in_sub / max(1, len(top5)),
            "lat": lat,
        }
    return out


def _summarize(per_probe, modes=MODES):
    """Mean each metric per mode over a list of per-probe dicts → a DataFrame."""
    rows, base = [], None
    for mode in modes:
        vals = [pp[mode] for pp in per_probe]
        hit5 = round(st.mean(v["hit5"] for v in vals), 3)
        if mode == "vector":
            base = hit5
        rows.append({
            "policy": mode, "hit@5": hit5,
            "hit@10": round(st.mean(v["hit10"] for v in vals), 3),
            "cand_recall": round(st.mean(v["cand_recall"] for v in vals), 3),
            # concentration, NOT relevance: fraction of the top-5 whose paper is in the
            # selected subgraph (how aggressively a policy focuses on the subgraph).
            "subgraph_conc@5": round(st.mean(v["subprec5"] for v in vals), 3),
            "Δhit@5_vs_vector": round(hit5 - base, 3),
            "guidance_ms_med": round(st.median(v["lat"] for v in vals), 3),
        })
    return pd.DataFrame(rows)


def eval_guidance(driver, retriever, df, probes, chunks_by_paper):
    """Returns (overall_df, by_regime: {regime: df})."""
    paper_of = df.set_index("chunk_id")["paper_id"].to_dict().get
    per_probe, regimes = [], {}
    for p in probes:
        scored = _score_probe(p, driver, retriever, paper_of, chunks_by_paper)
        per_probe.append(scored)
        regimes.setdefault(p["regime"], []).append(scored)
    overall = _summarize(per_probe)
    by_regime = {reg: _summarize(pp) for reg, pp in regimes.items()}
    return overall, by_regime


# --- main ------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    auth = (NEO4J_USER, NEO4J_PASSWORD) if NEO4J_PASSWORD else None
    driver = GraphDatabase.driver(NEO4J_URL, auth=auth)
    driver.verify_connectivity()
    index = get_index(driver)
    print(f"[eval] graph: {len(index.authors)} authors, {len(index.topics)} topics")

    df = load_chunks()
    print(f"[eval] corpus: {len(df)} chunks; building retriever (BGE + BM25)…")
    retriever = HybridRetriever(df, load_blessed_config())

    # --- Table 1: linking ---
    methods = ["fuzzy", "spacy"]
    if _backend_ok():
        methods.append("llm")
        print("[eval] LLM backend reachable → including llm row")
    else:
        print("[eval] no LLM backend → llm row marked n/a")
    lprobes = build_linking_probes(driver, index, df, rng)
    print(f"[eval] linking probes: {len(lprobes)} "
          f"({sum(p['kind']=='negative' for p in lprobes)} negatives, "
          f"{sum(p['kind']=='topic-heldout' for p in lprobes)} held-out topic paraphrases)")
    ldf = eval_linking(driver, index, lprobes, methods)
    if "llm" not in methods:
        ldf = pd.concat([ldf, pd.DataFrame([{
            "method": "llm", "precision": "n/a", "recall": "n/a", "f1": "n/a",
            "fp_rate_neg": "n/a", "heldout_topic_recall": "n/a", "median_ms": "n/a",
            "p95_ms": "n/a (no backend)"}])], ignore_index=True)
    print("\n=== Entity-linking comparison ===")
    print(ldf.to_string(index=False))

    # --- Table 2: guidance ---
    gprobes, chunks_by_paper = build_guidance_probes(driver, df, rng)
    n_spec = sum(p["regime"] == "specified" for p in gprobes)
    n_under = sum(p["regime"] == "underspecified" for p in gprobes)
    print(f"\n[eval] guidance probes: {len(gprobes)} ({n_spec} specified / {n_under} underspecified)")
    gdf, by_regime = eval_guidance(driver, retriever, df, gprobes, chunks_by_paper)
    print("\n=== Guidance-policy comparison — OVERALL (vs vector-only) ===")
    print(gdf.to_string(index=False))
    for reg, rdf in by_regime.items():
        print(f"\n--- regime: {reg} ---")
        print(rdf.to_string(index=False))

    # --- write artifacts ---
    ldf.to_csv(OUT / "graph_linking_comparison.csv", index=False)
    gdf.to_csv(OUT / "graph_guidance_comparison.csv", index=False)
    (OUT / "graph_linking_comparison.md").write_text(
        "# D3 Task-1 — Entity-linking comparison\n\n"
        f"Probe set: {len(lprobes)} labelled queries derived from the live graph (seed={SEED}); "
        "precision/recall/F1 over (label, node) pairs on the *linkable* probes (author "
        "plain/prose/typo-1/typo-3, in-vocabulary topics, held-out topic paraphrases, "
        "author+topic). Diagnostics: **fp_rate_neg** = fraction of non-linkable queries "
        "(generic questions + plausible *absent* author names) that wrongly linked something; "
        "**heldout_topic_recall** = recall on topic paraphrases deliberately NOT in the "
        "synonym table.\n\n"
        + ldf.to_markdown(index=False) + "\n\n"
        "**Reading it.** Fuzzy leads on precision, recall and latency; spaCy trails because "
        "PERSON NER misfires on typo'd/unusual author spans. The low `heldout_topic_recall` "
        "for *both* offline linkers is the honest limitation: topic linking is bounded by the "
        "hand-built synonym table, so a paraphrase it has never seen does not link. "
        "Embedding- or LLM-based topic linking is the future-work fix.\n")

    regime_md = "\n\n".join(
        f"### Regime: {reg}\n\n" + rdf.to_markdown(index=False) for reg, rdf in by_regime.items())
    (OUT / "graph_guidance_comparison.md").write_text(
        "# D3 Task-1 — Guidance-policy comparison\n\n"
        f"Probe set: {len(gprobes)} entity-anchored content questions "
        f"({n_spec} specified, {n_under} underspecified; seed={SEED}); candidate pool "
        f"k={CANDIDATE_K}. **Hit@k** = a chunk of the target paper reached the top-k; "
        "**cand_recall** = a target chunk is anywhere in the reshaped candidate list (the "
        "signal a downstream cross-encoder reranker sees); **subgraph_conc@5** = fraction of "
        "the top-5 whose paper is in the selected subgraph (concentration, *not* relevance). "
        "Policies apply to the hybrid pool; vector/hybrid carry no graph guidance.\n\n"
        "## Overall\n\n" + gdf.to_markdown(index=False) + "\n\n"
        "## By regime\n\n"
        "*Specified* queries already retrieve well (hit@5 near ceiling) → graph adds "
        "**precision**: filter pushes subgraph_conc@5 to ~1.0 with no recall loss; booster is "
        "the recall-safe middle ground. *Underspecified* queries miss the paper under vector "
        "search → graph adds **recall**: expansion raises cand_recall by injecting subgraph "
        "chunks the vector pool missed.\n\n"
        "**Honesty note on expansion.** For solo-author underspecified probes the target "
        "paper *is* the linked subgraph, so a correct link + expansion injects it by "
        "construction — `cand_recall→1.0` shows the mechanism and its ceiling (bounded by link "
        "precision), not a free recovery of arbitrary chunks. The non-tautological half is "
        "`subgraph_conc@5`: expansion's injected chunks land at the tail, so top-5 precision is "
        "unchanged — expansion buys candidate recall *without* polluting the top-k the reranker "
        "orders. cs.CL (dominant-topic) probes are included to show filter's limit: when the "
        "subgraph is 113/155 papers, filtering barely narrows the field.\n\n"
        + regime_md + "\n")
    print(f"\n[eval] wrote 4 artifacts to {OUT}/")
    driver.close()


if __name__ == "__main__":
    main()
