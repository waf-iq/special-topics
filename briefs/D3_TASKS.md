# D3 — Team Tasks (GraphRAG Executor, Evaluation & Safety)

D3 = Week 9, **15% of project**. Rubric: **GraphRAG pipeline 8%** · **Evaluation 5%** · **Safety 2%**. Full rubric in `CSAI415_Project_Brief.pdf`.

## How we work in D3 (read first)

**One big self-contained task per member. When you finish, you are done — no waves, no going back, no waiting on a teammate.**

This is possible because **H0 is already done** (scaffolded + verified in the repo): every interface the pipeline needs is frozen, and every stage has a working stub. So you build your slice against:
- the **frozen H0 contracts** (`graphrag.GraphRAGExecutor`, `Citation`/`AnswerResult`, `eval.evaluate_answers`, the `/ask` shape, the seam signatures), and
- the **live D2 services** (`POST /search`, the seeded Neo4j graph, `chunks.parquet`),

never against another member's unfinished code. You own your files outright (no shared-file edits), you produce your own evidence, you commit, you're done. One **integrator** (Ahmad Fraij, Task 7) does the single final pass that wires the six finished pieces together and writes the report — that's the only aggregation point, and it consumes already-committed work.

Your job is still to **dig in and compare approaches** (the menu under each task), measure, and defend the winner — every task ships a small comparison table. That depth is what the rubric + AI-log policy reward.

## Team decisions (locked — do NOT relitigate with your AI)

| Decision | Choice | Why |
|---|---|---|
| **Answer generator** | Local SLM **`Qwen2.5-3B-Instruct`** (D4's QLoRA target) via Ollama/transformers | D4 tunes a 1–3B model and compares zero-shot vs tuned *inside* GraphRAG. |
| **Quality-ceiling row (optional)** | **Groq `llama-3.3-70b-versatile`** — FREE tier | A strong-model reference row, zero cost. Not the graded answerer. |
| **Faithfulness/relevance judge** | **Groq `llama-3.3-70b-versatile`** via `langchain-groq` (FREE) | RAGAS needs an LLM judge; Groq's free tier covers our ~30–50-question volume. **Not** xAI Grok (paid since May 2025). |
| **Eval library** | **RAGAS** (faithfulness + answer-relevance) + our own latency p95 | Brief: "RAGAS or equivalent". |
| **GraphRAG corpus** | **arXiv cs.CL** (graph + page ranges for citations) | Only corpus with both a graph and page numbers. |
| **Retrieval-sanity corpus** | **SciFact** (carried from D2) | Keeps Recall@5 continuity; not used for GraphRAG/answering. |
| **Reranker** | **Included** (cross-encoder) | Doctor's Week 07 §5; rubric says "(re)ranking". |
| **Multi-objective HPO** | **Yes** — doctor's NSGA-II Pareto over GraphRAG knobs | Mirrors Week 07; lifts the Evaluation score. |
| **Agent layer** | **Fixed deterministic executor** + thin tool wrapper (`vector_search`, `cypher_query`, `read_pdf_page_range`) | Rubric says "executor". Tools exist so the deny-risky-tool-call demo has something to guard. No ReAct/LangGraph (D4 scope). |
| **Safety mitigations** | **3**: prompt-injection defense, source pinning + provenance filtering, deny risky tool calls | Brief names prompt-injection & retrieval-poisoning; all give clean before/after. |
| **Qdrant layout** | **1 collection + source-filter-at-query-time** (refactor D2's 3 collections) | Deferred from D2; the executor wants one retriever. |

**Models config (locked):** Groq is OpenAI-compatible — `GROQ_API_KEY`, base `https://api.groq.com/openai/v1`, model `llama-3.3-70b-versatile`. Free-tier ≈ 30 req/min, 1K req/day, 12K tokens/min → add retry/backoff in batch loops. Local SLM via `ollama pull qwen2.5:3b-instruct` (or transformers `Qwen/Qwen2.5-3B-Instruct`). Deps `ragas`, `langchain-groq`, `groq` are in `requirements.txt` (not imported by the scaffold).

## H0 contracts (DONE — committed + verified, build against these)

| Surface | Where | Frozen |
|---|---|---|
| Executor | `src/csai415/graphrag.py` | `GraphRAGExecutor.answer(query, k, mode, rerank) -> AnswerResult`; `Citation`, `AnswerResult` |
| Subgraph seam | `src/csai415/graph_select.py` | `SubgraphResult`, `select_subgraph(query, neo4j_driver=…)` |
| Rerank seam | `src/csai415/rerank.py` | `rerank(query, candidate_ids, chunk_text, top_n)` |
| Answer seam | `src/csai415/answer.py` | `generate_answer(query, citations, contexts) -> (text, used_idxs)` (`citations[i]`↔`contexts[i]`) |
| Eval | `src/csai415/eval.py` | `evaluate_answers(answer_fn, gold) -> {faithfulness, answer_relevance, recall5, p95_latency_ms, n}` |
| Tools/guard | `src/csai415/tools.py` | `is_write_cypher`, `RiskyToolCallDenied`, `cypher_query` guard |
| API | `src/csai415/api.py` | `POST /ask` + `AskRequest`/`AskResponse`/`CitationModel` |
| Gold schema | `data/gold/qa_answers.example.jsonl` | the v2 row shape |

Contract smoke: `tests/test_graphrag_contract.py` (3 tests, fake retriever — no model/Docker). Keep it green.

---

## Assignments — 7 self-contained tasks

Each task lists: **the files you own** (no one else edits them), **what you build against** (so you never wait), **the question you must answer by comparing approaches**, and **done when** (finish once).

### Task 1 — Graph-guided retrieval  ·  *GraphRAG 8%*  ·  **Owner: Abdurlahman Alali**
You own everything "graph": link the query to graph nodes, select the subgraph, and decide how the subgraph reshapes retrieval.
- **Files you own:** `src/csai415/graph_select.py` (fill the `select_subgraph` stub + add a `graph_guide(candidates, subgraph, mode, lookup)` policy function).
- **Build against:** the **live seeded Neo4j** (D2) + the existing `HybridRetriever`. The executor already calls your seam — no other member's code involved.
- **Compare:** entity linking (LLM-NER vs spaCy vs fuzzy match against node names); which of the 5 D2 Cypher templates to fire; and the guidance policy — graph-as-**filter** vs graph-as-**booster** vs graph-as-**expansion**.
- **Done when:** `select_subgraph` returns real `paper_ids`/`cypher_used` against the live graph; a table of (linking method × precision × latency) and (guidance policy × Recall@5/NDCG vs vector-only); graceful fallback when no node matches. The contract test still passes.
- **Starter Qs for your AI:** Is LLM entity extraction overkill at ~150 papers/760 authors? · Route to the right Cypher without an LLM — how? · graph-as-filter can starve recall — how do I tune the fallback so a bad subgraph never beats pure vector?

### Task 2 — Reranking  ·  *GraphRAG 8%*  ·  **Owner: Ahmed Soliman**
Is a cross-encoder reranker worth its latency?
- **Files you own:** `src/csai415/rerank.py` (fill the identity stub).
- **Build against:** candidates from the **live `/search`** (D2) or any chunk-text dict — fully standalone.
- **Compare:** `cross-encoder/ms-marco-MiniLM-L-6-v2` vs `BAAI/bge-reranker-base` vs MMR-diversity vs none; candidate-pool sweep (rerank top-{20,50,100}→5).
- **Done when:** `rerank()` reorders by a real model; a table of (option × NDCG@5 lift × p95 latency added); a stated verdict. Contract test green.
- **Starter Qs:** Will ms-marco underperform bge-reranker on cs.CL abstracts — how to test cleanly? · What candidate_k gives room to re-order without blowing 2s p95? · Is the forward pass the latency bottleneck on CPU?

### Task 3 — Answer generation + citations + page ranges  ·  *GraphRAG 8%*  ·  **Owner: WAFIQ**
Turn ranked chunks into a grounded answer with inline `[n]` citations → `{title, page_range}`.
- **Files you own:** `src/csai415/answer.py` (replace the extractive stub).
- **Build against:** `(citations, contexts)` from the **live `/search`** or a fixture list — standalone.
- **Compare:** the answerer (local `Qwen2.5-3B-Instruct` graded vs Groq `llama-3.3-70b` ceiling row); numbered-source prompting vs post-hoc citation attribution; extractive vs abstractive; the "context insufficient → refuse" path.
- **Done when:** `generate_answer(query, citations, contexts)` returns a cited answer with correct page ranges; a side-by-side of Qwen-3B vs Groq-70B on 5 example questions. Contract test green.
- **Starter Qs:** What prompt/max-tokens keeps a 3B model under a few seconds while still citing? · numbered-source vs post-hoc — which is more faithful on a small model, and how do I measure it? · How do I make it say "context doesn't answer this" instead of hallucinating?

### Task 4 — Evaluation harness + gold set  ·  *Eval 5%*  ·  **Owner: Yousef Alsakkaf**
Build the gold Q/A set and the RAGAS-via-Groq harness; demonstrate it.
- **Files you own:** the `evaluate_answers` body in `src/csai415/eval.py`, `data/gold/qa_answers.jsonl` (the real ~30–50 arXiv Q/A), `scripts/build_gold_answers.py`, `src/csai415/ragas_groq.py` (judge wiring).
- **Build against:** **any `answer_fn`** — demo on the current (stub/extractive) answerer that exists today; the harness is answerer-agnostic, so you never wait on Task 3.
- **Compare:** gold-set construction (hand-author vs LLM-gen + verify, doctor's `eval_set.json` pattern with `source_chunk_text`); RAGAS-with-Groq-judge vs a hand-rolled LLM-judge; sanity-check RAGAS faithfulness against a few hand labels.
- **Done when:** `evaluate_answers` returns **real** faithfulness/answer-relevance via Groq (replacing the H0 proxy); gold set committed; a demo run table; retry/backoff for the 30-rpm cap. Contract test green.
- **Starter Qs:** What exactly do I pass as `contexts` to RAGAS, and does chunk granularity bias the score? · ~40 questions is noisy — bootstrap a CI? · How do I cache judge responses so the eval is reproducible?

### Task 5 — Ablation + multi-objective HPO  ·  *Eval 5%*  ·  **Owner: Yehia Noureldin**
The required vector-vs-graph-vs-hybrid ablation + the doctor's Pareto method, as a reusable runner.
- **Files you own:** `src/csai415/graphrag_hpo.py`, `notebooks/04_graphrag_ablation.ipynb`.
- **Build against:** the **frozen executor + `evaluate_answers` contract** (both exist at H0) — your deliverable is the *runner*, so you build now and never wait. The coordinator runs your finished runner on the fully-wired pipeline for the report's final numbers.
- **Compare/produce:** the 3-way ablation × {rerank on/off} × {Qwen-3B, Groq-70B}; NSGA-II Pareto of (faithfulness or NDCG@5, latency) over `mode`/`hybrid_weight`/`candidate_k`/`rerank`/expansion-depth; fastest-acceptable / balanced / best-quality knee points (doctor's §7).
- **Done when:** a runnable ablation table + Pareto plot + a recommended config with justification, demonstrated on the current pipeline.
- **Starter Qs:** Faithfulness or NDCG@5 as the quality axis? · How do I stop the Pareto front overfitting to ~40 questions? · Precompute retrieval so each trial doesn't re-call the 3B model?

### Task 6 — Safety: 3 mitigations + before/after  ·  *Safety 2%*  ·  **Owner: Musab**
- **Files you own:** `src/csai415/safety.py`, `scripts/safety_eval.py`, `reports/D3/d3_safety.md`.
- **Build against:** the **tool guard (exists at H0)** + the current pipeline — your mitigations are wrappers/filters; you never wait on others.
- **Build:** (a) **prompt-injection defense** — plant `"ignore previous instructions…"`, show it obeys, add delimiting/instruction-hierarchy/sanitization → defended; (b) **source pinning + provenance filtering** — plant a poisoned/low-trust chunk, show corruption, filter by provenance/trust → defended; (c) **deny risky tool calls** — extend the `cypher_query` read-only guard, show a write/`DELETE` blocked. Compare ≥2 implementations per mitigation.
- **Done when:** an attack set (~10 cases) + a before/after table per mitigation (attack success ↓, benign answers unchanged) + a documented-limits paragraph.
- **Starter Qs:** Strongest *cheap* prompt-injection defense for a small model — how does each fail? · provenance allowlist vs trust-score — cleaner before/after without nuking recall? · How do I prove the mitigation didn't hurt benign answers (control set)?

### Task 7 — Integration + Qdrant refactor + `/ask` + report  ·  *spans all*  ·  **Owner: Ahmad Fraij** (integrator)
The single aggregation point — consumes the six finished pieces. Not member-to-member back-and-forth; one final pass.
- **Files you own:** the `graphrag.py` executor *shell* (graph branch delegates to Task 1's `graph_guide`), `src/csai415/qdrant_dense.py` + `src/csai415/api.py` (the 1-collection refactor + `/ask` wiring + Neo4j driver injection), `reports/D3/D3_report.md`, `notebooks/`-level glue.
- **Do:** collapse the 3 Qdrant collections → 1 + source-filter-at-query-time; wire the real seams into the executor (mechanical — contracts match); run Task 4's harness, Task 5's runner, and Task 6's safety eval on the **fully-wired** pipeline to get the report's headline numbers; write the 2-page report.
- **Done when:** `/ask` live on the single-collection stack; the report covers all three rubric areas with real numbers; `tests/test_graphrag_contract.py` + the D2 suite stay green.

---

## Dependency map (why there are no waves)

```
Live D2 services (/search, Neo4j, parquet)  +  frozen H0 contracts (in repo)
        │
        ├── Task 1  graph-guided retrieval        ┐
        ├── Task 2  reranking                     │  six independent slices —
        ├── Task 3  answer + citations            │  each builds against contracts/live
        ├── Task 4  eval harness + gold set        │  services + stubs, finishes once,
        ├── Task 5  ablation + HPO runner         │  no member-to-member dependency
        └── Task 6  safety mitigations            ┘
                          │
                          ▼
        Task 7 (integrator — Ahmad Fraij): one final integration pass — wire the six,
        run their finished harnesses on the real pipeline, write the report.
```
No member edits another member's files. No member runs in two waves. The only thing that needs everything finished is the **report**, and that's the coordinator's single task.

## Definition of done (D3 rubric checklist)

- [ ] **T1** `select_subgraph` real against live Neo4j + graph-guidance policy chosen with evidence.
- [ ] **T2** cross-encoder rerank chosen with a lift-vs-latency table.
- [ ] **T3** grounded answers with `[n]` citations + correct page ranges; Qwen-3B vs Groq-70B side-by-side.
- [ ] **T4** real RAGAS faithfulness + answer-relevance via Groq; `qa_answers.jsonl` committed (~30–50). Targets ≥ 0.8 / ≥ 0.8.
- [ ] **T5** vector vs graph vs hybrid (× rerank) ablation table + Pareto plot + recommended config.
- [ ] **T6** 3 mitigations with before/after evidence + documented limits.
- [ ] **T7** 1 Qdrant collection + `/ask` live; `reports/D3/D3_report.md` (→ PDF) covering all rubric areas with real numbers; suites green.
- [ ] Each member: ≥2 commits under their own identity; populated `ai_logs/D3/<name>.md` (share-link + the approaches they compared).
