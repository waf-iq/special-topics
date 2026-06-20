# D3 + D4 — Team Tasks (GraphRAG Executor → SLM Tuning & Final Package)

Combined build, **two graded deliverables (30% total)**:
- **D3** (Week 9, 15%): GraphRAG pipeline **8%** · Evaluation **5%** · Safety **2%**.
- **D4** (Week 10/11, 15%): Tuning & card **5%** · Integration & perf **5%** · Demo & report **5%**.

We combine the *build* because D4's entire infrastructure is D3's backbone — the executor is a swappable-answerer seam and `evaluate_answers` is answerer-agnostic, so "zero-shot vs tuned inside GraphRAG" is just running the same harness twice. **D4 adds no new people; it deepens three verticals** (answerer, data+eval, integration+report) and rolls D3's evidence into D4's bigger report.

## How we work (read first)

**One big self-contained task per member. Finish once — no waves, no going back, no waiting on a teammate.** You build against the **frozen H0 contracts** (in the repo) + the **live D2 services** (`/search`, Neo4j, `chunks.parquet`), never against another member's unfinished code. You own your files outright. One **integrator** (Abdurlahman Alali, Task 7) does the single final pass and the report.

**D3 → D4 phasing (not a wave — an inherent order):** D3 lands first; all 7 build their slices in parallel. The QLoRA tune (D4) comes *after* D3 because it needs the zero-shot baseline, the eval harness, and the training set to all exist. Only **three** verticals have a D4 phase — answerer (tune), data+eval (training set was already curated in D3), integration (final report+demo). The other four finish in D3 and feed the report.

Your job is still to **dig in and compare approaches**, measure, and defend the winner — every task ships a comparison table.

## Team decisions (locked — do NOT relitigate with your AI)

| Decision | Choice |
|---|---|
| **Answer generator (graded)** | Local SLM **`Qwen2.5-3B-Instruct`** via Ollama/transformers |
| **Quality-ceiling row (optional)** | **Groq `llama-3.3-70b-versatile`** — FREE tier |
| **Faithfulness/relevance judge** | **Groq `llama-3.3-70b-versatile`** via `langchain-groq` (FREE; **not** xAI Grok) |
| **Eval library** | **RAGAS** (faithfulness + answer-relevance) + our own latency p95 |
| **GraphRAG corpus** | **arXiv cs.CL** (graph + page ranges); **SciFact** kept for Recall@5 sanity |
| **Reranker** | **Included** (cross-encoder) |
| **Multi-objective HPO** | **Yes** — doctor's NSGA-II Pareto over GraphRAG knobs |
| **Agent layer** | **Fixed executor** + thin tool wrapper; no ReAct/LangGraph |
| **Safety mitigations** | **3**: prompt-injection · source pinning/provenance · deny risky tool calls |
| **Qdrant layout** | **1 collection + source-filter-at-query-time** (refactor D2's 3) |
| **SLM tuning (D4)** | **QLoRA on Qwen2.5-3B**, run on **Google Colab Pro** |
| **Tuned-model serving (D4)** | merge LoRA → **quantize to GGUF (4-bit) → serve via Ollama** as `qwen2.5-3b-csai415`; `answer.py` selects the model, so the executor/eval stay local & identical to zero-shot |
| **QLoRA training set (D4)** | **hand-curated ~100–300 arXiv Q/A**, **disjoint from `qa_answers.jsonl`** (no train/test leakage); Ahmad owns both data sets |
| **Tune owner (D4)** | **WAFIQ** — Phase B of the answerer vertical, after D3 |

**Models config:** Groq — OpenAI-compatible, `GROQ_API_KEY`, base `https://api.groq.com/openai/v1`, `llama-3.3-70b-versatile`, free-tier ≈ 30 rpm/1K rpd/12K tpm (add retry/backoff). Local SLM — `ollama pull qwen2.5:3b-instruct`. Deps `ragas`, `langchain-groq`, `groq` in `requirements.txt`.

## Contracts (DONE at H0 — build against these)

| Surface | Where | Frozen |
|---|---|---|
| Executor | `src/csai415/graphrag.py` | `GraphRAGExecutor.answer(query, k, mode, rerank) -> AnswerResult`; `Citation`, `AnswerResult` |
| Subgraph seam | `src/csai415/graph_select.py` | `SubgraphResult`, `select_subgraph(query, neo4j_driver=…)` |
| Rerank seam | `src/csai415/rerank.py` | `rerank(query, candidate_ids, chunk_text, top_n)` |
| Answer seam | `src/csai415/answer.py` | `generate_answer(query, citations, contexts) -> (text, used_idxs)`; **model selected by `CSAI415_ANSWERER` env** (`qwen2.5:3b-instruct` zero-shot / `qwen2.5-3b-csai415` tuned) — D4 tuned model is a config swap, **not a new interface** |
| Eval | `src/csai415/eval.py` | `evaluate_answers(answer_fn, gold) -> {faithfulness, answer_relevance, recall5, p95_latency_ms, n}` |
| Tools/guard | `src/csai415/tools.py` | `is_write_cypher`, `RiskyToolCallDenied`, `cypher_query` guard |
| API | `src/csai415/api.py` | `POST /ask` + `AskRequest`/`AskResponse`/`CitationModel` |
| Eval gold schema | `data/gold/qa_answers.example.jsonl` | v2 row shape |
| **Train schema (D4)** | `data/train/qa_train.example.jsonl` | QLoRA training-row shape (Ahmad freezes the real `qa_train.jsonl`) |

Contract smoke: `tests/test_graphrag_contract.py` (3 tests, no model/Docker). Keep green.

---

## Assignments — 7 self-contained tasks

### Task 1 — Graph-guided retrieval  ·  *D3 GraphRAG 8%*  ·  **Owner: Yousef Alsakkaf**
Own everything "graph": link query→nodes, select subgraph, decide how it reshapes retrieval.
- **Files:** `src/csai415/graph_select.py`. **Build against:** live Neo4j + `HybridRetriever`.
- **Compare:** entity linking (LLM-NER vs spaCy vs fuzzy); which Cypher template fires; guidance policy — graph-as-**filter** vs **booster** vs **expansion**.
- **Done when:** real `select_subgraph` against the live graph; tables (linking precision/latency; guidance policy vs vector-only); graceful no-hit fallback.

### Task 2 — Reranking  ·  *D3 GraphRAG 8%*  ·  **Owner: Ahmed Soliman**
- **Files:** `src/csai415/rerank.py`. **Build against:** live `/search` candidates.
- **Compare:** `ms-marco-MiniLM-L-6-v2` vs `bge-reranker-base` vs MMR vs none; candidate-pool sweep.
- **Done when:** real rerank + (option × NDCG@5 lift × p95) table + verdict.

### Task 3 — Answerer (zero-shot **+** QLoRA tune)  ·  *D3 GraphRAG 8% + D4 Tuning 5%*  ·  **Owner: WAFIQ**
The full model lifecycle — two phases, one owner, so the tuned model honors the same citation format with zero back-and-forth.
- **Files:** `src/csai415/answer.py`; D4 adds `notebooks/05_qlora_tune.ipynb` (Colab Pro) + `reports/D4/tuning_card.md`.
- **Phase A (D3, now):** zero-shot Qwen-3B answerer + `[n]` citations + page ranges. Compare numbered-source vs post-hoc attribution; extractive vs abstractive; context-insufficient→refuse. Side-by-side Qwen-3B vs Groq-70B ceiling.
- **Phase B (D4, after D3 + Ahmad's `qa_train.jsonl` land):** QLoRA tune on **Colab Pro** → merge LoRA → quantize **GGUF 4-bit** → serve via **Ollama** as `qwen2.5-3b-csai415` (drops into the same `answer.py` via `CSAI415_ANSWERER`). Write the **tuning card** (dataset size, epochs, lr, LoRA rank/alpha/dropout, hardware/time, base-model license). *Prep the Colab notebook + GGUF→Ollama path during D3 so the tune is push-button once the baseline + training set exist.*
- **Done when (D3):** zero-shot answerer shipped + comparison. **Done when (D4):** tuned GGUF served in Ollama + tuning card committed + base-vs-tuned compared via the eval harness.
- **Starter Qs:** prompt/max-tokens to keep a 3B model fast while citing? · numbered-source vs post-hoc faithfulness on a small model? · QLoRA rank/lr for ~100–300 examples without overfitting? · does 4-bit GGUF quantization erode faithfulness vs the merged fp16 model?

### Task 4 — Data (eval gold **+** train set) + evaluation harness  ·  *D3 Eval 5% + D4 data*  ·  **Owner: Ahmad Fraij**
Own all the data + the RAGAS harness. Curate BOTH sets in D3 so the training set is ready for WAFIQ's Phase B.
- **Files:** `eval.evaluate_answers` body, `data/gold/qa_answers.jsonl`, **`data/train/qa_train.jsonl`**, `scripts/build_gold_answers.py`, `src/csai415/ragas_groq.py`. **Build against:** any `answer_fn` (answerer-agnostic — demo on the current answerer).
- **Compare/build:** gold construction (hand vs Groq-gen+verify, doctor's `eval_set.json` pattern); RAGAS-with-Groq vs hand-rolled judge; **leakage check** (assert no question/chunk overlap between `qa_train.jsonl` and `qa_answers.jsonl`).
- **Done when:** real RAGAS faithfulness/relevance via Groq (replaces the H0 proxy); both data sets committed + leakage assertion passes; retry/backoff for the 30-rpm cap.

### Task 5 — Ablation + multi-objective HPO  ·  *D3 Eval 5%*  ·  **Owner: Yehia Noureldin**
- **Files:** `src/csai415/graphrag_hpo.py`, `notebooks/04_graphrag_ablation.ipynb`. **Build against:** the frozen executor + `evaluate_answers` contracts (deliverable is the *runner* — the integrator runs it on the final pipeline).
- **Build:** vector vs graph vs hybrid × {rerank on/off}; NSGA-II Pareto of (faithfulness or NDCG@5, latency); knee-point picks (doctor's §7). Leave a hook so the integrator can add the **base-vs-tuned** row in D4.
- **Done when:** ablation table + Pareto plot + recommended config, demonstrated on the current pipeline.

### Task 6 — Safety: 3 mitigations + before/after  ·  *D3 Safety 2%*  ·  **Owner: Musab**
- **Files:** `src/csai415/safety.py`, `scripts/safety_eval.py`, `reports/D3/d3_safety.md`. **Build against:** the H0 tool guard + current pipeline.
- **Build:** prompt-injection defense; source pinning + provenance filtering; deny risky tool calls (extend the `cypher_query` read-only guard). ≥2 implementations per mitigation.
- **Done when:** ~10-case attack set + before/after table per mitigation (attack success ↓, benign unchanged) + documented limits.

### Task 7 — Integration + Qdrant refactor + `/ask` + **final report & demo**  ·  *spans D3 + D4*  ·  **Owner: Abdurlahman Alali** (integrator)
The single aggregation point — consumes the finished pieces. The only vertical that needs everything, by design.
- **Files:** the `graphrag.py` executor *shell* (graph branch delegates to Task 1), `qdrant_dense.py` + `api.py` (1-collection refactor + `/ask` + Neo4j driver), `reports/D4/D4_report.md`, README/`.env.example`/smoke.
- **D3:** collapse 3 Qdrant collections → 1; wire the real seams (mechanical — contracts match); run Tasks 4/5/6 harnesses on the wired pipeline for D3 numbers.
- **D4:** run the eval harness with **zero-shot vs tuned** (`CSAI415_ANSWERER`) for the final quality/latency delta table (incl. quantize/cache effect); write the **8–10pp report** (architecture, experiments, ablations, failure cases, ethics & licensing, future work); **repo hygiene** (one-command README, `.env.example`, seeds, `pytest` smoke); coordinate the **8-min live demo** (each member contributes ~60–90s on their slice).
- **Done when (D3):** `/ask` live on single-collection stack + D3 numbers. **Done when (D4):** base-vs-tuned table + 8–10pp report + demo deck + green suites.

---

## Dependency map

```
Live D2 services + frozen H0 contracts
        │
PHASE 1 — D3 (parallel, finish once):
   T1 graph-guided · T2 rerank · T3a zero-shot answerer · T4 data+eval+TRAIN SET ·
   T5 ablation/HPO · T6 safety        →  T7 integrates → zero-shot baseline + harness live
        │
PHASE 2 — D4 (after D3; concentrated):
   T3b WAFIQ: QLoRA tune  (needs: zero-shot baseline + T4's qa_train.jsonl + eval harness)
              → merge → GGUF → Ollama  + tuning card
        │
   T7 Abdurlahman: run base-vs-tuned through the harness → final delta table
              → 8–10pp report + 8-min demo + repo hygiene
```
No member edits another's files. No member runs in two D3 waves. The D3→D4 order is the inherent "tune after you have a baseline," not back-and-forth.

## Definition of done (combined checklist)

**D3**
- [ ] T1 real subgraph selection + guidance policy chosen with evidence.
- [ ] T2 cross-encoder rerank chosen with a lift-vs-latency table.
- [ ] T3a zero-shot answers with `[n]` citations + page ranges; Qwen-3B vs Groq-70B side-by-side.
- [ ] T4 real RAGAS faithfulness/relevance via Groq; `qa_answers.jsonl` committed; targets ≥ 0.8 / ≥ 0.8.
- [ ] T5 vector/graph/hybrid (× rerank) ablation + Pareto plot + recommended config.
- [ ] T6 3 mitigations with before/after + documented limits.
- [ ] T7 1 Qdrant collection + `/ask` live; D3 evidence assembled; suites green.

**D4**
- [ ] T4 `qa_train.jsonl` committed (~100–300, hand-curated arXiv) + **leakage check passes** (disjoint from gold).
- [ ] T3b QLoRA tune on Colab Pro → GGUF in Ollama; **tuning card** complete (dataset size, epochs, lr, LoRA ranks, hardware/time, license).
- [ ] T7 **zero-shot vs tuned** quality/latency delta table (incl. quantize/cache); 8–10pp report (architecture, experiments, ablations, failure cases, ethics & licensing, future work); README one-command + `.env.example` + seeds + `pytest` smoke; 8-min demo.
- [ ] Each member: ≥2 commits under their own identity; populated `ai_logs/` with share-link + approaches compared, per deliverable.
