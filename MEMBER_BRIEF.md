# CSAI415 D1 — Shared Member Brief

**Purpose:** every team member pastes Sections 1–5 plus their own role section (6.A / 6.B / 6.C / 6.D) into their AI chat as a single first message, then has a *real, iterative* technical conversation about their slice.

**Pair labels reflect temporal order:** Pair A starts first (its outputs unblock everyone), Pair B starts when Pair A's data lands, Pair C starts last when both A and B are working.

**Hard rule from the course brief:** "Submitting no logs, identical logs across members, or logs that do not correspond to the deliverable will be treated as missing evidence." Grading focuses on **depth of engagement**, **critical thinking** (push back, verify, iterate), and **individual contribution**. So:
- Do **not** share AI sessions with another member.
- Do **not** paste your final code in and ask the AI to "explain it" — that reads as fake engagement.
- **Do** ask substantive questions, paste real errors when they happen, push back when the AI's answer is generic, and ask for tradeoffs.

---

## 1. Course context (paste verbatim to your AI)

We are a 6-person team in CSAI415 building a **PDF-Papers AI Agent**: an agent that answers questions over scientific PDFs using hybrid retrieval (lexical + dense), a knowledge graph (GraphRAG), online learning from feedback (River), and a small PEFT/QLoRA-tuned model. The full project is 60% of the module grade across 4 deliverables (D1–D4). I am working on **D1 only** right now.

Suggested stack (we will use a subset for D1): Python, FastAPI, MongoDB, Qdrant, Neo4j, sentence-transformers, River, MLflow, Optuna/FLAML, PEFT/QLoRA, pytest, Docker Compose. **For D1 we deliberately skip Mongo/Qdrant/Neo4j/Docker** — those are D2's scope. D1 uses local Parquet + JSONL only.

## 2. D1 scope (Week 5, 15% of project = 9% of module)

D1 grading breaks down as: AutoML design 6% · Online learning 6% · Report 3%.

Required outputs:
- **AutoML Track A**: supervised auto-tuned kNN retriever with Optuna. Search space: k, metric, SVD dim, normalization, hybrid weight.
- **Online learning**: a River component for **adaptive hybrid weight from click-helpful feedback (y/n)**. Includes ADWIN drift handling and a prequential metrics plot.
- **Short report (max 2 pages)**: baseline vs AutoML metrics (NDCG@5, Recall@5), p95 latency, prequential chart, decisions, pitfalls.
- **Repo artifacts**: runnable notebook/script + YAML run card for the winning config.

Baseline targets from brief: Recall@5 ≥ 0.60, p95 latency ≤ 2s on CPU, online learning shows > +5% relative improvement vs static after drift.

## 3. Team decisions (locked — do NOT relitigate with your AI)

| Decision | Choice |
|---|---|
| AutoML track | **A — supervised kNN with Optuna** |
| Online-learning task | **(ii) adaptive hybrid weight from feedback** |
| Stores for D1 | **Local Parquet + JSONL only** (no Mongo/Qdrant/Neo4j) |
| Dense embedder | **`BAAI/bge-small-en-v1.5`** (384-dim) — corrected from the original `sentence-transformers/bge-small-en`, which is not a real HuggingFace repo. Pair B **must** embed queries with this same model. |
| Lexical | **`rank-bm25`** (pure Python) |
| AutoML library | **Optuna** with TPE sampler + MedianPruner |
| Online learner | **River** (`river.linear_model` family) + `river.drift.ADWIN` |
| Corpus for D1 | **SciFact (BEIR)** via `ir_datasets` — 5,183 abstracts with **300 manually-annotated** claim→evidence judgments. PLUS 5 arXiv cs.CL PDFs ingested end-to-end to demonstrate the PDF pipeline. |
| Gold Q/A set | **SciFact's 300 claims** with real human relevance labels. No synthesis. |
| Repo | `github.com/waf-iq/CSAI415`, branch `main`, feature branches per pair |

## 4. Repo layout

```
CSAI415/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw_pdfs/              (gitignored)
│   ├── processed/chunks.parquet
│   └── gold/qa.jsonl
├── src/csai415/
│   ├── ingest.py              (Pair A)
│   ├── retrieve.py            (Pair B)
│   ├── automl.py              (Pair B)
│   ├── online.py              (Pair C)
│   ├── eval.py                (Pair A)
│   ├── runcard.py             (Pair A)
│   └── mlflow_tracking.py     (Solo: Musab)
├── notebooks/
│   ├── 01_automl.ipynb        (Pair B)
│   └── 02_online_learning.ipynb (Pair C)
├── configs/winning_runcard.yaml
├── reports/D1_report.pdf
├── tests/test_smoke.py
└── ai_logs/<member-name>.md   (one per member with your AI share-link)
```

## 5. Interface contracts (anyone changing these MUST notify the other pairs)

```python
# data/processed/chunks.parquet schema
# columns: paper_id (str), chunk_id (str), text (str),
#          embedding (list[float], 384-dim),
#          page_start (int|null), page_end (int|null),     # null for SciFact abstracts; populated for arXiv PDFs
#          title (str), authors (list[str]|null), year (int|null), topic (str|null),
#          source (str)  # "scifact" or "arxiv-demo"

# data/gold/qa.jsonl line format
# {"qid": "q001", "question": "<claim text>", "relevant_chunk_ids": ["..."], "topic": "..."}
# Built from SciFact's claims + qrels (test split). All 300 claims have human-annotated evidence.

# src/csai415/eval.py — Pair B and Pair C BOTH call this
def evaluate(
    retriever_fn,           # callable: (query: str, k: int, hybrid_weight: float) -> list[str]
    queries: list[dict],    # loaded from qa.jsonl
    k: int = 5,
) -> dict:
    """Returns {'ndcg5': float, 'recall5': float, 'p95_latency_ms': float}."""

# src/csai415/retrieve.py — Pair B produces variants of this signature
def retriever_fn(query: str, k: int, hybrid_weight: float) -> list[str]:
    """Hybrid retrieve. hybrid_weight=1.0 means pure dense, 0.0 means pure BM25.
    Returns list of chunk_ids ordered by relevance."""
```

## 6. Your slice (paste ONLY the subsection that matches your assignment)

---

### 6.A — Pair A: Corpus + Eval + Report (Abdurlahman Alali + Yousef Alsakkaf)

**Files you own:** `src/csai415/ingest.py`, `src/csai415/eval.py`, `src/csai415/runcard.py`, `README.md`, `requirements.txt`, `.env.example`, `tests/test_smoke.py`, `reports/D1_report.pdf`. **You are on the critical path — your outputs unblock Pairs B and C. Start NOW.**

**Sub-roles:**
- **Abdurlahman Alali**: ingest pipeline (SciFact load → chunks → embeddings → `chunks.parquet`) + gold Q/A builder from SciFact qrels.
- **Yousef Alsakkaf**: `eval.py` verification + `runcard.py` verification + report + README + smoke tests.

**Acceptance bar — first 3 hours:**
- SciFact (BEIR) loaded via `ir_datasets` → `data/processed/chunks.parquet` (~5,183 abstract chunks, can sub-chunk longer ones).
- 5 arXiv cs.CL PDFs downloaded to `data/raw_pdfs/`, parsed with PyMuPDF, appended to `chunks.parquet` with `source="arxiv-demo"` and real page_start/page_end. These are *not* used in eval — they're proof the PDF pipeline runs.
- `data/gold/qa.jsonl` built from SciFact's 300 test-split claims + qrels (1–N relevant doc_ids per claim → mapped to chunk_ids).
- `src/csai415/eval.py` verified working with `evaluate(retriever_fn, queries, k=5) -> dict`.

**Acceptance bar — hours 9–11:**
- 2-page PDF report at `reports/D1_report.pdf` with: (1) one paragraph methods, (2) baseline-vs-AutoML table, (3) prequential plot from Pair C, (4) decisions/pitfalls bullets.
- `README.md` with one-command setup.
- `tests/test_smoke.py` runs in < 30s and verifies ingest → retrieve → evaluate works end-to-end on 5 docs.

**Starter questions for your AI:**
1. Loading SciFact via `ir_datasets` — what's the right call? `ir_datasets.load("beir/scifact")` exposes `docs_iter()`, `queries_iter()`, and `qrels_iter()`. Should I use train+test or test-only for the gold set (test has the 300 evaluated claims)?
2. SciFact abstracts are typically 200–400 words. Should I chunk them (300 tokens / 50 overlap) or keep one chunk per abstract? Tradeoff: chunking helps when a claim matches a specific sentence; one-chunk-per-abstract preserves context.
3. For the 5 arXiv demo PDFs — which parser balances speed vs page-map fidelity for cs.CL papers: `pypdf`, `pdfplumber`, `PyMuPDF` (fitz), `unstructured`? PyMuPDF is usually the speed/fidelity winner. Defend or push back.
4. The chunks.parquet schema has nullable page_start/page_end (null for SciFact, populated for arXiv). Is nullable the right move, or should I use sentinel `-1`? What breaks downstream if Pair B or C sees nulls?
5. SciFact's qrels map claim_id → doc_id. If I sub-chunk abstracts, I need to map doc_id → multiple chunk_ids. What's the cleanest way: mark "all chunks of doc X are relevant" or "only the chunk containing the evidence sentence is relevant"? The latter is more rigorous but requires reading the evidence span.
6. For p95 latency, do I time end-to-end (embed query → BM25 → dense → fuse → top-k) or just the retrieval after embedding? Brief target is ≤ 2s on CPU — report both?
7. Report structure for 2 pages: what's the minimum viable layout? Abstract (3 lines), methods (1/4 page), results table, prequential figure, decisions/pitfalls bullets, references.
8. **After ingest finishes**, ask: "I got X chunks from Y SciFact abstracts + Z chunks from 5 arXiv PDFs. Mean chunk length is W tokens. Does that distribution suggest my chunker is splitting reasonably?"

**Integration boundary:**
- The schema of `chunks.parquet` and `qa.jsonl` (Section 5) is contractual. If you must change it, tell Pairs B and C *before* committing.
- `evaluate()`'s return dict keys are contractual.

---

### 6.B — Pair B: AutoML kNN + Optuna (WAFIQ Akram ABO DAKEN + Ahmad Fraij)

**Files you own:** `src/csai415/retrieve.py`, `src/csai415/automl.py`, `notebooks/01_automl.ipynb`, `configs/winning_runcard.yaml`.

**Sub-roles:**
- **Ahmad Fraij**: baseline kNN retriever (`HybridRetriever` class in `retrieve.py`) + search-space definition in `automl.py`.
- **WAFIQ**: Optuna study driver + winning-trial → YAML run-card writer + integration test.

**You start in Wave 2:** wait for Pair A's `chunks.parquet` to land before running against real data. You CAN build code against a synthetic fixture in the meantime.

**Acceptance bar:**
- A clear search space over: `candidate_k` (int [5,50], = pool size per backend before fusion; the eval k stays fixed at 5), `metric` ({cosine, l2, dot}), `svd_dim` ({None, 64, 128, 256}), `normalize` (bool), `hybrid_weight` (float [0,1]).
- ≥ 60 Optuna trials, MedianPruner enabled.
- Baseline-vs-AutoML table: rows = {baseline, winning}, cols = {NDCG@5, Recall@5, p95 latency ms}.
- `configs/winning_runcard.yaml` reproducible: seed, embedding model, all winning hyperparams, dataset hash.

**Starter questions for your AI (don't stop at the first answer):**
1. Given BAAI/bge-small-en-v1.5 (384-dim) embeddings over ~5,200 chunks and 300 SciFact queries, what's a sensible Optuna search space for hybrid kNN? What should be log-scale, what should be categorical, and what's the risk of overfitting the study to 300 queries?
2. Should the objective be raw NDCG@5, or NDCG@5 minus a latency penalty? If I add a latency penalty, how do I weight it without dominating the signal?
3. When I apply SVD before cosine similarity, are the post-SVD vectors already mean-centered? Do I still need explicit L2 normalization for cosine to be well-defined?
4. For hybrid blend, weighted-sum-of-scores requires score normalization (BM25 is unbounded, cosine is in [-1,1]). Is RRF (reciprocal rank fusion) safer? What does each break under?
5. With 300 gold queries, should I use cross-validation across queries (e.g., 5-fold), or trust a single split? What's the bias-variance tradeoff?
6. What minimum fields must the run-card YAML capture so D2 can reproduce the embedding + index + winning hyperparams from scratch?
7. **After running the study**, paste your best trial back and ask: "Best trial: k=X, svd=Y, hybrid_weight=Z, NDCG@5=W. Is W plausible for this corpus size, or am I overfitting? What ablation would reveal it?"

**Integration boundary you cannot cross without telling Pair A/C:**
- The signature of `retriever_fn`. Pair C's online learner consumes it.
- The schema of `chunks.parquet`. Pair A produces it.

---

### 6.C — Pair C: Online learning (Ahmed Soliman + Yehia Noureldin)

**Files you own:** `src/csai415/online.py`, `notebooks/02_online_learning.ipynb`, `reports/prequential.png`.

**You start in Wave 3:** you need both Pair A's `qa.jsonl` and Pair B's working `retriever_fn`. You CAN build skeleton code against a fake retriever in the meantime; swap in real one at the end.

**Sub-roles:**
- **Ahmed Soliman**: River learner that updates `hybrid_weight` from binary feedback (the model itself).
- **Yehia Noureldin**: drift-simulation harness (200-event stream with planted drift) + ADWIN integration + prequential plot.

**Acceptance bar:**
- A River-based model that takes (query_features, current_hybrid_weight, feedback ∈ {0,1}) and outputs the next `hybrid_weight` to use.
- An ADWIN detector wired in, with a documented response policy (e.g., "on drift detected, reset learner state" or "boost learning rate").
- Prequential NDCG@5 chart over the 200-event stream with the drift point marked and ADWIN firings annotated.
- Quantitative claim in the report: "online learner improves NDCG@5 by ≥ 5% vs static hybrid_weight after the drift point."

**Starter questions for your AI:**
1. For learning a continuous `hybrid_weight` ∈ [0,1] from binary click-helpful feedback, which River model best fits — `LinearRegression`, `LogisticRegression` treating it as a bandit, or a Hoeffding tree? What signal does the binary feedback actually give the regressor?
2. Frame this as a contextual bandit: the action is the hybrid_weight (discretized to ~5 values?), the context is query features (length, topic), the reward is the click. Does River have first-class bandit support, or should I roll my own ε-greedy on top of a regressor?
3. How should I set ADWIN's `delta`? For a 200-event stream with a planted drift at event 100, what false-positive rate is acceptable, and what does that imply for delta?
4. Drift simulation: which kind of shift produces the cleanest demo — (a) topic distribution shift (SciFact has multiple medical sub-topics), (b) query-length shift, (c) relevance-label noise increase? Which best motivates *adapting the hybrid weight* specifically?
5. Prequential chart: sliding window vs cumulative? For 200 events, what window size is interpretable on a 2-page report?
6. What's a defensible baseline to compare against? "Static hybrid_weight = winning AutoML value" is the obvious one — anything else worth showing?
7. **After running**, ask: "ADWIN fired at event 87 but my planted drift is at 100. Is that a false positive, or is the signal leading the drift label? How do I diagnose?"

**Integration boundary:**
- Consumes Pair B's `retriever_fn` and Pair A's `evaluate()` and `qa.jsonl`. Don't reimplement either.
- If your learner needs query features beyond raw text (length, topic), check whether Pair A's `qa.jsonl` already has them before adding fields.

---

### 6.D — Solo: Experiment tracking with MLflow (Musab)

**Files you own:** `src/csai415/mlflow_tracking.py`. (Plus you'll add a small section to `notebooks/01_automl.ipynb` that pulls the comparison view from MLflow for the report.)

**You start NOW** — your work is fully independent of all 3 pairs. You can read MLflow docs, set up the local backend, and draft the integration the moment you clone the repo. You converge with Pair B (WAFIQ) when their Optuna study is ready to run — at that point WAFIQ passes your callback into `run_study(callbacks=[cb])` (the parameter already exists on the function signature) and your tracking goes live automatically.

**Acceptance bar:**
- Local MLflow backend running (`file:./mlruns` is fine; `mlruns/` is gitignored).
- Optuna→MLflow callback wired: every trial logs `params`, `metrics` (ndcg5, recall5, p95_latency_ms), and the dataset SHA-256 hashes as tags.
- After the study finishes, the best trial's MLflow run is tagged `csai415.blessed=true` and has the runcard YAML + an optimization-history PNG + a parameter-importance PNG attached as artifacts.
- `reports/mlflow_top5.md` (or similar) — a markdown table comparing top-5 runs, generated programmatically from `mlflow.search_runs()`. This goes into the 2-page report.
- A screenshot of the MLflow UI's "Compare Runs" view, also for the report.

**Starter questions for your AI:**
1. **Tracking backend** — `file:./mlruns` (local files, no server needed) vs `sqlite:///mlruns.db` vs a real tracking server. For a 1-day local D1 run, what trade-offs?
2. **Use the built-in Optuna integration or write own callback** — `optuna.integration.mlflow.MLflowCallback` auto-logs each trial. What does it *not* log that's worth adding (dataset hashes, retriever config nesting, artifact paths)?
3. **Run hierarchy** — should the 60 Optuna trials be 60 sibling MLflow runs, or 1 parent run with 60 children? Which gives a better Compare-Runs view in the UI?
4. **The "blessed run" pattern** — after Optuna finishes, what's the cleanest way to mark the winner so D2 reviewers can find it? Custom tag, special run name, or MLflow Model Registry?
5. **Comparison-table export** — pull top-5 runs at end of study, write a markdown table to `reports/mlflow_top5.md`. Should the script be invoked from inside `run_study()`, or be a separate post-run command? What's better for reproducibility?
6. **Screenshot for the report** — the MLflow UI has Compare Runs (table), Parallel Coordinates, and Parameter Importance views. Which one tells the strongest story in a single screenshot for a 2-page report?
7. **After running the live integration**, ask: "I'm seeing N runs in MLflow but Optuna reports M trials — why the mismatch? Pruned trials?"

**Integration boundary:**
- Don't modify `automl.py` or `retrieve.py` — pass your callback from outside via `run_study(callbacks=[cb])`. The function already accepts that param.
- Don't change the runcard YAML schema — `runcard.write_runcard()` is Pair A's contract. You attach the same YAML to MLflow as an artifact, not duplicate its fields.

---

## 7. Coordination rules

- **Branches:** `pair-a-corpus`, `pair-b-automl`, `pair-c-online`, `solo-musab-mlflow`. PR into `main`. Request review from one teammate outside your pair.
- **Commits:** every member must have ≥ 2 commits under their own GitHub identity. Configure `git config user.name` and `user.email` before committing.
- **AI logs:** by hour 11, every member updates `ai_logs/<your-name>.md` with their AI share-link + a 2-sentence note on what you discussed. Empty file = missing evidence.
- **Cross-pair changes:** any change to Section 5 (interface contracts) requires a 1-line message in team chat to the other pairs.
- **If blocked:** post in team chat with the specific error + what you tried. Don't silently spin for > 30 min.

## 8. Definition of done (D1 rubric checklist)

| Rubric item | Owner | Done when |
|---|---|---|
| AutoML design (6%) | Pair B + Musab | Search space documented; ≥ 60 Optuna trials; run card YAML reproducible; MLflow tracking enabled with blessed-run tag + top-5 comparison table |
| Online learning (6%) | Pair C | River learner integrated; ADWIN drift response with evidence; prequential plot |
| Report quality (3%) | Pair A | 2-page PDF with tables/plots, baseline vs AutoML, pitfalls |
| Reproducibility | Pair A | One-command setup in README; `pytest tests/test_smoke.py` green |
| AI logs | Everyone | `ai_logs/<name>.md` has share-link, distinct content per person |
| Commits from all | Everyone | `git log --pretty=format:%an` shows all 7 names |
