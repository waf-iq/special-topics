# CSAI415 — PDF-Papers AI Agent

Hybrid retrieval + GraphRAG agent over scientific PDFs with online learning and AutoML. Full project brief in `CSAI415_Project_Brief.pdf`; D1 work plan in `MEMBER_BRIEF.md`.

## Team

Pair labels reflect temporal order: Pair A starts first, Pair C starts last.

| Member | D1 slice |
|---|---|
| Ahmed Soliman | Pair A — Corpus + Eval + Report (ingest pipeline + gold Q/A) |
| Yousef Alsakkaf | Pair A — Corpus + Eval + Report (eval/runcard/report/plumbing) |
| WAFIQ Akram ABO DAKEN | Pair B — AutoML (Optuna driver + run card + notebook) |
| Ahmad Fraij | Pair B — AutoML (HybridRetriever class + search space) |
| Abdurlahman Alali | Pair C — Online learning (River learner) |
| Yehia Noureldin | Pair C — Online learning (drift simulation + prequential plot) |
| Musab | Solo — Experiment tracking (MLflow integration around Optuna study) |

## D1 setup

- **Corpus:** SciFact (BEIR) via `ir_datasets` — 5,183 scientific abstracts. Plus 5 arXiv cs.CL PDFs ingested end-to-end to prove the PDF pipeline works (not used in eval).
- **Gold set:** SciFact's 300 manually-judged test claims with multi-doc relevance.
- **Embedder:** `BAAI/bge-small-en-v1.5` (384-dim). Queries get a BGE prefix; corpus does not.
- **Eval:** stratified 80/20 split of the 300 claims — TPE tunes on 240, single blind eval on 60 holdout.

## Quickstart

```bash
git clone https://github.com/waf-iq/special-topics.git
cd special-topics
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env

# chunks.parquet and qa.jsonl are committed — no need to re-run ingest unless you want to
# python -m csai415.ingest  # only if you want to regenerate from source

# Run the AutoML study end-to-end (writes configs/winning_runcard.yaml, ~20 min)
python -c "from csai415.automl import run_and_record; run_and_record()"

# Pair B notebook: produces reports/*.png from the runcard + study DB
jupyter notebook notebooks/01_automl.ipynb

# Pair C notebook: prequential plot from the online learner
jupyter notebook notebooks/02_online_learning.ipynb

# Smoke test (must stay green)
pytest tests/test_smoke.py
```

## Repo layout

```
src/csai415/
  ingest.py     — SciFact loader + arXiv PDF parser + embedding (Pair A)
  retrieve.py   — HybridRetriever (BM25 + dense kNN + optional SVD + fusion) (Pair B)
  automl.py     — Optuna study + run_and_record orchestration (Pair B)
  online.py     — River learner + ADWIN + prequential loop (Pair C)
  eval.py       — NDCG@5, Recall@5, p95 latency (Pair A)
  runcard.py    — Winning config -> YAML with git + env + split metadata (Pair A)
notebooks/      — 01_automl.ipynb (Pair B), 02_online_learning.ipynb (Pair C)
configs/        — winning_runcard.yaml, d1_split_indices.json (for Pair C reproducibility)
data/           — raw_pdfs/ (gitignored), processed/chunks.parquet, gold/qa.jsonl
studies/        — Optuna SQLite DBs (gitignored)
reports/        — D1_report.pdf + the figures it embeds
tests/          — smoke tests
```

## D1 deliverable checklist

- [x] `data/processed/chunks.parquet` (6,020 chunks: 5,663 SciFact + 357 arXiv-demo, 384-dim embeddings)
- [x] `data/gold/qa.jsonl` (300 SciFact test claims with multi-doc relevance)
- [x] `HybridRetriever` (BM25 + dense + optional SVD, weighted-sum fusion with per-query min-max scaling)
- [x] Optuna study (80 trials, multivariate TPE, NopPruner, stratified 80/20 split)
- [x] `configs/winning_runcard.yaml` reproducible (git SHA + env versions + dataset hashes + split indices)
- [ ] River + ADWIN online learner with prequential plot
- [ ] `reports/D1_report.pdf` (2 pages, baseline-vs-AutoML table + prequential chart)
- [x] `pytest tests/test_smoke.py` green (11 passed + 1 xpass)
- [ ] Every member has ≥2 commits under their own GitHub identity

## Reproducibility

All seeded with `seed=42` unless overridden. Run-card YAML at `configs/winning_runcard.yaml` captures:

- Git SHA + dirty flag at write time
- Python + package versions (optuna, numpy, pandas, scikit-learn, sentence-transformers)
- Dataset SHA-256 of `chunks.parquet` and `qa.jsonl`
- Optuna sampler/pruner config + storage URI
- 80/20 split strategy, seed, and indices path
- Winning hyperparameters + tune/holdout metrics for the winner + all three baselines (pure BM25, pure dense, default hybrid)

`configs/d1_split_indices.json` persists the exact tune/holdout indices so Pair C evaluates on the same 60 holdout queries Pair B held out.

## Licensing

Code: MIT — see `LICENSE`. Corpus: SciFact (CC BY-NC 2.0 via BEIR); arXiv demo PDFs are open-access cs.CL submissions referenced in `data/raw_pdfs/arxiv_meta.json` after ingest.
