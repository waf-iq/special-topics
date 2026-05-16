# CSAI415 — PDF-Papers AI Agent

Hybrid retrieval + GraphRAG agent over scientific PDFs with online learning and AutoML. Full project brief in `CSAI415_Project_Brief.pdf`; D1 work plan in `MEMBER_BRIEF.md`.

## Team

Pair labels reflect temporal order: Pair A starts first, Pair C starts last.

| Member | D1 slice |
|---|---|
| Abdurlahman Alali | Pair A — Corpus + Eval + Report (ingest pipeline + gold Q/A) |
| Yousef Alsakkaf | Pair A — Corpus + Eval + Report (eval/runcard/report/plumbing) |
| WAFIQ Akram ABO DAKEN | Pair B — AutoML (Optuna driver + run card) |
| Ahmad Fraij | Pair B — AutoML (baseline kNN + search space) |
| Ahmed Soliman | Pair C — Online learning (River learner) |
| Yehia Noureldin | Pair C — Online learning (drift simulation + prequential plot) |
| Musab | Solo — Experiment tracking (MLflow integration around Optuna study) |

## Quickstart

```bash
git clone https://github.com/waf-iq/CSAI415.git
cd CSAI415
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env

# Pair C runs ingest first to produce data/processed/chunks.parquet and data/gold/qa.jsonl
python -m csai415.ingest

# Pair A runs the Optuna study
jupyter notebook notebooks/01_automl.ipynb

# Pair B runs the online-learning prequential
jupyter notebook notebooks/02_online_learning.ipynb

# Smoke test (should pass on any clone)
pytest tests/test_smoke.py
```

## Repo layout

```
src/csai415/
  ingest.py     — PDF -> chunks -> embeddings (Pair C)
  retrieve.py   — BM25 + dense + hybrid retriever (Pair A)
  automl.py     — Optuna study (Pair A)
  online.py     — River learner + ADWIN (Pair B)
  eval.py       — NDCG@5, Recall@5, p95 latency (Pair C)
  runcard.py    — Winning config -> YAML (Pair C)
notebooks/      — driver notebooks per pair
configs/        — winning_runcard.yaml
data/           — raw_pdfs/ (gitignored), processed/, gold/
reports/        — D1_report.pdf
tests/          — smoke tests
ai_logs/        — one markdown per member with AI chat share-link
```

## D1 deliverable checklist

- [ ] `data/processed/chunks.parquet` produced from 30–50 arXiv PDFs
- [ ] `data/gold/qa.jsonl` with ≥ 20 questions
- [ ] Baseline kNN + Optuna study (≥ 60 trials)
- [ ] `configs/winning_runcard.yaml` reproducible
- [ ] River + ADWIN online learner with prequential plot
- [ ] `reports/D1_report.pdf` (2 pages, baseline-vs-AutoML table, prequential chart)
- [ ] `pytest tests/test_smoke.py` green
- [ ] Every member has commits under their own GitHub identity
- [ ] Every member has filled `ai_logs/<name>.md` with their AI share-link

## Reproducibility

All seeded with `seed=42` unless overridden. Embedding model pinned in `.env.example`. Run-card YAML captures dataset hash + all hyperparameters.

## Licensing

Code: MIT (to be confirmed by team). Corpus: open-access arXiv papers only; sources listed in `data/processed/sources.csv` after ingest.
