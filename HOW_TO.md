# How to reproduce D1 results

Step-by-step guide to regenerate every artifact in this repo from a fresh clone. The README's Quickstart is the short version; this doc is the detailed walkthrough with timings and per-slice commands.

## Setup

```bash
git clone https://github.com/waf-iq/special-topics.git
cd special-topics
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
cp .env.example .env   # no secrets needed for D1
```

First run will download `BAAI/bge-small-en-v1.5` (~150 MB) on demand and cache it under `~/.cache/huggingface/`.

## Pipeline data — Pair A

`data/processed/chunks.parquet` (6,020 chunks) and `data/gold/qa.jsonl` (300 SciFact test claims) are **committed** so D1 runs out of the box.

To regenerate from source:

```bash
python -m csai415.ingest                # ~5-8 min: downloads SciFact + 5 arXiv PDFs, embeds, writes parquet + qa.jsonl
```

## AutoML study — Pair B

Runs 80 Optuna trials (multivariate TPE, NopPruner, stratified 80/20 split), evaluates winner + 3 baselines on the 60-query holdout, writes the runcard:

```bash
python -c "from csai415.automl import run_and_record; run_and_record()"
# ~15-25 min on CPU
# outputs: configs/winning_runcard.yaml, configs/d1_split_indices.json, studies/csai415-d1-knn.db (gitignored)
```

Render the report figures from the study:

```bash
python -m nbconvert --to notebook --execute notebooks/01_automl.ipynb --inplace
# ~30 sec
# outputs: reports/optimization_history.png, reports/param_importances.png, reports/winner_vs_baselines.png, reports/winner_vs_baselines.csv
```

## Online learning prequential — Pair C

Runs the ε-greedy bandit vs static AutoML-winner baseline over a 200-event stream with a query-style drift at event 100:

```bash
python -m nbconvert --to notebook --execute notebooks/02_online_learning.ipynb --inplace
# ~2 min
# outputs: reports/prequential.png (depends on configs/winning_runcard.yaml from Pair B)
```

## MLflow tracking — Solo (Musab)

Replays the completed Optuna study into MLflow, tags the winner as blessed with artifacts, exports the top-5 comparison table + parallel-coordinates plot:

```bash
python -m csai415.mlflow_tracking
# ~30 sec
# outputs: mlruns.db (gitignored), reports/mlflow_top5.md, reports/mlflow_parallel_coords.png

mlflow ui --backend-store-uri sqlite:///mlruns.db
# browse at http://localhost:5000
```

## D1 report

The Markdown source is at `reports/D1_report.md`; the committed PDF was rendered via:

```bash
pandoc reports/D1_report.md -o reports/D1_report.pdf
```

(Any Markdown→PDF tool works; the figure paths are relative so `pandoc` handles them out of the box.)

## Smoke tests

```bash
pytest tests/test_smoke.py
# expect: 11 passed, 1 xpassed
```

## Full end-to-end from a blank clone

```bash
# 1. Setup
git clone https://github.com/waf-iq/special-topics.git && cd special-topics
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt && pip install -e .

# 2. Pair B AutoML (~20 min)
python -c "from csai415.automl import run_and_record; run_and_record()"
python -m nbconvert --to notebook --execute notebooks/01_automl.ipynb --inplace

# 3. Pair C prequential (~2 min)
python -m nbconvert --to notebook --execute notebooks/02_online_learning.ipynb --inplace

# 4. MLflow replay (~30 sec)
python -m csai415.mlflow_tracking

# 5. Verify
pytest tests/test_smoke.py
```

Total ≈ 25-30 minutes on a CPU laptop. After this, every `reports/*.png`, `reports/*.csv`, `reports/*.md`, and `configs/winning_runcard.yaml` is regenerated and `pytest` is green.
