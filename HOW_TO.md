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
# outputs: reports/D1/optimization_history.png, reports/D1/param_importances.png, reports/D1/winner_vs_baselines.png, reports/D1/winner_vs_baselines.csv
```

## Online learning prequential — Pair C

Runs the ε-greedy bandit vs static AutoML-winner baseline over a 200-event stream with a query-style drift at event 100:

```bash
python -m nbconvert --to notebook --execute notebooks/02_online_learning.ipynb --inplace
# ~2 min
# outputs: reports/D1/prequential.png (depends on configs/winning_runcard.yaml from Pair B)
```

## MLflow tracking — Solo (Musab)

Replays the completed Optuna study into MLflow, tags the winner as blessed with artifacts, exports the top-5 comparison table + parallel-coordinates plot:

```bash
python -m csai415.mlflow_tracking
# ~30 sec
# outputs: mlruns.db (gitignored), reports/D1/mlflow_top5.md, reports/D1/mlflow_parallel_coords.png

mlflow ui --backend-store-uri sqlite:///mlruns.db
# browse at http://localhost:5000
```

## D1 report

The Markdown source is at `reports/D1/D1_report.md`; the committed PDF was rendered via:

```bash
pandoc reports/D1/D1_report.md -o reports/D1/D1_report.pdf
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

---

# How to reproduce D2 results

D2 builds on top of D1's `chunks.parquet` + blessed BOHB runcard. The D2 work moves them onto a Docker-Compose stack (Mongo + Qdrant + Neo4j) behind a FastAPI `/search`. Full task plan lives in `D2_TASKS.md`; this section is the runbook for spinning the stack up and regenerating every D2 artifact.

## D2 prerequisites

- All D1 setup (see top of this file) — the `.venv`, the `chunks.parquet`, the `winning_runcard.yaml`.
- **Docker Desktop** (Windows/macOS) or Docker Engine + Compose (Linux). Single command brings up 4 containers.
- ~2 GB disk for Docker volumes + chunks.parquet (~48 MB) + arXiv PDFs (gitignored, regenerated only if you re-run D2-A1).

```bash
docker --version            # expect Docker version 24+
docker compose version      # expect Docker Compose v2+
```

## (Optional) Re-expand the arXiv corpus — D2-A1

Already on main as `chunks.parquet` (15,760 chunks: 5,663 SciFact + 357 arxiv-demo + 9,740 arxiv across 150 papers). To re-run from scratch:

```bash
python scripts/ingest_arxiv_batch.py --max 148 --sleep 5
# ~30 min: arxiv downloads + PyMuPDF parse + BGE embedding
# Idempotent — re-running skips paper_ids already in parquet.
# Output: data/processed/chunks.parquet (appended), data/raw_pdfs/arxiv_batch_meta.json
```

## Bring up the stack — D2-M1

```bash
cp .env.example .env        # localhost defaults — no edits needed for single-machine dev
make up                     # docker compose up -d for mongo + qdrant + neo4j
# OR
docker compose up -d        # same thing

# Wait ~30 sec for healthchecks to settle, then:
docker compose ps           # expect: 3 services healthy
```

For the FastAPI service too:

```bash
docker compose --profile api up -d --build
# Builds the api image (~1-2 min first time) then starts it.
# api container reads CSAI415_USE_QDRANT=1 from docker-compose.yml so /search uses live Qdrant.
```

## Seed all three stores — D2-INT1

`make seed` runs the seeders in dependency order (Mongo → Qdrant × 3 collections → Neo4j):

```bash
make seed
# ~2-3 min total. Expected counts:
#   Mongo:   5,338 papers / 15,760 chunks
#   Qdrant:  15,760 (full) / 5,663 (scifact) / 10,097 (arxiv) vectors
#   Neo4j:   155 papers / ~764 authors / ~13 topics
```

Or run individually:

```bash
python scripts/seed_mongo.py  --source all
python scripts/seed_qdrant.py                    # → single chunks_bge384 (source-filtered at query time)
python scripts/seed_neo4j.py  --source all
```

Idempotent — re-running is safe (Mongo upserts by `_id`; Qdrant recreates collections; Neo4j uses `MERGE`).

## Verify `/search` and `/healthz` — D2-B1 / D2-INT2

```bash
curl http://localhost:8000/healthz
# expect 200 + {"status":"ok"} (also pings all three Qdrant collections internally)

curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"language model pretraining","k":5,"source":"scifact"}'
# expect 5 hits, every chunk_id prefixed "scifact:"
```

Full verification (3 source paths + 20-request p95 latency) is in `reports/D2/d2_int2_verification.md`.

## Run the live-stack smoke test — D2-A4

```bash
D2_STACK_UP=1 pytest tests/test_smoke.py -v
# expect: 12 passed (was 11 + 1 skipped)
```

Without the env var the live test is skipped and the suite stays green offline.

## Regenerate retrieval metrics — D2-B3

```bash
python scripts/eval_search_metrics.py
# ~2 min: runs the 60-query SciFact holdout against the in-process retriever
# Outputs:
#   reports/D2/d2_search_metrics.csv  (3 configs × 4 metrics)
#   reports/D2/d2_topk_examples.md    (3 SciFact + 2 arXiv example queries)
# Expected: hybrid_blessed NDCG@5 = 0.561, Recall@5 = 0.649, p95 ~106ms
```

## Capture Cypher outputs — D2-C3

Requires Neo4j seeded (above):

```bash
python scripts/capture_cypher_outputs.py
# Runs all 5 cypher/*.cypher files against the live graph, captures results.
# Output: reports/D2/d2_cypher_examples.md
```

## Regenerate the dataflow diagram — D2-M1

```bash
make diagram   # needs Node + npx; uses @mermaid-js/mermaid-cli to render the PNG
```

## Regenerate the D2 report PDF — D2-A3

```bash
pip install markdown-pdf
python -c "
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path
import os
os.chdir('reports/D2')
md = Path('D2_report.md').read_text(encoding='utf-8')
pdf = MarkdownPdf(toc_level=0)
pdf.add_section(Section(md, paper_size='A4'))
pdf.meta['title'] = 'CSAI415 D2 Report'
pdf.meta['author'] = 'Ahmad Fraij, WAFIQ Akram ABO DAKEN'
pdf.save('D2_report.pdf')
"
```

(`pandoc reports/D2/D2_report.md -o reports/D2/D2_report.pdf` also works if pandoc is installed.)

## Render the D2 walkthrough notebook

`notebooks/03_d2_retrieval_stack.ipynb` is a **self-contained, in-memory** run of the **full** D2 pipeline, in the style of the Week 5/6 labs — **no Docker required**. It imports the team's *actual* production modules and runs ingest → store → search → graph in-process:

- **§2 ingestion** — parses + embeds real PDFs from `data/raw_pdfs/` via `ingest.parse_arxiv_pdfs` + `ingest.embed_chunks`, then retrieves over the just-ingested chunks;
- **§3–5 storage** — Mongo via `mongomock`, Qdrant via `QdrantClient(":memory:")` (seeded by `seed_collection_from_parquet`, 3 collections);
- **§6–8 retrieval** — the real FastAPI `/search` via `create_app(qdrant_client=...)` + `TestClient`: `/healthz`, all 3 routes (no-leakage asserts), and the holdout Recall@5/NDCG@5 reproduced against `d2_search_metrics.csv`;
- **§9–11 graph** — Neo4j when configured **or** a `networkx` fallback (reusing `seed_neo4j.seed_graph`), running the 5 documented queries **plus** 5 intermediate analytics (prolific authors, co-authorship pairs, interdisciplinary authors, topic co-occurrence, most-collaborative authors).

```bash
pip install qdrant-client mongomock networkx neo4j python-dotenv pymupdf   # all in requirements.txt
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 notebooks/03_d2_retrieval_stack.ipynb --inplace
# ~3-4 min (ingest 2 PDFs + seed 3 in-memory Qdrant collections + 60-query holdout eval; first BGE load cached).
```

To run **§9–11 as real Cypher** against your own Neo4j (Aura or the compose Neo4j), drop credentials into a `.env` at the repo root — the notebook calls `load_dotenv()` and accepts both the repo names (`NEO4J_URL`/`NEO4J_USER`) and Aura's (`NEO4J_URI`/`NEO4J_USERNAME`):

```ini
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

## Tear down (full reset)

```bash
make clean     # docker compose down -v — also deletes the named volumes
```

## Full end-to-end D2 from a blank clone

Assumes D1 has already been reproduced (top of this file) and `chunks.parquet` exists.

```bash
# 1. Stack up
docker compose --profile api up -d --build

# 2. Seed everything
make seed

# 3. Verify retrieval
curl http://localhost:8000/healthz
python scripts/eval_search_metrics.py

# 4. Capture graph outputs
python scripts/capture_cypher_outputs.py

# 5. Live-stack smoke
D2_STACK_UP=1 pytest tests/test_smoke.py -v
```

Total ≈ 10-15 minutes (compose image build dominates first time). All `reports/D2/d2_*.{csv,md,png}` regenerate; `/search` is live on `localhost:8000`.

## Troubleshooting

- **`/healthz` returns 503 with "qdrant collection ... unreachable"** → run `python scripts/seed_qdrant.py`. The D3 layout uses a single `chunks_bge384` collection (source-filtered at query time); the seeder drops any stale per-source collections unless `--keep-legacy` is passed.
- **`make seed` errors out on `seed_neo4j.py`** → check `docker compose ps neo4j` is healthy. Neo4j takes ~15-20 s longer than Mongo/Qdrant to settle on first boot.
- **Embedder download timing out on the API container** → `BAAI/bge-small-en-v1.5` is fetched at app startup. First boot needs internet; subsequent restarts use the HuggingFace cache mounted in the container.
- **`docker compose --profile api up` fails to build** → check the `Dockerfile` matches the python version in `.venv` (3.12 expected); the api image installs `requirements.txt` so any local-only deps would surface here.
- **First `/search` after restart is slow (~2.5 s)** → expected: Qdrant HNSW cold-cache. Subsequent queries are sub-500 ms; see `reports/D2/d2_int2_verification.md` for the full latency profile.
