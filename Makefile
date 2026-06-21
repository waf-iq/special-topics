# D2-M1 — one-command stack + seed helpers. See D2_TASKS.md §D2-M1.
#
# Seeds run host-side against the ports docker-compose publishes, so they use
# the project venv. Override the interpreter if yours lives elsewhere:
#   make seed PYTHON=.venv/bin/python
PYTHON ?= .venv/bin/python

.PHONY: up down logs seed seed-dev serve diagram clean

up:                ## Bring up the 3 data stores (mongo + qdrant + neo4j)
	docker compose up -d

serve:             ## Run the GraphRAG demo (FastAPI + 1-page UI) at http://localhost:$(API_PORT) — loads .env
	set -a; [ -f .env ] && . ./.env; set +a; \
	$(PYTHON) -m uvicorn csai415.api:app --host 0.0.0.0 --port $${API_PORT:-8000}

api:               ## Bring up the stores + FastAPI app (needs D2-B1's api.py)
	docker compose --profile api up -d --build

down:              ## Stop and remove containers (keeps named volumes)
	docker compose down

logs:              ## Tail logs for all services
	docker compose logs -f

seed:              ## Seed all three stores in dependency order (mongo -> qdrant -> neo4j)
	$(PYTHON) scripts/seed_mongo.py --source all
	$(PYTHON) scripts/seed_qdrant.py                    # D3: single collection, source-filtered at query time
	$(PYTHON) scripts/seed_neo4j.py --source all        # reads papers from Mongo — must run after seed_mongo

seed-dev:          ## Seed Neo4j only, from parquet (before Mongo is populated)
	$(PYTHON) scripts/seed_neo4j.py --from-parquet data/processed/chunks.parquet --source arxiv

diagram:           ## Render all reports/D2/*.mmd diagrams to PNG via mermaid-cli
	npx --yes @mermaid-js/mermaid-cli -i reports/D2/d2_dataflow.mmd -o reports/D2/d2_dataflow.png
	npx --yes @mermaid-js/mermaid-cli -i reports/D2/d2_graph_schema.mmd -o reports/D2/d2_graph_schema.png

graph-sample:      ## Render the Author/Paper/Topic subgraph PNG from parquet (no live Neo4j needed)
	$(PYTHON) scripts/render_graph_sample.py

clean:             ## Stop containers AND delete the data volumes (full reset)
	docker compose down -v
