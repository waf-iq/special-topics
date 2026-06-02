# D2-M1 — one-command stack + seed helpers. See D2_TASKS.md §D2-M1.
#
# Seeds run host-side against the ports docker-compose publishes, so they use
# the project venv. Override the interpreter if yours lives elsewhere:
#   make seed PYTHON=.venv/bin/python
PYTHON ?= .venv/bin/python

.PHONY: up down logs seed seed-dev diagram clean

up:                ## Bring up the 3 data stores (mongo + qdrant + neo4j)
	docker compose up -d

api:               ## Bring up the stores + FastAPI app (needs D2-B1's api.py)
	docker compose --profile api up -d --build

down:              ## Stop and remove containers (keeps named volumes)
	docker compose down

logs:              ## Tail logs for all services
	docker compose logs -f

seed:              ## Seed all three stores in dependency order (mongo -> qdrant -> neo4j)
	$(PYTHON) scripts/seed_mongo.py --source all
	$(PYTHON) scripts/seed_qdrant.py --source all
	$(PYTHON) scripts/seed_qdrant.py --source scifact   # D2-INT2: per-source collection
	$(PYTHON) scripts/seed_qdrant.py --source arxiv     # D2-INT2: per-source collection
	$(PYTHON) scripts/seed_neo4j.py --source all

seed-dev:          ## Seed Neo4j only, from parquet (before Mongo is populated)
	$(PYTHON) scripts/seed_neo4j.py --from-parquet data/processed/chunks.parquet --source arxiv

diagram:           ## Render the dataflow diagram PNG from the .mmd source
	npx --yes @mermaid-js/mermaid-cli -i reports/d2_dataflow.mmd -o reports/d2_dataflow.png

clean:             ## Stop containers AND delete the data volumes (full reset)
	docker compose down -v
