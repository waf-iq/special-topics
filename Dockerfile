# D2-M1 — image for the FastAPI `/search` app (D2-B1).
# Built by the `api` service in docker-compose.yml. Runs uvicorn against
# csai415.api:app. Heavy deps (torch, sentence-transformers) are pulled because
# the API embeds the query at request time with BGE-small.
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; build tools only if a wheel is missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Blessed BOHB config is loaded at startup from configs/winning_runcard.yaml by
# the app's lifespan hook (D2-B1), never from the request body.
CMD ["uvicorn", "csai415.api:app", "--host", "0.0.0.0", "--port", "8000"]
