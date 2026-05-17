"""Pair A — Writes the YAML run-card capturing the winning Optuna config produced by Pair B. See MEMBER_BRIEF.md §6.B acceptance bar."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RUNCARD_PATH = Path("configs/winning_runcard.yaml")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_runcard(
    best_params: dict[str, Any],
    best_value: float,
    n_trials: int,
    embedding_model: str,
    chunks_parquet: Path,
    gold_jsonl: Path,
    metrics: dict[str, float],
    out_path: Path = RUNCARD_PATH,
) -> Path:
    """Write a fully reproducible run-card. See §6.B question 6 for required fields."""
    card = {
        "schema_version": "1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deliverable": "D1",
        "track": "A-supervised-knn",
        "seed": 42,
        "embedding": {"model": embedding_model, "dim": 384},
        "dataset": {
            "chunks_parquet": str(chunks_parquet),
            "chunks_sha256_16": _hash_file(chunks_parquet) if chunks_parquet.exists() else None,
            "gold_jsonl": str(gold_jsonl),
            "gold_sha256_16": _hash_file(gold_jsonl) if gold_jsonl.exists() else None,
        },
        "automl": {
            "library": "optuna",
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
            "n_trials": n_trials,
            "best_value_ndcg5": best_value,
            "best_params": best_params,
        },
        "metrics": metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(card, f, sort_keys=False)
    return out_path
