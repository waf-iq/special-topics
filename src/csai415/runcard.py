"""Pair A — Writes the YAML run-card capturing the winning Optuna config produced by Pair B. See MEMBER_BRIEF.md §6.B acceptance bar."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RUNCARD_PATH = Path("configs/winning_runcard.yaml")

DEFAULT_SAMPLER = "TPESampler"
DEFAULT_PRUNER = "MedianPruner"

ENV_PACKAGES = ["optuna", "numpy", "pandas", "scikit-learn", "sentence-transformers"]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"sha": sha, "dirty": dirty}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def _env_info() -> dict:
    packages = {}
    for pkg in ENV_PACKAGES:
        try:
            packages[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            packages[pkg] = None
    return {
        "python": sys.version.split()[0],
        "packages": packages,
    }


def write_runcard(
    best_params: dict[str, Any],
    best_value: float,
    n_trials: int,
    embedding_model: str,
    chunks_parquet: Path,
    gold_jsonl: Path,
    metrics: dict[str, float],
    out_path: Path = RUNCARD_PATH,
    *,
    split: dict | None = None,
    sampler_config: dict | None = None,
    pruner_config: dict | None = None,
    study_storage: str | None = None,
    notes: str | None = None,
    # v3 (D1 rework — B3): records which HPO method's winner is blessed +
    # the full 5-method comparison so the marker can read it without opening
    # sampler_comparison.csv. All optional; if every v3 field is None the
    # runcard still has the new schema_version but no rework-specific data.
    blessed_method: str | None = None,
    comparison_table: list[dict[str, Any]] | None = None,
    comparison_csv: Path | str | None = None,
) -> Path:
    """Write a fully reproducible run-card.

    metrics should carry both tune and holdout numbers with suffixed keys
    (e.g. ndcg5_tune, ndcg5_holdout, p95_latency_ms_tune, ...).

    Schema is v3 from the D1 rework onward (see [[project-b1-blessed-winner]]).
    The rework-only fields (`blessed_method`, `comparison_table`,
    `comparison_csv`) live under `automl.*`. Callers writing pre-rework
    single-method runcards just leave them None.
    """
    automl_section: dict[str, Any] = {
        "library": "optuna",
        "sampler": sampler_config if sampler_config is not None else DEFAULT_SAMPLER,
        "pruner": pruner_config if pruner_config is not None else DEFAULT_PRUNER,
        "storage": study_storage,
        "n_trials": n_trials,
        "best_value_ndcg5_tune": best_value,
        "best_params": best_params,
    }
    if blessed_method is not None:
        automl_section["blessed_method"] = blessed_method
    if comparison_table is not None:
        automl_section["comparison"] = comparison_table
    if comparison_csv is not None:
        # Normalize to forward-slash so the runcard is portable across OSes.
        # (The pre-existing v2 dataset paths still use the native separator —
        # cleanup tracked separately.)
        automl_section["comparison_csv"] = Path(comparison_csv).as_posix()

    card = {
        "schema_version": "3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deliverable": "D1",
        "track": "A-supervised-knn",
        "seed": 42,
        "embedding": {"model": embedding_model, "dim": 384},
        "dataset": {
            # POSIX-normalize so the same runcard renders identically on
            # Windows and Unix (and its SHA — if anyone hashes it — matches).
            "chunks_parquet": Path(chunks_parquet).as_posix(),
            "chunks_sha256": _hash_file(chunks_parquet) if chunks_parquet.exists() else None,
            "gold_jsonl": Path(gold_jsonl).as_posix(),
            "gold_sha256": _hash_file(gold_jsonl) if gold_jsonl.exists() else None,
        },
        "code": _git_info(),
        "env": _env_info(),
        "split": split,
        "automl": automl_section,
        "metrics": metrics,
        "notes": notes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(card, f, sort_keys=False)
    return out_path
