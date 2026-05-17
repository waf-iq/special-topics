"""Solo (Musab) — MLflow experiment tracking around Pair B's Optuna study.

Integrates from the outside: Pair B (WAFIQ) passes the callback from
`make_mlflow_callback()` into `run_study(callbacks=[cb])`. After the study
finishes, `log_winning_run()` tags the best trial and attaches artifacts
(runcard YAML, optimization-history PNG, parameter-importance PNG).

See MEMBER_BRIEF.md §6.D for the full slice spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import optuna

EXPERIMENT_NAME = "csai415-d1-automl"
TRACKING_URI = "file:./mlruns"           # local file backend; sqlite:///mlruns.db also fine

BLESSED_TAG_KEY = "csai415.blessed"      # tag set on the best trial after study completes
DATASET_HASH_TAG_KEY = "csai415.dataset_sha256_16"


def setup_experiment(name: str = EXPERIMENT_NAME, tracking_uri: str = TRACKING_URI) -> str:
    """Set tracking URI and create-or-get the experiment. Returns experiment_id.

    See §6.D Q1 for choice of file:// vs sqlite:// backend.
    """
    raise NotImplementedError("Musab — minimal: mlflow.set_tracking_uri + mlflow.set_experiment.")


def make_mlflow_callback(metric_name: str = "ndcg5", experiment_name: str = EXPERIMENT_NAME) -> Callable:
    """Return an Optuna callback that logs every trial to MLflow.

    Optuna ships `optuna.integration.mlflow.MLflowCallback` — decide whether to use it
    or write your own. See §6.D Q2 for what the built-in misses (e.g., dataset hashes,
    artifact logging, run grouping).
    """
    raise NotImplementedError("Musab — start with optuna.integration.mlflow.MLflowCallback; extend if needed.")


def log_winning_run(
    study: optuna.Study,
    runcard_path: Path,
    dataset_hashes: dict[str, str],
    plots_dir: Path | None = None,
) -> str:
    """After study finishes: find the best trial's MLflow run, set the 'blessed' tag,
    attach the runcard YAML + Optuna plots as artifacts, log dataset hashes as tags.
    Returns the blessed run's run_id.

    See §6.D Q3 for the 'blessed run' pattern.
    """
    raise NotImplementedError("Musab — mlflow.search_runs() to find the best, then mlflow.set_tag + log_artifact.")


def export_comparison_table(top_n: int = 5, out_path: Path = Path("reports/mlflow_top5.md")) -> Path:
    """Pull top-N runs from MLflow by NDCG@5, write a markdown comparison table for the report.

    Columns suggestion: rank, run_id (short), ndcg5, recall5, p95_latency_ms,
    k, metric, svd_dim, normalize, hybrid_weight. See §6.D Q4.
    """
    raise NotImplementedError("Musab — mlflow.search_runs(order_by=['metrics.ndcg5 DESC']).head(top_n).")
