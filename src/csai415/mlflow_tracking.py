"""Solo (Musab) — MLflow experiment tracking around Pair B's Optuna AutoML study.

Replays the already-completed Optuna SQLite study into MLflow, tags the best
run as 'blessed', attaches the runcard + report figures as artifacts, and
exports a Markdown comparison table for the 2-page D1 report.

Replay (not live callback) because WAFIQ's run_and_record() already burned the
80-trial compute and the numbers are committed in configs/winning_runcard.yaml;
re-running with a live callback would risk number drift and ~20 min of waste.

Run all of it: python -m csai415.mlflow_tracking
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import optuna
import yaml

EXPERIMENT_NAME = "csai415-d1-automl"
TRACKING_URI = "sqlite:///mlruns.db"  # sqlite > file:./mlruns for search_runs() speed

STUDY_NAME = "csai415-d1-knn"
STUDY_STORAGE = "sqlite:///studies/csai415-d1-knn.db"

RUNCARD_PATH = Path("configs/winning_runcard.yaml")
PLOTS_DIR = Path("reports")
MARKDOWN_OUT = Path("reports/mlflow_top5.md")

BLESSED_TAG_KEY = "csai415.blessed"
ARTIFACT_PLOT_NAMES = (
    "optimization_history.png",
    "param_importances.png",
    "winner_vs_baselines.png",
)


def setup_experiment(name: str = EXPERIMENT_NAME, tracking_uri: str = TRACKING_URI) -> str:
    """Set the tracking URI, get-or-create the experiment, and make it active so
    later mlflow.search_runs() calls don't fall back to the default experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(name)
    return exp_id


def replay_study_to_mlflow(
    study_storage: str = STUDY_STORAGE,
    dataset_hashes: dict[str, str] | None = None,
) -> list[str]:
    """Iterate the completed SQLite study and log each trial as a nested MLflow run.

    Parent run groups the 80 trials; child runs carry params/metrics/tags.
    Returns the list of child run_ids.
    """
    study = optuna.load_study(study_name=STUDY_NAME, storage=study_storage)
    dataset_hashes = dataset_hashes or {}
    run_ids: list[str] = []

    with mlflow.start_run(run_name="optuna_tpe_study"):
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as child:
                mlflow.log_params(trial.params)
                metrics = {
                    k: v for k, v in trial.user_attrs.items()
                    if k in {"ndcg5", "recall5", "p95_latency_ms"}
                }
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.set_tag("trial_number", str(trial.number))
                if dataset_hashes:
                    mlflow.set_tags(dataset_hashes)
                run_ids.append(child.info.run_id)

    return run_ids


def log_winning_run(
    runcard_path: Path = RUNCARD_PATH,
    plots_dir: Path = PLOTS_DIR,
    study_storage: str = STUDY_STORAGE,
) -> str:
    """Find the MLflow run matching Optuna's best trial, set the 'blessed' tag,
    and attach the runcard + report PNGs as artifacts.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    study = optuna.load_study(study_name=STUDY_NAME, storage=study_storage)
    best_trial = study.best_trial

    runs = mlflow.search_runs(filter_string=f"tags.trial_number = '{best_trial.number}'")
    if runs.empty:
        raise ValueError(
            f"No MLflow run found for best trial #{best_trial.number}. "
            "Did replay_study_to_mlflow run first?"
        )
    best_run_id = runs.iloc[0].run_id

    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag(BLESSED_TAG_KEY, "true")
        if runcard_path.exists():
            mlflow.log_artifact(str(runcard_path))
        for plot_name in ARTIFACT_PLOT_NAMES:
            plot_path = plots_dir / plot_name
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path))

    return best_run_id


def export_comparison_table(top_n: int = 5, out_path: Path = MARKDOWN_OUT) -> Path:
    """Top-N MLflow runs by NDCG@5 -> markdown table for the D1 report."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(order_by=["metrics.ndcg5 DESC"])
    # Filter out the parent run (it has no metrics) by dropping NaN ndcg5
    runs_df = runs_df.dropna(subset=["metrics.ndcg5"]).head(top_n)

    runs_df["run_id"] = runs_df["run_id"].str[:7]
    runs_df["metrics.ndcg5"] = runs_df["metrics.ndcg5"].round(4)
    runs_df["metrics.recall5"] = runs_df["metrics.recall5"].round(4)
    runs_df["metrics.p95_latency_ms"] = runs_df["metrics.p95_latency_ms"].round(1)

    cols = {
        "run_id": "Run ID",
        "metrics.ndcg5": "NDCG@5",
        "metrics.recall5": "Recall@5",
        "metrics.p95_latency_ms": "p95 Latency (ms)",
        "params.candidate_k": "candidate_k",
        "params.metric": "Metric",
        "params.svd_dim": "SVD",
        "params.hybrid_weight": "Hybrid Wt.",
    }
    table = runs_df[list(cols)].rename(columns=cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table.to_markdown(index=False), encoding="utf-8")
    return out_path


def main() -> None:
    """Orchestration: replay study -> bless winner -> export top-5 table."""
    setup_experiment()

    with RUNCARD_PATH.open(encoding="utf-8") as f:
        card = yaml.safe_load(f)
    dataset_hashes = {
        "dataset.chunks_sha256": card["dataset"]["chunks_sha256"],
        "dataset.gold_sha256": card["dataset"]["gold_sha256"],
    }

    print("Replaying Optuna study to MLflow...")
    run_ids = replay_study_to_mlflow(dataset_hashes=dataset_hashes)
    print(f"  logged {len(run_ids)} completed trials")

    print("Tagging blessed run + attaching artifacts...")
    best = log_winning_run()
    print(f"  blessed run_id={best}")

    print("Exporting top-5 comparison table...")
    out = export_comparison_table()
    print(f"  wrote {out}")

    print("\nDone. UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == "__main__":
    main()
