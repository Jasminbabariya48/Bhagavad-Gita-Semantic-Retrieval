"""MLflow experiment tracking helpers.

Every model-comparison / training run should go through `mlflow_run()` so
params, metrics, and artifacts are logged consistently in one place instead
of each script hand-rolling its own `mlflow.start_run()` block.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def configure_mlflow(experiment_name: str | None = None):
    import mlflow

    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name or cfg.mlflow.experiment_name)
    return mlflow


@contextmanager
def mlflow_run(
    run_name: str,
    params: dict | None = None,
    tags: dict | None = None,
    experiment_name: str | None = None,
):
    """Usage:

        with mlflow_run("e5-small-run", params=config, tags={"dataset": "gita"}) as mlflow:
            ...
            mlflow.log_metric("Recall@1", 0.81)
            mlflow.log_artifact("outputs/eval.csv")
    """
    mlflow = configure_mlflow(experiment_name)
    try:
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)
            yield mlflow
        logger.info("MLflow run '%s' logged successfully", run_name)
    except Exception as exc:  # pragma: no cover - network/server dependent
        logger.warning(
            "MLflow logging failed (%s). Continuing without tracking — "
            "is the MLflow server running at the configured tracking_uri?",
            exc,
        )


def log_metrics_dict(mlflow_module, metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        mlflow_module.log_metric(k.replace("@", "_at_"), v)


def log_artifact_if_exists(mlflow_module, path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        mlflow_module.log_artifact(str(path))
    else:
        logger.warning("Artifact not found, skipping log_artifact: %s", path)
