#!/usr/bin/env python
"""Compare all embedding candidates in config.yaml (embedding.candidates)
on the same query_pairs.csv, logging each as its own MLflow run under the
comparison experiment, and print a leaderboard.

Usage:
    python scripts/run_compare_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version
from src.mlflow_utils import mlflow_run
from scripts.run_evaluate import evaluate_model

logger = get_logger(__name__)


def main():
    cfg = get_config()
    log_pipeline_version(logger)

    out_dir = cfg.resolve_path(cfg.paths.evaluation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_rows = []

    for model_name in cfg.embedding.candidates:
        logger.info("=" * 80)
        logger.info("Evaluating candidate: %s", model_name)
        try:
            with mlflow_run(
                f"compare_{model_name.replace('/', '__')}",
                params={"model_name": model_name},
                tags={"stage": "comparison"},
                experiment_name=cfg.mlflow.comparison_experiment_name,
            ) as mlflow:
                result, resolved_name = evaluate_model(model_name)
                df = result.to_dataframe()

                for _, row in df.iterrows():
                    mlflow.log_metric(str(row["Metric"]).replace("@", "_at_"), float(row["Score"]))

                row_dict = {"model": resolved_name}
                row_dict.update(dict(zip(df["Metric"], df["Score"])))
                leaderboard_rows.append(row_dict)
        except Exception as exc:
            logger.error("Failed to evaluate '%s': %s", model_name, exc)

    leaderboard = pd.DataFrame(leaderboard_rows)
    out_path = out_dir / "model_comparison_leaderboard.csv"
    leaderboard.to_csv(out_path, index=False)

    print("\n=== Model Comparison Leaderboard ===")
    print(leaderboard.to_string(index=False))
    logger.info("Saved leaderboard to %s", out_path)


if __name__ == "__main__":
    main()
