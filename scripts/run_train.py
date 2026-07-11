#!/usr/bin/env python
"""Fine-tune the embedding model on triplets.csv, with MLflow logging.

Usage:
    python scripts/run_train.py [--model NAME] [--epochs N] [--batch-size N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version
from src.mlflow_utils import log_artifact_if_exists, mlflow_run
from src.trainer import train_embedding_model

logger = get_logger(__name__)


def main():
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Base model name (default: config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loss", default=None, choices=["MultipleNegativesRankingLoss", "TripletLoss"])
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    log_pipeline_version(logger)

    model_name = args.model or cfg.embedding.active_model
    run_name = f"train_{model_name.replace('/', '__')}"

    if args.no_mlflow:
        result = train_embedding_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_name=args.loss,
        )
        logger.info("Training complete: %s", result)
        return

    params = {
        "model_name": model_name,
        "epochs": args.epochs or cfg.training.epochs,
        "batch_size": args.batch_size or cfg.training.batch_size,
        "loss": args.loss or cfg.training.loss,
    }
    with mlflow_run(run_name, params=params, tags={"stage": "training"}) as mlflow:
        result = train_embedding_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_name=args.loss,
        )
        mlflow.log_metric("num_training_examples", result.num_examples)
        mlflow.set_tag("output_path", result.output_path)

    logger.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
