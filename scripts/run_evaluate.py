#!/usr/bin/env python
"""Evaluate an embedding model on query_pairs.csv: Recall@K, MRR, nDCG, MAP,
Hit Rate, and query latency. Optionally logs everything to MLflow.

Usage:
    python scripts/run_evaluate.py [--model NAME] [--no-mlflow]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_config
from src.evaluator import evaluate_similarity_matrix, measure_query_latency
from src.logging_utils import get_logger, log_pipeline_version
from src.mlflow_utils import mlflow_run

logger = get_logger(__name__)


def evaluate_model(model_name: str | None = None, pairs_path: str | None = None):
    cfg = get_config()
    from src.embedding import EmbeddingModel

    embedding_model = EmbeddingModel(model_name_or_path=model_name)

    pairs_path = pairs_path or str(cfg.path("query_pairs"))
    pairs = pd.read_csv(pairs_path)

    queries = pairs["query"].astype(str).tolist()
    documents = pairs["document"].astype(str).tolist()

    logger.info("Encoding %d queries and %d documents", len(queries), len(documents))
    query_embeddings = embedding_model.encode_queries(queries)
    document_embeddings = embedding_model.encode_documents(documents)

    similarity_matrix = cosine_similarity(query_embeddings, document_embeddings)

    mean_ms, p95_ms = measure_query_latency(
        embedding_model.encode_query, queries[: min(50, len(queries))]
    )

    result = evaluate_similarity_matrix(
        similarity_matrix,
        k_values=cfg.evaluation.k_values,
        ndcg_k=cfg.evaluation.ndcg_k,
        map_k=cfg.evaluation.map_k,
        latency_ms_mean=mean_ms,
        latency_ms_p95=p95_ms,
    )
    return result, embedding_model.model_name_or_path


def main():
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Embedding model to evaluate")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    log_pipeline_version(logger)

    out_dir = cfg.resolve_path(cfg.paths.evaluation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_mlflow:
        result, model_name = evaluate_model(args.model)
        df = result.to_dataframe()
        out_path = out_dir / f"eval_{model_name.replace('/', '__')}.csv"
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        logger.info("Saved evaluation results to %s", out_path)
        return

    model_name = args.model or cfg.embedding.active_model
    with mlflow_run(
        f"eval_{model_name.replace('/', '__')}",
        params={"model_name": model_name},
        tags={"stage": "evaluation"},
    ) as mlflow:
        result, resolved_model_name = evaluate_model(args.model)
        df = result.to_dataframe()
        out_path = out_dir / f"eval_{resolved_model_name.replace('/', '__')}.csv"
        df.to_csv(out_path, index=False)

        for _, row in df.iterrows():
            mlflow.log_metric(str(row["Metric"]).replace("@", "_at_"), float(row["Score"]))
        mlflow.log_artifact(str(out_path))

    print(df.to_string(index=False))
    logger.info("Saved evaluation results to %s", out_path)


if __name__ == "__main__":
    main()
