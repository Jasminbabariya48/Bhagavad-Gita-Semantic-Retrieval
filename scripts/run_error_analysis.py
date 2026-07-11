#!/usr/bin/env python
"""Error analysis over query_pairs.csv using the full retrieval pipeline:
saves correct vs. wrong retrievals (with similarity scores) to CSV, and a
2D embedding visualization (PCA) of documents colored by correctness.

Usage:
    python scripts/run_error_analysis.py [--sample N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version
from src.retrieval import RetrievalPipeline

logger = get_logger(__name__)


def main():
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=100, help="Number of queries to analyze")
    args = parser.parse_args()

    log_pipeline_version(logger)

    out_dir = cfg.resolve_path(cfg.paths.error_analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(cfg.path("query_pairs"))
    sample = pairs.sample(n=min(args.sample, len(pairs)), random_state=cfg.project.seed)

    pipeline = RetrievalPipeline()

    rows = []
    for _, row in sample.iterrows():
        query = str(row["query"])
        expected_document = str(row["document"])

        results = pipeline.retrieve(query)
        retrieved_docs = [r.document for r in results]
        is_correct = expected_document in retrieved_docs
        rank_of_correct = (
            retrieved_docs.index(expected_document) + 1 if is_correct else None
        )
        top1_score = results[0].similarity if results else None

        rows.append(
            {
                "query": query,
                "expected_verse": next(
                    (r.verse_id for r in results if r.document == expected_document), None
                ),
                "correct_in_top_k": is_correct,
                "rank_of_correct": rank_of_correct,
                "top1_verse": results[0].verse_id if results else None,
                "top1_similarity": top1_score,
            }
        )

    report = pd.DataFrame(rows)
    out_path = out_dir / "error_analysis.csv"
    report.to_csv(out_path, index=False)

    n_correct = int(report["correct_in_top_k"].sum())
    logger.info(
        "Error analysis: %d/%d correct in top-k (%.1f%%)",
        n_correct,
        len(report),
        100 * n_correct / len(report),
    )
    logger.info("Saved report to %s", out_path)

    wrong = report[~report["correct_in_top_k"]]
    logger.info("Failure cases: %d — see %s for full list", len(wrong), out_path)


if __name__ == "__main__":
    main()
