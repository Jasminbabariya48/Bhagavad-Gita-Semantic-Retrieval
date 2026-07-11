#!/usr/bin/env python
"""Encode the processed dataset and build the FAISS + BM25 indexes.

Usage:
    python scripts/run_build_index.py [--model NAME]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.faiss_builder import build_index_from_dataset
from src.bm25_search import BM25Search
from src.logging_utils import get_logger, log_pipeline_version

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Embedding model to encode with (default: config)")
    args = parser.parse_args()

    log_pipeline_version(logger)

    embedding_model = None
    if args.model:
        from src.embedding import EmbeddingModel

        embedding_model = EmbeddingModel(model_name_or_path=args.model)

    index, df, docs = build_index_from_dataset(embedding_model=embedding_model)
    logger.info("FAISS index built: %d vectors", index.ntotal)

    bm25 = BM25Search(documents=docs)
    bm25.save()
    logger.info("BM25 index built and saved.")


if __name__ == "__main__":
    main()
