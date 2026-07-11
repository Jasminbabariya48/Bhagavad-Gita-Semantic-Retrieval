#!/usr/bin/env python
"""Interactive CLI: ask questions against the full retrieval pipeline.

Usage:
    python scripts/run_query.py
    python scripts/run_query.py --query "How to control the mind?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import get_logger, log_pipeline_version
from src.retrieval import RetrievalPipeline

logger = get_logger(__name__)


def print_results(query: str, results) -> None:
    print()
    print("=" * 100)
    print("QUERY:", query)
    print("=" * 100)
    for r in results:
        print()
        print(f"TOP {r.rank}  |  Verse {r.verse_id}  |  Score: {r.similarity:.4f}")
        print("-" * 80)
        print("Sanskrit:", r.sanskrit)
        print("Translation:", r.translation[:400])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=None, help="Single query to run non-interactively")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    log_pipeline_version(logger)
    pipeline = RetrievalPipeline()

    if args.query:
        results = pipeline.retrieve(args.query, top_k=args.top_k)
        print_results(args.query, results)
        return

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.strip().lower() == "exit":
            break
        results = pipeline.retrieve(query, top_k=args.top_k)
        print_results(query, results)


if __name__ == "__main__":
    main()
