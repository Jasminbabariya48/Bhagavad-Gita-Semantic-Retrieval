#!/usr/bin/env python
"""Cleaned dataset -> query pairs CSV (used for triplets + evaluation).

Usage:
    python scripts/run_generate_query_pairs.py [--in PATH] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import get_logger, log_pipeline_version
from src.query_pairs import run

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=None, help="Path to cleaned dataset CSV")
    parser.add_argument("--out", default=None, help="Path to write query pairs CSV")
    args = parser.parse_args()

    log_pipeline_version(logger)
    df = run(processed_dataset_path=args.in_path, out_path=args.out)
    logger.info("Done. %d query pairs generated.", len(df))


if __name__ == "__main__":
    main()
