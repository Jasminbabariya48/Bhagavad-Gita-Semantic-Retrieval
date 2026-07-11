#!/usr/bin/env python
"""Query pairs -> (anchor, positive, negative) triplets CSV for training.

Usage:
    python scripts/run_generate_triplets.py [--in PATH] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import get_logger, log_pipeline_version
from src.triplet_generation import run

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=None, help="Path to query pairs CSV")
    parser.add_argument("--out", default=None, help="Path to write triplets CSV")
    args = parser.parse_args()

    log_pipeline_version(logger)
    df = run(query_pairs_path=args.in_path, out_path=args.out)
    logger.info("Done. %d triplets generated.", len(df))


if __name__ == "__main__":
    main()
