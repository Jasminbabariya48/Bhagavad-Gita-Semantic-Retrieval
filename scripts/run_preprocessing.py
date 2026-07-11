#!/usr/bin/env python
"""Raw dataset -> cleaned dataset.

Usage:
    python scripts/run_preprocessing.py [--raw PATH] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import get_logger, log_pipeline_version
from src.preprocessing import run

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=None, help="Path to raw dataset CSV")
    parser.add_argument("--out", default=None, help="Path to write cleaned dataset CSV")
    args = parser.parse_args()

    log_pipeline_version(logger)
    df = run(raw_path=args.raw, out_path=args.out)
    logger.info("Done. %d rows in cleaned dataset.", len(df))


if __name__ == "__main__":
    main()
