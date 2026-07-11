"""
Build (anchor, positive, negative) triplets from query pairs.

Refactor of notebooks/01_create_triplets.ipynb. Logic preserved: for every
positive (query, document) pair, sample a random different document as the
negative. Seeded for reproducibility.
"""

from __future__ import annotations

import random

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def build_triplets(pairs_df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    cfg = get_config()
    seed = cfg.project.seed if seed is None else seed
    rng = random.Random(seed)

    all_documents = pairs_df["document"].unique().tolist()
    if len(all_documents) < 2:
        raise ValueError("Need at least 2 unique documents to sample negatives.")

    triplets = []
    for _, row in pairs_df.iterrows():
        anchor = row["query"]
        positive = row["document"]

        negative = rng.choice(all_documents)
        while negative == positive:
            negative = rng.choice(all_documents)

        triplets.append({"anchor": anchor, "positive": positive, "negative": negative})

    triplets_df = pd.DataFrame(triplets)
    logger.info("Built %d triplets from %d pairs", len(triplets_df), len(pairs_df))
    return triplets_df


def validate_triplets(triplets_df: pd.DataFrame) -> int:
    """Returns the count of malformed triplets (positive == negative)."""
    wrong = int((triplets_df["positive"] == triplets_df["negative"]).sum())
    if wrong:
        logger.warning("%d triplets have positive == negative", wrong)
    else:
        logger.info("All triplets valid (0 positive == negative collisions)")
    return wrong


def save_triplets(triplets_df: pd.DataFrame, path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("triplets"))
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    triplets_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved triplets to %s", path)
    return path


def run(query_pairs_path: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    query_pairs_path = query_pairs_path or str(cfg.path("query_pairs"))
    pairs_df = pd.read_csv(query_pairs_path)
    triplets_df = build_triplets(pairs_df)
    validate_triplets(triplets_df)
    save_triplets(triplets_df, out_path)
    return triplets_df
