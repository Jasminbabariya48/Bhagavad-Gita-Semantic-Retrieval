"""
Generate (query, document, label) training pairs from the cleaned dataset.

Refactor of notebooks/02_create_query_pairs.ipynb — logic is preserved
(same query templates, same use of the first sentence of the translation)
but parameterized and testable instead of hardcoded to a local Windows path.
"""

from __future__ import annotations

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


def generate_queries_for_row(verse_id: str, translation: str) -> list[str]:
    """English query templates for a single verse (mirrors the notebook)."""
    return [
        f"What is explained in verse {verse_id}?",
        "What does this verse say?",
        f"Explain verse {verse_id}",
        f"Meaning of verse {verse_id}",
        translation[:120],
    ]


def build_query_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """df must have the columns configured under `dataset:` in config.yaml
    (id/sanskrit/translation/document). Returns a shuffled DataFrame with
    columns [query, document, label]."""
    cfg = get_config()
    id_col = cfg.dataset.id_column
    sk_col = cfg.dataset.sanskrit_column
    tr_col = cfg.dataset.translation_column
    doc_col = cfg.dataset.document_column

    pairs: list[dict] = []
    for _, row in df.iterrows():
        verse_id = row[id_col]
        sanskrit = row[sk_col]
        translation = str(row[tr_col])
        document = row[doc_col]

        for q in generate_queries_for_row(str(verse_id), translation):
            pairs.append({"query": q, "document": document, "label": 1})

        # Sanskrit-language query for the same document
        pairs.append({"query": sanskrit, "document": document, "label": 1})

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sample(frac=1, random_state=cfg.project.seed).reset_index(
        drop=True
    )
    logger.info("Generated %d query-document pairs from %d verses", len(pairs_df), len(df))
    return pairs_df


def save_query_pairs(pairs_df: pd.DataFrame, path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("query_pairs"))
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pairs_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved query pairs to %s", path)
    return path


def run(processed_dataset_path: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    processed_dataset_path = processed_dataset_path or str(cfg.path("processed_dataset"))
    df = pd.read_csv(processed_dataset_path)
    pairs_df = build_query_pairs(df)
    save_query_pairs(pairs_df, out_path)
    return pairs_df
