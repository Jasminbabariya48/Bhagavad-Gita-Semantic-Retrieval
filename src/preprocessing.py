"""
Raw dataset -> cleaned dataset.

The notebooks the user supplied assume a `cleaned_dataset.csv` already
exists (with columns unique_key / sanskrit_data / translation / document)
but the notebook that produces it wasn't in the export. This module
reconstructs that step as a proper, testable pipeline so the project has no
missing link: raw CSV -> preprocess_dataset() -> cleaned_dataset.csv.

If your raw export already matches the target schema, this still runs
safely — cleaning is idempotent (null-dropping, whitespace/unicode
normalization, dedup) and a no-op on already-clean data.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_RAW_COLUMNS = {"unique_key", "sanskrit_data", "translation"}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: object) -> str:
    """Normalize a single text field: unicode NFC form, collapsed whitespace,
    stripped stray control characters. Returns "" for null/NaN input."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\x00", "")
    return _normalize_whitespace(text)


def load_raw_dataset(path: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    path = path or str(cfg.path("raw_dataset"))
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)

    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw dataset is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields, drop nulls/dupes, and build the `document` column
    (sanskrit + translation) that the embedding/BM25/FAISS steps index."""
    cfg = get_config()
    id_col = cfg.dataset.id_column
    sk_col = cfg.dataset.sanskrit_column
    tr_col = cfg.dataset.translation_column
    doc_col = cfg.dataset.document_column

    df = df.copy()
    n_before = len(df)

    for col in ("unique_key", "sanskrit_data", "translation"):
        df[col] = df[col].apply(clean_text)

    df = df[(df["sanskrit_data"] != "") & (df["translation"] != "")]
    df = df.drop_duplicates(subset=["unique_key"])
    df = df[df["unique_key"] != ""]

    df = df.rename(
        columns={
            "unique_key": id_col,
            "sanskrit_data": sk_col,
            "translation": tr_col,
        }
    )

    df[doc_col] = (
        df[sk_col].astype(str) + "\n" + df[tr_col].astype(str)
    ).apply(_normalize_whitespace)

    df = df.reset_index(drop=True)

    n_after = len(df)
    logger.info(
        "Preprocessing complete: %d -> %d rows (%d dropped as null/duplicate/empty)",
        n_before,
        n_after,
        n_before - n_after,
    )
    return df[[id_col, sk_col, tr_col, doc_col]]


def save_processed(df: pd.DataFrame, path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("processed_dataset"))
    out_path = path
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved processed dataset (%d rows) to %s", len(df), out_path)
    return out_path


def run(raw_path: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    """End-to-end: load raw -> clean -> save. Returns the cleaned DataFrame."""
    df_raw = load_raw_dataset(raw_path)
    df_clean = preprocess_dataset(df_raw)
    save_processed(df_clean, out_path)
    return df_clean
