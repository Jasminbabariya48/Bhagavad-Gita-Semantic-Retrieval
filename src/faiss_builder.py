from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import os

from src.config import get_config
from src.logging_utils import get_logger
import faiss

logger = get_logger(__name__)


def build_index(embeddings: np.ndarray):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("Built FAISS IndexFlatIP: %d vectors, dim=%d", index.ntotal, dimension)
    return index


def save_index(index, path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("faiss_index"))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    logger.info("Saved FAISS index to %s", path)
    return path


def load_index(path: str | None = None):
    cfg = get_config()
    path = path or str(cfg.path("faiss_index"))
    logger.info("Loading FAISS index from %s", path)
    return faiss.read_index(path)


def save_metadata(df: pd.DataFrame, path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("metadata"))
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [cfg.dataset.id_column, cfg.dataset.sanskrit_column, cfg.dataset.translation_column]
    df[cols].to_pickle(path)
    logger.info("Saved metadata (%d rows) to %s", len(df), path)
    return path


def load_metadata(path: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    path = path or str(cfg.path("metadata"))
    return pd.read_pickle(path)


def save_documents(documents: list[str], path: str | None = None) -> str:
    cfg = get_config()
    path = path or str(cfg.path("documents"))
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(documents, fh)
    logger.info("Saved %d documents to %s", len(documents), path)
    return path


def load_documents(path: str | None = None) -> list[str]:
    cfg = get_config()
    path = path or str(cfg.path("documents"))
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_index_from_dataset(
    df: pd.DataFrame | None = None,
    embedding_model=None,
) -> tuple:
    """End-to-end: read the processed dataset, embed the `document` column,
    build + save the FAISS index, and persist metadata/documents alongside
    it so dense_search.py can load everything by convention."""
    cfg = get_config()

    if df is None:
        df = pd.read_csv(cfg.path("processed_dataset"))

    if embedding_model is None:
        from src.embedding import EmbeddingModel

        embedding_model = EmbeddingModel()

    doc_col = cfg.dataset.document_column
    docs = df[doc_col].astype(str).tolist()

    logger.info("Encoding %d documents for FAISS index", len(docs))
    embeddings = embedding_model.encode_documents(docs)

    index = build_index(embeddings)
    save_index(index)
    save_metadata(df)
    save_documents(docs)

    return index, df, docs
