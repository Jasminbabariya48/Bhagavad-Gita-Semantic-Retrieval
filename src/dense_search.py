"""Dense (embedding + FAISS) retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredDoc:
    index: int
    score: float


class DenseSearch:
    """Loads a FAISS index + embedding model and searches by cosine similarity
    (index is built over L2-normalized vectors, so inner product == cosine)."""

    def __init__(self, embedding_model=None, index=None, top_k: int | None = None):
        cfg = get_config()
        self.top_k = top_k or cfg.retrieval.dense_top_k

        if embedding_model is None:
            from src.embedding import EmbeddingModel

            embedding_model = EmbeddingModel()
        self.embedding_model = embedding_model

        if index is None:
            from src.faiss_builder import load_index

            index = load_index()
        self.index = index

    def search(self, query: str, top_k: int | None = None) -> list[ScoredDoc]:
        top_k = top_k or self.top_k
        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = np.asarray([query_embedding], dtype=np.float32)

        import faiss

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results = [
            ScoredDoc(index=int(idx), score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]
        return results
