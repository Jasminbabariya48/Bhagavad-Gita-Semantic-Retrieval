"""Sparse (BM25) retrieval over the document corpus.

Uses rank_bm25's Okapi implementation. Tokenization is intentionally simple
(lowercase + split on non-alphanumeric) so it works reasonably for both the
transliterated Sanskrit and the English translation text without pulling in
a language-specific tokenizer.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredDoc:
    index: int
    score: float


class BM25Search:
    def __init__(self, documents: list[str] | None = None, top_k: int | None = None):
        cfg = get_config()
        self.top_k = top_k or cfg.retrieval.bm25_top_k

        if documents is None:
            from src.faiss_builder import load_documents

            documents = load_documents()
        self.documents = documents

        from rank_bm25 import BM25Okapi

        tokenized_corpus = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents", len(documents))

    def search(self, query: str, top_k: int | None = None) -> list[ScoredDoc]:
        top_k = top_k or self.top_k
        tokenized_query = tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]
        return [ScoredDoc(index=int(i), score=float(scores[i])) for i in ranked_indices]

    def save(self, path: str | None = None) -> str:
        cfg = get_config()
        path = path or str(cfg.path("bm25_index"))
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"bm25": self._bm25, "documents": self.documents}, fh)
        logger.info("Saved BM25 index to %s", path)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "BM25Search":
        cfg = get_config()
        path = path or str(cfg.path("bm25_index"))
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.top_k = cfg.retrieval.bm25_top_k
        obj.documents = data["documents"]
        obj._bm25 = data["bm25"]
        return obj
