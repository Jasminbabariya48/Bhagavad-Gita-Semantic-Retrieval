"""Cross-encoder re-ranking of hybrid-search candidates.

Takes the top-N fused (dense + BM25) candidates and re-scores each
(query, document) pair jointly with a CrossEncoder — much more accurate
than bi-encoder similarity for final ranking, at the cost of being too slow
to run over the full corpus (hence: retrieve broad, rerank narrow).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RerankedDoc:
    index: int
    rerank_score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        cfg = get_config()
        self.enabled = cfg.reranker.enabled
        self.model_name = model_name or cfg.reranker.model_name
        self.batch_size = batch_size or cfg.reranker.batch_size
        self._model = None  # lazy-loaded

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker '%s'", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidate_indices: list[int],
        candidate_documents: list[str],
        top_k: int | None = None,
    ) -> list[RerankedDoc]:
        cfg = get_config()
        top_k = top_k or cfg.retrieval.final_top_k

        if not self.enabled or not candidate_indices:
            return [
                RerankedDoc(index=idx, rerank_score=float(len(candidate_indices) - i))
                for i, idx in enumerate(candidate_indices[:top_k])
            ]

        model = self._ensure_loaded()
        pairs = [[query, doc] for doc in candidate_documents]
        scores = model.predict(pairs, batch_size=self.batch_size)

        scored = list(zip(candidate_indices, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RerankedDoc(index=int(idx), rerank_score=float(score))
            for idx, score in scored[:top_k]
        ]
