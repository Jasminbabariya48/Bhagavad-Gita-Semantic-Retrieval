"""Hybrid search: dense retrieval + BM25, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from src.config import get_config
from src.logging_utils import get_logger
from src.rrf import FusedDoc, reciprocal_rank_fusion

logger = get_logger(__name__)


class HybridSearch:
    def __init__(self, dense_search=None, bm25_search=None):
        if dense_search is None:
            from src.dense_search import DenseSearch

            dense_search = DenseSearch()
        if bm25_search is None:
            from src.bm25_search import BM25Search

            bm25_search = BM25Search()

        self.dense_search = dense_search
        self.bm25_search = bm25_search

    def search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
        fusion_top_n: int | None = None,
    ) -> list[FusedDoc]:
        cfg = get_config()
        dense_top_k = dense_top_k or cfg.retrieval.dense_top_k
        bm25_top_k = bm25_top_k or cfg.retrieval.bm25_top_k
        rrf_k = rrf_k or cfg.retrieval.rrf_k
        fusion_top_n = fusion_top_n or cfg.retrieval.fusion_top_n

        dense_results = self.dense_search.search(query, top_k=dense_top_k)
        bm25_results = self.bm25_search.search(query, top_k=bm25_top_k)

        ranked_lists = {
            "dense": [d.index for d in dense_results],
            "bm25": [d.index for d in bm25_results],
        }

        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=fusion_top_n)
        logger.debug(
            "Hybrid search '%s': dense=%d bm25=%d fused_top_n=%d",
            query,
            len(dense_results),
            len(bm25_results),
            len(fused),
        )
        return fused
