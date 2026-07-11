"""
The end-to-end retrieval pipeline:

    User Query
      -> preprocessing (whitespace/unicode normalize)
      -> query expansion
      -> dense embedding + FAISS search (top 50)   \
                                                      -> Reciprocal Rank Fusion -> top 20
      -> BM25 search (top 50)                       /
      -> Cross-encoder re-ranking -> top 5
      -> formatted results (verse ID, similarity, Sanskrit, translation)

This is the single object app/streamlit_app.py and any API layer should
import — it owns wiring together every other src/ module so callers never
have to assemble dense/bm25/rrf/reranker themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version
from src.preprocessing import clean_text

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    verse_id: str
    sanskrit: str
    translation: str
    document: str
    similarity: float
    rank: int
    stage_scores: dict = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(
        self,
        embedding_model=None,
        dense_search=None,
        bm25_search=None,
        hybrid_search=None,
        reranker=None,
        query_expander=None,
        metadata=None,
    ):
        cfg = get_config()
        log_pipeline_version(logger)

        from src.embedding import EmbeddingModel

        self.embedding_model = embedding_model or EmbeddingModel()

        from src.dense_search import DenseSearch

        self.dense_search = dense_search or DenseSearch(embedding_model=self.embedding_model)

        from src.bm25_search import BM25Search

        self.bm25_search = bm25_search or BM25Search()

        from src.hybrid_search import HybridSearch

        self.hybrid_search = hybrid_search or HybridSearch(
            dense_search=self.dense_search, bm25_search=self.bm25_search
        )

        from src.reranker import CrossEncoderReranker

        self.reranker = reranker or CrossEncoderReranker()

        from src.query_expansion import QueryExpander

        self.query_expander = query_expander or QueryExpander()

        if metadata is None:
            from src.faiss_builder import load_metadata

            metadata = load_metadata()
        self.metadata = metadata.reset_index(drop=True)

        self.documents = self.bm25_search.documents
        self.cfg = cfg

    def retrieve(
        self,
        query: str,
        use_query_expansion: bool | None = None,
        use_reranker: bool | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        cfg = self.cfg
        top_k = top_k or cfg.retrieval.final_top_k
        use_query_expansion = (
            cfg.query_expansion.enabled if use_query_expansion is None else use_query_expansion
        )
        use_reranker = cfg.reranker.enabled if use_reranker is None else use_reranker

        clean_query = clean_text(query)
        if not clean_query:
            return []

        search_query = clean_query
        if use_query_expansion:
            search_query = self.query_expander.expand(clean_query)

        fused = self.hybrid_search.search(search_query)
        candidate_indices = [d.index for d in fused]
        candidate_docs = [self.documents[i] for i in candidate_indices]
        fused_scores = {d.index: d.rrf_score for d in fused}

        if use_reranker and candidate_indices:
            from src.reranker import CrossEncoderReranker

            reranker = self.reranker if isinstance(self.reranker, CrossEncoderReranker) else self.reranker
            reranked = reranker.rerank(clean_query, candidate_indices, candidate_docs, top_k=top_k)
            final = [(r.index, r.rerank_score) for r in reranked]
        else:
            final = [(idx, fused_scores[idx]) for idx in candidate_indices[:top_k]]

        id_col = cfg.dataset.id_column
        sk_col = cfg.dataset.sanskrit_column
        tr_col = cfg.dataset.translation_column

        results = []
        for rank, (idx, score) in enumerate(final, start=1):
            row = self.metadata.iloc[idx]
            results.append(
                RetrievalResult(
                    verse_id=str(row[id_col]),
                    sanskrit=str(row[sk_col]),
                    translation=str(row[tr_col]),
                    document=self.documents[idx],
                    similarity=float(score),
                    rank=rank,
                    stage_scores={"rrf_score": fused_scores.get(idx)},
                )
            )
        return results
