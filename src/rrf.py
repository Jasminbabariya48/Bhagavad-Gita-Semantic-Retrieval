"""Reciprocal Rank Fusion (RRF).

Fuses multiple ranked lists (e.g. dense + BM25) into a single ranking
without needing to normalize/calibrate each retriever's raw scores against
each other — RRF only looks at *rank position*, which is what makes it a
robust default for hybrid search.

    RRF_score(doc) = sum over rankers r of  1 / (k + rank_r(doc))

Standard choice is k=60 (from the original RRF paper); exposed via config.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class FusedDoc:
    index: int
    rrf_score: float
    source_ranks: dict[str, int]  # ranker name -> 1-indexed rank (if present)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedDoc]:
    """
    ranked_lists: mapping of ranker name -> list of document indices, already
                  sorted best-first (rank 1 = ranked_lists[name][0]).
    k: RRF damping constant.
    top_n: if set, truncate the fused ranking to the top N results.
    """
    scores: dict[int, float] = defaultdict(float)
    source_ranks: dict[int, dict[str, int]] = defaultdict(dict)

    for ranker_name, doc_indices in ranked_lists.items():
        for rank, doc_idx in enumerate(doc_indices, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
            source_ranks[doc_idx][ranker_name] = rank

    fused = [
        FusedDoc(index=doc_idx, rrf_score=score, source_ranks=source_ranks[doc_idx])
        for doc_idx, score in scores.items()
    ]
    fused.sort(key=lambda d: d.rrf_score, reverse=True)

    if top_n is not None:
        fused = fused[:top_n]
    return fused
