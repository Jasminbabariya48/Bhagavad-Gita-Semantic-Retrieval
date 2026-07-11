import numpy as np

from src.evaluator import (
    average_precision_at_k,
    compute_all_metrics,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def _identity_similarity(n: int) -> np.ndarray:
    """Perfect similarity matrix: query i's best match is document i."""
    return np.eye(n) + 0.01  # avoid ties


def test_perfect_retrieval_scores_are_maximal():
    sim = _identity_similarity(10)
    assert recall_at_k(sim, 1) == 1.0
    assert mean_reciprocal_rank(sim) == 1.0
    assert ndcg_at_k(sim, 10) == 1.0
    assert average_precision_at_k(sim, 10) == 1.0
    assert hit_rate_at_k(sim, 1) == 1.0


def test_worst_case_recall_zero():
    # every query's correct doc has the lowest similarity
    n = 4
    sim = np.ones((n, n))
    np.fill_diagonal(sim, 0.0)
    assert recall_at_k(sim, 1) == 0.0


def test_mrr_partial_credit():
    # query 0's correct doc (index 0) is ranked 2nd
    sim = np.array(
        [
            [0.5, 0.9],  # doc1 > doc0 -> correct doc0 ranked 2nd -> RR = 1/2
            [0.1, 0.9],  # doc1 > doc0 -> correct doc1 ranked 1st -> RR = 1
        ]
    )
    assert abs(mean_reciprocal_rank(sim) - 0.75) < 1e-9


def test_compute_all_metrics_keys_present():
    sim = _identity_similarity(5)
    metrics = compute_all_metrics(sim, k_values=[1, 5], ndcg_k=5, map_k=5)
    for key in ["Recall@1", "Recall@5", "HitRate@1", "HitRate@5", "MRR", "nDCG@5", "MAP@5"]:
        assert key in metrics
