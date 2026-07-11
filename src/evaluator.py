"""
Retrieval evaluation metrics.

Refactor of notebooks/05_evaluate_model.ipynb (Recall@K, MRR, nDCG@K logic
preserved exactly) plus the additional metrics called out in the
requirements: MAP, Hit Rate, and query latency.

All metric functions take a `similarity_matrix` of shape
(n_queries, n_documents) where the correct document for query i is assumed
to be at index i (this matches how query_pairs / triplets are constructed:
one document per verse, evaluated 1:1). Pure numpy — no model dependency —
so these are fast to unit test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def recall_at_k(similarity_matrix: np.ndarray, k: int) -> float:
    correct = 0
    for i in range(len(similarity_matrix)):
        top_k = np.argsort(similarity_matrix[i])[::-1][:k]
        if i in top_k:
            correct += 1
    return correct / len(similarity_matrix)


def mean_reciprocal_rank(similarity_matrix: np.ndarray) -> float:
    rr = []
    for i in range(len(similarity_matrix)):
        ranking = np.argsort(similarity_matrix[i])[::-1]
        rank = np.where(ranking == i)[0][0] + 1
        rr.append(1 / rank)
    return float(np.mean(rr))


def ndcg_at_k(similarity_matrix: np.ndarray, k: int = 10) -> float:
    scores = []
    for i in range(len(similarity_matrix)):
        ranking = np.argsort(similarity_matrix[i])[::-1][:k]
        if i in ranking:
            rank = np.where(ranking == i)[0][0]
            dcg = 1 / np.log2(rank + 2)
        else:
            dcg = 0.0
        idcg = 1.0  # single relevant document per query -> ideal DCG is 1
        scores.append(dcg / idcg)
    return float(np.mean(scores))


def average_precision_at_k(similarity_matrix: np.ndarray, k: int = 10) -> float:
    """MAP@K with a single relevant document per query: precision at the
    rank where the correct document appears (0 if it's outside top-k)."""
    scores = []
    for i in range(len(similarity_matrix)):
        ranking = np.argsort(similarity_matrix[i])[::-1][:k]
        hit = np.where(ranking == i)[0]
        if len(hit) == 0:
            scores.append(0.0)
        else:
            rank = hit[0] + 1  # 1-indexed
            scores.append(1.0 / rank)
    return float(np.mean(scores))


def hit_rate_at_k(similarity_matrix: np.ndarray, k: int) -> float:
    """Fraction of queries where the correct document appears anywhere in
    top-k. Numerically identical to recall_at_k under the single-relevant-doc
    assumption used here, kept as a separate named metric because it's a
    standard, explicitly requested reporting metric distinct from recall in
    multi-relevant-document settings."""
    return recall_at_k(similarity_matrix, k)


@dataclass
class EvaluationResult:
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms_mean: float | None = None
    latency_ms_p95: float | None = None
    n_queries: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"Metric": k, "Score": v} for k, v in self.metrics.items()]
        if self.latency_ms_mean is not None:
            rows.append({"Metric": "Latency_ms_mean", "Score": self.latency_ms_mean})
            rows.append({"Metric": "Latency_ms_p95", "Score": self.latency_ms_p95})
        return pd.DataFrame(rows)


def compute_all_metrics(
    similarity_matrix: np.ndarray,
    k_values: list[int] | None = None,
    ndcg_k: int = 10,
    map_k: int = 10,
) -> dict[str, float]:
    k_values = k_values or [1, 5, 10]
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k(similarity_matrix, k)
        metrics[f"HitRate@{k}"] = hit_rate_at_k(similarity_matrix, k)
    metrics["MRR"] = mean_reciprocal_rank(similarity_matrix)
    metrics[f"nDCG@{ndcg_k}"] = ndcg_at_k(similarity_matrix, ndcg_k)
    metrics[f"MAP@{map_k}"] = average_precision_at_k(similarity_matrix, map_k)
    return metrics


def measure_query_latency(encode_fn, queries: list[str]) -> tuple[float, float]:
    """Times `encode_fn(query)` per query. Returns (mean_ms, p95_ms)."""
    latencies = []
    for q in queries:
        start = time.perf_counter()
        encode_fn(q)
        latencies.append((time.perf_counter() - start) * 1000)
    arr = np.array(latencies)
    return float(arr.mean()), float(np.percentile(arr, 95))


def evaluate_similarity_matrix(
    similarity_matrix: np.ndarray,
    k_values: list[int] | None = None,
    ndcg_k: int = 10,
    map_k: int = 10,
    latency_ms_mean: float | None = None,
    latency_ms_p95: float | None = None,
) -> EvaluationResult:
    metrics = compute_all_metrics(similarity_matrix, k_values, ndcg_k, map_k)
    return EvaluationResult(
        metrics=metrics,
        latency_ms_mean=latency_ms_mean,
        latency_ms_p95=latency_ms_p95,
        n_queries=len(similarity_matrix),
    )
