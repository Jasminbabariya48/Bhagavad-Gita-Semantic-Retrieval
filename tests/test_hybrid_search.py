from dataclasses import dataclass

from src.hybrid_search import HybridSearch


@dataclass
class _FakeScoredDoc:
    index: int
    score: float


class _FakeDenseSearch:
    def search(self, query, top_k=None):
        return [_FakeScoredDoc(1, 0.9), _FakeScoredDoc(2, 0.8), _FakeScoredDoc(3, 0.7)]


class _FakeBM25Search:
    def search(self, query, top_k=None):
        return [_FakeScoredDoc(2, 5.0), _FakeScoredDoc(4, 3.0), _FakeScoredDoc(1, 2.0)]


def test_hybrid_search_fuses_both_rankers():
    hybrid = HybridSearch(dense_search=_FakeDenseSearch(), bm25_search=_FakeBM25Search())
    fused = hybrid.search("test query")
    # doc 2 appears highly ranked in both -> should be first
    assert fused[0].index == 2
    fused_indices = {d.index for d in fused}
    assert fused_indices == {1, 2, 3, 4}
