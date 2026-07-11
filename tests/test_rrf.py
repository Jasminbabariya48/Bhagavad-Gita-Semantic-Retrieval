from src.rrf import reciprocal_rank_fusion


def test_single_ranker_preserves_order():
    fused = reciprocal_rank_fusion({"dense": [3, 1, 2]}, k=60)
    assert [d.index for d in fused] == [3, 1, 2]


def test_agreement_boosts_rank():
    # doc 5 is ranked highly by both rankers -> should win over docs that
    # only appear in one list, even if that list ranks them #1.
    dense = [1, 5, 2]
    bm25 = [5, 3, 4]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)
    assert fused[0].index == 5


def test_top_n_truncates():
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3, 4, 5]}, k=60, top_n=2)
    assert len(fused) == 2


def test_source_ranks_recorded():
    fused = reciprocal_rank_fusion({"dense": [1, 2], "bm25": [2, 1]}, k=60)
    doc1 = next(d for d in fused if d.index == 1)
    assert doc1.source_ranks == {"dense": 1, "bm25": 2}


def test_score_formula():
    # single doc, single ranker, rank 1: score = 1 / (k + 1)
    fused = reciprocal_rank_fusion({"dense": [7]}, k=60)
    assert abs(fused[0].rrf_score - (1 / 61)) < 1e-9


def test_empty_rankers():
    fused = reciprocal_rank_fusion({}, k=60)
    assert fused == []
