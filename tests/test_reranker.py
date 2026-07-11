from src.reranker import CrossEncoderReranker


def test_reranker_disabled_preserves_input_order():
    reranker = CrossEncoderReranker()
    reranker.enabled = False  # force fallback path, no model load
    result = reranker.rerank(
        "query",
        candidate_indices=[5, 2, 9],
        candidate_documents=["doc5", "doc2", "doc9"],
        top_k=2,
    )
    assert [r.index for r in result] == [5, 2]


def test_reranker_disabled_empty_candidates():
    reranker = CrossEncoderReranker()
    reranker.enabled = False
    result = reranker.rerank("query", [], [], top_k=5)
    assert result == []
