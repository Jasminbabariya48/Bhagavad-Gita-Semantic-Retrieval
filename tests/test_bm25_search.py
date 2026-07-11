from src.bm25_search import BM25Search, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Control the Mind!") == ["control", "the", "mind"]


def test_bm25_finds_relevant_document():
    documents = [
        "Krishna teaches about karma and duty in the battlefield.",
        "The recipe requires flour, sugar, and butter.",
        "Arjuna asks Krishna how to control the restless mind.",
        "The stock market closed higher today.",
    ]
    bm25 = BM25Search(documents=documents, top_k=4)
    results = bm25.search("how to control the mind")
    assert results[0].index == 2  # the mind-control document should rank first


def test_bm25_save_and_load(tmp_path):
    documents = [
        "alpha beta gamma alpha alpha",
        "delta epsilon zeta",
        "unrelated filler text about nothing in particular",
    ]
    bm25 = BM25Search(documents=documents, top_k=2)
    path = str(tmp_path / "bm25.pkl")
    bm25.save(path)

    loaded = BM25Search.load(path)
    results = loaded.search("alpha")
    assert results[0].index == 0
