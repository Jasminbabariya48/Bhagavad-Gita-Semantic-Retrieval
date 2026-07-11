import pandas as pd

from src.query_pairs import build_query_pairs
from src.triplet_generation import build_triplets, validate_triplets


def _sample_dataset():
    return pd.DataFrame(
        {
            "unique_key": ["1", "2", "3"],
            "sanskrit_data": ["sk1", "sk2", "sk3"],
            "translation": ["Translation one.", "Translation two.", "Translation three."],
            "document": ["sk1\nTranslation one.", "sk2\nTranslation two.", "sk3\nTranslation three."],
        }
    )


def test_build_query_pairs_generates_multiple_queries_per_verse():
    df = _sample_dataset()
    pairs = build_query_pairs(df)
    # 5 english templates + 1 sanskrit query per verse = 6 * 3 = 18
    assert len(pairs) == 18
    assert set(pairs["document"]) == set(df["document"])
    assert (pairs["label"] == 1).all()


def test_build_triplets_negative_never_equals_positive():
    df = _sample_dataset()
    pairs = build_query_pairs(df)
    triplets = build_triplets(pairs, seed=42)
    assert len(triplets) == len(pairs)
    assert validate_triplets(triplets) == 0
    assert (triplets["positive"] != triplets["negative"]).all()


def test_build_triplets_is_reproducible_with_seed():
    df = _sample_dataset()
    pairs = build_query_pairs(df)
    t1 = build_triplets(pairs, seed=123)
    t2 = build_triplets(pairs, seed=123)
    assert t1["negative"].tolist() == t2["negative"].tolist()
