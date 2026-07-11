import pandas as pd

from src.preprocessing import clean_text, preprocess_dataset


def test_clean_text_handles_null():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  \n") == "hello world"


def test_preprocess_dataset_builds_document_column():
    df = pd.DataFrame(
        {
            "unique_key": ["1", "2", "2", ""],
            "sanskrit_data": ["श्लोक १", "श्लोक २", "श्लोक २", "x"],
            "translation": ["Verse one meaning.", "Verse two meaning.", "Verse two meaning.", "y"],
        }
    )
    cleaned = preprocess_dataset(df)
    # duplicate id "2" and empty id "" should be dropped
    assert len(cleaned) == 2
    assert "document" in cleaned.columns
    assert "श्लोक १" in cleaned.iloc[0]["document"]
    assert "Verse one meaning." in cleaned.iloc[0]["document"]


def test_preprocess_dataset_drops_empty_translation():
    df = pd.DataFrame(
        {
            "unique_key": ["1", "2"],
            "sanskrit_data": ["a", "b"],
            "translation": ["", "valid translation"],
        }
    )
    cleaned = preprocess_dataset(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["unique_key"] == "2"
