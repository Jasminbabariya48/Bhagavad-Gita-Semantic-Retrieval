from src.config import get_config


def test_attribute_access():
    cfg = get_config()
    assert cfg.project.name == "immverse_ai"
    assert cfg.project.seed == 42


def test_nested_dict_returns_config_object():
    cfg = get_config()
    assert cfg.embedding.active_model == "intfloat/multilingual-e5-small"
    assert isinstance(cfg.embedding.candidates, list)


def test_path_resolution_is_absolute():
    cfg = get_config()
    path = cfg.path("processed_dataset")
    assert path.is_absolute()
    assert str(path).endswith("cleaned_dataset.csv")


def test_missing_key_raises_attribute_error():
    cfg = get_config()
    try:
        _ = cfg.nonexistent_section
        assert False, "expected AttributeError"
    except AttributeError:
        pass


def test_get_with_default():
    cfg = get_config()
    assert cfg.get("nonexistent_section", "fallback") == "fallback"
