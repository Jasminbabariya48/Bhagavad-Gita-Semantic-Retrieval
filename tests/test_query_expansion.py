from src.query_expansion import QueryExpander


def test_expand_adds_related_terms():
    expander = QueryExpander()
    expanded = expander.expand("How to control anger?")
    assert expanded.startswith("How to control anger?")
    assert "mind" in expanded or "discipline" in expanded


def test_expand_no_match_returns_original():
    expander = QueryExpander()
    expanded = expander.expand("xyzabc nonword query")
    assert expanded == "xyzabc nonword query"


def test_expand_respects_max_terms():
    expander = QueryExpander(max_terms=1)
    expanded = expander.expand("mind and karma and dharma")
    added_tokens = expanded.replace("mind and karma and dharma", "").split()
    assert len(added_tokens) <= 1
