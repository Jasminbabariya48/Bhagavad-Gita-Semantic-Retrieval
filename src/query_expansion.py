"""
Query expansion.

Expands a short user query with related terms before retrieval, e.g.
"How to control anger?" -> "How to control anger? mind self control
discipline emotion". Backed by a small, curated YAML thesaurus
(config/query_synonyms.yaml) rather than a live LLM call, so expansion is
fast, deterministic, and has no external dependency — swap in an LLM-backed
expander later by implementing the same `expand(query) -> str` interface.
"""

from __future__ import annotations

import functools
import re

import yaml

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


@functools.lru_cache(maxsize=1)
def _load_synonyms() -> dict[str, list[str]]:
    cfg = get_config()
    path = cfg.resolve_path(cfg.query_expansion.synonyms_path)
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {k.lower(): v for k, v in data.items()}


class QueryExpander:
    def __init__(self, max_terms: int | None = None):
        cfg = get_config()
        self.enabled = cfg.query_expansion.enabled
        self.max_terms = max_terms or cfg.query_expansion.max_expansion_terms
        self.synonyms = _load_synonyms()

    def expand(self, query: str) -> str:
        if not self.enabled:
            return query

        tokens = {t.lower() for t in _TOKEN_RE.findall(query)}
        expansion_terms: list[str] = []
        for token in tokens:
            for term in self.synonyms.get(token, []):
                if term not in expansion_terms and term.lower() not in tokens:
                    expansion_terms.append(term)

        expansion_terms = expansion_terms[: self.max_terms]
        if not expansion_terms:
            return query

        expanded = f"{query} {' '.join(expansion_terms)}"
        logger.debug("Expanded query '%s' -> '%s'", query, expanded)
        return expanded
