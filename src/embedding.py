"""
Thin wrapper around SentenceTransformer so the rest of the codebase never
touches the library directly. Centralizes:
  - e5-family "query: " / "passage: " prefixing (these models need it —
    forgetting it is one of the most common silent accuracy bugs)
  - embedding normalization
  - batch size / device handling

Import is lazy (inside functions) so modules that only need pure-Python
logic (RRF, config, preprocessing) can be unit tested without requiring
torch / sentence-transformers to be installed.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.config import get_config
from src.logging_utils import get_logger

logger = get_logger(__name__)

# Model name substrings that require e5-style instruction prefixes.
_E5_FAMILY_MARKERS = ("e5",)


def model_needs_e5_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return any(marker in name for marker in _E5_FAMILY_MARKERS)


class EmbeddingModel:
    """Loads a SentenceTransformer and exposes encode_queries / encode_documents
    with the correct prefixing/normalization for the configured model."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        use_e5_prefixes: bool | None = None,
        normalize: bool | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ):
        cfg = get_config()
        self.model_name_or_path = model_name_or_path or cfg.embedding.active_model
        self.use_e5_prefixes = (
            model_needs_e5_prefix(self.model_name_or_path)
            if use_e5_prefixes is None
            else use_e5_prefixes
        )
        self.normalize = cfg.embedding.normalize_embeddings if normalize is None else normalize
        self.batch_size = batch_size or cfg.embedding.batch_size
        self.device = device

        logger.info(
            "Loading embedding model '%s' (e5_prefix=%s, normalize=%s)",
            self.model_name_or_path,
            self.use_e5_prefixes,
            self.normalize,
        )
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(self.model_name_or_path, device=self.device)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _prefix(self, texts: Iterable[str], kind: str) -> list[str]:
        if not self.use_e5_prefixes:
            return list(texts)
        prefix = "query: " if kind == "query" else "passage: "
        return [f"{prefix}{t}" for t in texts]

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        texts = self._prefix(list(queries), "query")
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def encode_documents(self, documents: Iterable[str]) -> np.ndarray:
        texts = self._prefix(list(documents), "passage")
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def save(self, path: str) -> None:
        self._model.save(path)
        logger.info("Saved embedding model to %s", path)

    @property
    def raw_model(self):
        """Escape hatch for callers (e.g. trainer.py) that need the underlying
        SentenceTransformer for .fit()."""
        return self._model
