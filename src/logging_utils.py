"""Consistent logging setup shared by every module and script.

Also emits a PIPELINE_VERSION line on first use in a process, which is the
cheapest way to catch a stale deployment (old code / old container image
running against new config) in production logs.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.config import get_config

_CONFIGURED = False
_VERSION_LOGGED = False


def _configure_root_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    cfg = get_config()
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    fmt = cfg.logging.format
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configured from config/config.yaml."""
    _configure_root_logging()
    logger = logging.getLogger(name)

    global _VERSION_LOGGED
    if not _VERSION_LOGGED:
        cfg = get_config()
        logger.info("PIPELINE_VERSION=%s", cfg.pipeline_version)
        _VERSION_LOGGED = True

    return logger


def log_pipeline_version(logger: Optional[logging.Logger] = None) -> None:
    """Explicitly emit the pipeline version — call this at the top of every
    entrypoint script so stale deployments show up immediately in logs."""
    cfg = get_config()
    logger = logger or get_logger(__name__)
    logger.info(
        "PIPELINE_VERSION=%s | active_embedding_model=%s | reranker_enabled=%s",
        cfg.pipeline_version,
        cfg.embedding.active_model,
        cfg.reranker.enabled,
    )
