"""Small shared helpers used across scripts/ and app/."""

from __future__ import annotations

import functools
import time

from src.logging_utils import get_logger

logger = get_logger(__name__)


def timer(func):
    """Decorator that logs how long a function took to run (DEBUG level)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("%s took %.2f ms", func.__name__, elapsed_ms)
        return result

    return wrapper


def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)
