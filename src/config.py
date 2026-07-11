"""
Central configuration loader.

Every other module reads its settings through `get_config()` instead of
hardcoding paths or hyperparameters. This keeps the project config-driven:
change config/config.yaml and every script picks up the new value.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the `src/` directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Thin, dict-like wrapper around the parsed YAML config.

    Supports attribute-style access (`cfg.embedding.active_model`) and
    resolves every path in the `paths:` section to an absolute path rooted
    at the project root, so callers never have to think about cwd.
    """

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self._root = root

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"No config key '{name}'") from exc
        if isinstance(value, dict):
            return Config(value, self._root)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a path from the `paths:` section against the project root."""
        return (self._root / relative_path).resolve()

    def path(self, key: str) -> Path:
        """Look up `paths.<key>` and resolve it to an absolute Path."""
        relative = self._data if key not in self._data else None
        raw = self.paths._data.get(key)  # type: ignore[attr-defined]
        if raw is None:
            raise KeyError(f"No path configured for '{key}'")
        return self.resolve_path(raw)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


@functools.lru_cache(maxsize=None)
def get_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load (and cache) the project configuration.

    Cached with lru_cache so repeated calls across modules return the same
    object without re-reading/parsing the YAML file each time.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, root=config_path.resolve().parent.parent)


def reload_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Bypass the cache — useful in tests that swap in a temp config file."""
    get_config.cache_clear()
    return get_config(config_path)
