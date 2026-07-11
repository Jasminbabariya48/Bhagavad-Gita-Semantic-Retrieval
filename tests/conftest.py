import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.config import reload_config


@pytest.fixture(autouse=True)
def _fresh_config():
    """Reload config fresh for every test so config-mutating tests don't leak."""
    reload_config()
    yield
