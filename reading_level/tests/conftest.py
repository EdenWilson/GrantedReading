import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reading_level import config


@pytest.fixture(scope="session")
def table():
    from reading_level.frequency import get_default_table

    return get_default_table()


@pytest.fixture(scope="session")
def fixture_text():
    def _load(name):
        return (config.FIXTURES_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()

    return _load
