import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.append(str(TESTS / "features"))  # the in-memory Supabase fake and the synthetic races
sys.path.append(str(TESTS / "ingestion"))  # the synthetic OpenF1 tables and the loaders of test_session

import pytest  # noqa: E402
from live_helpers import write_valid_race  # noqa: E402


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """A raw folder with the synthetic race 2024 round 1 (Q and R, valid)."""
    write_valid_race(tmp_path)
    return tmp_path
