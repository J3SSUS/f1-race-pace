from pathlib import Path

import pytest
from synthetic_race import write_race


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Three synthetic races (2024 rounds 1 to 3: clean, safety car, red flag) in the layout of `data/raw`."""
    for round, scenario in enumerate(("clean", "safety_car", "red_flag"), start=1):
        write_race(tmp_path, 2024, round, scenario)
    return tmp_path
