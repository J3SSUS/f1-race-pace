import json
import shutil

import pytest

from sitegen.inputs import ROOT


@pytest.fixture
def site_root(tmp_path):
    """A repository root with the inputs of the build: team colors, `explain.json` and a card with run ids."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "team_colors.json", root / "config" / "team_colors.json")
    (root / "site" / "charts").mkdir(parents=True)
    (root / "site" / "charts" / "explain.json").write_text(json.dumps({"model_version_id": 4, "importance": []}))
    (root / "MODEL_CARD.md").write_text(
        "# Card\n\n2026-11-02 · explain run d5bb2da8b777423eb46c949c88b44739"
        " · evaluate run 100bba09a6604cf895095eb3143b4e30 · sampling run e64c04d8897c44d283da0422f2e818ce\n\n"
        "## 1. What it predicts\n\nThe pace.\n\n## 2. Results\n\nDetails.\n"
    )
    return root
