import json
import shutil
import subprocess

import pytest

from sitegen.inputs import ROOT


@pytest.fixture
def site_root(tmp_path):
    """A git repository with the inputs of the build: team colors, `explain.json` (ignored, as after SPEC 17 step 12)
    and a card with run ids. `site/.gitkeep` is the one file git would publish under `site/`."""
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
    (root / ".gitignore").write_text("site/charts/explain.json\n")
    (root / "site" / ".gitkeep").write_text("")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    return root
