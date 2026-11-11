import json
import shutil

import pytest

from sitegen.inputs import ROOT


@pytest.fixture
def root(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "team_colors.json", tmp_path / "config" / "team_colors.json")
    (tmp_path / "site" / "charts").mkdir(parents=True)
    (tmp_path / "site" / "charts" / "explain.json").write_text(json.dumps({"model_version_id": 4, "importance": []}))
    (tmp_path / "MODEL_CARD.md").write_text(
        "# Card\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n![Importance](site/charts/permutation_importance.svg)\n"
    )
    return tmp_path
