import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from features.store import StoreError
from training.explain import Case, Champion, Explanation

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "explain.py"
KEY = "service-key-that-must-never-be-printed"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("explain_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(tmp_path):
    names = [f"f{i}" for i in range(8)]
    importance = pd.DataFrame(
        {"feature": names, "mae_increase_mean": [0.8 - 0.1 * i for i in range(8)], "mae_increase_std": 0.01}
    )
    cases = [
        Case("fastest_predicted", 1, 2025, 24, "VER", -0.9, -0.7, -0.2, 0.0, {}),
        Case("slowest_predicted", 2, 2025, 24, "SAR", 1.1, 1.4, -0.3, 0.0, {}),
    ]
    explanation = Explanation(
        shap=pd.DataFrame(),
        base_value=0.05,
        importance=importance,
        coefficients=None,
        calibration={"slope": 0.95, "intercept": 0.01, "n_rows": 500, "n_races": 20},
        cases=cases,
    )
    files = [tmp_path / "shap_summary.svg", tmp_path / "explain.json"]
    return {
        "run_id": "abc123",
        "champion": Champion(4, "2", "run", "sampling", "ridge", None),
        "explanation": explanation,
        "files": files,
        "shap_rows": 6000,
    }


def run(cli, runner):
    lines: list[str] = []
    return cli.main([], runner=runner, out=lines.append), lines


def test_it_prints_the_champion_the_top_five_the_calibration_the_cases_and_the_files(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", KEY)
    code, lines = run(cli, lambda **kwargs: fake_result(tmp_path))
    text = "\n".join(lines)
    assert code == 0
    assert "version 4 (ridge)" in text and "linear explainer" in text
    assert "6000" in text
    assert sum(line.startswith("  f") and "±" in line for line in lines) == 5  # five features, no more
    assert "f0" in text and "f5" not in text
    assert "0.950" in text and "500 out-of-fold rows, 20 races" in text
    assert "VER" in text and "SAR" in text and "actual +1.400" in text
    assert "shap_summary.svg" in text and "explain.json" in text and "abc123" in text
    assert KEY not in text


@pytest.mark.parametrize(
    "error",
    [
        ValueError("f1-race-pace has no 'champion' alias: run scripts/promote.py first"),
        ValueError("algorithm 'baseline' has no explainer; allowed: ['catboost']"),
        StoreError("read from features failed: APIError"),
        RuntimeError("upsert into shap_values (on_conflict entry_id,model_version_id,feature_id) failed: APIError"),
        MlflowException("no such run"),
    ],
)
def test_a_failure_prints_one_stopped_line_and_exits_1_without_the_key(cli, error, monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", KEY)

    def broken(**kwargs):
        raise error

    code, lines = run(cli, broken)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("STOPPED: ")
    assert KEY not in lines[0]


def test_the_charts_dir_defaults_to_site_charts_of_the_repository_and_can_be_changed(cli, tmp_path):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return fake_result(tmp_path)

    run(cli, spy)
    assert seen["charts_dir"] == SCRIPT.parents[1] / "site" / "charts"
    lines: list[str] = []
    cli.main(["--charts-dir", str(tmp_path / "x")], runner=spy, out=lines.append)
    assert seen["charts_dir"] == tmp_path / "x"
