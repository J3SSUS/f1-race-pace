# ruff: noqa: F811  (the shared fixtures are imported from test_run_evaluation and used as arguments)
import importlib.util
from functools import partial
from pathlib import Path

import pytest
from mlflow import MlflowClient
from test_run_evaluation import parent, run, seven, supabase  # noqa: F401

from training.evaluation import test_metric_rows
from training.registry import REGISTERED_MODEL

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "promote.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("promote_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluated(uri, parent, supabase, seven):
    """The seven versions after `run_evaluation`: ridge and the baseline have test metrics.

    The models of the fixture are identical, so ridge's test MAE is set to 80% of the baseline's: a real improvement.
    """
    run(parent, uri, supabase)
    maes = [r for r in supabase.tables["metrics"] if r["split"] == "test" and r["metric_name"] == "mae"]
    baseline = next(r for r in maes if r["model_version_id"] == seven["baseline"].model_version_id)
    next(r for r in maes if r["model_version_id"] == seven["ridge"].model_version_id)["value"] = baseline["value"] * 0.8
    return seven


def promote(cli, uri, supabase, version_id):
    return cli.promote(version_id, client=supabase, tracking_uri=uri)


def main(cli, uri, supabase, version_id):
    lines: list[str] = []
    code = cli.main(
        [str(version_id)], promoter=partial(cli.promote, client=supabase, tracking_uri=uri), out=lines.append
    )
    return code, lines


def add_test_mae(supabase, version_id, mae):
    supabase.tables["metrics"].extend(
        test_metric_rows(version_id, {"mae": mae, "rmse": mae, "r2": 0.0, "pairwise_accuracy": 0.5})
    )


def alias(uri):
    try:
        return str(MlflowClient(uri).get_model_version_by_alias(REGISTERED_MODEL, "champion").version)
    except Exception:
        return None


def current(supabase):
    return [r for r in supabase.tables.get("champion_history", []) if r.get("demoted_at") is None]


def test_the_alias_is_set_and_one_current_row_is_written(cli, uri, supabase, evaluated):
    version = evaluated["ridge"]
    result = promote(cli, uri, supabase, version.model_version_id)
    assert alias(uri) == version.mlflow_version
    (row,) = current(supabase)
    assert row["model_version_id"] == version.model_version_id and row["promoted_by"] == "manual"
    maes = {
        r["model_version_id"]: r["value"]
        for r in supabase.tables["metrics"]
        if r["split"] == "test" and r["metric_name"] == "mae"
    }
    baseline_mae, champion_mae = maes[evaluated["baseline"].model_version_id], maes[version.model_version_id]
    assert row["test_mae_improvement_pct"] == pytest.approx((baseline_mae - champion_mae) / baseline_mae * 100)
    assert row["test_mae_improvement_pct"] == pytest.approx(20.0)
    assert result["changed"] and result["alias_set"] and result["demoted"] == []


def test_a_second_promotion_closes_the_first_row_before_inserting(cli, uri, supabase, evaluated):
    first, second = evaluated["ridge"], evaluated["elasticnet"]
    promote(cli, uri, supabase, first.model_version_id)
    add_test_mae(supabase, second.model_version_id, 0.1)
    result = promote(cli, uri, supabase, second.model_version_id)
    rows = supabase.tables["champion_history"]
    assert len(rows) == 2 and result["demoted"] == [rows[0]["champion_id"]]
    assert rows[0]["demoted_at"] is not None and rows[1].get("demoted_at") is None
    assert [r["model_version_id"] for r in current(supabase)] == [second.model_version_id]
    assert alias(uri) == second.mlflow_version


def test_promoting_the_current_champion_changes_nothing(cli, uri, supabase, evaluated):
    version = evaluated["ridge"]
    promote(cli, uri, supabase, version.model_version_id)
    before = (list(supabase.writes), [dict(r) for r in supabase.tables["champion_history"]])
    result = promote(cli, uri, supabase, version.model_version_id)
    assert result["changed"] is False and result["alias_set"] is False
    assert (supabase.writes, supabase.tables["champion_history"]) == before
    code, lines = main(cli, uri, supabase, version.model_version_id)
    assert code == 0 and "already the champion" in lines[0]


def test_an_unknown_version_raises_before_writing(cli, uri, supabase, evaluated):
    writes = list(supabase.writes)
    with pytest.raises(ValueError, match="does not exist"):
        promote(cli, uri, supabase, 999)
    assert supabase.writes == writes and alias(uri) is None


def test_a_version_without_test_metrics_raises_before_writing(cli, uri, supabase, evaluated):
    writes = list(supabase.writes)
    with pytest.raises(ValueError, match="no test metrics"):
        promote(cli, uri, supabase, evaluated["elasticnet"].model_version_id)  # only train metrics
    assert supabase.writes == writes and alias(uri) is None and supabase.count("champion_history") == 0


def test_before_any_evaluation_nothing_can_be_promoted(cli, uri, supabase, seven):
    with pytest.raises(ValueError, match="no test metrics"):
        promote(cli, uri, supabase, seven["ridge"].model_version_id)
    assert alias(uri) is None


def test_a_sampling_run_without_a_baseline_with_test_metrics_raises_before_writing(cli, uri, supabase, evaluated):
    baseline_id = evaluated["baseline"].model_version_id
    supabase.tables["metrics"] = [
        r for r in supabase.tables["metrics"] if not (r["model_version_id"] == baseline_id and r["split"] == "test")
    ]
    with pytest.raises(ValueError, match="0 baseline versions"):
        promote(cli, uri, supabase, evaluated["ridge"].model_version_id)
    assert alias(uri) is None and supabase.count("champion_history") == 0


def test_a_non_positive_improvement_warns_and_still_promotes(cli, uri, supabase, evaluated):
    worse = evaluated["lightgbm"]
    add_test_mae(supabase, worse.model_version_id, 5.0)  # the baseline's test MAE is far lower
    code, lines = main(cli, uri, supabase, worse.model_version_id)
    assert code == 0
    assert any(line.startswith("WARNING") for line in lines)
    assert alias(uri) == worse.mlflow_version and len(current(supabase)) == 1
    assert current(supabase)[0]["test_mae_improvement_pct"] < 0


def test_a_positive_improvement_has_no_warning(cli, uri, supabase, evaluated):
    code, lines = main(cli, uri, supabase, evaluated["ridge"].model_version_id)
    assert code == 0 and not any("WARNING" in line for line in lines)
    assert lines[0].startswith(f"promoted model version {evaluated['ridge'].model_version_id}")


def test_if_supabase_fails_the_alias_stays_and_a_new_call_completes_the_row(cli, uri, supabase, evaluated, monkeypatch):
    version = evaluated["ridge"]

    def boom(*args, **kwargs):
        raise RuntimeError("insert into champion_history failed: ConnectionError")

    monkeypatch.setattr(cli, "insert_rows", boom)
    code, lines = main(cli, uri, supabase, version.model_version_id)
    assert code == 1 and lines == ["STOPPED: insert into champion_history failed: ConnectionError"]
    assert alias(uri) == version.mlflow_version and current(supabase) == []

    monkeypatch.undo()
    result = promote(cli, uri, supabase, version.model_version_id)
    assert result["changed"] and result["alias_set"] is False  # the alias was not set a second time
    assert [r["model_version_id"] for r in current(supabase)] == [version.model_version_id]


def test_an_unknown_version_prints_stopped_and_exits_1(cli, uri, supabase, evaluated):
    code, lines = main(cli, uri, supabase, 999)
    assert code == 1 and len(lines) == 1 and lines[0].startswith("STOPPED:")
