# ruff: noqa: F811  (the shared fixtures are imported from test_run_evaluation and used as arguments)
import importlib.util
from functools import partial
from pathlib import Path

import pytest
from mlflow import MlflowClient
from test_run_evaluation import parent, run, seven, supabase  # noqa: F401

from training import promotion
from training.evaluation import test_metric_rows
from training.registry import REGISTERED_MODEL
from training.selection import SIMPLICITY

pytestmark = pytest.mark.slow  # logs real models in a local MLflow store: needs --runslow

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

    monkeypatch.setattr(promotion, "insert_rows", boom)
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


# ---- the shadow mark and the challenger alias (SPEC 25) ----


def shadows(supabase):
    return sorted(r["model_version_id"] for r in supabase.tables["model_versions"] if r.get("is_shadow"))


def set_challenger(uri, version):
    MlflowClient(uri).set_registered_model_alias(REGISTERED_MODEL, "challenger", version.mlflow_version)


def challenger_alias(uri):
    try:
        return str(MlflowClient(uri).get_model_version_by_alias(REGISTERED_MODEL, "challenger").version)
    except Exception:
        return None


@pytest.fixture
def three(evaluated, supabase):
    """Three versions with test metrics, in the order they are promoted in the tests."""
    first, second, third = evaluated["ridge"], evaluated[SIMPLICITY[1]], evaluated[SIMPLICITY[2]]
    add_test_mae(supabase, second.model_version_id, 0.1)
    add_test_mae(supabase, third.model_version_id, 0.1)
    return first, second, third


def test_the_first_champion_marks_no_shadow(cli, uri, supabase, three):
    first, _, _ = three
    result = promote(cli, uri, supabase, first.model_version_id)
    assert shadows(supabase) == [] and result["shadow_set"] is None and result["shadow_cleared"] == []


def test_a_second_promotion_marks_the_champion_that_leaves(cli, uri, supabase, three):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    result = promote(cli, uri, supabase, second.model_version_id)
    assert shadows(supabase) == [first.model_version_id]
    assert result["shadow_set"] == first.model_version_id and result["shadow_cleared"] == []


def test_a_third_promotion_moves_the_mark_to_the_new_outgoing_champion(cli, uri, supabase, three):
    first, second, third = three
    for version in (first, second, third):
        result = promote(cli, uri, supabase, version.model_version_id)
    assert shadows(supabase) == [second.model_version_id]  # one shadow at a time
    assert result["shadow_set"] == second.model_version_id and result["shadow_cleared"] == [first.model_version_id]
    assert [r["model_version_id"] for r in current(supabase)] == [third.model_version_id]


def test_the_champion_is_never_the_shadow(cli, uri, supabase, three):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    promote(cli, uri, supabase, second.model_version_id)
    promote(cli, uri, supabase, first.model_version_id)  # back to the older champion, by hand
    assert shadows(supabase) == [second.model_version_id]
    assert [r["model_version_id"] for r in current(supabase)] == [first.model_version_id]


def test_an_automatic_promotion_marks_the_shadow_too(cli, uri, supabase, three):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    cli.promote(second.model_version_id, "auto", client=supabase, tracking_uri=uri)
    assert shadows(supabase) == [first.model_version_id]
    assert current(supabase)[0]["promoted_by"] == "auto"


def test_promoted_by_must_be_manual_or_auto(cli, uri, supabase, three):
    writes = list(supabase.writes)
    with pytest.raises(ValueError, match="promoted_by"):
        cli.promote(three[0].model_version_id, "robot", client=supabase, tracking_uri=uri)
    assert supabase.writes == writes and alias(uri) is None


def test_the_challenger_alias_is_removed_from_the_promoted_version(cli, uri, supabase, three):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    set_challenger(uri, second)
    result = promote(cli, uri, supabase, second.model_version_id)
    assert challenger_alias(uri) is None and result["challenger_alias_removed"] is True
    assert alias(uri) == second.mlflow_version


def test_the_challenger_alias_on_another_version_stays(cli, uri, supabase, three):
    first, second, third = three
    promote(cli, uri, supabase, first.model_version_id)
    set_challenger(uri, third)
    result = promote(cli, uri, supabase, second.model_version_id)
    assert challenger_alias(uri) == third.mlflow_version and result["challenger_alias_removed"] is False


def test_if_the_mark_fails_a_new_call_completes_it_without_a_second_alias_change(
    cli, uri, supabase, three, monkeypatch
):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    real_update = promotion.update_rows

    def update_or_fail(client, table, values, match):
        if table == "model_versions":
            raise RuntimeError("update of model_versions failed: ConnectionError")
        return real_update(client, table, values, match)

    monkeypatch.setattr(promotion, "update_rows", update_or_fail)
    code, lines = main(cli, uri, supabase, second.model_version_id)
    assert code == 1 and lines == ["STOPPED: update of model_versions failed: ConnectionError"]
    assert alias(uri) == second.mlflow_version and shadows(supabase) == []
    assert [r["model_version_id"] for r in current(supabase)] == [second.model_version_id]

    monkeypatch.undo()
    result = promote(cli, uri, supabase, second.model_version_id)
    assert result["changed"] is True and result["alias_set"] is False and result["demoted"] == []
    assert shadows(supabase) == [first.model_version_id] and supabase.count("champion_history") == 2
    again = promote(cli, uri, supabase, second.model_version_id)
    assert again["changed"] is False  # nothing is left to complete


def test_promoting_the_current_champion_with_a_complete_mark_changes_nothing(cli, uri, supabase, three):
    first, second, _ = three
    promote(cli, uri, supabase, first.model_version_id)
    promote(cli, uri, supabase, second.model_version_id)
    before = (list(supabase.writes), [dict(r) for r in supabase.tables["model_versions"]])
    result = promote(cli, uri, supabase, second.model_version_id)
    assert result["changed"] is False and result["shadow_set"] == first.model_version_id
    assert (supabase.writes, supabase.tables["model_versions"]) == before


# ---- auto_promote (SPEC 25) ----

RACES = [{"race_id": 100 + i, "year": 2024, "round": i + 1} for i in range(8)]


def add_walk_forward(supabase, version_id, maes: dict[int, float]):
    supabase.tables["metrics"].extend(
        {
            "model_version_id": version_id,
            "race_id": race_id,
            "split": "walk_forward",
            "segment_type": "overall",
            "segment_value": None,
            "metric_name": "mae",
            "value": value,
        }
        for race_id, value in maes.items()
    )


@pytest.fixture
def duel(cli, uri, supabase, three):
    """Version 1 is the champion and version 2 the challenger, both with `walk_forward` MAE on the races 103 to 107.

    The challenger has no test metrics, like a refit (SPEC 23).
    """
    champion, challenger, other = three
    supabase.tables["races"] = RACES
    supabase.tables["metrics"] = [
        r
        for r in supabase.tables["metrics"]
        if not (r["split"] == "test" and r["model_version_id"] == challenger.model_version_id)
    ]
    promote(cli, uri, supabase, champion.model_version_id)
    set_challenger(uri, challenger)
    add_walk_forward(supabase, champion.model_version_id, {100 + i: 1.0 for i in range(3, 8)})
    add_walk_forward(supabase, challenger.model_version_id, {100 + i: 0.5 for i in range(3, 8)})
    return champion, challenger, other


def auto(cli, uri, supabase, version_id, dry_run=False):
    return promotion.auto_promote(version_id, client=supabase, tracking_uri=uri, dry_run=dry_run)


def test_the_challenger_that_passes_is_promoted_with_the_forward_improvement(cli, uri, supabase, duel):
    champion, challenger, _ = duel
    result = auto(cli, uri, supabase, challenger.model_version_id)
    assert result["promoted"] is True and result["passes"] is True and result["common_races"] == 5
    assert alias(uri) == challenger.mlflow_version and challenger_alias(uri) is None
    (row,) = current(supabase)
    assert row["model_version_id"] == challenger.model_version_id and row["promoted_by"] == "auto"
    assert row["test_mae_improvement_pct"] == pytest.approx(50.0)  # (1.0 - 0.5) / 1.0, forward, no test metrics
    assert shadows(supabase) == [champion.model_version_id]
    assert result["test_mae"] is None and result["shadow_set"] == champion.model_version_id


def test_a_challenger_without_five_common_races_is_not_promoted_and_nothing_is_written(cli, uri, supabase, duel):
    champion, challenger, _ = duel
    supabase.tables["metrics"] = [
        r for r in supabase.tables["metrics"] if not (r["split"] == "walk_forward" and r["race_id"] == 107)
    ]
    before = (list(supabase.writes), [dict(r) for r in supabase.tables["champion_history"]])
    result = auto(cli, uri, supabase, challenger.model_version_id)
    assert result["promoted"] is False and result["common_races"] == 4 and "only 4 common races" in result["reason"]
    assert (supabase.writes, supabase.tables["champion_history"]) == before
    assert alias(uri) == champion.mlflow_version and challenger_alias(uri) == challenger.mlflow_version


def test_a_challenger_that_is_worse_in_three_races_is_not_promoted(cli, uri, supabase, duel):
    champion, challenger, _ = duel
    for row in supabase.tables["metrics"]:
        if row["model_version_id"] == challenger.model_version_id and row["race_id"] in (103, 104, 105):
            row["value"] = 1.2
    before = list(supabase.writes)
    result = auto(cli, uri, supabase, challenger.model_version_id)
    assert result["promoted"] is False and result["races_worse"] == 3 and "worse than the champion" in result["reason"]
    assert supabase.writes == before and alias(uri) == champion.mlflow_version


def test_a_dry_run_that_passes_writes_nothing(cli, uri, supabase, duel):
    champion, challenger, _ = duel
    before = (list(supabase.writes), [dict(r) for r in supabase.tables["champion_history"]])
    result = auto(cli, uri, supabase, challenger.model_version_id, dry_run=True)
    assert result["passes"] is True and result["promoted"] is False and result["dry_run"] is True
    assert "dry run" in result["reason"] and result["improvement_pct"] == pytest.approx(50.0)
    assert (supabase.writes, supabase.tables["champion_history"]) == before
    assert alias(uri) == champion.mlflow_version and challenger_alias(uri) == challenger.mlflow_version
    assert shadows(supabase) == []


def test_a_version_without_the_challenger_alias_is_refused_before_writing(cli, uri, supabase, duel):
    _, _, other = duel
    add_walk_forward(supabase, other.model_version_id, {100 + i: 0.5 for i in range(3, 8)})
    before = list(supabase.writes)
    with pytest.raises(ValueError, match="does not hold the `challenger` alias"):
        auto(cli, uri, supabase, other.model_version_id)
    assert supabase.writes == before


def test_the_champion_cannot_be_promoted_again_automatically(cli, uri, supabase, duel):
    champion, _, _ = duel
    set_challenger(uri, champion)
    with pytest.raises(ValueError, match="already the champion"):
        auto(cli, uri, supabase, champion.model_version_id)


def test_an_unknown_version_is_refused(cli, uri, supabase, duel):
    with pytest.raises(ValueError, match="does not exist"):
        auto(cli, uri, supabase, 999)


def test_without_a_current_champion_it_is_refused(cli, uri, supabase, duel):
    _, challenger, _ = duel
    supabase.tables["champion_history"] = []
    with pytest.raises(ValueError, match="expected one current champion, found 0"):
        auto(cli, uri, supabase, challenger.model_version_id)


def test_a_manual_promotion_still_needs_test_metrics_and_refuses_a_given_improvement(cli, uri, supabase, duel):
    _, challenger, third = duel
    with pytest.raises(ValueError, match="no test metrics"):
        promote(cli, uri, supabase, challenger.model_version_id)  # the challenger has none: only auto skips them
    with pytest.raises(ValueError, match="only given by an automatic promotion"):
        cli.promote(third.model_version_id, "manual", client=supabase, tracking_uri=uri, improvement_pct=10.0)
