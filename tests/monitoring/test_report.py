"""Build and store a drift report (SPEC 26, step 7): the numbers, the rows, the MLflow run and the idempotence."""

import json

import numpy as np
import pytest
from fake_supabase import FakeSupabase
from mlflow import MlflowClient
from monitoring_helpers import CATALOG, CHAMPION_ID, RACES, client, mlflow_for, splitter, wide

from monitoring.drift import EXPERIMENT, FeatureDrift, FeatureDriftResult
from monitoring.inputs import load_inputs
from monitoring.report import DriftReport, build_report, feature_rows, report_row, write_report

IDS = {name: i for i, name in enumerate(CATALOG.names, start=1)}
WINDOW = [RACES[(2026, n)] for n in (1, 2, 3, 4, 5)]  # ids 50, 12, 44, 5, 31: not chronological
HTML = "<html><body>track_temp</body></html>"


def features(drifted: int = 0) -> list[FeatureDrift]:
    """One result per feature of the catalog; the first `drifted` of them have drifted."""
    return [
        FeatureDrift(name, IDS[name], "ks", None, 0.001 if i < drifted else 0.5, i < drifted)
        for i, name in enumerate(CATALOG.names)
    ]


def stub_drift(monkeypatch, drifted: int = 0, html: str | None = HTML):
    """Evidently is covered in `test_feature_drift`; here the drift is given so every expectation is by hand."""
    monkeypatch.setattr(
        "monitoring.report.feature_drift",
        lambda reference, current, catalog, ids: FeatureDriftResult(features(drifted), html),
    )


def inputs(**kwargs):
    return load_inputs(client(**kwargs), ml=mlflow_for(), splitter=splitter, reader=lambda c, v: wide())


def make_report(**over) -> DriftReport:
    base = {
        "model_version_id": CHAMPION_ID,
        "from_race_id": WINDOW[0],
        "through_race_id": WINDOW[-1],
        "n_races": 5,
        "current_rows": 10,
        "reference_rows": 4,
        "current_mae": 0.6333,
        "reference_mae": 0.43,
        "reference_source": "test",
        "mae_change_pct": 47.3,
        "n_drifted_features": 2,
        "rule_met": True,
        "features": features(2),
    }
    return DriftReport(**(base | over))


# ------------------------------------------------------------------------------------------------ build_report


def test_the_report_has_the_pooled_mae_the_row_counts_and_the_window_ends(monkeypatch):
    stub_drift(monkeypatch)
    report, html = build_report(inputs(), WINDOW, CATALOG, IDS)
    # HAM 0.1*round, VER 0.3*round, VER of round 1 has no error: 9 rows, 0.1 + 0.8 + 1.2 + 1.6 + 2.0 = 5.7
    assert report.current_mae == pytest.approx(5.7 / 9)
    assert report.reference_mae == 0.43 and report.reference_source == "test"
    assert report.mae_change_pct == pytest.approx((5.7 / 9 / 0.43 - 1) * 100)
    assert (report.from_race_id, report.through_race_id, report.n_races) == (RACES[(2026, 1)], RACES[(2026, 5)], 5)
    assert report.current_rows == 10 and report.reference_rows == 4  # valid rows of 5 races; 2 train races x 2
    assert report.model_version_id == CHAMPION_ID and html == HTML


def test_the_rule_holds_by_the_mae_alone(monkeypatch):
    stub_drift(monkeypatch, drifted=0)
    report, _ = build_report(inputs(), WINDOW, CATALOG, IDS)
    assert report.mae_change_pct > 15 and report.n_drifted_features == 0 and report.rule_met is True


def test_the_rule_holds_by_three_drifted_features_alone(monkeypatch):
    stub_drift(monkeypatch, drifted=3)
    report, _ = build_report(inputs(test_mae=0.6), WINDOW, CATALOG, IDS)  # 0.6333 / 0.6 = +5.6%
    assert report.mae_change_pct < 15 and report.n_drifted_features == 3 and report.rule_met is True


def test_the_rule_does_not_hold_below_both_thresholds(monkeypatch):
    stub_drift(monkeypatch, drifted=2)
    report, _ = build_report(inputs(test_mae=0.6), WINDOW, CATALOG, IDS)
    assert report.rule_met is False and report.n_drifted_features == 2


def test_a_report_without_a_testable_feature_has_no_html(monkeypatch):
    stub_drift(monkeypatch, html=None)
    assert build_report(inputs(), WINDOW, CATALOG, IDS)[1] is None


# ------------------------------------------------------------------------------------------------- the rows


def test_the_rows_are_plain_python_numbers_that_json_can_carry():
    report = make_report(
        model_version_id=np.int64(4),
        current_mae=np.float64(0.5),
        reference_rows=np.int64(9),
        rule_met=np.bool_(True),
        features=[FeatureDrift("x", np.int64(1), "ks", None, np.float64(0.2), np.bool_(False))],
    )
    row = report_row(report, "run-1", "dag-1")
    json.dumps(row)
    json.dumps(feature_rows(report, 7))
    assert row["triggered_retrain"] is False and row["dag_run_id"] == "dag-1" and row["mlflow_run_id"] == "run-1"


def test_a_missing_p_value_is_a_null_and_the_statistic_always_is():
    report = make_report(features=[FeatureDrift("x", 1, "chisquare", None, None, False)])
    assert feature_rows(report, 7) == [
        {
            "drift_report_id": 7,
            "feature_id": 1,
            "stattest": "chisquare",
            "statistic": None,
            "p_value": None,
            "is_drifted": False,
        }
    ]


# ----------------------------------------------------------------------------------------------- write_report


def runs(uri: str) -> list:
    ml = MlflowClient(uri)
    return ml.search_runs([ml.get_experiment_by_name(EXPERIMENT).experiment_id])


def test_a_new_report_makes_a_run_with_its_html_a_row_and_one_row_per_feature(uri):
    db = FakeSupabase()
    written = write_report(db, make_report(), HTML, dag_run_id="dag-1", ml=MlflowClient(uri))
    assert written.created is True and written.features_written == len(CATALOG.names)

    (run,) = runs(uri)
    assert run.info.run_id == written.mlflow_run_id and run.info.status == "FINISHED"
    assert run.data.tags["step"] == "drift" and run.data.tags["through_race_id"] == str(WINDOW[-1])
    assert run.data.tags["model_version_id"] == str(CHAMPION_ID) and run.data.metrics["mae_change_pct"] == 47.3
    ml = MlflowClient(uri)
    assert [a.path for a in ml.list_artifacts(run.info.run_id)] == ["drift_report.html"]

    (row,) = db.tables["drift_reports"]
    assert row["drift_report_id"] == written.drift_report_id and row["mlflow_run_id"] == run.info.run_id
    assert row["triggered_retrain"] is False and row["rule_met"] is True and row["dag_run_id"] == "dag-1"
    assert (row["from_race_id"], row["through_race_id"], row["n_races"]) == (WINDOW[0], WINDOW[-1], 5)
    results = db.tables["drift_feature_results"]
    assert len(results) == 15 and {r["drift_report_id"] for r in results} == {written.drift_report_id}
    assert sum(r["is_drifted"] for r in results) == 2 and all(r["statistic"] is None for r in results)


def test_a_report_with_no_html_has_a_run_but_no_artifact(uri):
    write_report(FakeSupabase(), make_report(), None, ml=MlflowClient(uri))
    (run,) = runs(uri)
    assert MlflowClient(uri).list_artifacts(run.info.run_id) == []


def test_the_same_report_again_writes_no_row_and_no_run(uri):
    db = FakeSupabase()
    first = write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    writes = list(db.writes)
    again = write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    assert again.created is False and again.features_written == 0
    assert again.drift_report_id == first.drift_report_id and again.mlflow_run_id == first.mlflow_run_id
    assert db.writes == writes and len(db.tables["drift_reports"]) == 1 and len(runs(uri)) == 1


def test_a_report_of_another_through_race_or_version_is_a_new_one(uri):
    db = FakeSupabase()
    write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    write_report(db, make_report(through_race_id=RACES[(2026, 6)]), HTML, ml=MlflowClient(uri))
    write_report(db, make_report(model_version_id=5), HTML, ml=MlflowClient(uri))
    assert (
        len(db.tables["drift_reports"]) == 3 and len(runs(uri)) == 3 and len(db.tables["drift_feature_results"]) == 45
    )


def test_a_report_whose_feature_rows_failed_is_completed_without_a_new_row_or_run(uri):
    db = FakeSupabase()
    first = write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    db.tables["drift_feature_results"] = db.tables["drift_feature_results"][:11]  # four feature rows were lost
    again = write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    assert again.created is False and again.features_written == 4 and len(db.tables["drift_feature_results"]) == 15
    assert (
        len(db.tables["drift_reports"]) == 1 and len(runs(uri)) == 1 and again.drift_report_id == first.drift_report_id
    )


class FailingInsert(FakeSupabase):
    """Refuses the first insert into `drift_reports`, as a Supabase outage right after the MLflow run would."""

    failed = False

    def table(self, name):
        query = super().table(name)
        if name == "drift_reports" and not self.failed:

            def insert(rows):
                self.failed = True
                raise ConnectionError("down")

            query.insert = insert
        return query


def test_a_failure_between_the_run_and_the_row_is_completed_by_the_next_call_with_the_same_run(uri):
    db = FailingInsert()
    with pytest.raises(RuntimeError, match="insert into drift_reports failed: ConnectionError"):
        write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    assert db.tables.get("drift_reports", []) == [] and len(runs(uri)) == 1

    written = write_report(db, make_report(), HTML, ml=MlflowClient(uri))
    assert written.created is True and len(db.tables["drift_reports"]) == 1
    assert len(runs(uri)) == 1 and written.mlflow_run_id == runs(uri)[0].info.run_id  # the run was reused


def test_a_run_that_failed_is_not_reused(uri):
    ml = MlflowClient(uri)
    experiment = ml.get_experiment_by_name(EXPERIMENT).experiment_id
    broken = ml.create_run(experiment, tags={"model_version_id": str(CHAMPION_ID), "through_race_id": str(WINDOW[-1])})
    ml.set_terminated(broken.info.run_id, "FAILED")
    written = write_report(FakeSupabase(), make_report(), HTML, ml=ml)
    assert written.mlflow_run_id != broken.info.run_id and len(runs(uri)) == 2
