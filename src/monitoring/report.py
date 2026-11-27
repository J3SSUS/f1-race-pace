"""Build a drift report from the shared inputs and store it (SPEC 26).

`build_report` joins the pieces of `monitoring.inputs` with `monitoring.drift`. `write_report` stores one report: the
Evidently HTML as an artifact of a run in the `drift` experiment of MLflow, then the row of `drift_reports` and its
rows of `drift_feature_results`. It is idempotent: a report that already exists (same model version and through race)
writes no row and no run, one that failed halfway is completed, and `triggered_retrain` is always false here (SPEC 27
is the one that sets it).
"""

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.supabase_io import insert_rows, upsert_rows
from features.catalog import Catalog
from features.store import read_table
from monitoring.drift import EXPERIMENT, FeatureDrift, feature_drift, mae_change_pct, rule_met
from monitoring.inputs import Inputs, current_rows, pooled_mae
from training.sampling import resolve_tracking_uri

HTML_ARTIFACT = "drift_report.html"
RESULT_KEY = "drift_report_id,feature_id"


@dataclass(frozen=True)
class DriftReport:
    model_version_id: int
    from_race_id: int
    through_race_id: int
    n_races: int
    current_rows: int
    reference_rows: int
    current_mae: float
    reference_mae: float
    reference_source: str
    mae_change_pct: float
    n_drifted_features: int
    rule_met: bool
    features: list[FeatureDrift]


@dataclass(frozen=True)
class Written:
    drift_report_id: int
    mlflow_run_id: str
    created: bool  # False if the report row was already there
    features_written: int  # `drift_feature_results` rows this call wrote (0 for a report that was complete)


def build_report(
    inputs: Inputs, window: list[int], catalog: Catalog, feature_ids: dict[str, int]
) -> tuple[DriftReport, str | None]:
    """The report of the races in `window` (oldest first) and its Evidently HTML (None if no feature could be tested).

    The current MAE is pooled over the champion's scored rows of the window, the current rows are the valid rows of
    its races, and both are compared with the champion's reference. Nothing is written.
    """
    current = current_rows(inputs.wide, window, inputs.race_order, catalog)
    current_mae = pooled_mae(inputs.results, window)
    drift = feature_drift(inputs.reference.rows, current, catalog, feature_ids)
    change = mae_change_pct(current_mae, inputs.reference.mae)
    report = DriftReport(
        model_version_id=inputs.champion.model_version_id,
        from_race_id=window[0],
        through_race_id=window[-1],
        n_races=len(window),
        current_rows=len(current),
        reference_rows=len(inputs.reference.rows),
        current_mae=current_mae,
        reference_mae=inputs.reference.mae,
        reference_source=inputs.reference.source,
        mae_change_pct=change,
        n_drifted_features=drift.n_drifted,
        rule_met=rule_met(change, drift.n_drifted),
        features=drift.features,
    )
    return report, drift.html


def _experiment_id(ml: MlflowClient) -> str:
    experiment = ml.get_experiment_by_name(EXPERIMENT)
    return experiment.experiment_id if experiment is not None else ml.create_experiment(EXPERIMENT)


def _find_run(ml: MlflowClient, experiment_id: str, report: DriftReport) -> str | None:
    """The finished run of this report, if an earlier call made it and failed before storing the row."""
    runs = ml.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.model_version_id = '{report.model_version_id}' "
            f"and tags.through_race_id = '{report.through_race_id}' and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _log_run(ml: MlflowClient, experiment_id: str, report: DriftReport, html: str | None) -> str:
    """One run of the `drift` experiment with the numbers of the report and the HTML as an artifact."""
    tags = {
        "step": "drift",
        "model_version_id": str(report.model_version_id),
        "through_race_id": str(report.through_race_id),
        "rule_met": str(report.rule_met).lower(),
    }
    run_id = ml.create_run(experiment_id, tags=tags, run_name=f"drift-v{report.model_version_id}").info.run_id
    try:
        for name, value in (
            ("mae_change_pct", report.mae_change_pct),
            ("current_mae", report.current_mae),
            ("reference_mae", report.reference_mae),
            ("n_drifted_features", report.n_drifted_features),
        ):
            ml.log_metric(run_id, name, float(value))
        if html is not None:
            ml.log_text(run_id, html, HTML_ARTIFACT)
    except Exception:
        ml.set_terminated(run_id, "FAILED")
        raise
    ml.set_terminated(run_id, "FINISHED")
    return run_id


def report_row(report: DriftReport, mlflow_run_id: str, dag_run_id: str | None) -> dict:
    """The `drift_reports` row, with plain Python numbers (the client serialises them as JSON)."""
    return {
        "model_version_id": int(report.model_version_id),
        "through_race_id": int(report.through_race_id),
        "from_race_id": int(report.from_race_id),
        "n_races": int(report.n_races),
        "current_rows": int(report.current_rows),
        "reference_rows": int(report.reference_rows),
        "current_mae": float(report.current_mae),
        "reference_mae": float(report.reference_mae),
        "reference_source": report.reference_source,
        "mae_change_pct": float(report.mae_change_pct),
        "n_drifted_features": int(report.n_drifted_features),
        "rule_met": bool(report.rule_met),
        "triggered_retrain": False,  # only SPEC 27 sets it, when it calls Airflow
        "mlflow_run_id": mlflow_run_id,
        "dag_run_id": dag_run_id,
    }


def feature_rows(report: DriftReport, drift_report_id: int) -> list[dict]:
    return [
        {
            "drift_report_id": drift_report_id,
            "feature_id": int(f.feature_id),
            "stattest": f.stattest,
            "statistic": None if f.statistic is None else float(f.statistic),
            "p_value": None if f.p_value is None else float(f.p_value),
            "is_drifted": bool(f.is_drifted),
        }
        for f in report.features
    ]


def write_report(
    client,
    report: DriftReport,
    html: str | None,
    *,
    dag_run_id: str | None = None,
    ml: MlflowClient | None = None,
    tracking_uri: str | None = None,
) -> Written:
    """Store `report`: the MLflow run, then its `drift_reports` row, then its `drift_feature_results` rows.

    A report whose row already exists (same `model_version_id` and `through_race_id`) is not stored again and gets no
    new run; only the feature rows it lacks are written, so a call after a failure completes it. A failure between the
    run and the row leaves a finished run that the next call finds by its tags and reuses. Raises on a failure of
    MLflow or Supabase; what was written stays.
    """
    existing = [
        r
        for r in read_table(client, "drift_reports", "drift_report_id,model_version_id,through_race_id,mlflow_run_id")
        if (r["model_version_id"], r["through_race_id"]) == (report.model_version_id, report.through_race_id)
    ]
    if existing:
        report_id, run_id, created = existing[0]["drift_report_id"], existing[0]["mlflow_run_id"], False
    else:
        ml = ml if ml is not None else MlflowClient(resolve_tracking_uri(tracking_uri))
        try:
            experiment_id = _experiment_id(ml)
            run_id = _find_run(ml, experiment_id, report) or _log_run(ml, experiment_id, report, html)
        except MlflowException as exc:
            raise RuntimeError(f"MLflow run of the drift report failed: {type(exc).__name__}") from None
        report_id = insert_rows(client, "drift_reports", [report_row(report, run_id, dag_run_id)])[0]["drift_report_id"]
        created = True

    stored = {
        r["feature_id"]
        for r in read_table(client, "drift_feature_results", "drift_report_id,feature_id")
        if r["drift_report_id"] == report_id
    }
    missing = [row for row in feature_rows(report, report_id) if row["feature_id"] not in stored]
    if missing:
        upsert_rows(client, "drift_feature_results", missing, RESULT_KEY)
    return Written(report_id, run_id, created, len(missing))
