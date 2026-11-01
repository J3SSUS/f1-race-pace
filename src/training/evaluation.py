"""The evaluation of the candidate and the baseline on the test races.

This is the only module that reads the test target. The test is touched once, only for the candidate and the
baseline, after the selection has been made from the out-of-fold predictions of train.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from common.config import TrainingConfig, load_config
from common.supabase_io import get_client, upsert_rows
from features.store import read_table, read_wide
from training.baseline import BASELINE_FEATURE
from training.data import NOT_FEATURES, TARGET, TrainSet
from training.metrics import score
from training.registry import METRICS_KEY, REGISTERED_MODEL, metric_rows
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri
from training.selection import BASELINE, Selection, load_oof, select
from training.step import finished_run_id

EVALUATE = "evaluate"  # run name of the step
N_BOOT = 1000
REPORTED = ("mae", "rmse", "r2", "pairwise_accuracy")  # the SPEC 07 metrics that go to `metrics`


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class

    name: str
    metrics: dict[str, float]  # mae, rmse, r2, pairwise_accuracy (SPEC 07 `score`)
    intervals: dict[str, tuple[float, float]]  # 2.5 and 97.5 percentiles, block bootstrap by race
    mean_error: float  # bias
    error_quantiles: dict[str, float]  # median, p90, p95, max of |y_pred - y_true|
    n_rows: int
    n_races: int


def load_test(
    run_id: str,
    client=None,
    *,
    version: str = "v1",
    tracking_uri: str | None = None,
    reader: Callable = read_wide,
    splitter: Callable = read_split,
) -> TrainSet:
    """Only the entries of the test races of the sampling run `run_id` whose row is valid.

    Same shape as `load_train`, on the other side of the split. The split is read first, so an unknown run stops
    before Supabase is touched. Train races and invalid rows are dropped before the `TrainSet` is built.
    """
    split = splitter(run_id, tracking_uri)
    test_races = pd.MultiIndex.from_frame(split.loc[split["split"] == "test", ["year", "round"]])
    wide = reader(client if client is not None else get_client(), version)

    keep = pd.MultiIndex.from_frame(wide[["year", "round"]]).isin(test_races) & wide["is_valid_row"].to_numpy()
    rows = wide[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"The test races of run '{run_id}' have no valid row in feature set {version}")

    features = [c for c in rows.columns if c not in NOT_FEATURES]
    return TrainSet(
        X=rows[features],
        y=rows[TARGET].astype("float64"),
        meta=rows[["entry_id", "year", "round", "driver"]],
    )


def _bootstrap_intervals(y_true, y_pred, races: pd.DataFrame, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """2.5 and 97.5 percentiles of each reported metric over `n_boot` resamples of whole races.

    A resample can repeat a race, so every drawn block gets its own label before `score` groups the rows by race.
    """
    rows_of_race = list(races.groupby(["year", "round"]).indices.values())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rows_of_race), size=(n_boot, len(rows_of_race)))
    draws: dict[str, list[float]] = {name: [] for name in REPORTED}
    for pick in picks:
        index = np.concatenate([rows_of_race[block] for block in pick])
        blocks = np.repeat(np.arange(len(pick)), [len(rows_of_race[block]) for block in pick])
        labels = pd.DataFrame({"year": blocks, "round": 0})
        resampled = score(y_true[index], y_pred[index], labels)
        for name in REPORTED:
            draws[name].append(resampled[name])
    return {name: tuple(float(v) for v in np.percentile(values, [2.5, 97.5])) for name, values in draws.items()}


def evaluate_on_test(model, test: TrainSet, n_boot: int = 1000, seed: int = 42, *, name: str = "") -> TestResult:
    """The test metrics of a fitted `model`, with a block bootstrap interval by race for each.

    Counts only the rows of `test` (valid rows). The qualifying baseline has no imputer, so a `NaN` in its feature
    raises here, before predicting, instead of failing in the one evaluation that cannot be repeated.
    """
    if isinstance(model, Pipeline) and "select" in model.named_steps and test.X[BASELINE_FEATURE].isna().any():
        missing = int(test.X[BASELINE_FEATURE].isna().sum())
        raise ValueError(f"the baseline cannot predict: {missing} test rows have no {BASELINE_FEATURE}")
    y_true = test.y.to_numpy(dtype="float64")
    y_pred = np.asarray(model.predict(test.X), dtype="float64")
    races = test.meta[["year", "round"]].reset_index(drop=True)
    scores = score(y_true, y_pred, races)
    abs_error = np.abs(y_pred - y_true)
    return TestResult(
        name=name,
        metrics={metric: scores[metric] for metric in REPORTED},
        intervals=_bootstrap_intervals(y_true, y_pred, races, n_boot, seed),
        mean_error=scores["mean_error"],
        error_quantiles={
            "median": float(np.median(abs_error)),
            "p90": float(np.percentile(abs_error, 90)),
            "p95": float(np.percentile(abs_error, 95)),
            "max": float(abs_error.max()),
        },
        n_rows=len(y_true),
        n_races=len(races.drop_duplicates()),
    )


def test_metric_rows(model_version_id: int, metrics: dict[str, float]) -> list[dict]:
    """The 4 `metrics` rows of a version on the test races: `split = 'test'`, overall."""
    return metric_rows(model_version_id, metrics, split="test")


def _locate(name: str, sampling_run_id: str, ml: MlflowClient, supabase, uri: str) -> tuple[str, int]:
    """`(MLflow run id, model_version_id)` of the finished run `name` of the sampling run, or `ValueError`."""
    run_id = finished_run_id(sampling_run_id, name, uri)
    if run_id is None:
        raise ValueError(f"no finished run of {name} for the sampling run {sampling_run_id}")
    versions = [
        v for v in ml.search_model_versions(f"name = '{REGISTERED_MODEL}'") if v.tags.get("mlflow_run_id") == run_id
    ]
    if not versions:
        raise ValueError(f"{name} (run {run_id}) is not a version of {REGISTERED_MODEL}: run scripts/train.py")
    rows = [
        r
        for r in read_table(supabase, "model_versions", "model_version_id,mlflow_run_id")
        if r["mlflow_run_id"] == run_id
    ]
    if not rows:
        raise ValueError(f"{name} (run {run_id}) has no model_versions row: run scripts/train.py")
    return run_id, rows[0]["model_version_id"]


def _selection_json(selection: Selection, n_boot: int, seed: int) -> dict:
    def row(c):
        return {
            "name": c.name,
            "cv_mae": c.cv_mae,
            "diff": c.diff,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "not_worse": c.not_worse,
        }

    return {
        "n_boot": n_boot,
        "seed": seed,
        "reference": selection.reference,
        "candidate": selection.candidate,
        "beats_baseline": selection.beats_baseline,
        "table": [row(c) for c in selection.table],
        "vs_baseline": row(selection.vs_baseline) if selection.vs_baseline else None,
    }


def _log_result(result: TestResult, prefix: str) -> None:
    """The test metrics of one model; `prefix` is empty for the candidate and `baseline_` for the baseline."""
    for metric, value in result.metrics.items():
        low, high = result.intervals[metric]
        mlflow.log_metrics(
            {
                f"{prefix}test_{metric}": value,
                f"{prefix}test_{metric}_ci_low": low,
                f"{prefix}test_{metric}_ci_high": high,
            }
        )
    mlflow.log_metric(f"{prefix}test_mean_error", result.mean_error)
    for name, value in result.error_quantiles.items():
        mlflow.log_metric(f"{prefix}test_err_{name}", value)


def run_evaluation(
    sampling_run_id: str,
    config: TrainingConfig | None = None,
    client=None,
    tracking_uri: str | None = None,
    loader: Callable = load_test,
) -> dict:
    """The `evaluate` step: select the candidate from the OOF, score it and the baseline once on the test races.

    The test is touched once: a FINISHED `evaluate` run for this sampling run makes it raise; a FAILED one does not
    block. Everything that can fail without the test (OOF, selection, versions, models) runs first. The 8 `metrics`
    rows go to Supabase before the MLflow run closes, so a Supabase failure leaves the run FAILED and the retry
    safe. It never promotes: no alias, no `champion_history`, no `shap_values`, no `walk_forward` metric.
    Returns `{"selection", "results", "versions", "run_id"}`.
    """
    config = config or load_config()
    uri = resolve_tracking_uri(tracking_uri)
    ml = MlflowClient(uri)
    try:
        experiment_id = ml.get_run(sampling_run_id).info.experiment_id
    except MlflowException as exc:
        raise ValueError(f"unknown sampling run {sampling_run_id}") from exc
    if finished_run_id(sampling_run_id, EVALUATE, uri) is not None:
        raise ValueError(
            f"the sampling run {sampling_run_id} already has a finished evaluate run: the test is touched once "
            "(to evaluate again, sample again)"
        )
    client = client if client is not None else get_client()
    seed = config.optuna.seed

    selection = select(load_oof(sampling_run_id, uri), N_BOOT, seed)
    names = [selection.candidate, BASELINE]
    located = {name: _locate(name, sampling_run_id, ml, client, uri) for name in names}
    mlflow.set_tracking_uri(uri)
    models = {name: mlflow.sklearn.load_model(f"runs:/{located[name][0]}/model") for name in names}

    tags = {"step": EVALUATE, SAMPLING_TAG: sampling_run_id}
    with mlflow.start_run(run_name=EVALUATE, experiment_id=experiment_id, tags=tags) as run:
        test = loader(sampling_run_id, client, tracking_uri=uri)
        results = {name: evaluate_on_test(models[name], test, N_BOOT, seed, name=name) for name in names}
        rows = [row for name in names for row in test_metric_rows(located[name][1], results[name].metrics)]
        upsert_rows(client, "metrics", rows, METRICS_KEY)

        mlflow.set_tags(
            {"candidate": selection.candidate or "", "beats_baseline": str(selection.beats_baseline).lower()}
        )
        mlflow.log_params(
            {"n_boot": N_BOOT, "seed": seed, "reference": selection.reference, "candidate": selection.candidate or ""}
        )
        for c in selection.table:
            mlflow.log_metrics(
                {
                    f"cv_mae_{c.name}": c.cv_mae,
                    f"diff_{c.name}": c.diff,
                    f"ci_low_{c.name}": c.ci_low,
                    f"ci_high_{c.name}": c.ci_high,
                }
            )
        _log_result(results[selection.candidate], "")
        _log_result(results[BASELINE], "baseline_")
        mlflow.log_dict(_selection_json(selection, N_BOOT, seed), "selection.json")
        mlflow.log_dict(
            {
                name: {
                    "metrics": r.metrics,
                    "intervals": {k: list(v) for k, v in r.intervals.items()},
                    "mean_error": r.mean_error,
                    "error_quantiles": r.error_quantiles,
                    "n_rows": r.n_rows,
                    "n_races": r.n_races,
                }
                for name, r in results.items()
            },
            "test_metrics.json",
        )
    return {
        "selection": selection,
        "results": results,
        "versions": {name: located[name][1] for name in names},
        "run_id": run.info.run_id,
    }


test_metric_rows.__test__ = False  # the name starts with "test" but it is not a pytest test
