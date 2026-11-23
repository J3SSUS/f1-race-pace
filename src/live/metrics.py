"""The `walk_forward` metrics of a year, recomputed from what Supabase holds about its predictions and ground truth.

`evaluation_frame` only reads. Every model version that has predictions in the year is scored on its own rows
(valid ground truth with a target and a non-null baseline prediction), and so is its baseline, with
`training.metrics.score`, the only implementation of the metrics.
"""

from dataclasses import dataclass

import pandas as pd
from mlflow import MlflowClient

from common.supabase_io import get_client, upsert_rows
from features.store import StoreError, read_table
from live.prediction import PredictionError, find_baseline
from training.metrics import score
from training.registry import METRIC_NAMES, METRICS_KEY, REFIT_STEP, metric_rows
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

SPLIT = "walk_forward"


class MetricsError(Exception):
    """The metrics of a year cannot be computed or stored; the message is worded for the user and has no secret."""


FRAME_COLUMNS = ["race_id", "year", "round", "driver", "model_version_id", "predicted", "baseline", "actual"]


def evaluation_frame(client, year: int) -> pd.DataFrame:
    """One row per prediction of `year` that can be scored, with `FRAME_COLUMNS`.

    The row's entry needs a valid `ground_truth` with a target and the prediction needs a non-null
    `baseline_pace_delta_pct`, so both models are compared on the same drivers. Sorted by `(round, driver,
    model_version_id)`; with nothing to score it is an empty frame with the same columns. Nothing is written.
    """
    races = pd.DataFrame(read_table(client, "races", "race_id,year,round"), columns=["race_id", "year", "round"])
    races = races[races["year"] == year]
    truth = pd.DataFrame(
        read_table(client, "ground_truth", "entry_id,actual_pace_delta_pct,is_valid_row"),
        columns=["entry_id", "actual_pace_delta_pct", "is_valid_row"],
    )
    truth = truth[truth["is_valid_row"].fillna(False).astype(bool) & truth["actual_pace_delta_pct"].notna()]
    predictions = pd.DataFrame(
        read_table(
            client,
            "predictions",
            "entry_id,model_version_id,predicted_pace_delta_pct,baseline_pace_delta_pct",
        ),
        columns=["entry_id", "model_version_id", "predicted_pace_delta_pct", "baseline_pace_delta_pct"],
    )
    predictions = predictions[predictions["baseline_pace_delta_pct"].notna()]
    if races.empty or truth.empty or predictions.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS).astype({"race_id": "int64", "year": "int64", "round": "int64"})

    entries = pd.DataFrame(read_table(client, "race_entries", "entry_id,race_id,driver_id"))
    drivers = pd.DataFrame(read_table(client, "drivers", "driver_id,code")).rename(columns={"code": "driver"})
    frame = (
        predictions.merge(truth, on="entry_id")
        .merge(entries, on="entry_id")
        .merge(races, on="race_id")
        .merge(drivers, on="driver_id")
        .rename(
            columns={
                "predicted_pace_delta_pct": "predicted",
                "baseline_pace_delta_pct": "baseline",
                "actual_pace_delta_pct": "actual",
            }
        )
    )
    frame = frame[FRAME_COLUMNS].astype(
        {
            "race_id": "int64",
            "year": "int64",
            "round": "int64",
            "model_version_id": "int64",
            "predicted": "float64",
            "baseline": "float64",
            "actual": "float64",
        }
    )
    return frame.sort_values(["round", "driver", "model_version_id"], ignore_index=True)


def _model_rows(frame: pd.DataFrame, column: str, model_version_id: int, label: str) -> list[dict]:
    """The aggregate set (`race_id` NULL) and one set per race of one model, scored on `column` against `actual`.

    The aggregate pools every row and calls `score` once. A race with fewer than two rows, or with no pair of
    different actual values, gets no set of its own but is still in the aggregate.
    """
    if frame.empty:
        raise ValueError(f"there are no evaluated rows for {label}")
    rows = [
        {**row, "race_id": None}
        for row in metric_rows(model_version_id, score(frame["actual"], frame[column], frame), SPLIT)
    ]
    for race_id, race in frame.groupby("race_id", sort=True):
        if len(race) < 2 or race["actual"].nunique() < 2:
            continue
        rows += [
            {**row, "race_id": int(race_id)}
            for row in metric_rows(model_version_id, score(race["actual"], race[column], race), SPLIT)
        ]
    return rows


def walk_forward_rows(frame: pd.DataFrame, version_id: int, baseline_version_id: int) -> list[dict]:
    """The `metrics` rows of one model version and its baseline over the rows of `frame` (from `evaluation_frame`).

    Each model gets one aggregate set (`race_id` NULL) and one set per race, four `metric_name` values each, with
    `split = 'walk_forward'` and `segment_type = 'overall'`. The version is scored on `predicted` and the baseline
    on `baseline`, both on the same rows (those of `version_id`) and with `training.metrics.score`.
    Raises `ValueError` when the version has no rows or no race has two drivers of different actual value.
    """
    frame = frame[frame["model_version_id"] == version_id].reset_index(drop=True)
    return _model_rows(frame, "predicted", version_id, f"model version {version_id}") + _model_rows(
        frame, "baseline", baseline_version_id, f"the baseline of model version {version_id}"
    )


@dataclass(frozen=True)
class ModelMetrics:
    model_version_id: int
    algorithm: str  # "ridge" for a model, "baseline" for a baseline
    races: int  # races with at least one evaluated row of the version
    rows: int  # evaluated rows (the same ones for the version and its baseline)
    scores: dict[str, float]  # the aggregate over all evaluated rows: mae, rmse, r2, pairwise_accuracy
    baseline_of: int | None = None  # for a baseline, the model version it is scored against; None for a model


@dataclass(frozen=True)
class MetricsResult:
    year: int
    models: tuple[ModelMetrics, ...]  # each model version with predictions, followed by its baseline
    rows_written: int  # metrics rows upserted


def _aggregate(rows: list[dict], version_id: int) -> dict[str, float]:
    """The aggregate scores of one version, under the SPEC 07 names (`pairwise_acc` is `pairwise_accuracy`)."""
    names = {stored: name for name, stored in METRIC_NAMES.items()}
    return {
        names[r["metric_name"]]: r["value"]
        for r in rows
        if r["model_version_id"] == version_id and r["race_id"] is None
    }


def walk_forward_metrics(client=None, *, year: int, tracking_uri: str | None = None) -> MetricsResult:
    """Recompute and upsert the `walk_forward` metrics of `year` for every model version with predictions.

    Each version found in the predictions of the year (`evaluation_frame`) is scored with its own baseline, found
    from its sampling run (`model_versions`, then the tags of its MLflow run) with `find_baseline` of its own kind:
    the baseline of a refit for a refit version, the other one otherwise. Raises `MetricsError`, writing nothing,
    when no race has both a prediction and a valid ground truth, when a version has no `model_versions` row or run
    tag, when a baseline cannot be found, when two versions would share one baseline (its rows would be scored on
    two sets of predictions) or when Supabase or MLflow fail (reported by type). It writes only `metrics`, upserted
    on `METRICS_KEY`: rerunning duplicates nothing.
    """
    try:
        client = client if client is not None else get_client()
        frame = evaluation_frame(client, year)
        if frame.empty:
            raise MetricsError(f"no race of {year} has both a prediction and a valid ground truth")
        versions = sorted(int(v) for v in frame["model_version_id"].unique())
        ml = MlflowClient(resolve_tracking_uri(tracking_uri))
        model_versions = read_table(client, "model_versions", "model_version_id,mlflow_run_id,algorithm")
        by_id = {r["model_version_id"]: r for r in model_versions}
        pairs: list[tuple[dict, dict]] = []
        for version_id in versions:
            version = by_id.get(version_id)
            if version is None:
                raise MetricsError(f"model version {version_id} has no model_versions row")
            tags = ml.get_run(version["mlflow_run_id"]).data.tags
            sampling_run_id = tags.get(SAMPLING_TAG, "")
            if not sampling_run_id:
                raise MetricsError(f"the run of model version {version_id} has no {SAMPLING_TAG} tag")
            baseline = find_baseline(
                client, ml, sampling_run_id, rows=model_versions, refit=tags.get("step") == REFIT_STEP
            )
            shared = next((v for v, b in pairs if b["model_version_id"] == baseline["model_version_id"]), None)
            if shared is not None:
                raise MetricsError(
                    f"model versions {shared['model_version_id']} and {version_id} share the baseline "
                    f"{baseline['model_version_id']}: it cannot be scored on two sets of predictions"
                )
            pairs.append((version, baseline))
        rows = [
            row
            for version, baseline in pairs
            for row in walk_forward_rows(frame, version["model_version_id"], baseline["model_version_id"])
        ]
        stored = upsert_rows(client, "metrics", rows, METRICS_KEY)
    except MetricsError:
        raise
    except (
        PredictionError,
        StoreError,
        ValueError,
        RuntimeError,
    ) as exc:  # the messages of all are worded for the user
        raise MetricsError(str(exc)) from None
    except Exception as exc:  # MLflow or Supabase down: the type only, never a body
        raise MetricsError(f"the metrics of {year} could not be computed ({type(exc).__name__})") from None

    models: list[ModelMetrics] = []
    for version, baseline in pairs:
        mine = frame[frame["model_version_id"] == version["model_version_id"]]
        races, n = int(mine["race_id"].nunique()), len(mine)
        version_id, baseline_id = version["model_version_id"], baseline["model_version_id"]
        models.append(ModelMetrics(version_id, version["algorithm"], races, n, _aggregate(rows, version_id)))
        models.append(
            ModelMetrics(baseline_id, "baseline", races, n, _aggregate(rows, baseline_id), baseline_of=version_id)
        )
    return MetricsResult(year, tuple(models), len(stored))
