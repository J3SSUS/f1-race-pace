"""The `walk_forward` metrics of a year, recomputed from what Supabase holds about its predictions and ground truth.

`evaluation_frame` only reads. The champion and the baseline are scored on the same rows (valid ground truth with a
target and a non-null baseline prediction), with `training.metrics.score`, the only implementation of the metrics.
"""

import pandas as pd

from features.store import read_table
from training.metrics import score
from training.registry import metric_rows

SPLIT = "walk_forward"
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


def walk_forward_rows(frame: pd.DataFrame, champion_version_id: int, baseline_version_id: int) -> list[dict]:
    """The `metrics` rows of the champion and the baseline over the rows of `frame` (from `evaluation_frame`).

    Each model gets one aggregate set (`race_id` NULL) and one set per race, four `metric_name` values each, with
    `split = 'walk_forward'` and `segment_type = 'overall'`. The champion is scored on `predicted` and the baseline
    on `baseline`, both on the same rows (those of the champion's version) and with `training.metrics.score`.
    Raises `ValueError` when the champion has no rows or no race has two drivers of different actual value.
    """
    frame = frame[frame["model_version_id"] == champion_version_id].reset_index(drop=True)
    return _model_rows(frame, "predicted", champion_version_id, "the champion") + _model_rows(
        frame, "baseline", baseline_version_id, "the baseline"
    )
