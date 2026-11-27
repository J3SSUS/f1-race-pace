"""What a drift report reads (SPEC 26): the champion, its reference MAE and rows, and the current MAE and rows.

Everything here only reads, from Supabase and MLflow; nothing is written. `load_inputs` reads once what every report of
a run shares (the champion, its scored predictions, the wide feature matrix and the reference), so the backfill of a
season does not read the feature store once per report. The window and the rule are in `monitoring.drift`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common.supabase_io import get_client
from features.catalog import Catalog, load_catalog
from features.store import StoreError, read_table, read_wide
from monitoring.drift import DRIFT_WINDOW, select_window
from training.data import load_train
from training.promotion import version_test_mae
from training.refit import load_all_races
from training.registry import REFIT_STEP, feature_set_id_of
from training.sampling import SAMPLING_TAG, read_split, resolve_tracking_uri


class DriftError(Exception):
    """A drift report cannot be built; the message is worded for the user and has no secret."""


@dataclass(frozen=True)
class Champion:
    model_version_id: int
    mlflow_run_id: str
    sampling_run_id: str
    is_refit: bool  # its run has `step = refit`: it trained on every race up to its cutoff, test included
    trained_through_race_id: int | None
    promoted_at: pd.Timestamp


@dataclass(frozen=True)
class Reference:
    mae: float
    source: str  # "test" (a champion that is not a refit) or "challenger" (a refit one)
    window_races: list[int]  # the races the reference MAE was measured on; empty when it is the test MAE
    base_race_id: int | None  # the race after which the reports of this champion count
    rows: pd.DataFrame  # the valid rows the champion trained on, one column per feature


@dataclass(frozen=True)
class Inputs:
    champion: Champion
    race_order: dict[int, tuple[int, int]]  # race_id -> (year, round)
    results: pd.DataFrame  # the champion's scored predictions: race_id, year, round, driver, abs_error
    wide: pd.DataFrame  # the wide feature matrix of the champion's feature set
    reference: Reference
    last_report_race_id: int | None  # through_race_id of this champion's latest report


def race_order(client) -> dict[int, tuple[int, int]]:
    return {r["race_id"]: (r["year"], r["round"]) for r in read_table(client, "races", "race_id,year,round")}


def load_champion(client, ml: MlflowClient) -> Champion:
    """The current champion: the one open row of `champion_history`, its `model_versions` row and its MLflow run."""
    history = [
        r
        for r in read_table(client, "champion_history", "champion_id,model_version_id,promoted_at,demoted_at")
        if r["demoted_at"] is None
    ]
    if len(history) != 1:
        raise DriftError(f"champion_history: expected one current champion, found {len(history)}")
    current = history[0]
    rows = [
        r
        for r in read_table(
            client, "model_versions", "model_version_id,mlflow_run_id,feature_set_id,trained_through_race_id"
        )
        if r["model_version_id"] == current["model_version_id"]
    ]
    if not rows:
        raise DriftError(f"the champion (model version {current['model_version_id']}) has no model_versions row")
    row = rows[0]
    try:
        tags = ml.get_run(row["mlflow_run_id"]).data.tags
    except MlflowException:
        raise DriftError(f"the champion's MLflow run {row['mlflow_run_id']} cannot be read") from None
    sampling_run_id = tags.get(SAMPLING_TAG, "")
    if not sampling_run_id:
        raise DriftError(f"the champion's run {row['mlflow_run_id']} has no {SAMPLING_TAG} tag")
    expected = feature_set_id_of(client, load_catalog().version)
    if row["feature_set_id"] != expected:
        raise DriftError(
            f"the champion was trained with feature_set_id {row['feature_set_id']}, not the one of the catalog "
            f"({expected}): its features are not the ones that are read"
        )
    return Champion(
        model_version_id=row["model_version_id"],
        mlflow_run_id=row["mlflow_run_id"],
        sampling_run_id=sampling_run_id,
        is_refit=tags.get("step") == REFIT_STEP,
        trained_through_race_id=row["trained_through_race_id"],
        promoted_at=pd.Timestamp(current["promoted_at"]),
    )


def champion_results(client, champion: Champion, order: dict[int, tuple[int, int]]) -> pd.DataFrame:
    """The champion's predictions that can be scored (a valid row with an error), from `prediction_results`.

    Columns `race_id`, `year`, `round`, `driver` and `abs_error`, sorted by `(year, round, driver)`. Races are matched
    by `(year, round)` because the view has no race id.
    """
    frame = pd.DataFrame(
        read_table(client, "prediction_results", "year,round,driver,model_version_id,abs_error,is_valid_row"),
        columns=["year", "round", "driver", "model_version_id", "abs_error", "is_valid_row"],
    )
    frame = frame[
        (frame["model_version_id"] == champion.model_version_id)
        & frame["is_valid_row"].fillna(False).astype(bool)
        & frame["abs_error"].notna()
    ]
    ids = {value: race_id for race_id, value in order.items()}
    frame = frame.assign(race_id=[ids.get((y, r)) for y, r in zip(frame["year"], frame["round"], strict=True)])
    if frame["race_id"].isna().any():
        raise DriftError("a prediction belongs to a race that is not in races")
    frame = frame.astype({"race_id": "int64", "abs_error": "float64"})
    return frame[["race_id", "year", "round", "driver", "abs_error"]].sort_values(
        ["year", "round", "driver"], ignore_index=True
    )


def pooled_mae(results: pd.DataFrame, races: list[int]) -> float:
    """The MAE over the valid rows of `races` pooled, like the test MAE (not the mean of per-race MAEs)."""
    rows = results[results["race_id"].isin(races)]
    if rows.empty:
        raise DriftError(f"no scored prediction in races {races}")
    return float(rows["abs_error"].mean())


def current_rows(
    wide: pd.DataFrame, races: list[int], order: dict[int, tuple[int, int]], catalog: Catalog
) -> pd.DataFrame:
    """The valid rows of `races` with the feature columns of the catalog, in catalog order."""
    wanted = pd.MultiIndex.from_tuples([order[r] for r in races], names=["year", "round"])
    keep = pd.MultiIndex.from_frame(wide[["year", "round"]]).isin(wanted) & wide["is_valid_row"].to_numpy()
    rows = wide[keep][catalog.names].reset_index(drop=True)
    if rows.empty:
        raise DriftError(f"races {races} have no valid row of features")
    return rows


def _ground_truth_ingested(client) -> pd.Series:
    """The moment each race's ground truth was complete: the latest `ingested_at` of its rows, by `race_id`."""
    entries = pd.DataFrame(read_table(client, "race_entries", "entry_id,race_id"))
    truth = pd.DataFrame(read_table(client, "ground_truth", "entry_id,ingested_at"))
    if entries.empty or truth.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    merged = truth.merge(entries, on="entry_id")
    merged["ingested_at"] = pd.to_datetime(merged["ingested_at"], utc=True, format="ISO8601")
    return merged.groupby("race_id")["ingested_at"].max()


def refit_window(client, champion: Champion, results: pd.DataFrame, order: dict[int, tuple[int, int]]) -> list[int]:
    """The last `DRIFT_WINDOW` races the refit champion predicted whose ground truth was in before it was promoted.

    These are the races it never saw and that its promotion was judged on (SPEC 25); its MAE on them is its reference.
    """
    done = _ground_truth_ingested(client)
    before = [r for r in results["race_id"].unique() if r in done.index and done[r] < champion.promoted_at]
    window = select_window(before, order)
    if not window.due:
        raise DriftError(
            f"the refit champion (model version {champion.model_version_id}) predicted {window.pending} races with "
            f"ground truth before its promotion, {DRIFT_WINDOW} are needed for its reference MAE"
        )
    return window.races


def load_reference(
    client,
    champion: Champion,
    results: pd.DataFrame,
    wide: pd.DataFrame,
    order: dict[int, tuple[int, int]],
    catalog: Catalog,
    tracking_uri: str | None = None,
    splitter: Callable = read_split,
) -> Reference:
    """The reference of the champion: its MAE and the rows it trained on.

    A champion that is not a refit has its overall `test` MAE and the valid rows of the train races of its sampling
    run. A refit one has no test MAE (it trained on the test races), so its reference MAE is its own over the races of
    `refit_window`, and its rows are the valid rows of every race of its sampling run, up to the cutoff.
    """
    reader = lambda _client, _version: wide  # noqa: E731  # the matrix is already read, once per run
    if champion.is_refit:
        window = refit_window(client, champion, results, order)
        mae, source, base = pooled_mae(results, window), "challenger", window[-1]
        rows = load_all_races(
            champion.sampling_run_id,
            client,
            version=catalog.version,
            reader=reader,
            splitter=splitter,
            tracking_uri=tracking_uri,
        ).X
    else:
        metrics = read_table(client, "metrics", "model_version_id,split,metric_name,value")
        test_mae = version_test_mae(metrics, champion.model_version_id)
        if test_mae is None:
            raise DriftError(
                f"the champion (model version {champion.model_version_id}) has no test MAE: run scripts/evaluate.py"
            )
        window, mae, source, base = [], test_mae, "test", champion.trained_through_race_id
        rows = load_train(
            champion.sampling_run_id,
            client,
            version=catalog.version,
            reader=reader,
            splitter=splitter,
            tracking_uri=tracking_uri,
        ).X
    return Reference(mae=mae, source=source, window_races=window, base_race_id=base, rows=rows[catalog.names])


def last_report_race(client, champion: Champion, order: dict[int, tuple[int, int]]) -> int | None:
    """`through_race_id` of the latest report of this champion (by year and round), or None. Reports of other
    versions do not count: the counter starts again when the champion changes."""
    races = [
        r["through_race_id"]
        for r in read_table(client, "drift_reports", "model_version_id,through_race_id")
        if r["model_version_id"] == champion.model_version_id
    ]
    return max(races, key=order.__getitem__) if races else None


def load_inputs(
    client=None,
    tracking_uri: str | None = None,
    *,
    ml: MlflowClient | None = None,
    splitter: Callable = read_split,
    reader: Callable = read_wide,
) -> Inputs:
    """Everything the reports of one run share, read once. Raises `DriftError` or `StoreError` with a clear message."""
    client = client if client is not None else get_client()
    ml = ml if ml is not None else MlflowClient(resolve_tracking_uri(tracking_uri))
    catalog = load_catalog()
    order = race_order(client)
    champion = load_champion(client, ml)
    results = champion_results(client, champion, order)
    try:
        wide = reader(client, catalog.version)
    except StoreError as exc:
        raise DriftError(str(exc)) from None
    reference = load_reference(client, champion, results, wide, order, catalog, tracking_uri, splitter)
    return Inputs(champion, order, results, wide, reference, last_report_race(client, champion, order))
