"""Prediction of one race from the data up to Saturday: the champion's pace for every entry, stored in `predictions`.

The race is created and its sessions are ingested with the shared code (`ensure_race_entries`, `ingest_session`),
earlier races without ground truth are filled in with `write_ground_truth`, and the features are built with
`race_features`. Only the champion predicts; the qualifying baseline of its sampling run goes in a column of the same
row. A race already predicted by the champion's version is never overwritten.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from common.supabase_io import get_client, insert_rows, upsert_rows
from features import race_features
from features.catalog import load_catalog
from features.store import FeatureSet, RaceRef, StoreError, feature_value_rows, read_table
from ingestion.openf1_client import SessionNotAvailable
from ingestion.reference import ReferenceDataError, ensure_race_entries
from ingestion.schedule import expected_sessions, load_schedule
from ingestion.session import RAW_DIR, Loader, ingest_session
from live.ground_truth import GroundTruthError, read_race_ref, write_ground_truth
from live.history import missing_ground_truth, read_history
from live.sessions import ingest_checked
from training.explain import ALIAS, Champion, load_champion
from training.registry import REGISTERED_MODEL, feature_set_id_of
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

EXTRA_SESSIONS = ("FP2", "S")  # read by the features besides Q (which ensure_race_entries ingests); R is never ingested


class PredictionError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""


@dataclass(frozen=True)
class Models:
    champion: Champion  # training.explain.Champion: model_version_id, mlflow_run_id, sampling_run_id, algorithm, model
    baseline: Pipeline  # the qualifying baseline of the champion's sampling run


def _check_and_find(client, uri: str) -> tuple[str, str]:
    """`(champion run id, baseline run id)`, after every check that needs no model; nothing is loaded here."""
    ml = MlflowClient(uri)
    try:
        version = ml.get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
    except MlflowException:
        raise PredictionError(f"{REGISTERED_MODEL} has no '{ALIAS}' alias: run scripts/promote.py first") from None
    run_id = version.tags.get("mlflow_run_id", "")
    sampling_run_id = version.tags.get(SAMPLING_TAG, "")
    if not run_id or not sampling_run_id:
        raise PredictionError(
            f"version {version.version} of {REGISTERED_MODEL} has no mlflow_run_id or sampling_run_id tag"
        )

    rows = read_table(client, "model_versions", "model_version_id,mlflow_run_id,algorithm,feature_set_id")
    champion = next((r for r in rows if r["mlflow_run_id"] == run_id), None)
    if champion is None:
        raise PredictionError(f"the champion (run {run_id}) has no model_versions row: run scripts/train.py")
    catalog = load_catalog()
    expected = feature_set_id_of(client, catalog.version)
    if champion["feature_set_id"] != expected:
        raise PredictionError(
            f"the champion was trained with feature_set_id {champion['feature_set_id']}, not the one of the catalog "
            f"({catalog.version}, feature_set_id {expected}): predicting would feed it other features"
        )

    for row in sorted((r for r in rows if r["algorithm"] == "baseline"), key=lambda r: -r["model_version_id"]):
        try:
            tags = ml.get_run(row["mlflow_run_id"]).data.tags
        except MlflowException:
            continue  # a model_versions row whose run is gone cannot be it
        if tags.get(SAMPLING_TAG) == sampling_run_id:
            return run_id, row["mlflow_run_id"]
    raise PredictionError(
        f"there is no baseline model version for the champion's sampling run {sampling_run_id}: run scripts/baseline.py"
    )


def load_models(client=None, tracking_uri: str | None = None) -> Models:
    """The champion (`training.explain.load_champion`) and the baseline of the same sampling run.

    The baseline is the `model_versions` row with algorithm `baseline` whose run carries the champion's
    `sampling_run_id` tag (a baseline of another sampling run is never picked; the newest one wins if there are
    several). Both are loaded through `runs:/<mlflow_run_id>/model`. Raises `PredictionError` when there is no
    `champion` alias, the champion's feature set is not the one of the catalog, or there is no baseline for that
    sampling run; all of those are checked before any model is loaded. Nothing is written.
    """
    uri = resolve_tracking_uri(tracking_uri)
    try:
        client = client if client is not None else get_client()
        _, baseline_run_id = _check_and_find(client, uri)
        champion = load_champion(uri, client)
        baseline = mlflow.sklearn.load_model(f"runs:/{baseline_run_id}/model")
    except PredictionError:
        raise
    except (ValueError, StoreError) as exc:  # the messages of both are worded for the user
        raise PredictionError(str(exc)) from None
    except Exception as exc:  # MLflow or Supabase down: the type only, never a body
        raise PredictionError(f"the models could not be loaded ({type(exc).__name__})") from None
    return Models(champion, baseline)


def race_start(schedule: pd.DataFrame, year: int, round: int) -> pd.Timestamp:
    """`session_date_utc` (timezone-aware UTC) of the `R` session of the event; `PredictionError` if it has none."""
    race = schedule[(schedule["year"] == year) & (schedule["round"] == round) & (schedule["session"] == "R")]
    if race.empty:
        raise PredictionError(f"{year} round {round} has no R session in the OpenF1 schedule")
    return pd.Timestamp(race["session_date_utc"].iloc[0])


def ingest_extra_sessions(
    schedule: pd.DataFrame,
    year: int,
    round: int,
    *,
    run_id: str,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """`FP2` and `S` with `ingest_session`, only when the schedule lists them for the event.

    Each is `downloaded`, `skipped` (already on disk and valid) or `not in the event` (a sprint weekend has no
    `FP2`, a normal one has no `S`). `R` is never requested. A session that is not published, is blocked by a live
    session or is not valid raises `PredictionError` naming it.
    """
    listed = set(expected_sessions(schedule, year, round))
    return {
        session: (
            ingest_checked(
                year, round, session, run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep, error=PredictionError
            )
            if session in listed
            else "not in the event"
        )
        for session in EXTRA_SESSIONS
    }


@dataclass(frozen=True)
class RaceFeatures:
    ref: RaceRef  # the race and the entry_id of each driver code
    frame: pd.DataFrame  # `race_features`: identification columns and the features in catalog order
    sessions: dict[str, str]  # "FP2" and "S" -> "downloaded" | "skipped" | "not in the event"
    ground_truth_filled: list[tuple[int, int]]  # earlier races whose ground truth this call wrote


def build_race_features(
    year: int,
    round: int,
    schedule: pd.DataFrame,
    *,
    client,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    run_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    check: Callable[[RaceRef], None] | None = None,
) -> RaceFeatures:
    """The Saturday features of one race, with everything they need on disk and in Supabase.

    In this order: `Q` is ingested (so a blocked or invalid one is reported by name) and `ensure_race_entries`
    creates the circuit, drivers, teams, race and entries; `check(ref)` runs, if given, as soon as the race exists
    (the guard against predicting twice); `FP2` and `S` are ingested when the schedule lists them; the ground truth
    of every earlier race that lacks it is written with `write_ground_truth`, in order, stopping with the race
    named if one fails; the history is read from Supabase and `race_features` builds the features. `R` of this race
    is never ingested or read. Raises `PredictionError`; writes only what those functions write.
    """
    run_id = run_id or uuid.uuid4().hex
    label = f"{year} round {round}"
    ingest_checked(year, round, "Q", run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep, error=PredictionError)
    try:
        ensure_race_entries(
            year,
            round,
            client=client,
            run_id=run_id,
            schedule=schedule,
            ingest=partial(ingest_session, loader=loader, sleep=sleep),
            raw_dir=raw_dir,
        )
        ref = read_race_ref(client, year, round)
    except (ReferenceDataError, GroundTruthError) as exc:
        raise PredictionError(str(exc)) from None
    if check is not None:
        check(ref)

    sessions = ingest_extra_sessions(schedule, year, round, run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep)

    filled = []
    for earlier_year, earlier_round in missing_ground_truth(client, year, round):
        try:
            write_ground_truth(
                earlier_year, earlier_round, client=client, raw_dir=raw_dir, loader=loader, run_id=run_id, sleep=sleep
            )
        except (GroundTruthError, StoreError) as exc:
            raise PredictionError(
                f"the ground truth of {earlier_year} round {earlier_round} could not be filled in: {exc}"
            ) from None
        filled.append((earlier_year, earlier_round))

    history = read_history(client, year, round)
    try:
        frame = race_features(ref.info, history, raw_dir)
    except Exception as exc:
        raise PredictionError(f"the features of {label} could not be built ({type(exc).__name__}): {exc}") from None
    return RaceFeatures(ref, frame, sessions, filled)


def predict_entries(models: Models, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per entry of `frame`: `driver`, `predicted_pace_delta_pct` and `baseline_pace_delta_pct`.

    The champion gets the features of the catalog in catalog order (never the identification columns) and predicts
    every entry, valid or not; its imputer handles a missing form feature. The baseline gets `quali_delta_pct` only
    and is NaN where that is missing (it has no imputer: a number would be invented). A NaN champion prediction
    raises `PredictionError`, because `predictions.predicted_pace_delta_pct` is `NOT NULL`. Nothing is written.
    """
    names = load_catalog().names
    features = frame[names]
    champion = pd.Series(models.champion.model.predict(features), index=frame.index, dtype="float64")
    if champion.isna().any():
        drivers = ", ".join(frame.loc[champion.isna(), "driver"])
        raise PredictionError(f"the champion predicted NaN for {drivers}: nothing was written")

    baseline = pd.Series(float("nan"), index=frame.index, dtype="float64")
    has_quali = frame["quali_delta_pct"].notna()
    if has_quali.any():
        baseline[has_quali] = models.baseline.predict(frame.loc[has_quali, ["quali_delta_pct"]])
    return pd.DataFrame(
        {
            "driver": frame["driver"].to_numpy(),
            "predicted_pace_delta_pct": champion.to_numpy(),
            "baseline_pace_delta_pct": baseline.to_numpy(),
        }
    )


@dataclass(frozen=True)
class PredictionResult:
    year: int
    round: int
    run_id: str  # dag_run_id of prediction_runs and pipeline_run_id of feature_values
    prediction_run_id: int
    model_version_id: int  # the champion's
    entries: int  # predictions written
    is_late: bool
    race_start: str  # ISO UTC, the R session of the schedule
    ground_truth_filled: list[tuple[int, int]]  # earlier races whose ground truth this run wrote
    sessions: dict[str, str]  # "FP2" and "S" -> "downloaded" | "skipped" | "not in the event"


def _read_feature_set(client) -> FeatureSet:
    """The catalog's feature set as it is in Supabase (read only: the champion check already needs it to exist)."""
    catalog = load_catalog()
    set_id = feature_set_id_of(client, catalog.version)
    ids = {f["name"]: f["feature_id"] for f in read_table(client, "features", "feature_id,name")}
    missing = [name for name in catalog.names if name not in ids]
    if missing:
        raise StoreError(f"features {missing} are not in the features table: run scripts/load_features.py first")
    return FeatureSet(set_id, {name: ids[name] for name in catalog.names})


def _refuse_if_predicted(client, model_version_id: int, label: str):
    """The guard: `check` of `build_race_features`; a race already predicted by this model version stops the run."""

    def check(ref: RaceRef) -> None:
        entries = set(ref.entry_ids.values())
        done = [
            p
            for p in read_table(client, "predictions", "entry_id,model_version_id,generated_at")
            if p["model_version_id"] == model_version_id and p["entry_id"] in entries
        ]
        if done:
            when = min(str(p["generated_at"]) for p in done)
            raise PredictionError(
                f"{label} already has predictions of model version {model_version_id}, generated at {when}: "
                "nothing was written; a Saturday prediction is never replaced (delete them by hand to predict again)"
            )

    return check


def predict_race(
    year: int,
    round: int,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    schedule: pd.DataFrame | None = None,
    models: Models | None = None,
    tracking_uri: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PredictionResult:
    """Predict one race with the champion from the data up to Saturday and store it.

    The champion and the baseline are loaded first (so a missing alias stops the run before anything is written),
    then the schedule, then `build_race_features` with the guard: a race that already has predictions of the
    champion's version stops right after the race exists, writing nothing more. The features are upserted into
    `feature_values` only after every entry has a prediction; then one `prediction_runs` row and one `predictions`
    row per entry are inserted, with `generated_at` and `is_late` taken from the same `now`. If the `predictions`
    insert fails after the run, the leftover run is harmless: the guard reads `predictions`, so a rerun works.
    Every failure is a `PredictionError`; Supabase errors are reported by type only.
    """
    run_id = run_id or uuid.uuid4().hex
    label = f"{year} round {round}"
    try:
        client = client if client is not None else get_client()
        models = models if models is not None else load_models(client, tracking_uri)
        champion_id = models.champion.model_version_id
        if schedule is None:
            try:
                schedule = load_schedule(year)
            except SessionNotAvailable:
                raise PredictionError(
                    "the OpenF1 schedule is blocked: a session is live (access is restricted from 30 minutes before "
                    "a session until 30 minutes after it ends); retry 30 minutes after it ends"
                ) from None
            except Exception as exc:
                raise PredictionError(f"the OpenF1 schedule could not be loaded ({type(exc).__name__})") from None
        start = race_start(schedule, year, round)

        built = build_race_features(
            year,
            round,
            schedule,
            client=client,
            raw_dir=raw_dir,
            loader=loader,
            run_id=run_id,
            sleep=sleep,
            check=_refuse_if_predicted(client, champion_id, label),
        )
        predicted = predict_entries(models, built.frame)

        catalog = load_catalog()
        upsert_rows(
            client,
            "feature_values",
            feature_value_rows(built.frame, built.ref.entry_ids, _read_feature_set(client), catalog, run_id),
            "entry_id,feature_id,feature_set_id",
        )

        now = now or datetime.now(UTC)
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        generated_at = now.isoformat()
        is_late = now > start.to_pydatetime()
        races = read_table(client, "races", "race_id,year,round")
        race_id = next(r["race_id"] for r in races if (r["year"], r["round"]) == (year, round))
        run_row = insert_rows(
            client,
            "prediction_runs",
            [
                {
                    "race_id": race_id,
                    "model_version_id": champion_id,
                    "dag_run_id": run_id,
                    "generated_at": generated_at,
                    "is_late": is_late,
                }
            ],
        )[0]
        rows = [
            {
                "prediction_run_id": run_row["prediction_run_id"],
                "entry_id": built.ref.entry_ids[record["driver"]],
                "model_version_id": champion_id,
                "predicted_pace_delta_pct": float(record["predicted_pace_delta_pct"]),
                "baseline_pace_delta_pct": (
                    None if pd.isna(record["baseline_pace_delta_pct"]) else float(record["baseline_pace_delta_pct"])
                ),
                "generated_at": generated_at,
            }
            for record in predicted.to_dict("records")
        ]
        insert_rows(client, "predictions", rows)
    except PredictionError:
        raise
    except (StoreError, RuntimeError) as exc:  # the messages of both name the table and the error type only
        raise PredictionError(str(exc)) from None
    return PredictionResult(
        year,
        round,
        run_id,
        run_row["prediction_run_id"],
        champion_id,
        len(rows),
        is_late,
        start.isoformat(),
        built.ground_truth_filled,
        built.sessions,
    )
