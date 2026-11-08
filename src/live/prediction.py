"""Prediction of one race from the data up to Saturday: the champion's pace for every entry, stored in `predictions`.

The race is created and its sessions are ingested with the shared code (`ensure_race_entries`, `ingest_session`),
earlier races without ground truth are filled in with `write_ground_truth`, and the features are built with
`race_features`. Only the champion predicts; the qualifying baseline of its sampling run goes in a column of the same
row. A race already predicted by the champion's version is never overwritten.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from common.supabase_io import get_client
from features.catalog import load_catalog
from features.store import StoreError, read_table
from ingestion.schedule import expected_sessions
from ingestion.session import RAW_DIR, Loader
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
