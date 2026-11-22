"""Prediction of one race from the data up to Saturday: the champion's pace for every entry, stored in `predictions`.

The race is created and its sessions are ingested with the shared code (`ensure_race_entries`, `ingest_session`),
earlier races without ground truth are filled in with `write_ground_truth`, and the features are built with
`race_features`. Only the champion predicts; the qualifying baseline of its sampling run goes in a column of the same
row. A race already predicted by a model's version is never overwritten.
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
from training.registry import CHALLENGER_ALIAS, REFIT_STEP, REGISTERED_MODEL, feature_set_id_of
from training.sampling import SAMPLING_TAG, resolve_tracking_uri

EXTRA_SESSIONS = ("FP2", "S")  # read by the features besides Q (which ensure_race_entries ingests); R is never ingested


class PredictionError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""


@dataclass(frozen=True)
class ActiveModel:
    """A model that predicts in the flow: its role, its version, its own baseline and where its training ended."""

    role: str  # "champion" | "challenger"
    version: Champion  # training.explain.Champion: model_version_id, mlflow_run_id, sampling_run_id, algorithm, model
    baseline: Pipeline  # its own baseline: `find_baseline` by the kind of its run
    trained_through_race_id: int | None = None  # the challenger's cutoff race; None for the champion (never skipped)


@dataclass(frozen=True)
class Models:
    champion: Champion
    baseline: Pipeline  # the qualifying baseline of the champion's sampling run
    challenger: ActiveModel | None = None  # None when no version has the `challenger` alias
    challenger_error: str = ""  # why the alias exists but the challenger could not be loaded; "" otherwise

    @property
    def active(self) -> tuple[ActiveModel, ...]:
        """The champion first and, if it loaded, the challenger second."""
        first = ActiveModel("champion", self.champion, self.baseline)
        return (first, self.challenger) if self.challenger is not None else (first,)


def find_baseline(
    client, ml: MlflowClient, sampling_run_id: str, rows: list[dict] | None = None, *, refit: bool = False
) -> dict:
    """The `model_versions` row (`model_version_id`, `mlflow_run_id`) of the baseline of a sampling run.

    It is the row with algorithm `baseline` whose MLflow run carries the `sampling_run_id` tag; a baseline of another
    sampling run is never picked and the newest one wins. A sampling run that was refit has two baselines with the
    same tag, told apart by the `step` tag of their runs: `refit = True` takes only the one with `step = refit` (the
    baseline of a refit version), `refit = False` (the default) only the ones without it (the baseline of a version
    trained on the train part). `rows` are the `model_versions` rows when the caller has already read them. Raises
    `PredictionError` naming the sampling run when there is none. Nothing is written.
    """
    if rows is None:
        rows = read_table(client, "model_versions", "model_version_id,mlflow_run_id,algorithm")
    for row in sorted((r for r in rows if r["algorithm"] == "baseline"), key=lambda r: -r["model_version_id"]):
        try:
            tags = ml.get_run(row["mlflow_run_id"]).data.tags
        except MlflowException:
            continue  # a model_versions row whose run is gone cannot be it
        if tags.get(SAMPLING_TAG) == sampling_run_id and (tags.get("step") == REFIT_STEP) == refit:
            return row
    if refit:
        raise PredictionError(
            f"there is no refit baseline model version for the sampling run {sampling_run_id}: run scripts/refit.py"
        )
    raise PredictionError(
        f"there is no baseline model version for the champion's sampling run {sampling_run_id}: run scripts/baseline.py"
    )


def _check_feature_set(client, row: dict, who: str) -> None:
    """Raise `PredictionError` if the `model_versions` row was trained with another feature set than the catalog's."""
    catalog = load_catalog()
    expected = feature_set_id_of(client, catalog.version)
    if row["feature_set_id"] != expected:
        raise PredictionError(
            f"{who} was trained with feature_set_id {row['feature_set_id']}, not the one of the catalog "
            f"({catalog.version}, feature_set_id {expected}): predicting would feed it other features"
        )


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
    _check_feature_set(client, champion, "the champion")

    is_refit = ml.get_run(run_id).data.tags.get("step") == REFIT_STEP  # a promoted refit keeps its own kind of baseline
    baseline = find_baseline(client, ml, sampling_run_id, rows=rows, refit=is_refit)
    return run_id, baseline["mlflow_run_id"]


def _load_challenger(client, uri: str, champion_run_id: str) -> tuple[ActiveModel | None, str]:
    """`(challenger, "")`, `(None, "")` when there is none, or `(None, why)` when the alias exists but it cannot load.

    A failure here never touches the champion: it comes back as text for `predict_race` to report after the champion
    was stored. Its baseline is the one of its own kind (`find_baseline` with `refit` true when its run is a refit).
    A version that also holds the `champion` alias has nothing to add and is not a challenger. Nothing is written.
    """
    ml = MlflowClient(uri)
    try:  # the alias list tells "no challenger" from a failure: a missing alias is not a distinct MLflow error
        if CHALLENGER_ALIAS not in ml.get_registered_model(REGISTERED_MODEL).aliases:
            return None, ""
        version = ml.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER_ALIAS)
    except MlflowException as exc:
        return None, f"the '{CHALLENGER_ALIAS}' alias could not be read ({type(exc).__name__})"
    try:
        run_id = version.tags.get("mlflow_run_id", "")
        sampling_run_id = version.tags.get(SAMPLING_TAG, "")
        if not run_id or not sampling_run_id:
            raise PredictionError(
                f"version {version.version} of {REGISTERED_MODEL} has no mlflow_run_id or sampling_run_id tag"
            )
        if run_id == champion_run_id:
            return None, ""
        rows = read_table(
            client, "model_versions", "model_version_id,mlflow_run_id,algorithm,feature_set_id,trained_through_race_id"
        )
        row = next((r for r in rows if r["mlflow_run_id"] == run_id), None)
        if row is None:
            raise PredictionError(f"the challenger (run {run_id}) has no model_versions row")
        _check_feature_set(client, row, "the challenger")
        is_refit = ml.get_run(run_id).data.tags.get("step") == REFIT_STEP
        baseline_row = find_baseline(client, ml, sampling_run_id, rows=rows, refit=is_refit)
        mlflow.set_tracking_uri(uri)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        baseline = mlflow.sklearn.load_model(f"runs:/{baseline_row['mlflow_run_id']}/model")
    except (PredictionError, ValueError, StoreError) as exc:  # the messages of all are worded for the user
        return None, str(exc)
    except Exception as exc:  # MLflow or Supabase down: the type only, never a body
        return None, f"the challenger could not be loaded ({type(exc).__name__})"
    loaded = Champion(row["model_version_id"], str(version.version), run_id, sampling_run_id, row["algorithm"], model)
    return ActiveModel("challenger", loaded, baseline, row["trained_through_race_id"]), ""


def load_models(client=None, tracking_uri: str | None = None) -> Models:
    """The champion (`training.explain.load_champion`), its baseline and, if there is one, the challenger.

    The champion's baseline is the `model_versions` row with algorithm `baseline` whose run carries the champion's
    `sampling_run_id` tag and is not a refit (a baseline of another sampling run is never picked; the newest one wins
    if there are several). Both are loaded through `runs:/<mlflow_run_id>/model`. Raises `PredictionError` when there
    is no `champion` alias, the champion's feature set is not the one of the catalog, or there is no baseline for
    that sampling run; all of those are checked before any model is loaded.

    The challenger is whatever version holds the `challenger` alias, loaded after the champion. If it cannot load
    (another feature set, no `model_versions` row, no baseline of its kind, a model that does not load) the champion
    is still returned, with `challenger` None and the reason in `challenger_error`. Nothing is written.
    """
    uri = resolve_tracking_uri(tracking_uri)
    try:
        client = client if client is not None else get_client()
        champion_run_id, baseline_run_id = _check_and_find(client, uri)
        champion = load_champion(uri, client)
        baseline = mlflow.sklearn.load_model(f"runs:/{baseline_run_id}/model")
    except PredictionError:
        raise
    except (ValueError, StoreError) as exc:  # the messages of both are worded for the user
        raise PredictionError(str(exc)) from None
    except Exception as exc:  # MLflow or Supabase down: the type only, never a body
        raise PredictionError(f"the models could not be loaded ({type(exc).__name__})") from None
    challenger, challenger_error = _load_challenger(client, uri, champion_run_id)
    return Models(champion, baseline, challenger, challenger_error)


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


def _predict_with(model: Champion, baseline_model: Pipeline, frame: pd.DataFrame, who: str) -> pd.DataFrame:
    """One row per entry of `frame`: `driver`, `predicted_pace_delta_pct` and `baseline_pace_delta_pct`.

    `model` gets the features of the catalog in catalog order (never the identification columns) and predicts every
    entry, valid or not; its imputer handles a missing form feature. `baseline_model` gets `quali_delta_pct` only and
    is NaN where that is missing (it has no imputer: a number would be invented). A NaN prediction raises
    `PredictionError` naming `who`, because `predictions.predicted_pace_delta_pct` is `NOT NULL`. Nothing is written.
    """
    names = load_catalog().names
    features = frame[names]
    predicted = pd.Series(model.model.predict(features), index=frame.index, dtype="float64")
    if predicted.isna().any():
        drivers = ", ".join(frame.loc[predicted.isna(), "driver"])
        raise PredictionError(f"{who} predicted NaN for {drivers}: nothing was written")

    baseline = pd.Series(float("nan"), index=frame.index, dtype="float64")
    has_quali = frame["quali_delta_pct"].notna()
    if has_quali.any():
        baseline[has_quali] = baseline_model.predict(frame.loc[has_quali, ["quali_delta_pct"]])
    return pd.DataFrame(
        {
            "driver": frame["driver"].to_numpy(),
            "predicted_pace_delta_pct": predicted.to_numpy(),
            "baseline_pace_delta_pct": baseline.to_numpy(),
        }
    )


def predict_entries(models: Models, frame: pd.DataFrame) -> pd.DataFrame:
    """`_predict_with` for the champion and its baseline (SPEC 14)."""
    return _predict_with(models.champion, models.baseline, frame, "the champion")


@dataclass(frozen=True)
class ModelPrediction:
    """What one active model did in a call: `predicted` (rows written) or `skipped` (and why)."""

    role: str  # "champion" | "challenger"
    model_version_id: int
    status: str  # "predicted" | "skipped"
    reason: str  # why it was skipped; "" when predicted
    prediction_run_id: int | None
    entries: int  # predictions written


@dataclass(frozen=True)
class PredictionResult:
    year: int
    round: int
    run_id: str  # dag_run_id of prediction_runs and pipeline_run_id of feature_values
    prediction_run_id: int | None  # the champion's; None if it was skipped
    model_version_id: int  # the champion's
    entries: int  # the champion's predictions written (0 if it was skipped)
    is_late: bool
    race_start: str  # ISO UTC, the R session of the schedule
    ground_truth_filled: list[tuple[int, int]]  # earlier races whose ground truth this run wrote
    sessions: dict[str, str]  # "FP2" and "S" -> "downloaded" | "skipped" | "not in the event"
    models: tuple[ModelPrediction, ...] = ()  # one per active model, the champion first
    feature_entries: int = 0  # entries whose features were upserted (the whole race, whichever model wrote)


class ChallengerFailed(PredictionError):
    """The champion was stored but the challenger could not be: `result` says what was written.

    `scripts/predict.py` prints the lines of `result` and then `STOPPED:` with this message, exit code 1.
    """

    def __init__(self, message: str, result: PredictionResult):
        super().__init__(message)
        self.result = result


def _read_feature_set(client) -> FeatureSet:
    """The catalog's feature set as it is in Supabase (read only: the champion check already needs it to exist)."""
    catalog = load_catalog()
    set_id = feature_set_id_of(client, catalog.version)
    ids = {f["name"]: f["feature_id"] for f in read_table(client, "features", "feature_id,name")}
    missing = [name for name in catalog.names if name not in ids]
    if missing:
        raise StoreError(f"features {missing} are not in the features table: run scripts/load_features.py first")
    return FeatureSet(set_id, {name: ids[name] for name in catalog.names})


@dataclass(frozen=True)
class RacePlan:
    todo: list[ActiveModel]  # the models that write predictions for this race, the champion first
    skipped: dict[str, str]  # role -> why it writes nothing (already predicted, or the race is not after its cutoff)
    broken: dict[str, str]  # role -> why it cannot predict (a challenger that did not load, or has no cutoff)


def plan_race(client, models: Models, entry_ids: set[int], year: int, round: int) -> RacePlan:
    """Which active models write predictions for the race: not the ones that already did, and the challenger only
    when the race is after its `trained_through_race_id` by `(year, round)`. `entry_ids` are the entries of the race
    (empty if it is not in Supabase yet). Reads `predictions` and `races` only."""
    entries = entry_ids
    done: dict[int, str] = {}
    for p in read_table(client, "predictions", "entry_id,model_version_id,generated_at"):
        if p["entry_id"] in entries:
            done[p["model_version_id"]] = min(str(p["generated_at"]), done.get(p["model_version_id"], "~"))
    races = {r["race_id"]: (r["year"], r["round"]) for r in read_table(client, "races", "race_id,year,round")}
    todo, skipped, broken = [], {}, {}
    if models.challenger_error:
        broken["challenger"] = models.challenger_error
    for active in models.active:
        version_id = active.version.model_version_id
        if version_id in done:
            skipped[active.role] = f"already has predictions generated at {done[version_id]}"
        elif active.role == "challenger":
            cutoff = races.get(active.trained_through_race_id)
            if cutoff is None:
                broken[active.role] = f"model version {version_id} has no known trained_through_race_id"
            elif (year, round) <= cutoff:
                skipped[active.role] = (
                    f"the race is at or before its cutoff ({cutoff[0]} round {cutoff[1]}), which it trained on"
                )
            else:
                todo.append(active)
        else:
            todo.append(active)
    return RacePlan(todo, skipped, broken)


def _nothing_to_predict(plan: RacePlan, models: Models, label: str) -> PredictionError:
    """The error of a race that no active model writes: the guard of SPEC 14, now for every active model."""
    if len(models.active) == 1 and not plan.broken:  # only the champion: the wording of SPEC 14
        version_id = models.champion.model_version_id
        when = plan.skipped["champion"].removeprefix("already has predictions generated at ")
        return PredictionError(
            f"{label} already has predictions of model version {version_id}, generated at {when}: "
            "nothing was written; a Saturday prediction is never replaced (delete them by hand to predict again)"
        )
    reasons = [f"{role}: {reason}" for role, reason in {**plan.skipped, **plan.broken}.items()]
    return PredictionError(
        f"{label} has nothing to predict ({'; '.join(reasons)}): nothing was written; "
        "a Saturday prediction is never replaced (delete them by hand to predict again)"
    )


def _store_predictions(
    client, model_version_id: int, race_id: int, entry_ids: dict[str, int], predicted: pd.DataFrame, **meta
) -> tuple[int, int]:
    """Insert one `prediction_runs` row and one `predictions` row per entry of a model: `(run id, rows)`."""
    run_row = insert_rows(
        client,
        "prediction_runs",
        [
            {
                "race_id": race_id,
                "model_version_id": model_version_id,
                "dag_run_id": meta["run_id"],
                "generated_at": meta["generated_at"],
                "is_late": meta["is_late"],
            }
        ],
    )[0]
    rows = [
        {
            "prediction_run_id": run_row["prediction_run_id"],
            "entry_id": entry_ids[record["driver"]],
            "model_version_id": model_version_id,
            "predicted_pace_delta_pct": float(record["predicted_pace_delta_pct"]),
            "baseline_pace_delta_pct": (
                None if pd.isna(record["baseline_pace_delta_pct"]) else float(record["baseline_pace_delta_pct"])
            ),
            "generated_at": meta["generated_at"],
        }
        for record in predicted.to_dict("records")
    ]
    insert_rows(client, "predictions", rows)
    return run_row["prediction_run_id"], len(rows)


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
    """Predict one race with every active model from the data up to Saturday and store it.

    The models are loaded first (so a missing champion alias stops the run before anything is written), then the
    schedule, then `build_race_features` with the guard `plan_race`: the models that already predicted the race, and a
    challenger whose cutoff is not before the race, write nothing; if no model writes, the run stops right after the
    race exists with nothing more written. The features are upserted into `feature_values` once, after the first
    model to write has every entry predicted; then each model gets one `prediction_runs` row and one `predictions`
    row per entry, the champion first, with `generated_at` and `is_late` taken from the same `now`. A champion that
    cannot predict stops everything. A challenger that cannot (it did not load, predicts NaN, or Supabase fails on
    it) leaves the champion's rows in place and raises `ChallengerFailed` carrying what was written. If an insert
    fails after its run, the leftover run is harmless: the guard reads `predictions`, so a rerun works. Every failure
    is a `PredictionError`; Supabase errors are reported by type only.
    """
    run_id = run_id or uuid.uuid4().hex
    label = f"{year} round {round}"
    plan: dict[str, RacePlan] = {}
    try:
        client = client if client is not None else get_client()
        models = models if models is not None else load_models(client, tracking_uri)
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

        def guard(ref: RaceRef) -> None:
            plan["plan"] = plan_race(client, models, set(ref.entry_ids.values()), year, round)
            if not plan["plan"].todo:
                raise _nothing_to_predict(plan["plan"], models, label)

        built = build_race_features(
            year,
            round,
            schedule,
            client=client,
            raw_dir=raw_dir,
            loader=loader,
            run_id=run_id,
            sleep=sleep,
            check=guard,
        )
        todo, skipped, broken = plan["plan"].todo, plan["plan"].skipped, plan["plan"].broken
        champion = next((m for m in todo if m.role == "champion"), None)
        champion_predicted = predict_entries(models, built.frame) if champion else None

        catalog = load_catalog()
        upsert_rows(
            client,
            "feature_values",
            feature_value_rows(built.frame, built.ref.entry_ids, _read_feature_set(client), catalog, run_id),
            "entry_id,feature_id,feature_set_id",
        )

        now = now or datetime.now(UTC)
        now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        meta = {"run_id": run_id, "generated_at": now.isoformat(), "is_late": now > start.to_pydatetime()}
        races = read_table(client, "races", "race_id,year,round")
        race_id = next(r["race_id"] for r in races if (r["year"], r["round"]) == (year, round))

        outcomes: list[ModelPrediction] = []
        if champion is not None:
            run_number, n = _store_predictions(
                client, champion.version.model_version_id, race_id, built.ref.entry_ids, champion_predicted, **meta
            )
            outcomes.append(
                ModelPrediction("champion", champion.version.model_version_id, "predicted", "", run_number, n)
            )
        else:
            outcomes.append(
                ModelPrediction("champion", models.champion.model_version_id, "skipped", skipped["champion"], None, 0)
            )
    except PredictionError:
        raise
    except (StoreError, RuntimeError) as exc:  # the messages of both name the table and the error type only
        raise PredictionError(str(exc)) from None

    failures: list[str] = []
    if "challenger" in broken:
        failures.append(f"challenger failed: {broken['challenger']}")
    challenger = next((m for m in todo if m.role == "challenger"), None)
    if models.challenger is not None and challenger is None and "challenger" in skipped:
        outcomes.append(
            ModelPrediction(
                "challenger", models.challenger.version.model_version_id, "skipped", skipped["challenger"], None, 0
            )
        )
    if challenger is not None:
        version_id = challenger.version.model_version_id
        try:
            predicted = _predict_with(challenger.version, challenger.baseline, built.frame, "the challenger")
            run_number, n = _store_predictions(client, version_id, race_id, built.ref.entry_ids, predicted, **meta)
            outcomes.append(ModelPrediction("challenger", version_id, "predicted", "", run_number, n))
        except (PredictionError, StoreError, RuntimeError) as exc:
            failures.append(f"challenger (model version {version_id}) failed: {exc}")

    first = outcomes[0]
    result = PredictionResult(
        year,
        round,
        run_id,
        first.prediction_run_id,
        first.model_version_id,
        first.entries,
        meta["is_late"],
        start.isoformat(),
        built.ground_truth_filled,
        built.sessions,
        tuple(outcomes),
        len(built.frame),
    )
    if failures:
        stored = " (the champion's predictions were stored)" if champion is not None else ""
        raise ChallengerFailed("; ".join(failures) + stored, result)
    return result
