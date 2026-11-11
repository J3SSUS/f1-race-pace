"""Backtest of one season: race by race, predict from the Saturday data, then write the ground truth.

The loop only calls `predict_race` (SPEC 14) and `write_ground_truth` (SPEC 13); it adds the order, the skipping of
what is done and the stop at the first failure. The `walk_forward` metrics of the year are recomputed at the end with
`walk_forward_metrics`. A race predicted by the champion's version is never predicted again.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.supabase_io import get_client
from features.store import StoreError, read_table
from ingestion.openf1_client import SessionNotAvailable
from ingestion.schedule import held_events, load_schedule
from ingestion.session import RAW_DIR, Loader
from live.ground_truth import GroundTruthError, write_ground_truth
from live.metrics import MetricsResult, walk_forward_metrics
from live.prediction import Models, PredictionError, load_models, predict_race


class BacktestError(Exception):
    """The backtest stopped; the message names the race and has no secret."""


@dataclass(frozen=True)
class RaceOutcome:
    year: int
    round: int
    prediction: str  # "predicted" | "skipped" (the champion's version had already predicted it)
    ground_truth: str  # "written" | "present"
    prediction_run_id: int | None  # of the run this call made; None when skipped
    is_late: bool | None  # of that run; None when skipped


@dataclass(frozen=True)
class BacktestResult:
    year: int
    run_id: str  # dag_run_id of the prediction_runs and pipeline_run_id of the feature_values it writes
    races: list[RaceOutcome]  # the races held in the range, in order; empty with metrics_only
    not_held: list[int]  # rounds of the range that are not held yet: left alone
    metrics: MetricsResult


def _race_state(client, year: int, round: int, champion_id: int) -> tuple[bool, bool]:
    """`(predicted, has_ground_truth)` of one race in Supabase; a race that is not there has neither."""
    race = next(
        (r for r in read_table(client, "races", "race_id,year,round") if (r["year"], r["round"]) == (year, round)), None
    )
    if race is None:
        return False, False
    entries = {
        e["entry_id"] for e in read_table(client, "race_entries", "entry_id,race_id") if e["race_id"] == race["race_id"]
    }
    if not entries:
        return False, False
    predicted = any(
        p["model_version_id"] == champion_id and p["entry_id"] in entries
        for p in read_table(client, "predictions", "entry_id,model_version_id")
    )
    with_truth = {g["entry_id"] for g in read_table(client, "ground_truth", "entry_id")}
    return predicted, entries <= with_truth


def backtest_season(
    year: int,
    *,
    from_round: int | None = None,
    to_round: int | None = None,
    metrics_only: bool = False,
    client=None,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    schedule: pd.DataFrame | None = None,
    models: Models | None = None,
    tracking_uri: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_race: Callable[[RaceOutcome], None] | None = None,
) -> BacktestResult:
    """Backtest the races held of `year` in `[from_round, to_round]`, then recompute the `walk_forward` metrics.

    Per race, in round order: predict unless the champion's version already did, then write the ground truth unless
    the race has it. The first failure raises `BacktestError` naming the race; what earlier races wrote stays and a
    rerun continues. `on_race` is called as each race ends. With `metrics_only` nothing is ingested, predicted or
    loaded: only the metrics are recomputed. `MetricsError` of the final step is raised as it is.
    """
    run_id = run_id or uuid.uuid4().hex
    client = client if client is not None else get_client()
    races: list[RaceOutcome] = []
    not_held: list[int] = []
    if not metrics_only:
        try:
            models = models if models is not None else load_models(client, tracking_uri)
        except (PredictionError, StoreError, RuntimeError) as exc:
            raise BacktestError(str(exc)) from None
        if schedule is None:
            try:
                schedule = load_schedule(year)
            except SessionNotAvailable:
                raise BacktestError(
                    "the OpenF1 schedule is blocked: a session is live (access is restricted from 30 minutes "
                    "before a session until 30 minutes after it ends); retry 30 minutes after it ends"
                ) from None
            except Exception as exc:
                raise BacktestError(f"the OpenF1 schedule could not be loaded ({type(exc).__name__})") from None
        now = now or datetime.now(UTC)
        in_range = sorted(int(r) for r in schedule.loc[schedule["year"] == year, "round"].dropna().unique())
        in_range = [
            r for r in in_range if (from_round is None or r >= from_round) and (to_round is None or r <= to_round)
        ]
        held = set(held_events(schedule, now).query("year == @year")["round"].astype(int))
        not_held = [r for r in in_range if r not in held]
        champion_id = models.champion.model_version_id

        for round in (r for r in in_range if r in held):
            label = f"{year} round {round}"
            try:
                predicted, has_truth = _race_state(client, year, round, champion_id)
                result = None
                if not predicted:
                    result = predict_race(
                        year,
                        round,
                        client=client,
                        raw_dir=raw_dir,
                        loader=loader,
                        schedule=schedule,
                        models=models,
                        run_id=run_id,
                        now=now,
                        sleep=sleep,
                    )
                if not has_truth:
                    write_ground_truth(
                        year, round, client=client, raw_dir=raw_dir, loader=loader, run_id=run_id, sleep=sleep
                    )
                    truth = "written"
                else:
                    truth = "present"
            except (PredictionError, GroundTruthError, StoreError, RuntimeError) as exc:
                raise BacktestError(f"{label}: {exc}") from None
            outcome = RaceOutcome(
                year,
                round,
                "skipped" if result is None else "predicted",
                truth,
                None if result is None else result.prediction_run_id,
                None if result is None else result.is_late,
            )
            races.append(outcome)
            if on_race is not None:
                on_race(outcome)

    metrics = walk_forward_metrics(client, year=year, tracking_uri=tracking_uri)
    return BacktestResult(year, run_id, races, not_held, metrics)
