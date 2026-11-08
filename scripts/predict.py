"""Predict one race with the champion from the data up to Saturday and store it in `predictions`.

Creates the race if it is not in Supabase, ingests `Q`, `FP2` and `S` (only the ones the schedule lists; never `R`),
writes the ground truth of any earlier race that lacks it, stores the features in `feature_values` and inserts one
`prediction_runs` row and one `predictions` row per entry (with the qualifying baseline in its own column). It never
overwrites: a race already predicted by the champion's model version stops it. Exits with code 1 and a `STOPPED:`
line, writing no `prediction_runs` or `predictions` row, if there is no `champion` alias or it was trained with
another feature set than the catalog's, if a session is not published yet, is blocked by a live OpenF1 session
(retry 30 minutes after it ends) or is not valid, if an earlier ground truth cannot be ingested, if the features do
not pass their schema, if the race is already predicted or if Supabase fails.

    uv run python scripts/predict.py --year 2026 --round 17
"""

import argparse
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.supabase_io import get_client  # noqa: E402
from features.catalog import load_catalog  # noqa: E402
from features.store import StoreError  # noqa: E402
from ingestion.session import RAW_DIR, Loader  # noqa: E402
from live.prediction import Models, PredictionError, load_models, predict_race  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="season of the race")
    parser.add_argument("--round", type=int, required=True, help="round of the race in its season")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    schedule: pd.DataFrame | None = None,
    models: Models | None = None,
    tracking_uri: str | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    args = parse_args(argv)
    run_id = uuid.uuid4().hex
    out(f"run {run_id}")
    try:
        client = client or get_client()
        models = models or load_models(client, tracking_uri)
        result = predict_race(
            args.year,
            args.round,
            client=client,
            raw_dir=raw_dir,
            loader=loader,
            schedule=schedule,
            models=models,
            run_id=run_id,
            now=now,
            sleep=sleep,
        )
    except (PredictionError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    filled = ", ".join(f"{year} round {round}" for year, round in result.ground_truth_filled) or "none"
    out(f"ground truth filled: {filled}")
    out("sessions: " + ", ".join(f"{session} {state}" for session, state in result.sessions.items()))
    features = len(load_catalog().names)
    out(f"feature_values: {result.entries * features} rows ({result.entries} entries x {features} features)")
    out(
        f"{result.year} round {result.round}: {result.entries} predictions of model version "
        f"{result.model_version_id} ({models.champion.algorithm}), late: {'yes' if result.is_late else 'no'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
