"""Predict one race with the champion, and the challenger if there is one, from the data up to Saturday.

Creates the race if it is not in Supabase, ingests `Q`, `FP2` and `S` (only the ones the schedule lists; never `R`),
writes the ground truth of any earlier race that lacks it, stores the features in `feature_values` and inserts, for
each model that writes, one `prediction_runs` row and one `predictions` row per entry (with the qualifying baseline
of that model in its own column). The challenger is the version with the `challenger` alias (SPEC 24): it predicts
only races after its cutoff, and a line says why when it is skipped. It never overwrites: a model that already
predicted the race is skipped, and the run stops only if no model has anything to write. The champion is stored
first. Exits with code 1 and a `STOPPED:` line, writing no `prediction_runs` or `predictions` row, if there is no
`champion` alias or it was trained with another feature set than the catalog's, if a session is not published yet,
is blocked by a live OpenF1 session (retry 30 minutes after it ends) or is not valid, if an earlier ground truth
cannot be ingested, if the features do not pass their schema, if every model already predicted the race or if
Supabase fails. If only the challenger fails, the champion stays stored, the lines of what was written are printed
and then `STOPPED: challenger ... failed: ...` with exit code 1.

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
from live.prediction import (  # noqa: E402
    ChallengerFailed,
    Models,
    PredictionError,
    PredictionResult,
    load_models,
    predict_race,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="season of the race")
    parser.add_argument("--round", type=int, required=True, help="round of the race in its season")
    return parser.parse_args(argv)


def report(result: PredictionResult, models: Models, out: Callable[[str], None]) -> None:
    """The lines of what a call did: ground truth, sessions, the features, and one line per active model.

    A model that wrote says how many predictions; one that did not says why. The champion's line has the wording of
    SPEC 14, and the challenger's adds `, challenger` after its algorithm.
    """
    filled = ", ".join(f"{year} round {round}" for year, round in result.ground_truth_filled) or "none"
    out(f"ground truth filled: {filled}")
    out("sessions: " + ", ".join(f"{session} {state}" for session, state in result.sessions.items()))
    features = len(load_catalog().names)
    entries = result.feature_entries
    out(f"feature_values: {entries * features} rows ({entries} entries x {features} features)")
    algorithms = {m.role: m.version.algorithm for m in models.active}
    late = "yes" if result.is_late else "no"
    for model in result.models:
        label = algorithms[model.role] + (", challenger" if model.role == "challenger" else "")
        who = f"model version {model.model_version_id} ({label})"
        if model.status == "predicted":
            out(f"{result.year} round {result.round}: {model.entries} predictions of {who}, late: {late}")
        else:
            out(f"{result.year} round {result.round}: {who} skipped: {model.reason}")


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
    except ChallengerFailed as exc:  # the champion is stored: say what was written, then stop
        report(exc.result, models, out)
        out(f"STOPPED: {exc}")
        return 1
    except (PredictionError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    report(result, models, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
