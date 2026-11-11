"""Backtest a season race by race and recompute its `walk_forward` metrics.

For every race already held in the range (the last session of its event ended), in round order: predicts it with the
champion from the data up to Saturday unless the champion's version already did (never overwritten), then writes its
ground truth unless it has it. Then recomputes the `walk_forward` metrics of the champion and the qualifying baseline
on the same rows and stores them in `metrics`. A rerun skips what is done. `--metrics-only` ingests and predicts
nothing: it only recomputes the metrics (MLflow is still needed to find the baseline). Exits with code 1 and a
`STOPPED:` line at the first race that fails (what earlier races wrote stays), if the OpenF1 schedule is blocked by a
live session (retry 30 minutes after it ends), if there is no valid champion, or if the metrics cannot be computed.

    uv run python scripts/backtest.py --year 2026 [--from-round 1] [--to-round 16] [--metrics-only]
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
from features.store import StoreError  # noqa: E402
from ingestion.session import RAW_DIR, Loader  # noqa: E402
from live.backtest import BacktestError, RaceOutcome, backtest_season  # noqa: E402
from live.metrics import MetricsError, MetricsResult  # noqa: E402
from live.prediction import Models  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True, help="season to backtest")
    parser.add_argument("--from-round", type=int, help="first round of the range (default: the first)")
    parser.add_argument("--to-round", type=int, help="last round of the range (default: the last)")
    parser.add_argument("--metrics-only", action="store_true", help="only recompute the metrics")
    return parser.parse_args(argv)


def race_line(outcome: RaceOutcome) -> str:
    if outcome.prediction == "skipped":
        prediction = "skipped (already predicted)"
    else:
        prediction = f"predicted ({'late' if outcome.is_late else 'on time'})"
    return f"{outcome.year} round {outcome.round}: {prediction}, ground truth {outcome.ground_truth}"


def rounds_text(rounds: list[int]) -> str:
    """`18 to 24` for consecutive rounds, otherwise the list."""
    if len(rounds) > 2 and rounds == list(range(rounds[0], rounds[-1] + 1)):
        return f"rounds {rounds[0]} to {rounds[-1]}"
    return ("round " if len(rounds) == 1 else "rounds ") + ", ".join(map(str, rounds))


def metrics_lines(metrics: MetricsResult) -> list[str]:
    champion, baseline = metrics.champion, metrics.baseline
    return [
        f"metrics: {champion.races} races, {champion.rows} rows",
        f"model version {champion.model_version_id} ({champion.algorithm}): MAE {champion.scores['mae']:.3f}, "
        f"baseline (version {baseline.model_version_id}): MAE {baseline.scores['mae']:.3f}",
    ]


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
    if not args.metrics_only:
        out(f"run {run_id}")
    try:
        client = client or get_client()
        result = backtest_season(
            args.year,
            from_round=args.from_round,
            to_round=args.to_round,
            metrics_only=args.metrics_only,
            client=client,
            raw_dir=raw_dir,
            loader=loader,
            schedule=schedule,
            models=models,
            tracking_uri=tracking_uri,
            run_id=run_id,
            now=now,
            sleep=sleep,
            on_race=lambda outcome: out(race_line(outcome)),
        )
    except (BacktestError, MetricsError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    if result.not_held:
        out(f"not held yet: {rounds_text(result.not_held)}")
    for line in metrics_lines(result.metrics):
        out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
