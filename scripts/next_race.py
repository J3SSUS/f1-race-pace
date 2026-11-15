"""Print the race a DAG runs for, as JSON: `{"year": 2026, "round": 21}`. Reads only the OpenF1 schedule.

`--which next` is the first round of the current year (UTC) whose `R` session has not started, so from the qualifying
Saturday until the lights go out it points to that race. `--which last` is the latest round of the current year whose
last session ended. `--year` and `--round`, given together, return that race after checking it is in the schedule (an
empty string counts as not given, which is how a null DAG param arrives). With `--output PATH` the JSON is also
written there (the DAGs pass `/airflow/xcom/return.json`). Exits with code 1 and a `STOPPED:` line when only one of
`--year` and `--round` is given, when the race is not in the schedule, when no race is left or none has been held, or
when OpenF1 has no schedule to give (a live session blocks it: retry 30 minutes after it ends).

    uv run python scripts/next_race.py --which next
    uv run python scripts/next_race.py --which last --output /airflow/xcom/return.json
    uv run python scripts/next_race.py --which next --year 2026 --round 20
"""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ingestion.openf1_client import OpenF1Error  # noqa: E402
from ingestion.schedule import load_schedule, resolve_race  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", choices=["next", "last"], required=True, help="which race when none is given")
    parser.add_argument("--year", default="", help="season of the race (with --round); empty means not given")
    parser.add_argument("--round", default="", help="round of the race (with --year); empty means not given")
    parser.add_argument("--output", type=Path, help="also write the JSON to this file")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    schedule_loader: Callable[[int], pd.DataFrame] = load_schedule,
    now: datetime | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    args = parse_args(argv)
    now = now or datetime.now(UTC)
    try:
        year = int(args.year) if str(args.year).strip() else None
        schedule_year = year if year is not None else now.year
        schedule = schedule_loader(schedule_year)
        race = resolve_race(schedule, args.which, args.year, args.round, now)
    except (ValueError, OpenF1Error) as exc:
        out(f"STOPPED: {exc}")
        return 1
    text = json.dumps(race)
    if args.output:
        args.output.write_text(text)
    out(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
