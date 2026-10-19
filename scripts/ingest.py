"""Backfill the raw OpenF1 data of every session of the events already held, then report the coverage.

Sessions already on disk are skipped, so the same command resumes after an interruption. It writes
`data/reports/coverage.csv` and exits with code 1 if fewer than 95% of the expected sessions are valid.

    uv run python scripts/ingest.py                          # 2023 to the current year
    uv run python scripts/ingest.py --from 2024 --to 2024 --round 8
    uv run python scripts/ingest.py --force                  # download again what is already on disk
"""

import argparse
import logging
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.coverage import REPORT_PATH, build_coverage, format_summary, summary, write_report  # noqa: E402
from ingestion.schedule import held_events, load_schedules  # noqa: E402
from ingestion.session import RAW_DIR, ingest_session  # noqa: E402

FIRST_YEAR = 2023


def parse_args(argv: list[str] | None, current_year: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first_year", type=int, default=FIRST_YEAR, help="first season (default 2023)")
    parser.add_argument("--to", dest="last_year", type=int, default=current_year, help="last season (default: current)")
    parser.add_argument("--round", type=int, default=None, help="only this round of each season")
    parser.add_argument("--force", action="store_true", help="download again sessions already on disk")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    ingest: Callable = ingest_session,
    schedule_loader: Callable = load_schedules,
    now: datetime | None = None,
    raw_dir: Path = RAW_DIR,
    report_path: Path = REPORT_PATH,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)  # flush: progress shows up when stdout goes to a file
    now = now or datetime.now(UTC)
    args = parse_args(argv, now.year)
    run_id = uuid.uuid4().hex

    events = held_events(schedule_loader(args.first_year, args.last_year), now)
    if args.round is not None:
        events = events[events["round"] == args.round].reset_index(drop=True)
    out(f"run {run_id}: {len(events)} sessions expected in {args.first_year}-{args.last_year}")

    for i, event in events.iterrows():
        year, rnd, session = int(event["year"]), int(event["round"]), event["session"]
        result = ingest(year, rnd, session, run_id=run_id, force=args.force)
        detail = "" if result.valid is None else f" ({'valid' if result.valid else 'INVALID'})"
        out(f"[{i + 1}/{len(events)}] {year} round {rnd:02d} {session}: {result.status}{detail}")

    coverage = build_coverage(events, raw_dir)
    out(f"report: {write_report(coverage, report_path)}")
    result = summary(coverage)
    out(format_summary(result))
    return 0 if result.meets_target else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
