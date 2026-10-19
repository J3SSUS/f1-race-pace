"""Load circuits, drivers, teams, races and race_entries into Supabase for every event already held.

Calls `ensure_race_entries` for each race, prints one line per race and a final count, and exits with
code 1 if any race failed (the others still load). Safe to run again: all writes are upserts.

    uv run python scripts/load_reference.py                          # 2023 to the current year
    uv run python scripts/load_reference.py --from 2024 --to 2024 --round 8
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

from common.supabase_io import get_client  # noqa: E402
from ingestion.reference import ensure_race_entries  # noqa: E402
from ingestion.schedule import held_events, load_schedules  # noqa: E402

FIRST_YEAR = 2023


def parse_args(argv: list[str] | None, current_year: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first_year", type=int, default=FIRST_YEAR, help="first season (default 2023)")
    parser.add_argument("--to", dest="last_year", type=int, default=current_year, help="last season (default: current)")
    parser.add_argument("--round", type=int, default=None, help="only this round of each season")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    ensure: Callable = ensure_race_entries,
    schedule_loader: Callable = load_schedules,
    client=None,
    now: datetime | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    now = now or datetime.now(UTC)
    args = parse_args(argv, now.year)
    run_id = uuid.uuid4().hex

    schedule = schedule_loader(args.first_year, args.last_year)
    events = held_events(schedule, now)
    if args.round is not None:
        events = events[events["round"] == args.round]
    races = events[["year", "round"]].drop_duplicates().itertuples(index=False)
    races = [(int(r.year), int(r.round)) for r in races]
    out(f"run {run_id}: {len(races)} races in {args.first_year}-{args.last_year}")

    client = client or get_client()
    loaded, failed = 0, []
    for i, (year, rnd) in enumerate(races, start=1):
        try:
            result = ensure(year, rnd, client=client, run_id=run_id, schedule=schedule)
        except Exception as exc:
            failed.append((year, rnd))
            out(f"[{i}/{len(races)}] {year} round {rnd:02d}: FAILED {exc}")
            continue
        loaded += 1
        state = "created" if result.created else "unchanged"
        out(f"[{i}/{len(races)}] {year} round {rnd:02d}: {state}, {result.entries} entries")

    out(f"{loaded} of {len(races)} races loaded, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
