"""Store what actually happened in one race: the real `race_pace_delta_pct` of each driver, in `ground_truth`.

Ingests the `Q` and `R` sessions of the race (a session already on disk is skipped), computes the actual target with
the target function of `src/features/` and upserts one row per entry. It makes no prediction and writes nothing else.
Run it again for the same race and no value changes. Exits with code 1 and a `STOPPED:` line, writing nothing to
Supabase, if the race has no entries (run `scripts/load_reference.py` first), if OpenF1 has no data for the session
yet or blocks the request because a session is live (retry 30 minutes after it ends), if a session is not valid, if
a driver has no `race_entries` row or if Supabase fails.

    uv run python scripts/ground_truth.py --year 2026 --round 18
"""

import argparse
import sys
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.supabase_io import get_client  # noqa: E402
from features.store import StoreError  # noqa: E402
from ingestion.session import RAW_DIR, Loader  # noqa: E402
from live.ground_truth import GroundTruthError, write_ground_truth  # noqa: E402


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
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] | None = None,
) -> int:
    out = out or partial(print, flush=True)
    args = parse_args(argv)
    run_id = uuid.uuid4().hex
    out(f"run {run_id}")
    try:
        result = write_ground_truth(
            args.year,
            args.round,
            client=client or get_client(),
            raw_dir=raw_dir,
            loader=loader,
            run_id=run_id,
            sleep=sleep,
        )
    except (GroundTruthError, StoreError, RuntimeError) as exc:
        out(f"STOPPED: {exc}")
        return 1
    out(", ".join(f"{session} {state}" for session, state in result.sessions.items()))
    out(f"{result.year} round {result.round}: {result.entries} entries, {result.valid_rows} valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
