"""Draft config/circuits.yml from the OpenF1 schedule, for a person to review.

OpenF1 only says `Permanent` or `Temporary - ...`. `Permanent` becomes `permanent` and `Temporary - Street`
becomes `street`; anything else (`Temporary - Road`, a missing label) is written as `review`, and so are
hybrids, which are a human judgement. The loader rejects `review`, so edit each one before loading.

    uv run python scripts/seed_circuits.py                 # 2023 to the current year
    uv run python scripts/seed_circuits.py --force         # overwrite a file that already exists
"""

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.reference_config import CIRCUITS_PATH  # noqa: E402
from ingestion.schedule import load_schedules  # noqa: E402

FIRST_YEAR = 2023
OPENF1_TYPES = {"Permanent": "permanent", "Temporary - Street": "street"}
HEADER = (
    "# Drafted by scripts/seed_circuits.py from the OpenF1 schedule.\n"
    "# circuit_type: street | permanent | hybrid. Replace every `review` by hand; the loader rejects it.\n"
)


def build_draft(schedule) -> list[dict]:
    """One entry per `circuit_key` of the schedule, ordered by key."""
    circuits = schedule.drop_duplicates("circuit_key").sort_values("circuit_key")
    return [
        {
            "circuit_key": int(row.circuit_key),
            "name": row.circuit_short_name,
            "country": row.country_name,
            "circuit_type": OPENF1_TYPES.get(row.circuit_type, "review"),
        }
        for row in circuits.itertuples()
    ]


def main(
    argv: list[str] | None = None,
    *,
    schedule_loader: Callable = load_schedules,
    path: Path = CIRCUITS_PATH,
    now: datetime | None = None,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite the file if it already exists")
    args = parser.parse_args(argv)

    if path.exists() and not args.force:
        out(f"{path} already exists and may hold reviewed entries; use --force to overwrite it")
        return 1
    now = now or datetime.now(UTC)
    entries = build_draft(schedule_loader(FIRST_YEAR, now.year))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    pending = sum(e["circuit_type"] == "review" for e in entries)
    out(f"wrote {len(entries)} circuits to {path}; {pending} still marked review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
