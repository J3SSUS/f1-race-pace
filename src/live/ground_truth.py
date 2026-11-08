"""Ground truth of one race: what actually happened, computed from the race session and stored in `ground_truth`.

It makes no prediction. The prediction script uses data up to Saturday and never writes `ground_truth`; neither
touches the other's table.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from common.supabase_io import get_client, upsert_rows
from features.race_info import RaceInfo
from features.store import LOAD_REFERENCE, RaceRef, ground_truth_rows, read_table
from features.target import race_target
from ingestion.session import RAW_DIR, Loader
from live.sessions import ingest_checked

SESSIONS = ("Q", "R")  # qualifying defines the entries and the valid-row rule; the race gives the target


class GroundTruthError(Exception):
    """A reason to stop, worded for the user. Never carries a response body or a key."""


def read_race_ref(client, year: int, round: int) -> RaceRef:
    """The `RaceInfo` and the `entry_id` of each driver code of this one race.

    Unlike `load_reference_index`, a race that is not loaded yet (no entries) elsewhere does not matter: only the
    requested race is checked. A missing race, circuit or entries, or an entry whose driver is missing, raises
    `GroundTruthError` naming the race; the remedy is `scripts/load_reference.py`. Nothing is written.
    """
    label = f"{year} round {round}"
    races = read_table(client, "races", "race_id,year,round,circuit_id")
    race = next((r for r in races if r["year"] == year and r["round"] == round), None)
    if race is None:
        raise GroundTruthError(f"race {label} is not in races: {LOAD_REFERENCE}")

    circuits = read_table(client, "circuits", "circuit_id,name,circuit_type")
    circuit = next((c for c in circuits if c["circuit_id"] == race["circuit_id"]), None)
    if circuit is None:
        raise GroundTruthError(f"race {label} has a circuit that is not in circuits: {LOAD_REFERENCE}")

    entries = [
        e for e in read_table(client, "race_entries", "entry_id,race_id,driver_id") if e["race_id"] == race["race_id"]
    ]
    if not entries:
        raise GroundTruthError(f"race {label} has no race_entries: {LOAD_REFERENCE}")

    codes = {d["driver_id"]: d["code"] for d in read_table(client, "drivers", "driver_id,code")}
    entry_ids: dict[str, int] = {}
    for entry in entries:
        if entry["driver_id"] not in codes:
            raise GroundTruthError(
                f"race {label}: entry {entry['entry_id']} has a driver that is not in drivers: {LOAD_REFERENCE}"
            )
        entry_ids[codes[entry["driver_id"]]] = entry["entry_id"]
    return RaceRef(RaceInfo(year, round, circuit["name"], circuit["circuit_type"]), entry_ids)


def ingest_race_sessions(
    year: int,
    round: int,
    *,
    run_id: str,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """`ingest_session` for `Q` and then `R`: each one is `downloaded` or `skipped` (already on disk and valid).

    It stops at the first session that is not published, is blocked by a live session, fails after its retries or is
    not valid, with a `GroundTruthError` that names it. It reads no table.
    """
    return {
        session: ingest_checked(
            year, round, session, run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep, error=GroundTruthError
        )
        for session in SESSIONS
    }


@dataclass(frozen=True)
class GroundTruthResult:
    year: int
    round: int
    run_id: str  # also written to ground_truth.dag_run_id and to the meta.json of the sessions it downloads
    entries: int  # rows written
    valid_rows: int  # rows with is_valid_row = true
    sessions: dict[str, str]  # "Q" and "R" -> "downloaded" | "skipped"


def write_ground_truth(
    year: int,
    round: int,
    *,
    client=None,
    raw_dir: Path = RAW_DIR,
    loader: Loader | None = None,
    run_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> GroundTruthResult:
    """Read the race, ingest `Q` and `R`, compute the actual target and upsert one `ground_truth` row per entry.

    Every row is built before anything is written, and the only write is one upsert on `entry_id`, so a failure never
    leaves a race with a partial ground truth. An existing row of the race is overwritten (`dag_run_id` takes this
    run); the values do not change because the target is deterministic. Raises `GroundTruthError`, or `StoreError`
    for a driver of the race with no `race_entries` row; the Supabase error is reported by type only.
    """
    run_id = run_id or uuid.uuid4().hex
    client = client if client is not None else get_client()
    ref = read_race_ref(client, year, round)
    sessions = ingest_race_sessions(year, round, run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep)
    try:
        frame = race_target(ref.info, raw_dir)
    except Exception as exc:
        raise GroundTruthError(
            f"race {year} round {round}: the target could not be computed ({type(exc).__name__}): {exc}"
        ) from exc
    rows = ground_truth_rows(frame.assign(year=year, round=round), ref.entry_ids, run_id)
    try:
        upsert_rows(client, "ground_truth", rows, "entry_id")
    except RuntimeError as exc:
        raise GroundTruthError(str(exc)) from None
    return GroundTruthResult(year, round, run_id, len(rows), sum(row["is_valid_row"] for row in rows), sessions)
