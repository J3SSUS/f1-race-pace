"""Ground truth of one race: what actually happened, computed from the race session and stored in `ground_truth`.

It makes no prediction. The prediction script uses data up to Saturday and never writes `ground_truth`; neither
touches the other's table.
"""

import time
from collections.abc import Callable
from pathlib import Path

from features.race_info import RaceInfo
from features.store import LOAD_REFERENCE, RaceRef, read_table
from ingestion.openf1_client import SessionNotAvailable
from ingestion.session import RAW_DIR, Loader, ingest_session, read_meta

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


def _ingest_one(
    year: int, round: int, session: str, *, run_id: str, raw_dir: Path, loader: Loader | None, sleep: Callable
) -> str:
    """`downloaded` or `skipped` for one session; raises `GroundTruthError` naming it if it cannot be used."""
    label = f"session {session} of {year} round {round}"
    base = loader
    if base is None:
        from ingestion.openf1_loader import load_session_tables as base  # the default of `ingest_session`

    blocked = False

    def watching(y: int, r: int, s: str):
        nonlocal blocked
        try:
            return base(y, r, s)
        except SessionNotAvailable:  # `ingest_session` turns every error into `failed`: keep the cause
            blocked = True
            raise

    result = ingest_session(year, round, session, run_id=run_id, loader=watching, raw_dir=raw_dir, sleep=sleep)
    if result.status == "failed":
        if blocked:
            raise GroundTruthError(
                f"{label} is blocked by OpenF1: a session is live (access is restricted from 30 minutes before "
                "a session until 30 minutes after it ends); retry 30 minutes after it ends"
            )
        raise GroundTruthError(
            f"{label} is not available: OpenF1 has no published data for it yet, or the download failed; retry later"
        )
    if not result.valid:
        errors = "; ".join((read_meta(result.path) or {}).get("errors", []))
        raise GroundTruthError(
            f"{label} is not valid: {errors}; download it again with "
            f"scripts/ingest.py --from {year} --to {year} --round {round} --force"
        )
    return result.status


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
        session: _ingest_one(year, round, session, run_id=run_id, raw_dir=raw_dir, loader=loader, sleep=sleep)
        for session in SESSIONS
    }
