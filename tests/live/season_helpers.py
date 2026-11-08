"""Three synthetic races of 2024 as the prediction tests need them: raw sessions, schedule and loaded reference rows."""

import json
from pathlib import Path

import pandas as pd
from fake_supabase import FakeSupabase
from synthetic_race import practice_tables, qualifying_tables, write_race, write_session
from test_schedule import NORMAL, meeting, schedule_of

from features.race_info import RaceInfo
from ingestion.reference import ensure_race_entries
from ingestion.reference_config import Circuit
from ingestion.session import IngestResult

CIRCUITS = {
    k: Circuit(circuit_key=k, name="Synthetic", country="Nowhere", circuit_type="permanent") for k in (101, 102, 103)
}
INFO = {r: RaceInfo(2024, r, "Synthetic", "permanent") for r in (1, 2, 3)}


def mark_valid(raw_dir: Path, round: int, session: str) -> None:
    meta_path = raw_dir / "2024" / f"{round:02d}" / session / "meta.json"
    meta_path.write_text(json.dumps(json.loads(meta_path.read_text()) | {"valid": True, "errors": []}))


def write_season(raw_dir: Path) -> None:
    """Rounds 1 and 2 with Q and R (clean, safety car) and round 3 with Saturday only: Q and FP2, no R."""
    for round, scenario in ((1, "clean"), (2, "safety_car")):
        write_race(raw_dir, 2024, round, scenario)
        mark_valid(raw_dir, round, "Q")
        mark_valid(raw_dir, round, "R")
    write_session(raw_dir, 2024, 3, "Q", qualifying_tables())
    write_session(raw_dir, 2024, 3, "FP2", practice_tables())
    mark_valid(raw_dir, 3, "Q")
    mark_valid(raw_dir, 3, "FP2")


def season_schedule() -> pd.DataFrame:
    """Three normal weekends one week apart; the race of round 3 starts on 2024-03-19 00:00 UTC."""
    return schedule_of(
        meeting(1, NORMAL, "2024-03-01"), meeting(2, NORMAL, "2024-03-08"), meeting(3, NORMAL, "2024-03-15")
    )


def skipped(year, round, session, **kwargs):
    return IngestResult(year, round, session, "skipped", True, Path("."))


def season_client(raw_dir: Path, schedule: pd.DataFrame) -> FakeSupabase:
    """Supabase with rounds 1 and 2 loaded (race, entries) but no ground truth; round 3 does not exist yet."""
    client = FakeSupabase()
    for round in (1, 2):
        ensure_race_entries(
            2024,
            round,
            client=client,
            schedule=schedule,
            ingest=skipped,
            raw_dir=raw_dir,
            circuits=CIRCUITS,
            aliases={},
        )
    client.writes.clear()
    return client
