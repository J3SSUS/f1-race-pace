"""`ensure_race_entries`: the race, its circuit, drivers, teams and `race_entries` in Supabase.

Entries come from the Q session on disk: the drivers of its `drivers` table that also have a row in its
`session_result`. Everything is validated before the first write, so a race that cannot load leaves
Supabase untouched. Writes are upserts on the natural keys of the schema, so a rerun changes nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.supabase_io import get_client, upsert_rows
from ingestion.reference_config import Circuit, canonical_team, load_circuits, load_team_aliases
from ingestion.schedule import load_schedule
from ingestion.session import RAW_DIR, ingest_session, read_meta, session_dir

# Tables of the Q session that `race_entries` needs; other tables may be invalid (e.g. no laps)
NEEDED_TABLES = ("drivers", "session_result")


class ReferenceDataError(Exception):
    """A race whose reference rows cannot be built; nothing was written for it."""


@dataclass
class EntriesResult:
    year: int
    round: int
    race_id: int
    entries: int  # rows in race_entries for this race after the call
    created: bool  # False when the race and every entry already existed


def _race_label(year: int, round: int) -> str:
    return f"{year} round {round}"


def _check_q_usable(meta: dict | None, label: str) -> None:
    if meta is None:
        raise ReferenceDataError(f"{label}: the Q session is not on disk")
    for table in NEEDED_TABLES:
        if meta["rows"].get(table, 0) < 1:
            raise ReferenceDataError(f"{label}: the Q session has no rows in `{table}`")
    errors = [e for e in meta["errors"] if e.startswith(NEEDED_TABLES)]
    if errors:
        raise ReferenceDataError(f"{label}: the Q session has invalid `drivers` or `session_result`: {errors}")


def _qualifiers(q_dir: Path, label: str) -> pd.DataFrame:
    """Drivers of the Q session that have a `session_result` row, one per driver code."""
    drivers = pd.read_parquet(q_dir / "drivers.parquet")
    results = pd.read_parquet(q_dir / "session_result.parquet")
    drivers = drivers[drivers["driver_number"].isin(results["driver_number"])]
    missing = drivers[drivers[["name_acronym", "team_name"]].isna().any(axis=1)]
    if not missing.empty:
        numbers = sorted(missing["driver_number"].tolist())
        raise ReferenceDataError(f"{label}: drivers {numbers} have no code or team in the Q `drivers` table")
    names = drivers.groupby("name_acronym")["full_name"].nunique(dropna=False)
    clash = sorted(names[names > 1].index)
    if clash:
        raise ReferenceDataError(f"{label}: driver code(s) {clash} arrive with two different names")
    return drivers.drop_duplicates("name_acronym").reset_index(drop=True)


def _event(schedule: pd.DataFrame, year: int, round: int, label: str) -> pd.DataFrame:
    event = schedule[(schedule["year"] == year) & (schedule["round"] == round)]
    if event.empty:
        raise ReferenceDataError(f"{label}: not in the OpenF1 schedule")
    return event


def _utc_date(event: pd.DataFrame, session: str) -> str | None:
    rows = event[event["session"] == session]
    return None if rows.empty else rows["session_date_utc"].iloc[0].date().isoformat()


def _existing(client, race: dict) -> tuple[int | None, set[int]]:
    """The race_id of an existing race and the driver ids already entered in it."""
    found = client.table("races").select("race_id").eq("year", race["year"]).eq("round", race["round"]).execute().data
    if not found:
        return None, set()
    race_id = found[0]["race_id"]
    entries = client.table("race_entries").select("driver_id").eq("race_id", race_id).execute().data
    return race_id, {row["driver_id"] for row in entries}


def ensure_race_entries(
    year: int,
    round: int,
    *,
    client=None,
    run_id: str | None = None,
    schedule: pd.DataFrame | None = None,
    circuits: dict[int, Circuit] | None = None,
    aliases: dict[str, str] | None = None,
    ingest: Callable = ingest_session,
    raw_dir: Path = RAW_DIR,
) -> EntriesResult:
    label = _race_label(year, round)

    result = ingest(year, round, "Q", run_id=run_id, raw_dir=raw_dir)
    if result.status == "failed":
        raise ReferenceDataError(f"{label}: the Q session could not be ingested")
    q_dir = session_dir(year, round, "Q", raw_dir)
    _check_q_usable(read_meta(q_dir), label)
    qualifiers = _qualifiers(q_dir, label)

    event = _event(load_schedule(year) if schedule is None else schedule, year, round, label)
    circuit_key = int(event["circuit_key"].iloc[0])
    circuits = load_circuits() if circuits is None else circuits
    if circuit_key not in circuits:
        raise ReferenceDataError(f"{label}: circuit_key {circuit_key} is not in config/circuits.yml")
    circuit = circuits[circuit_key]
    aliases = load_team_aliases() if aliases is None else aliases
    teams = qualifiers["team_name"].map(lambda name: canonical_team(name, aliases))

    client = client or get_client()
    race = {
        "year": year,
        "round": round,
        "event_name": event["event_name"].iloc[0],
        "quali_date": _utc_date(event, "Q"),
        "race_date": _utc_date(event, "R"),
        "is_sprint_weekend": bool((event["session"] == "S").any()),
    }
    existing_race_id, existing_drivers = _existing(client, race)

    circuit_row = {"name": circuit.name, "country": circuit.country, "circuit_type": circuit.circuit_type}
    circuit_id = upsert_rows(client, "circuits", [circuit_row], "name")[0]["circuit_id"]
    stored_drivers = upsert_rows(
        client,
        "drivers",
        [{"code": c, "full_name": n} for c, n in zip(qualifiers["name_acronym"], qualifiers["full_name"], strict=True)],
        "code",
    )
    driver_id = {row["code"]: row["driver_id"] for row in stored_drivers}
    stored_teams = upsert_rows(client, "teams", [{"name": n} for n in sorted(set(teams))], "name")
    team_id = {row["name"]: row["team_id"] for row in stored_teams}
    race_id = upsert_rows(client, "races", [race | {"circuit_id": circuit_id}], "year,round")[0]["race_id"]
    entries = upsert_rows(
        client,
        "race_entries",
        [
            {"race_id": race_id, "driver_id": driver_id[code], "team_id": team_id[team]}
            for code, team in zip(qualifiers["name_acronym"], teams, strict=True)
        ],
        "race_id,driver_id",
    )
    created = existing_race_id is None or not {row["driver_id"] for row in entries} <= existing_drivers
    return EntriesResult(year, round, race_id, len(entries), created)
