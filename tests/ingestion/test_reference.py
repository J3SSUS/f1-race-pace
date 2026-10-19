import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from test_schedule import NORMAL, SPRINT_2024, meeting

from ingestion.reference import ReferenceDataError, ensure_race_entries
from ingestion.reference_config import Circuit
from ingestion.schedule import build_schedule
from ingestion.session import IngestResult

CIRCUITS = {101: Circuit(circuit_key=101, name="Monte Carlo", country="Monaco", circuit_type="street")}
ALIASES = {"AlphaTauri": "RB"}
DRIVERS = [
    (1, "VER", "Max VERSTAPPEN", "Red Bull Racing"),
    (4, "NOR", "Lando NORRIS", "McLaren"),
    (22, "TSU", "Yuki TSUNODA", "AlphaTauri"),
]
ID_COLUMNS = {"circuits": "circuit_id", "drivers": "driver_id", "teams": "team_id", "races": "race_id"}


class FakeSupabase:
    """In-memory tables with identity ids, upsert on the conflict columns, and select().eq()."""

    def __init__(self):
        self.tables: dict[str, list[dict]] = {}
        self.writes: list[str] = []
        self._table = None
        self._filters: list[tuple[str, object]] = []
        self._select = None
        self._pending = None

    def table(self, name):
        self._table, self._filters, self._select, self._pending = name, [], None, None
        return self

    def select(self, columns):
        self._select = columns
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def upsert(self, rows, on_conflict):
        self._pending = (rows, on_conflict.split(","))
        return self

    def execute(self):
        rows = self.tables.setdefault(self._table, [])
        if self._pending is None:
            data = [r for r in rows if all(r.get(c) == v for c, v in self._filters)]
            return type("R", (), {"data": [dict(r) for r in data]})()
        self.writes.append(self._table)
        out = []
        for new in self._pending[0]:
            key = {c: new[c] for c in self._pending[1]}
            found = next((r for r in rows if all(r.get(c) == v for c, v in key.items())), None)
            if found is None:
                found = {}
                if self._table in ID_COLUMNS:
                    found[ID_COLUMNS[self._table]] = len(rows) + 1
                rows.append(found)
            found.update(new)
            out.append(dict(found))
        return type("R", (), {"data": out})()

    def count(self, table):
        return len(self.tables.get(table, []))


def write_q(raw_dir: Path, year=2024, round=1, drivers=DRIVERS, result_numbers=None, rows=None, errors=None):
    """A Q session on disk: drivers, session_result and a meta.json."""
    path = raw_dir / str(year) / f"{round:02d}" / "Q"
    path.mkdir(parents=True)
    pd.DataFrame(
        [{"driver_number": n, "name_acronym": c, "full_name": f, "team_name": t} for n, c, f, t in drivers]
    ).to_parquet(path / "drivers.parquet")
    numbers = [d[0] for d in drivers] if result_numbers is None else result_numbers
    pd.DataFrame({"driver_number": numbers}).to_parquet(path / "session_result.parquet")
    meta = {
        "rows": rows or {"drivers": len(drivers), "session_result": len(numbers)},
        "errors": errors or [],
        "valid": not errors,
    }
    (path / "meta.json").write_text(json.dumps(meta))


def fake_ingest(status="skipped"):
    calls = []

    def ingest(year, round, session, **kwargs):
        calls.append((year, round, session))
        return IngestResult(year, round, session, status, True, Path("."))

    ingest.calls = calls
    return ingest


def schedule(names=NORMAL, year=2024):
    return build_schedule(
        meeting(1, names, "2024-05-24"),
        [{"meeting_key": 1, "meeting_name": "Monaco Grand Prix", "circuit_type": "Temporary - Street"}],
        year,
    )


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


def run(raw_dir, client, **kwargs):
    options = {
        "schedule": schedule(),
        "circuits": CIRCUITS,
        "aliases": ALIASES,
        "ingest": fake_ingest(),
        "raw_dir": raw_dir,
    }
    return ensure_race_entries(2024, 1, client=client, **(options | kwargs))


def test_first_call_creates_the_race_and_entries(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    result = run(raw_dir, client)
    assert (result.year, result.round, result.entries, result.created) == (2024, 1, 3, True)
    race = client.tables["races"][0]
    assert race["race_id"] == result.race_id
    assert race["event_name"] == "Monaco Grand Prix"
    assert (race["quali_date"], race["race_date"]) == ("2024-05-27", "2024-05-28")
    assert race["is_sprint_weekend"] is False
    assert race["circuit_id"] == client.tables["circuits"][0]["circuit_id"]
    assert client.tables["circuits"][0] == {
        "circuit_id": 1,
        "name": "Monte Carlo",
        "country": "Monaco",
        "circuit_type": "street",
    }
    assert {d["code"] for d in client.tables["drivers"]} == {"VER", "NOR", "TSU"}


def test_second_call_changes_nothing(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    first = run(raw_dir, client)
    sizes = {t: client.count(t) for t in ("circuits", "drivers", "teams", "races", "race_entries")}
    second = run(raw_dir, client)
    assert {t: client.count(t) for t in sizes} == sizes
    assert (second.race_id, second.entries, second.created) == (first.race_id, 3, False)


def test_a_new_driver_in_an_existing_race_counts_as_created(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    run(raw_dir, client)
    shutil.rmtree(raw_dir / "2024")
    write_q(raw_dir, drivers=[*DRIVERS, (81, "PIA", "Oscar PIASTRI", "McLaren")])
    assert run(raw_dir, client).created is True
    assert client.count("race_entries") == 4


def test_calls_ingest_session_for_the_q_session(raw_dir):
    write_q(raw_dir)
    ingest = fake_ingest()
    run(raw_dir, FakeSupabase(), ingest=ingest)
    assert ingest.calls == [(2024, 1, "Q")]


def test_a_failed_ingest_raises_and_writes_nothing(raw_dir):
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match="2024 round 1"):
        run(raw_dir, client, ingest=fake_ingest("failed"))
    assert client.writes == []


def test_sprint_weekend_flag(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    run(raw_dir, client, schedule=schedule(SPRINT_2024))
    assert client.tables["races"][0]["is_sprint_weekend"] is True


def test_the_team_is_aliased(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    run(raw_dir, client)
    assert {t["name"] for t in client.tables["teams"]} == {"Red Bull Racing", "McLaren", "RB"}


def test_unknown_circuit_raises_with_its_key_and_writes_nothing(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match="circuit_key 101"):
        run(raw_dir, client, circuits={})
    assert client.writes == []


def test_a_race_missing_from_the_schedule_raises(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match="schedule"):
        run(raw_dir, client, schedule=schedule().iloc[0:0])
    assert client.writes == []


@pytest.mark.parametrize(
    "rows, errors",
    [
        ({"drivers": 0, "session_result": 3}, None),
        ({"drivers": 3, "session_result": 0}, None),
        (None, ["drivers: column 'team_name' not in dataframe"]),
        (None, ["session_result: table has no rows (failure case: False)"]),
    ],
)
def test_unusable_drivers_or_session_result_raises_and_writes_nothing(raw_dir, rows, errors):
    write_q(raw_dir, rows=rows, errors=errors)
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match="2024 round 1"):
        run(raw_dir, client)
    assert client.writes == []


def test_a_q_session_invalid_only_because_of_laps_is_accepted(raw_dir):
    write_q(raw_dir, errors=["laps: table has no rows (failure case: False)"])
    assert run(raw_dir, FakeSupabase()).entries == 3


def test_a_missing_q_session_raises(raw_dir):
    with pytest.raises(ReferenceDataError, match="not on disk"):
        run(raw_dir, FakeSupabase())


def test_only_drivers_with_a_session_result_get_an_entry(raw_dir):
    write_q(raw_dir, result_numbers=[1, 4])
    client = FakeSupabase()
    assert run(raw_dir, client).entries == 2
    assert {d["code"] for d in client.tables["drivers"]} == {"VER", "NOR"}


def test_a_driver_without_a_team_raises_and_writes_nothing(raw_dir):
    write_q(raw_dir, drivers=[*DRIVERS[:2], (22, "TSU", "Yuki TSUNODA", None)])
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match="no code or team"):
        run(raw_dir, client)
    assert client.writes == []


def test_one_driver_code_with_two_names_raises_and_writes_nothing(raw_dir):
    write_q(raw_dir, drivers=[*DRIVERS, (99, "VER", "Someone ELSE", "McLaren")])
    client = FakeSupabase()
    with pytest.raises(ReferenceDataError, match=r"VER.*two different names"):
        run(raw_dir, client)
    assert client.writes == []


def test_a_known_code_with_a_new_name_is_updated(raw_dir):
    write_q(raw_dir)
    client = FakeSupabase()
    run(raw_dir, client)
    client.tables["drivers"][0]["full_name"] = "Old Name"
    run(raw_dir, client)
    assert client.tables["drivers"][0]["full_name"] == "Max VERSTAPPEN"
    assert client.count("drivers") == 3
