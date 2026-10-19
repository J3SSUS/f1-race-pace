import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_schedule import NORMAL, meeting

from ingestion.reference_config import canonical_team, load_circuits, load_team_aliases
from ingestion.schedule import build_schedule

SEED_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_circuits.py"

MONACO = {"circuit_key": 22, "name": "Monte Carlo", "country": "Monaco", "circuit_type": "street"}
SAKHIR = {"circuit_key": 63, "name": "Sakhir", "country": "Bahrain", "circuit_type": "permanent"}


def write(tmp_path, entries):
    path = tmp_path / "circuits.yml"
    path.write_text(yaml.safe_dump(entries))
    return path


def test_accepts_a_good_file(tmp_path):
    circuits = load_circuits(write(tmp_path, [MONACO, SAKHIR | {"circuit_type": "hybrid"}]))
    assert circuits[22].name == "Monte Carlo"
    assert circuits[22].circuit_type == "street"
    assert circuits[63].circuit_type == "hybrid"


def test_rejects_review(tmp_path):
    with pytest.raises(ValidationError, match="review"):
        load_circuits(write(tmp_path, [MONACO | {"circuit_type": "review"}]))


def test_rejects_an_unknown_type(tmp_path):
    with pytest.raises(ValidationError):
        load_circuits(write(tmp_path, [MONACO | {"circuit_type": "oval"}]))


def test_rejects_a_duplicate_circuit_key(tmp_path):
    with pytest.raises(ValueError, match="duplicate circuit_key 22"):
        load_circuits(write(tmp_path, [MONACO, SAKHIR | {"circuit_key": 22}]))


def test_rejects_a_duplicate_name(tmp_path):
    with pytest.raises(ValueError, match="duplicate circuit name"):
        load_circuits(write(tmp_path, [MONACO, SAKHIR | {"name": "Monte Carlo"}]))


@pytest.mark.parametrize("missing", ["circuit_key", "name", "country", "circuit_type"])
def test_rejects_a_missing_key(tmp_path, missing):
    entry = {k: v for k, v in MONACO.items() if k != missing}
    with pytest.raises(ValidationError, match=missing):
        load_circuits(write(tmp_path, [entry]))


@pytest.fixture(scope="module")
def seed():
    spec = importlib.util.spec_from_file_location("seed_circuits", SEED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule_with_types(*types):
    """One normal weekend per type, each at its own circuit (key 101, 102, ...)."""
    sessions, meetings = [], []
    for i, circuit_type in enumerate(types, start=1):
        sessions += meeting(i, NORMAL, f"2024-0{i}-01")
        meetings.append({"meeting_key": i, "meeting_name": f"Grand Prix {i}", "circuit_type": circuit_type})
    return build_schedule(sessions, meetings, 2024)


def test_draft_maps_openf1_types_and_leaves_the_rest_for_review(seed):
    draft = seed.build_draft(schedule_with_types("Permanent", "Temporary - Street", "Temporary - Road", None))
    assert [e["circuit_type"] for e in draft] == ["permanent", "street", "review", "review"]
    assert draft[0] == {"circuit_key": 101, "name": "Circuit 1", "country": "Country 1", "circuit_type": "permanent"}


def test_draft_has_one_entry_per_circuit(seed):
    draft = seed.build_draft(schedule_with_types("Permanent", "Permanent"))
    assert [e["circuit_key"] for e in draft] == [101, 102]
    assert len(seed.build_draft(schedule_with_types("Permanent").iloc[:0])) == 0


def test_seed_writes_a_file_the_loader_rejects_until_reviewed(seed, tmp_path):
    path = tmp_path / "circuits.yml"
    sched = schedule_with_types("Permanent", "Temporary - Road")
    messages = []
    code = seed.main(
        [], schedule_loader=lambda a, b: sched, path=path, now=datetime(2024, 12, 1, tzinfo=UTC), out=messages.append
    )
    assert code == 0
    assert "1 still marked review" in messages[0]
    with pytest.raises(ValidationError, match="review"):
        load_circuits(path)

    entries = yaml.safe_load(path.read_text())
    entries[1]["circuit_type"] = "hybrid"
    path.write_text(yaml.safe_dump(entries))
    assert set(load_circuits(path)) == {101, 102}


def test_seed_does_not_overwrite_a_reviewed_file_without_force(seed, tmp_path):
    path = write(tmp_path, [MONACO])
    sched = schedule_with_types("Permanent")
    kwargs = {"schedule_loader": lambda a, b: sched, "path": path, "out": lambda m: None}
    assert seed.main([], **kwargs) == 1
    assert yaml.safe_load(path.read_text()) == [MONACO]
    assert seed.main(["--force"], **kwargs) == 0
    assert yaml.safe_load(path.read_text())[0]["circuit_key"] == 101


def test_repo_teams_file_maps_the_renamed_lineages():
    aliases = load_team_aliases()
    assert canonical_team("Alfa Romeo", aliases) == "Kick Sauber"
    assert canonical_team("AlphaTauri", aliases) == "RB"
    assert canonical_team("Racing Bulls", aliases) == "RB"
    assert canonical_team("Kick Sauber", aliases) == "Kick Sauber"
    assert canonical_team("RB", aliases) == "RB"


def test_canonical_team_keeps_a_name_that_is_not_aliased():
    assert canonical_team("Ferrari", {"Alfa Romeo": "Kick Sauber"}) == "Ferrari"
    assert canonical_team("Ferrari") == "Ferrari"


def test_rejects_a_chained_alias(tmp_path):
    path = tmp_path / "teams.yml"
    path.write_text(yaml.safe_dump({"aliases": {"A": "B", "B": "C"}}))
    with pytest.raises(ValueError, match="also an alias"):
        load_team_aliases(path)


def test_rejects_a_teams_file_without_aliases(tmp_path):
    path = tmp_path / "teams.yml"
    path.write_text("Alfa Romeo: Kick Sauber\n")
    with pytest.raises(ValueError, match="aliases"):
        load_team_aliases(path)
