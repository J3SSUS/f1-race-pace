import pytest
from fake_supabase import FakeSupabase

from features.race_info import RaceInfo
from live.ground_truth import GroundTruthError, read_race_ref


def reference() -> dict[str, list[dict]]:
    """Race 2024 round 1 (Monaco) and round 2 (Silverstone) with entries; round 3 is loaded without entries."""
    return {
        "circuits": [
            {"circuit_id": 1, "name": "Monaco", "circuit_type": "street"},
            {"circuit_id": 2, "name": "Silverstone", "circuit_type": "permanent"},
        ],
        "drivers": [{"driver_id": 1, "code": "VER"}, {"driver_id": 2, "code": "NOR"}, {"driver_id": 3, "code": "LEC"}],
        "races": [
            {"race_id": 30, "year": 2024, "round": 3, "circuit_id": 2},
            {"race_id": 20, "year": 2024, "round": 2, "circuit_id": 2},
            {"race_id": 10, "year": 2024, "round": 1, "circuit_id": 1},
        ],
        "race_entries": [
            {"entry_id": 100, "race_id": 10, "driver_id": 1},
            {"entry_id": 101, "race_id": 10, "driver_id": 2},
            {"entry_id": 200, "race_id": 20, "driver_id": 2},
            {"entry_id": 201, "race_id": 20, "driver_id": 3},
        ],
    }


def test_it_returns_the_race_info_and_the_entry_of_each_driver_of_that_race_only():
    client = FakeSupabase(reference())
    ref = read_race_ref(client, 2024, 2)
    assert ref.info == RaceInfo(2024, 2, "Silverstone", "permanent")
    assert ref.entry_ids == {"NOR": 200, "LEC": 201}
    assert read_race_ref(client, 2024, 1).entry_ids == {"VER": 100, "NOR": 101}


def test_a_race_loaded_without_entries_does_not_stop_another_race():
    assert read_race_ref(FakeSupabase(reference()), 2024, 1).entry_ids == {"VER": 100, "NOR": 101}


def test_it_reads_past_the_first_page_of_entries():
    tables = reference()
    tables["race_entries"] = [{"entry_id": 1000 + i, "race_id": 10, "driver_id": 1} for i in range(2500)] + tables[
        "race_entries"
    ][2:]
    assert read_race_ref(FakeSupabase(tables), 2024, 2).entry_ids == {"NOR": 200, "LEC": 201}


def test_a_race_that_is_not_in_races_names_the_race_and_load_reference():
    with pytest.raises(GroundTruthError, match=r"2024 round 9 is not in races.*scripts/load_reference\.py"):
        read_race_ref(FakeSupabase(reference()), 2024, 9)


def test_a_race_without_entries_names_the_race_and_load_reference():
    with pytest.raises(GroundTruthError, match=r"2024 round 3 has no race_entries.*scripts/load_reference\.py"):
        read_race_ref(FakeSupabase(reference()), 2024, 3)


def test_a_race_with_a_missing_circuit_names_the_race():
    tables = reference()
    tables["circuits"] = tables["circuits"][:1]
    with pytest.raises(GroundTruthError, match=r"2024 round 2 has a circuit that is not in circuits"):
        read_race_ref(FakeSupabase(tables), 2024, 2)


def test_an_entry_with_a_missing_driver_names_the_race_and_the_entry():
    tables = reference()
    tables["drivers"] = tables["drivers"][:2]  # LEC is gone
    with pytest.raises(GroundTruthError, match=r"2024 round 2: entry 201 has a driver that is not in drivers"):
        read_race_ref(FakeSupabase(tables), 2024, 2)


def test_it_writes_nothing():
    client = FakeSupabase(reference())
    read_race_ref(client, 2024, 1)
    assert client.writes == []
