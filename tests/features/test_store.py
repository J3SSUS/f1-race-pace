from dataclasses import replace

import pytest
from fake_supabase import FakeSupabase

from features.catalog import load_catalog
from features.race_info import RaceInfo
from features.store import (
    PAGE_SIZE,
    LoadResult,
    StoreError,
    load_reference_index,
    read_table,
    write_feature_set,
)

SECRET = "service-key-that-must-not-leak"


def entries(count):
    return {"race_entries": [{"entry_id": i, "race_id": 1} for i in range(1, count + 1)]}


def test_read_table_returns_every_row_in_order_across_pages():
    client = FakeSupabase(entries(2500))
    rows = read_table(client, "race_entries")
    assert [row["entry_id"] for row in rows] == list(range(1, 2501))
    assert client.reads == [("race_entries", 0, 1000), ("race_entries", 1000, 1000), ("race_entries", 2000, 500)]


def test_a_table_of_exactly_one_page_costs_one_extra_empty_read():
    client = FakeSupabase(entries(PAGE_SIZE))
    assert len(read_table(client, "race_entries")) == PAGE_SIZE
    assert client.reads == [("race_entries", 0, 1000), ("race_entries", 1000, 0)]


def test_an_empty_or_missing_table_gives_no_rows():
    assert read_table(FakeSupabase(), "ground_truth") == []


def test_read_table_passes_the_columns_through():
    client = FakeSupabase(entries(3))
    assert read_table(client, "race_entries", "entry_id") == [{"entry_id": 1}, {"entry_id": 2}, {"entry_id": 3}]


def test_a_failed_read_names_the_table_but_not_the_key():
    class Broken(FakeSupabase):
        def table(self, name):
            raise ConnectionError(f"boom with key {SECRET}")

    with pytest.raises(StoreError) as error:
        read_table(Broken(), "feature_values")
    assert "feature_values" in str(error.value)
    assert SECRET not in str(error.value) and SECRET not in repr(error.value.__cause__)


def test_load_result_is_a_frozen_record_of_the_run():
    result = LoadResult("abc", races=2, entries=16, feature_values=240, ground_truth=16, valid_rows=14)
    assert result.valid_rows == 14
    with pytest.raises(AttributeError):
        result.races = 3


CATALOG = load_catalog()


def snapshot(client):
    return {name: [dict(row) for row in rows] for name, rows in client.tables.items()}


def test_write_feature_set_fills_features_the_set_and_its_positions():
    client = FakeSupabase()
    written = write_feature_set(client, CATALOG)
    assert [row["name"] for row in client.tables["features"]] == CATALOG.names
    assert {
        row["name"]: (row["feature_group"], row["dtype"], row["known_on_saturday"]) for row in client.tables["features"]
    } == {f.name: (f.group, f.dtype, f.known_on_saturday) for f in CATALOG.features}
    assert client.tables["feature_sets"] == [{"feature_set_id": 1, "version": "v1", "schema_hash": CATALOG.schema_hash}]
    positions = {row["feature_id"]: row["position"] for row in client.tables["feature_set_items"]}
    assert [positions[written.feature_ids[name]] for name in CATALOG.names] == list(range(1, 16))
    assert written.feature_set_id == 1
    assert set(written.feature_ids) == set(CATALOG.names)


def test_rerunning_write_feature_set_changes_nothing():
    client = FakeSupabase()
    first = write_feature_set(client, CATALOG)
    before = snapshot(client)
    assert write_feature_set(client, CATALOG) == first
    assert snapshot(client) == before


def test_another_schema_hash_stops_before_writing_and_keeps_v1_as_it_was():
    client = FakeSupabase()
    write_feature_set(client, CATALOG)
    before, writes = snapshot(client), list(client.writes)
    with pytest.raises(StoreError, match="v1.*new version"):
        write_feature_set(client, replace(CATALOG, schema_hash="another"))
    assert snapshot(client) == before and client.writes == writes


def test_a_feature_set_with_an_empty_hash_is_not_overwritten_either():
    client = FakeSupabase({"feature_sets": [{"feature_set_id": 1, "version": "v1", "schema_hash": None}]})
    with pytest.raises(StoreError):
        write_feature_set(client, CATALOG)
    assert client.writes == [] and client.count("features") == 0


def test_a_new_version_gets_its_own_set_and_shares_the_features():
    client = FakeSupabase()
    write_feature_set(client, CATALOG)
    second = write_feature_set(client, replace(CATALOG, schema_hash="another"), version="v2")
    assert second.feature_set_id == 2
    assert client.count("features") == 15 and client.count("feature_set_items") == 30


def test_a_feature_that_exists_with_another_definition_stops_before_writing():
    client = FakeSupabase()
    write_feature_set(client, CATALOG)
    before, writes = snapshot(client), list(client.writes)
    features = (replace(CATALOG.features[0], group="practice", dtype="int"), *CATALOG.features[1:])
    with pytest.raises(StoreError, match="quali_delta_pct.*feature_group, dtype.*new name"):
        write_feature_set(client, replace(CATALOG, features=features, schema_hash="another"), version="v2")
    assert snapshot(client) == before and client.writes == writes


def test_a_new_version_can_add_a_feature_and_keeps_the_old_ones_as_they_were():
    client = FakeSupabase()
    write_feature_set(client, CATALOG)
    extra = replace(CATALOG.features[0], name="grid_penalty", position=16)
    second = write_feature_set(
        client, replace(CATALOG, features=(*CATALOG.features, extra), schema_hash="another"), "v2"
    )
    assert client.count("features") == 16 and client.count("feature_set_items") == 31
    assert len(second.feature_ids) == 16


def reference():
    return {
        "circuits": [
            {"circuit_id": 1, "name": "Monaco", "circuit_type": "street"},
            {"circuit_id": 2, "name": "Silverstone", "circuit_type": "permanent"},
        ],
        "drivers": [{"driver_id": 1, "code": "VER"}, {"driver_id": 2, "code": "NOR"}, {"driver_id": 3, "code": "LEC"}],
        "races": [
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


def test_the_reference_index_gives_the_race_info_and_entry_of_each_driver_in_order():
    index = load_reference_index(FakeSupabase(reference()))
    assert list(index) == [(2024, 1), (2024, 2)]
    assert index[(2024, 1)].info == RaceInfo(2024, 1, "Monaco", "street")
    assert index[(2024, 1)].entry_ids == {"VER": 100, "NOR": 101}
    assert index[(2024, 2)].info == RaceInfo(2024, 2, "Silverstone", "permanent")
    assert index[(2024, 2)].entry_ids == {"NOR": 200, "LEC": 201}


def test_a_race_without_entries_points_to_load_reference():
    tables = reference()
    tables["race_entries"] = [e for e in tables["race_entries"] if e["race_id"] != 20]
    with pytest.raises(StoreError, match=r"2024 round 2 has no race_entries.*scripts/load_reference\.py"):
        load_reference_index(FakeSupabase(tables))


def test_a_race_with_a_missing_circuit_points_to_load_reference():
    tables = reference()
    tables["circuits"] = tables["circuits"][:1]
    with pytest.raises(StoreError, match=r"2024 round 2 has a circuit.*scripts/load_reference\.py"):
        load_reference_index(FakeSupabase(tables))


def test_an_entry_with_a_missing_driver_points_to_load_reference():
    tables = reference()
    tables["drivers"] = tables["drivers"][:2]
    with pytest.raises(StoreError, match=r"entry 201 has a driver.*scripts/load_reference\.py"):
        load_reference_index(FakeSupabase(tables))


def test_the_reference_index_reads_past_the_first_page():
    tables = reference()
    tables["race_entries"] += [{"entry_id": 1000 + i, "race_id": 20, "driver_id": 3} for i in range(1200)]
    index = load_reference_index(FakeSupabase(tables))
    assert index[(2024, 2)].entry_ids["LEC"] == 2199
