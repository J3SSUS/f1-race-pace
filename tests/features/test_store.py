import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest
from fake_supabase import FakeSupabase

from features import build_race
from features.catalog import load_catalog
from features.form import HISTORY_COLUMNS, empty_history
from features.race_info import RaceInfo
from features.store import (
    PAGE_SIZE,
    LoadResult,
    StoreError,
    feature_value_rows,
    ground_truth_rows,
    load_features,
    load_reference_index,
    read_table,
    read_wide,
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


RUN = "run-1"
CODES = ["AAA", "BBB", "CCC", "HHH", "DDD", "EEE", "FFF", "GGG"]
ENTRY_IDS = {code: 100 + i for i, code in enumerate(CODES)}
TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]
TEAM_OF = dict(zip(CODES, ["Alpha", "Alpha", "Beta", "Delta", "Beta", "Gamma", "Gamma", "Delta"], strict=True))


@pytest.fixture
def built(raw_dir):
    """Round 1 of the synthetic races: HHH (12 clean laps) and GGG (8) are not valid; there is no FP2 or sprint."""
    frame = build_race(RaceInfo(2024, 1, "Monaco", "street"), empty_history(), raw_dir)
    return frame, write_feature_set(FakeSupabase(), CATALOG)


def value_of(rows, written, driver, name):
    (row,) = [r for r in rows if r["entry_id"] == ENTRY_IDS[driver] and r["feature_id"] == written.feature_ids[name]]
    return row


def test_feature_values_have_one_row_per_entry_and_feature(built):
    frame, written = built
    rows = feature_value_rows(frame, ENTRY_IDS, written, CATALOG, RUN)
    assert len(rows) == 8 * 15
    assert {(r["entry_id"], r["feature_id"]) for r in rows} == {
        (e, f) for e in ENTRY_IDS.values() for f in written.feature_ids.values()
    }
    assert all(
        set(r) == {"entry_id", "feature_id", "feature_set_id", "value_num", "value_text", "pipeline_run_id"}
        for r in rows
    )
    assert {(r["feature_set_id"], r["pipeline_run_id"]) for r in rows} == {(written.feature_set_id, RUN)}


def test_numbers_and_flags_go_to_value_num_and_the_category_to_value_text(built):
    frame, written = built
    rows = feature_value_rows(frame, ENTRY_IDS, written, CATALOG, RUN)
    assert value_of(rows, written, "AAA", "reached_q3")["value_num"] == 1.0
    assert value_of(rows, written, "DDD", "reached_q3")["value_num"] == 0.0
    assert value_of(rows, written, "AAA", "rain_flag")["value_num"] == 0.0
    assert value_of(rows, written, "BBB", "grid_position")["value_num"] == 2.0
    assert value_of(rows, written, "GGG", "quali_delta_pct")["value_num"] == 3.0
    assert value_of(rows, written, "AAA", "reached_q3")["value_text"] is None
    circuit_type = value_of(rows, written, "AAA", "circuit_type")
    assert (circuit_type["value_num"], circuit_type["value_text"]) == (None, "street")


def test_a_missing_feature_keeps_its_row_with_both_values_null(built):
    frame, written = built
    rows = feature_value_rows(frame, ENTRY_IDS, written, CATALOG, RUN)
    for name in ("fp_tyre_deg", "sprint_pace_delta_pct", "driver_pace_last3"):
        missing = value_of(rows, written, "AAA", name)
        assert (missing["value_num"], missing["value_text"]) == (None, None)


def test_the_rows_are_plain_python_values_that_json_can_carry(built):
    frame, written = built
    json.dumps(feature_value_rows(frame, ENTRY_IDS, written, CATALOG, RUN))
    json.dumps(ground_truth_rows(frame, ENTRY_IDS, RUN))


def test_ground_truth_has_one_row_per_entry_and_no_target_when_not_valid(built):
    frame, _ = built
    rows = {r["entry_id"]: r for r in ground_truth_rows(frame, ENTRY_IDS, RUN)}
    assert len(rows) == 8
    assert rows[ENTRY_IDS["AAA"]] == {
        "entry_id": 100,
        "actual_pace_delta_pct": pytest.approx(-1.315789, abs=1e-6),
        "clean_laps": 27,
        "is_valid_row": True,
        "dag_run_id": RUN,
    }
    for code, laps in (("HHH", 12), ("GGG", 8)):
        row = rows[ENTRY_IDS[code]]
        assert (row["actual_pace_delta_pct"], row["clean_laps"], row["is_valid_row"]) == (None, laps, False)


def test_a_driver_without_an_entry_names_the_race_and_the_code(built):
    frame, written = built
    entry_ids = {code: i for code, i in ENTRY_IDS.items() if code != "FFF"}
    for build in (
        lambda: feature_value_rows(frame, entry_ids, written, CATALOG, RUN),
        lambda: ground_truth_rows(frame, entry_ids, RUN),
    ):
        with pytest.raises(StoreError, match=r"2024 round 1: driver FFF.*scripts/load_reference\.py"):
            build()


def synthetic_reference(rounds=(1, 2, 3)):
    """The reference tables of Supabase for the synthetic races: one circuit, 8 drivers, 8 entries per race."""
    return {
        "circuits": [{"circuit_id": 1, "name": "Synthetic", "circuit_type": "permanent"}],
        "drivers": [{"driver_id": i + 1, "code": code} for i, code in enumerate(CODES)],
        "teams": [{"team_id": i + 1, "name": name} for i, name in enumerate(TEAMS)],
        "races": [{"race_id": r, "year": 2024, "round": r, "circuit_id": 1} for r in rounds],
        "race_entries": [
            {"entry_id": r * 100 + i, "race_id": r, "driver_id": i + 1, "team_id": TEAMS.index(TEAM_OF[code]) + 1}
            for r in rounds
            for i, code in enumerate(CODES)
        ],
    }


def build_in_order(raw_dir, rounds):
    """The synthetic races built one by one with `build_race`, each with the valid rows of the earlier ones."""
    history, frames = empty_history(), []
    for round in rounds:
        frame = build_race(RaceInfo(2024, round, "Synthetic", "permanent"), history, raw_dir)
        valid = frame[frame["is_valid_row"]][HISTORY_COLUMNS]
        history = valid if history.empty else pd.concat([history, valid], ignore_index=True)
        frames.append(frame)
    return frames


def test_load_features_writes_the_hand_counted_rows_of_two_races(raw_dir):
    client = FakeSupabase(synthetic_reference((1, 2)))
    lines = []
    result = load_features(client, raw_dir=raw_dir, run_id=RUN, log=lines.append)
    first, second = (int(f["is_valid_row"].sum()) for f in build_in_order(raw_dir, (1, 2)))
    assert first == 6  # AAA, BBB, CCC, DDD, EEE, FFF; HHH has 12 clean laps and GGG 8
    assert result == LoadResult(
        RUN, races=2, entries=16, feature_values=240, ground_truth=16, valid_rows=first + second
    )
    assert (client.count("features"), client.count("feature_values"), client.count("ground_truth")) == (15, 240, 16)
    assert client.tables["feature_sets"][0]["version"] == "v1"
    assert lines == [f"2024 round 1: 8 entries, {first} valid", f"2024 round 2: 8 entries, {second} valid"]
    assert {row["pipeline_run_id"] for row in client.tables["feature_values"]} == {RUN}
    assert {row["dag_run_id"] for row in client.tables["ground_truth"]} == {RUN}


def test_a_second_run_leaves_every_table_as_it_was(raw_dir):
    client = FakeSupabase(synthetic_reference((1, 2)))
    load_features(client, raw_dir=raw_dir, run_id=RUN)
    before = snapshot(client)
    load_features(client, raw_dir=raw_dir, run_id=RUN)
    assert snapshot(client) == before


def test_another_run_id_only_changes_the_run_columns(raw_dir):
    client = FakeSupabase(synthetic_reference((1, 2)))
    load_features(client, raw_dir=raw_dir, run_id="one")
    first = snapshot(client)
    result = load_features(client, raw_dir=raw_dir, run_id="two")
    second = snapshot(client)
    assert result.run_id == "two"
    assert {row["pipeline_run_id"] for row in second["feature_values"]} == {"two"}
    for table, column in (("feature_values", "pipeline_run_id"), ("ground_truth", "dag_run_id")):
        assert [{k: v for k, v in r.items() if k != column} for r in first[table]] == [
            {k: v for k, v in r.items() if k != column} for r in second[table]
        ]


def test_the_form_features_of_a_race_use_the_valid_rows_of_the_earlier_ones(raw_dir):
    client = FakeSupabase(synthetic_reference())
    load_features(client, raw_dir=raw_dir, run_id=RUN)
    third = build_in_order(raw_dir, (1, 2, 3))[2].set_index("driver")
    feature_id = {r["name"]: r["feature_id"] for r in client.tables["features"]}["driver_pace_last3"]
    stored = {r["entry_id"]: r["value_num"] for r in client.tables["feature_values"] if r["feature_id"] == feature_id}
    expected = third.loc["AAA", "driver_pace_last3"]
    assert not pd.isna(expected)
    assert stored[300] == pytest.approx(expected, abs=1e-12)


def test_a_race_that_fails_stops_the_load_and_the_later_race_is_not_written(raw_dir):
    shutil.rmtree(raw_dir / "2024" / "02")
    client = FakeSupabase(synthetic_reference())
    with pytest.raises(StoreError, match="race 2024 round 2 failed"):
        load_features(client, raw_dir=raw_dir, run_id=RUN)
    assert {r["entry_id"] // 100 for r in client.tables["feature_values"]} == {1}
    assert {r["entry_id"] // 100 for r in client.tables["ground_truth"]} == {1}


def test_a_missing_entry_stops_before_writing_anything(raw_dir):
    tables = synthetic_reference()
    tables["race_entries"] = [e for e in tables["race_entries"] if e["race_id"] != 2]
    client = FakeSupabase(tables)
    with pytest.raises(StoreError, match=r"scripts/load_reference\.py"):
        load_features(client, raw_dir=raw_dir, run_id=RUN)
    assert client.writes == []


def test_a_driver_without_an_entry_names_the_race_and_the_driver(raw_dir):
    tables = synthetic_reference((1,))
    tables["drivers"] = [d for d in tables["drivers"] if d["code"] != "FFF"]
    tables["race_entries"] = [e for e in tables["race_entries"] if e["driver_id"] != 7]
    with pytest.raises(StoreError, match=r"race 2024 round 1: driver FFF"):
        load_features(FakeSupabase(tables), raw_dir=raw_dir, run_id=RUN)


def test_a_feature_set_with_another_hash_stops_before_any_value(raw_dir):
    tables = synthetic_reference((1,))
    tables["feature_sets"] = [{"feature_set_id": 1, "version": "v1", "schema_hash": "old"}]
    client = FakeSupabase(tables)
    with pytest.raises(StoreError, match="new version"):
        load_features(client, raw_dir=raw_dir, run_id=RUN)
    assert client.count("feature_values") == 0 and client.count("ground_truth") == 0
    assert client.tables["feature_sets"] == [{"feature_set_id": 1, "version": "v1", "schema_hash": "old"}]


def test_the_year_limits_only_change_what_is_written(raw_dir):
    client = FakeSupabase(synthetic_reference())
    result = load_features(client, raw_dir=raw_dir, run_id=RUN, last_year=2023)
    assert (result.races, result.feature_values) == (0, 0) and client.count("feature_values") == 0
    result = load_features(client, raw_dir=raw_dir, run_id=RUN, first_year=2024, last_year=2024)
    assert (result.races, result.feature_values, result.ground_truth) == (3, 360, 24)


def test_read_wide_equals_what_build_race_returned(raw_dir):
    client = FakeSupabase(synthetic_reference())
    load_features(client, raw_dir=raw_dir, run_id=RUN)
    wide = read_wide(client)
    expected = pd.concat(build_in_order(raw_dir, (1, 2, 3)), ignore_index=True)
    expected = expected.sort_values(["year", "round", "driver"], ignore_index=True)
    assert list(wide.columns) == ["entry_id", *expected.columns]
    assert wide["entry_id"].tolist() == [r * 100 + CODES.index(code) for r in (1, 2, 3) for code in sorted(CODES)]
    pd.testing.assert_frame_equal(wide.drop(columns="entry_id"), expected)


def wide_tables(entries=1200):
    """A hand-made feature set: positions opposite to the feature ids, two races inserted in reverse order."""
    rows = range(1, entries + 1)
    return {
        "circuits": [{"circuit_id": 1, "name": "C1"}],
        "drivers": [{"driver_id": 1, "code": "ZZZ"}, {"driver_id": 2, "code": "AAA"}],
        "teams": [{"team_id": 1, "name": "T1"}],
        "races": [
            {"race_id": 2, "year": 2024, "round": 2, "circuit_id": 1},
            {"race_id": 1, "year": 2024, "round": 1, "circuit_id": 1},
        ],
        "race_entries": [
            {"entry_id": i, "race_id": 1 + i % 2, "driver_id": 1 + (i // 2) % 2, "team_id": 1} for i in rows
        ],
        "features": [
            {"feature_id": 1, "name": "x", "dtype": "float"},
            {"feature_id": 2, "name": "kind", "dtype": "category"},
        ],
        "feature_sets": [{"feature_set_id": 1, "version": "v1"}, {"feature_set_id": 2, "version": "v2"}],
        "feature_set_items": [
            {"feature_set_id": 1, "feature_id": 1, "position": 2},
            {"feature_set_id": 1, "feature_id": 2, "position": 1},
            {"feature_set_id": 2, "feature_id": 1, "position": 1},
        ],
        "feature_values": [
            *(
                {"entry_id": i, "feature_id": 1, "feature_set_id": 1, "value_num": float(i), "value_text": None}
                for i in rows
            ),
            *(
                {"entry_id": i, "feature_id": 2, "feature_set_id": 1, "value_num": None, "value_text": f"k{i % 3}"}
                for i in rows
            ),
            {"entry_id": 1, "feature_id": 1, "feature_set_id": 2, "value_num": 999.0, "value_text": None},
        ],
        "ground_truth": [
            {
                "entry_id": i,
                "actual_pace_delta_pct": None if i % 7 == 0 else i / 10,
                "clean_laps": 20,
                "is_valid_row": i % 7 != 0,
            }
            for i in rows
        ],
    }


def test_read_wide_orders_columns_by_position_and_rows_by_race_and_driver():
    wide = read_wide(FakeSupabase(wide_tables()))
    assert len(wide) == 1200  # past the 1,000 rows of one request, and not the value of set v2
    assert list(wide.columns) == [
        *["entry_id", "year", "round", "circuit", "driver", "team"],
        *["race_pace_delta_pct", "clean_laps", "is_valid_row", "kind", "x"],
    ]
    keys = list(zip(wide["year"], wide["round"], wide["driver"], strict=True))
    assert keys == sorted(keys)
    row = wide[wide["entry_id"] == 5].iloc[0]
    assert (row["round"], row["driver"], row["kind"], row["x"], row["race_pace_delta_pct"]) == (
        2,
        "ZZZ",
        "k2",
        5.0,
        0.5,
    )
    assert wide["x"].dtype == "float64" and wide["kind"].dtype == object  # not in features.yml: the dtype of the table
    invalid = wide[~wide["is_valid_row"]]
    assert invalid["race_pace_delta_pct"].isna().all() and len(invalid) == 171


def test_read_wide_of_a_set_that_does_not_exist_or_has_no_values_points_to_load_features():
    with pytest.raises(StoreError, match=r"feature set v9 does not exist.*load_features\.py"):
        read_wide(FakeSupabase(wide_tables(10)), "v9")
    with pytest.raises(StoreError, match=r"feature set v2 has no values.*load_features\.py"):
        read_wide(FakeSupabase({**wide_tables(10), "feature_values": []}), "v2")


def test_entries_with_features_but_no_ground_truth_are_reported():
    tables = wide_tables(10)
    tables["ground_truth"] = [g for g in tables["ground_truth"] if g["entry_id"] not in (3, 4)]
    with pytest.raises(StoreError, match=r"2 entries have features but no ground_truth"):
        read_wide(FakeSupabase(tables))
