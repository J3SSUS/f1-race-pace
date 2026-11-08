import pytest
from fake_supabase import FakeSupabase
from live_helpers import ENTRY, reference
from synthetic_race import DRIVERS

from features.race_info import RaceInfo
from features.store import StoreError, ground_truth_rows
from features.target import race_target
from live.ground_truth import GroundTruthError, GroundTruthResult, write_ground_truth

FIELD_MEDIAN = 91.2  # the `clean` scenario of the synthetic race, derived by hand in `synthetic_race`
VALID = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}  # HHH has 12 clean laps and GGG 8: below the 15 needed


def run(raw_dir, client, **kw):
    return write_ground_truth(2024, 1, client=client, raw_dir=raw_dir, sleep=lambda s: None, **kw)


def stored(client) -> dict[int, dict]:
    return {row["entry_id"]: row for row in client.tables["ground_truth"]}


def test_one_row_per_entry_with_the_hand_derived_values(raw):
    client = FakeSupabase(reference())
    run(raw, client, run_id="r1")
    rows = stored(client)
    assert len(rows) == 8
    for driver in DRIVERS:
        row = rows[ENTRY[driver.code]]
        assert row["dag_run_id"] == "r1"
        if driver.code in VALID:
            assert row["is_valid_row"] is True
            assert row["actual_pace_delta_pct"] == pytest.approx((driver.base / FIELD_MEDIAN - 1) * 100)
        else:
            assert row["is_valid_row"] is False
            assert row["actual_pace_delta_pct"] is None
    assert rows[ENTRY["AAA"]]["clean_laps"] == 27
    assert rows[ENTRY["DDD"]]["clean_laps"] == 29
    assert (rows[ENTRY["HHH"]]["clean_laps"], rows[ENTRY["GGG"]]["clean_laps"]) == (12, 8)  # filled when invalid


def test_the_rows_equal_those_of_race_target_and_ground_truth_rows(raw):
    client = FakeSupabase(reference())
    run(raw, client, run_id="r1")
    expected = ground_truth_rows(race_target(RaceInfo(2024, 1, "Synthetic", "permanent"), raw), ENTRY, "r1")
    got = [{k: v for k, v in r.items()} for r in client.tables["ground_truth"]]
    assert sorted(got, key=lambda r: r["entry_id"]) == sorted(expected, key=lambda r: r["entry_id"])


def test_the_result_reports_what_was_written(raw):
    client = FakeSupabase(reference())
    result = run(raw, client, run_id="r1")
    assert result == GroundTruthResult(2024, 1, "r1", 8, 6, {"Q": "skipped", "R": "skipped"})


def test_a_run_id_is_generated_when_none_is_given(raw):
    client = FakeSupabase(reference())
    result = run(raw, client)
    assert len(result.run_id) == 32
    assert {r["dag_run_id"] for r in client.tables["ground_truth"]} == {result.run_id}


def test_it_writes_once_and_only_to_ground_truth(raw):
    client = FakeSupabase(reference())
    run(raw, client)
    assert client.writes == [("ground_truth", 8)]


def test_a_driver_without_a_race_entry_raises_before_any_write(raw):
    tables = reference()
    tables["race_entries"] = [e for e in tables["race_entries"] if e["entry_id"] != ENTRY["FFF"]]
    client = FakeSupabase(tables)
    with pytest.raises(StoreError, match=r"race 2024 round 1: driver FFF.*scripts/load_reference\.py"):
        run(raw, client)
    assert client.writes == []


def test_running_it_twice_gives_the_same_rows_and_only_the_run_changes(raw):
    client = FakeSupabase(reference())
    run(raw, client, run_id="r1")
    first = stored(client)
    run(raw, client, run_id="r2")
    second = stored(client)
    assert client.count("ground_truth") == 8
    for entry_id, row in second.items():
        assert row["dag_run_id"] == "r2"
        assert {k: v for k, v in row.items() if k != "dag_run_id"} == {
            k: v for k, v in first[entry_id].items() if k != "dag_run_id"
        }


def test_a_supabase_error_is_reported_by_type_without_its_text(raw):
    class Broken(FakeSupabase):
        def table(self, name):
            query = super().table(name)
            if name == "ground_truth":

                def boom(*args, **kwargs):
                    raise ValueError("SECRET-SERVICE-KEY in the response body")

                query.upsert = boom
            return query

    with pytest.raises(GroundTruthError) as error:
        run(raw, Broken(reference()))
    assert "ValueError" in str(error.value) and "ground_truth" in str(error.value)
    assert "SECRET" not in str(error.value)


def test_a_target_that_cannot_be_computed_names_the_race_and_writes_nothing(raw):
    (raw / "2024" / "01" / "R" / "laps.parquet").unlink()
    client = FakeSupabase(reference())
    with pytest.raises(GroundTruthError, match=r"race 2024 round 1: the target could not be computed"):
        run(raw, client)
    assert client.writes == []


def test_a_race_that_is_not_loaded_stops_before_ingesting(raw):
    calls = []

    def loader(year, rnd, session):
        calls.append(session)
        raise AssertionError("must not be called")

    tables = reference()
    tables["race_entries"] = []
    with pytest.raises(GroundTruthError, match=r"no race_entries"):
        run(raw, FakeSupabase(tables), loader=loader)
    assert calls == []
