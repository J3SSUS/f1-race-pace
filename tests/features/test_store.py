import pytest
from fake_supabase import FakeSupabase

from features.store import PAGE_SIZE, LoadResult, StoreError, read_table

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
