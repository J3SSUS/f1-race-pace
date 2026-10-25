import pytest
from fake_supabase import PAGE_LIMIT, FakeSupabase


def test_upserting_the_same_key_twice_leaves_one_row():
    client = FakeSupabase()
    client.table("drivers").upsert([{"code": "VER", "full_name": "Max"}], on_conflict="code").execute()
    client.table("drivers").upsert([{"code": "VER", "full_name": "Max Verstappen"}], on_conflict="code").execute()
    assert client.tables["drivers"] == [{"driver_id": 1, "code": "VER", "full_name": "Max Verstappen"}]


def test_new_rows_get_the_next_identity_and_upsert_returns_them_in_order():
    client = FakeSupabase({"teams": [{"team_id": 7, "name": "Ferrari"}]})
    stored = client.table("teams").upsert([{"name": "McLaren"}, {"name": "Ferrari"}], on_conflict="name").execute().data
    assert stored == [{"team_id": 8, "name": "McLaren"}, {"team_id": 7, "name": "Ferrari"}]


def test_a_composite_conflict_key_matches_on_every_column():
    client = FakeSupabase()
    rows = [{"entry_id": 1, "feature_id": 1, "feature_set_id": 1, "value_num": 0.5}]
    for value in (0.5, 0.7):
        rows[0]["value_num"] = value
        client.table("feature_values").upsert(rows, on_conflict="entry_id,feature_id,feature_set_id").execute()
    client.table("feature_values").upsert(
        [{"entry_id": 1, "feature_id": 2, "feature_set_id": 1, "value_num": 1.0}],
        on_conflict="entry_id,feature_id,feature_set_id",
    ).execute()
    assert [row["value_num"] for row in client.tables["feature_values"]] == [0.7, 1.0]


def test_two_rows_with_the_same_key_in_one_request_are_refused():
    client = FakeSupabase()
    with pytest.raises(ValueError, match="share the key"):
        client.table("teams").upsert([{"name": "Haas"}, {"name": "Haas"}], on_conflict="name").execute()
    assert client.count("teams") == 0


def test_a_read_returns_at_most_1000_rows_per_request():
    client = FakeSupabase({"race_entries": [{"entry_id": i} for i in range(1, 2501)]})
    assert len(client.table("race_entries").select("*").execute().data) == PAGE_LIMIT
    assert len(client.table("race_entries").select("*").range(0, 2499).execute().data) == PAGE_LIMIT
    third = client.table("race_entries").select("*").range(2000, 2999).execute().data
    assert [row["entry_id"] for row in third] == list(range(2001, 2501))
    assert client.reads == [("race_entries", 0, 1000), ("race_entries", 0, 1000), ("race_entries", 2000, 500)]


def test_select_keeps_only_the_asked_columns_and_returns_copies():
    client = FakeSupabase({"races": [{"race_id": 1, "year": 2024, "round": 1, "event_name": "Bahrain"}]})
    data = client.table("races").select("race_id, year").execute().data
    assert data == [{"race_id": 1, "year": 2024}]
    data[0]["year"] = 1999
    assert client.tables["races"][0]["year"] == 2024


def test_reading_a_table_that_does_not_exist_returns_no_rows():
    assert FakeSupabase().table("ground_truth").select("*").execute().data == []
