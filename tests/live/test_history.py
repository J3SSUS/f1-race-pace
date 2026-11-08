import pandas as pd
from fake_supabase import FakeSupabase

from features.form import HISTORY_COLUMNS
from live.history import missing_ground_truth, read_history


def tables() -> dict[str, list[dict]]:
    """2024 rounds 1 to 3 and 2025 round 1, with two drivers each. Ground truth exists for 2024 rounds 1 and 3
    (round 3 with one invalid row and one row without a target) and for 2025 round 1; 2024 round 2 has none."""
    circuits = [
        {"circuit_id": 1, "name": "Monaco"},
        {"circuit_id": 2, "name": "Silverstone"},
    ]
    drivers = [{"driver_id": 1, "code": "VER"}, {"driver_id": 2, "code": "NOR"}]
    teams = [{"team_id": 1, "name": "Red Bull"}, {"team_id": 2, "name": "McLaren"}]
    races = [
        {"race_id": 40, "year": 2025, "round": 1, "circuit_id": 1},  # listed out of order on purpose
        {"race_id": 10, "year": 2024, "round": 1, "circuit_id": 1},
        {"race_id": 30, "year": 2024, "round": 3, "circuit_id": 2},
        {"race_id": 20, "year": 2024, "round": 2, "circuit_id": 2},
    ]
    entries, truth = [], []
    for race in races:
        base = race["race_id"] * 10
        entries += [
            {"entry_id": base + 1, "race_id": race["race_id"], "driver_id": 1, "team_id": 1},
            {"entry_id": base + 2, "race_id": race["race_id"], "driver_id": 2, "team_id": 2},
        ]
    for entry_id, value, valid in [
        (101, -0.5, True),
        (102, 0.4, True),
        (301, 0.1, True),
        (302, 0.9, False),  # not a valid row
        (401, -0.7, True),
        (402, None, True),  # valid flag but no target
    ]:
        truth.append({"entry_id": entry_id, "actual_pace_delta_pct": value, "is_valid_row": valid})
    return {
        "circuits": circuits,
        "drivers": drivers,
        "teams": teams,
        "races": races,
        "race_entries": entries,
        "ground_truth": truth,
    }


def test_missing_ground_truth_lists_the_earlier_races_without_it_in_order():
    t = tables()
    t["ground_truth"] = [g for g in t["ground_truth"] if g["entry_id"] not in (101, 301)]  # round 1 and 3 lose one row
    assert missing_ground_truth(FakeSupabase(t), 2025, 1) == [(2024, 1), (2024, 2), (2024, 3)]


def test_a_complete_race_is_not_listed():
    assert missing_ground_truth(FakeSupabase(tables()), 2025, 1) == [(2024, 2)]


def test_one_entry_without_a_row_is_enough_to_list_a_race():
    t = tables()
    t["ground_truth"] = [g for g in t["ground_truth"] if g["entry_id"] != 102]
    assert missing_ground_truth(FakeSupabase(t), 2025, 1) == [(2024, 1), (2024, 2)]


def test_the_same_race_and_later_races_are_ignored():
    client = FakeSupabase(tables())
    assert missing_ground_truth(client, 2024, 2) == []  # round 2 itself has none but is not earlier
    assert missing_ground_truth(client, 2024, 3) == [(2024, 2)]
    assert missing_ground_truth(client, 2024, 1) == []


def test_a_race_without_entries_is_not_listed():
    t = tables()
    t["races"].append({"race_id": 50, "year": 2024, "round": 4, "circuit_id": 1})
    assert missing_ground_truth(FakeSupabase(t), 2025, 1) == [(2024, 2)]


def test_read_history_returns_exactly_the_history_columns_with_names_and_codes():
    history = read_history(FakeSupabase(tables()), 2025, 1)
    assert list(history.columns) == HISTORY_COLUMNS
    assert history.to_dict("records") == [
        {"year": 2024, "round": 1, "circuit": "Monaco", "driver": "NOR", "team": "McLaren", "race_pace_delta_pct": 0.4},
        {
            "year": 2024,
            "round": 1,
            "circuit": "Monaco",
            "driver": "VER",
            "team": "Red Bull",
            "race_pace_delta_pct": -0.5,
        },
        {
            "year": 2024,
            "round": 3,
            "circuit": "Silverstone",
            "driver": "VER",
            "team": "Red Bull",
            "race_pace_delta_pct": 0.1,
        },
    ]


def test_read_history_keeps_only_valid_rows_with_a_target():
    history = read_history(FakeSupabase(tables()), 2026, 1)
    assert set(history["race_pace_delta_pct"]) == {-0.5, 0.4, 0.1, -0.7}
    assert history["race_pace_delta_pct"].notna().all()


def test_read_history_takes_only_earlier_races():
    history = read_history(FakeSupabase(tables()), 2024, 3)
    assert set(zip(history["year"], history["round"], strict=True)) == {(2024, 1)}


def test_the_team_is_the_one_of_the_entry_not_a_fixed_one():
    t = tables()
    t["race_entries"] = [dict(e, team_id=2) if e["entry_id"] == 101 else e for e in t["race_entries"]]
    history = read_history(FakeSupabase(t), 2025, 1)
    assert history[history["driver"] == "VER"]["team"].tolist() == ["McLaren", "Red Bull"]


def test_the_first_race_has_an_empty_history_with_the_columns():
    history = read_history(FakeSupabase(tables()), 2024, 1)
    assert history.empty and list(history.columns) == HISTORY_COLUMNS


def test_it_reads_past_the_first_page():
    t = tables()
    for i in range(2500):  # many valid rows of round 1 on other entries
        t["race_entries"].append({"entry_id": 5000 + i, "race_id": 10, "driver_id": 1, "team_id": 1})
        t["ground_truth"].append({"entry_id": 5000 + i, "actual_pace_delta_pct": 0.0, "is_valid_row": True})
    history = read_history(FakeSupabase(t), 2025, 1)
    assert len(history) == 3 + 2500
    assert missing_ground_truth(FakeSupabase(t), 2025, 1) == [(2024, 2)]


def test_both_functions_write_nothing():
    client = FakeSupabase(tables())
    missing_ground_truth(client, 2025, 1)
    read_history(client, 2025, 1)
    assert client.writes == []


def test_history_dtypes_match_what_form_features_expects():
    history = read_history(FakeSupabase(tables()), 2025, 1)
    assert isinstance(history, pd.DataFrame)
    assert history["race_pace_delta_pct"].dtype == "float64" and history["year"].dtype == "int64"
